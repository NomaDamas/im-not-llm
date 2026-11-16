//! Core ML calls: compiling a package and running one function of it on the
//! Neural Engine (`MLComputeUnits.cpuAndNeuralEngine`; the GPU is never used).
//!
//! A [`Function`] owns page-aligned fp16 input buffers and the `MLMultiArray`s
//! wrapping them (no copy on our side), so the steady state of a call is:
//! fill the buffers, predict, read the output.

use crate::package::FunctionSpec;
use half::f16;
use inl_core::{InlError, Result};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};
use std::alloc::Layout;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

fn ns_error(what: &str, error: &NSError) -> InlError {
    InlError::Inference(format!("{what}: {}", error.localizedDescription()))
}

fn file_url(path: &Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()))
}

/// Compile an `.mlpackage` for this Mac; returns the `.mlmodelc` Core ML
/// wrote into a temporary directory (the caller moves it).
pub(crate) fn compile(mlpackage: &Path) -> Result<PathBuf> {
    autoreleasepool(|_| {
        let url = file_url(mlpackage);
        #[allow(deprecated)]
        // SAFETY: a valid file URL; the call is synchronous.
        let compiled = unsafe { MLModel::compileModelAtURL_error(&url) }
            .map_err(|error| ns_error("Core ML compile", &error))?;
        let path = compiled
            .path()
            .ok_or_else(|| InlError::Inference("Core ML compile returned no path".into()))?;
        Ok(PathBuf::from(path.to_string()))
    })
}

/// Page-aligned fp16 storage that Core ML reads in place.
struct Buffer {
    ptr: NonNull<f16>,
    len: usize,
    layout: Layout,
}

// SAFETY: plain owned memory.
unsafe impl Send for Buffer {}

impl Buffer {
    const ALIGN: usize = 16 * 1024;

    fn zeroed(len: usize) -> Self {
        let layout = Layout::from_size_align((len * 2).max(2), Self::ALIGN).expect("buffer layout");
        // SAFETY: the layout has a non-zero size.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw.cast::<f16>())
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len, layout }
    }

    fn as_mut_slice(&mut self) -> &mut [f16] {
        // SAFETY: `len` initialized (zeroed) f16 values owned by self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `zeroed`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr().cast(), self.layout) }
    }
}

fn numbers(values: &[usize]) -> Retained<NSArray<NSNumber>> {
    let numbers: Vec<Retained<NSNumber>> = values
        .iter()
        .map(|&value| NSNumber::new_isize(value as isize))
        .collect();
    NSArray::from_retained_slice(&numbers)
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

/// One function of a compiled package, loaded for the Neural Engine, with its
/// inputs bound to owned buffers.
pub(crate) struct Function {
    // Dropped before the buffers the arrays point into.
    model: Retained<MLModel>,
    provider: Retained<MLDictionaryFeatureProvider>,
    output: Retained<NSString>,
    rows: usize,
    width: usize,
    names: Vec<String>,
    buffers: Vec<Buffer>,
}

// SAFETY: Core ML models and feature providers may be used from any thread;
// a `Function` is only ever used by one thread at a time (`&mut self`).
unsafe impl Send for Function {}

impl Function {
    pub(crate) fn load(mlmodelc: &Path, name: &str, spec: &FunctionSpec) -> Result<Self> {
        autoreleasepool(|_| {
            let what = format!("Core ML load of {name}");
            let [output] = spec.outputs.as_slice() else {
                return Err(InlError::Inference(format!("{what}: expected one output")));
            };
            let (rows, width) = match output.shape.as_slice() {
                [rows, width] => (*rows, *width),
                other => {
                    return Err(InlError::Inference(format!(
                        "{what}: output shape {other:?} is not [rows, width]"
                    )))
                }
            };
            // SAFETY: plain Core ML configuration and load calls.
            let model = unsafe {
                let config = MLModelConfiguration::new();
                config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine);
                config.setFunctionName(Some(&NSString::from_str(name)));
                MLModel::modelWithContentsOfURL_configuration_error(&file_url(mlmodelc), &config)
            }
            .map_err(|error| ns_error(&what, &error))?;
            let mut buffers = Vec::with_capacity(spec.inputs.len());
            let mut keys = Vec::with_capacity(spec.inputs.len());
            let mut values: Vec<Retained<AnyObject>> = Vec::with_capacity(spec.inputs.len());
            for input in &spec.inputs {
                if input.dtype != "float16" {
                    return Err(InlError::Inference(format!(
                        "{what}: input {} is {}, expected float16",
                        input.name, input.dtype
                    )));
                }
                let buffer = Buffer::zeroed(input.elements());
                // SAFETY: the buffer holds `elements` f16 values laid out
                // contiguously and outlives the array (field order of Self).
                let array = unsafe {
                    MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                        MLMultiArray::alloc(),
                        buffer.ptr.cast(),
                        &numbers(&input.shape),
                        MLMultiArrayDataType::Float16,
                        &numbers(&contiguous_strides(&input.shape)),
                        None,
                    )
                }
                .map_err(|error| ns_error(&what, &error))?;
                // SAFETY: wrapping a valid multi-array.
                let value = unsafe { MLFeatureValue::featureValueWithMultiArray(&array) };
                keys.push(NSString::from_str(&input.name));
                values.push(Retained::into_super(Retained::into_super(value)));
                buffers.push(buffer);
            }
            let key_refs: Vec<&NSString> = keys.iter().map(|key| &**key).collect();
            let dictionary = NSDictionary::from_retained_objects(&key_refs, &values);
            // SAFETY: a dictionary of feature values keyed by input name.
            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &dictionary,
                )
            }
            .map_err(|error| ns_error(&what, &error))?;
            Ok(Self {
                model,
                provider,
                output: NSString::from_str(&output.name),
                rows,
                width,
                names: spec.inputs.iter().map(|input| input.name.clone()).collect(),
                buffers,
            })
        })
    }

    /// The buffer of input `name`.
    pub(crate) fn input(&mut self, name: &str) -> &mut [f16] {
        let index = self
            .names
            .iter()
            .position(|candidate| candidate == name)
            .unwrap_or_else(|| panic!("function has no input {name}"));
        self.buffers[index].as_mut_slice()
    }

    /// Run the function on the current inputs; `out` receives the output
    /// row-major as f32.
    pub(crate) fn predict(&mut self, out: &mut Vec<f32>) -> Result<()> {
        autoreleasepool(|_| {
            // SAFETY: the provider's arrays point into live buffers.
            let result = unsafe {
                self.model
                    .predictionFromFeatures_error(ProtocolObject::from_ref(&*self.provider))
            }
            .map_err(|error| ns_error("Core ML prediction", &error))?;
            // SAFETY: reading a named feature of the prediction.
            let array = unsafe { result.featureValueForName(&self.output) }
                .and_then(|value| unsafe { value.multiArrayValue() })
                .ok_or_else(|| InlError::Inference("Core ML prediction has no embedding".into()))?;
            // SAFETY: plain accessors.
            let (shape, strides, data_type) =
                unsafe { (array.shape(), array.strides(), array.dataType()) };
            let shape: Vec<usize> = shape.iter().map(|n| n.as_isize() as usize).collect();
            let strides: Vec<usize> = strides.iter().map(|n| n.as_isize() as usize).collect();
            if shape != [self.rows, self.width] {
                return Err(InlError::Inference(format!(
                    "Core ML output shape {shape:?}, expected [{}, {}]",
                    self.rows, self.width
                )));
            }
            out.clear();
            out.reserve(self.rows * self.width);
            #[allow(deprecated)]
            // SAFETY: the array stays alive inside this pool; indices follow
            // its own strides and data type.
            unsafe {
                let base = array.dataPointer().as_ptr();
                for row in 0..self.rows {
                    for column in 0..self.width {
                        let at = row * strides[0] + column * strides[1];
                        let value = if data_type == MLMultiArrayDataType::Float32 {
                            *base.cast::<f32>().add(at)
                        } else if data_type == MLMultiArrayDataType::Float16 {
                            (*base.cast::<f16>().add(at)).to_f32()
                        } else {
                            return Err(InlError::Inference(format!(
                                "Core ML output data type {data_type:?} is not float32/float16"
                            )));
                        };
                        out.push(value);
                    }
                }
            }
            Ok(())
        })
    }
}
