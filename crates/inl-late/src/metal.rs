//! Metal kernels for the Qwen3.5 gated-delta-net block: the chunked delta rule's triangular inverse,
//! the depthwise causal convolution in front of it, the per-token (recurrent) form of the rule for
//! short sequences, and the gated RMSNorm that follows the rule - one fused kernel each over
//! `late.metal`.
//!
//! One [`Kernels`] is built per model (per device) from the model dtype: it keeps that dtype's conv,
//! recurrent and gated-norm pipelines (`ComputePipeline` is a ref-counted Metal handle) plus the
//! always-f32 inverse, and the ops below clone the pipeline they need and reject a tensor of another
//! dtype. Arithmetic runs in `MTLMathModeSafe`: no product, sum or conversion is contracted or
//! reassociated, so every operation rounds in the tensor's dtype, exactly like the un-fused candle op
//! sequence with its one kernel per op; math *functions* keep candle's default `metal::fast` tables.
//! Deps, `cfg(target_os = "macos")` only: `candle-metal-kernels 0.11.1`, `objc2-metal 0.3.2`,
//! `objc2 0.6.3` (so this file also compiles where those crates are absent).

#[cfg(target_os = "macos")]
use candle_core::{CpuStorage, CustomOp1, CustomOp2, CustomOp3, Layout, MetalStorage, Shape};
use candle_core::{DType, Device, Error, Result, Tensor};
#[cfg(target_os = "macos")]
use candle_metal_kernels::metal::{ComputePipeline, Library};

/// Largest side the inverse kernel's threadgroup tile holds: the model's chunk size. `late.metal`'s
/// `LATE_MAX_SIZE` must match.
#[cfg(target_os = "macos")]
const MAX_SIZE: usize = 64;

/// Largest key/value dimension the recurrent kernel's staged rows and per-column state hold.
/// `late.metal`'s `LATE_MAX_DIM` must match.
#[cfg(target_os = "macos")]
const MAX_DIM: usize = 128;

/// `late.metal`, embedded so the crate needs no build script or resource path.
#[cfg(target_os = "macos")]
const SOURCE: &str = include_str!("late.metal");

/// `late.metal`'s always-f32 inverse kernel; the other pipelines come from the model dtype.
#[cfg(target_os = "macos")]
const INVERSE_KERNEL: &str = "unit_lower_inverse_f32";

/// The compiled `late.metal` pipelines of one Metal device for one model dtype. Build once per model
/// with [`Kernels::new`] and share.
#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(super) struct Kernels {
    /// Held so the compiled functions outlive the pipelines built from them.
    _library: Library,
    /// The dtype the conv, recurrent and gated-norm pipelines were compiled for; the ops reject a
    /// tensor of another dtype. The inverse pipeline is always f32.
    dtype: DType,
    /// The inverse (f32), then the conv, the recurrent rule and the gated norm of `dtype`.
    pipelines: [ComputePipeline; 4],
}

/// Without Metal there is nothing to build; the type still exists so callers need no `cfg`.
#[cfg(not(target_os = "macos"))]
#[derive(Clone)]
pub(super) struct Kernels;

impl Kernels {
    /// Compiles `late.metal` for `device` and builds the four pipelines of `dtype` (`f32`, `f16` or
    /// `bf16`). Metal compile errors, and a dtype the source cannot instantiate, propagate here rather
    /// than on first use.
    pub(super) fn new(device: &Device, dtype: DType) -> Result<Self> {
        if !device.is_metal() {
            return Err(Error::msg("late metal kernels: `device` is not Metal"));
        }
        #[cfg(target_os = "macos")]
        let built = {
            let suffix = match dtype {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                other => {
                    return Err(Error::msg(format!(
                        "late metal kernels: unsupported dtype {other:?}, expected f32, f16 or bf16"
                    )))
                }
            };
            match device {
                Device::Metal(device) => compile(device, suffix).map(|(library, pipelines)| Self {
                    _library: library,
                    dtype,
                    pipelines,
                }),
                _ => unreachable!("`is_metal` was checked above"),
            }
        };
        #[cfg(not(target_os = "macos"))]
        let _ = dtype;
        #[cfg(not(target_os = "macos"))]
        let built = Err(Error::msg("late metal kernels: Metal is macOS only"));
        built
    }
}

/// `X = (I + A)^-1` for strictly lower-triangular `A: [N, size, size]` on a Metal tensor: f32,
/// contiguous, `size` a power of two up to [`MAX_SIZE`] (the chunk the caller is working on).
pub(super) fn unit_lower_inverse(a: &Tensor, size: usize, kernels: &Kernels) -> Result<Tensor> {
    if !a.device().is_metal() {
        return Err(Error::msg("unit_lower_inverse: Metal-only"));
    }
    #[cfg(target_os = "macos")]
    {
        if a.dtype() != DType::F32 {
            return Err(Error::msg(format!(
                "unit_lower_inverse: the inverse kernel is f32, got {:?}",
                a.dtype()
            )));
        }
        let pipeline = kernels.pipelines[0].clone();
        a.contiguous()?
            .apply_op1_no_bwd(&UnitLowerInverse { pipeline, size })
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (size, kernels);
    #[cfg(not(target_os = "macos"))]
    return Err(Error::msg("unit_lower_inverse: Metal is macOS only"));
}

/// `silu(conv1d(xs))` with left zero padding, cut to the input length, on Metal. `xs` is `[B, T, C]` in
/// any layout (it is read through its strides) and `weights` is `[C, K]` (made contiguous); both must be
/// the model dtype the kernels were built for, and the result is `[B, T, C]` in that dtype.
pub(super) fn causal_conv(xs: &Tensor, weights: &Tensor, kernels: &Kernels) -> Result<Tensor> {
    if !xs.device().is_metal() {
        return Err(Error::msg("causal_conv: Metal-only"));
    }
    let weights = weights.contiguous()?;
    #[cfg(target_os = "macos")]
    {
        if xs.dtype() != kernels.dtype {
            return Err(Error::msg(format!(
                "causal_conv: xs is {:?} but the kernels were built for {:?}",
                xs.dtype(),
                kernels.dtype
            )));
        }
        let pipeline = kernels.pipelines[1].clone();
        xs.apply_op2_no_bwd(&weights, &CausalConv { pipeline })
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (weights, kernels);
    #[cfg(not(target_os = "macos"))]
    return Err(Error::msg("causal_conv: Metal is macOS only"));
}

/// The per-token gated delta rule over the convolution's `q`/`k`/`v`. `mixed` is
/// `[B, T, 2*nk*dk + nv*dv]` in the model dtype (the channel split of `qwen3_5.rs`, `nk`-headed `q`/`k`
/// repeated to `nv` as `head / (nv / nk)`), `g` and `beta` are f32 `[B, T, nv]`, and the result is f32
/// `[B, T, nv, dv]`. For the short sequences the caller routes here instead of the chunked form;
/// `nv` must be a multiple of `nk`, `dk`/`dv` at most `MAX_DIM`.
#[allow(clippy::too_many_arguments)] // the reference's own signature
pub(super) fn recurrent_delta(
    mixed: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernels: &Kernels,
) -> Result<Tensor> {
    if !mixed.device().is_metal() {
        return Err(Error::msg("recurrent_delta: Metal-only"));
    }
    let (g, beta) = (g.contiguous()?, beta.contiguous()?);
    #[cfg(target_os = "macos")]
    {
        if mixed.dtype() != kernels.dtype {
            return Err(Error::msg(format!(
                "recurrent_delta: mixed is {:?} but the kernels were built for {:?}",
                mixed.dtype(),
                kernels.dtype
            )));
        }
        let pipeline = kernels.pipelines[2].clone();
        let op = RecurrentDelta {
            pipeline,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        };
        mixed.apply_op3_no_bwd(&g, &beta, &op)
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (g, beta, key_heads, value_heads, key_dim, value_dim, kernels);
    #[cfg(not(target_os = "macos"))]
    return Err(Error::msg("recurrent_delta: Metal is macOS only"));
}

/// The gated RMSNorm `qwen3_5.rs` applies after the delta rule, in one Metal op. `core` is the rule's
/// output already cast to the model dtype and `z` the `in_proj_z` output, both `[B, T, nv, dv]`
/// contiguous in the model dtype; `weight` is `[dv]` in that dtype; `eps` is added to the variance
/// before the square root. The result is the model dtype `[B, T, nv, dv]`.
pub(super) fn gated_norm(
    core: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f64,
    kernels: &Kernels,
) -> Result<Tensor> {
    if !core.device().is_metal() {
        return Err(Error::msg("gated_norm: Metal-only"));
    }
    let weight = weight.contiguous()?;
    #[cfg(target_os = "macos")]
    {
        if core.dtype() != kernels.dtype {
            return Err(Error::msg(format!(
                "gated_norm: core is {:?} but the kernels were built for {:?}",
                core.dtype(),
                kernels.dtype
            )));
        }
        let pipeline = kernels.pipelines[3].clone();
        let op = GatedNorm {
            pipeline,
            eps: eps as f32,
        };
        core.apply_op3_no_bwd(z, &weight, &op)
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (z, weight, eps, kernels);
    #[cfg(not(target_os = "macos"))]
    return Err(Error::msg("gated_norm: Metal is macOS only"));
}

/// Compiles `late.metal` and builds the pipelines of `suffix` (`"f32"`, `"f16"` or `"bf16"`): the
/// always-f32 inverse, then the conv, the recurrent rule and the gated norm.
#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn compile(
    device: &candle_core::MetalDevice,
    suffix: &str,
) -> Result<(Library, [ComputePipeline; 4])> {
    use objc2::available;
    use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode};

    // `Safe` disables the unsafe floating-point optimizations (contraction, reassociation) so every
    // operation rounds in the tensor dtype; the math functions stay on candle's default `metal::fast`
    // tables.
    let options = MTLCompileOptions::new();
    if available!(macos = 15, ios = 18) {
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
    } else {
        // The pre-`mathMode` spelling of the same choice: `NO` is IEEE-strict.
        options.setFastMathEnabled(false);
    }
    let library: Library = device
        .metal_device()
        .new_library_with_source(SOURCE, Some(&options))
        .map_err(|error| Error::msg(format!("late.metal cannot be compiled: {error}")))?;
    let kernel = |name: &str| -> Result<ComputePipeline> {
        let function = library.get_function(name, None).map_err(|error| {
            Error::msg(format!("late.metal has no {name} for this device: {error}"))
        })?;
        device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| Error::msg(format!("late.metal: {name} pipeline: {error}")))
    };
    let pipelines = [
        kernel(INVERSE_KERNEL)?,
        kernel(&format!("causal_conv_{suffix}"))?,
        kernel(&format!("recurrent_delta_{suffix}"))?,
        kernel(&format!("gated_norm_{suffix}"))?,
    ];
    Ok((library, pipelines))
}

/// Unary op carrying its pipeline, cloned from [`Kernels`], and the chunk size.
#[cfg(target_os = "macos")]
struct UnitLowerInverse {
    pipeline: ComputePipeline,
    size: usize,
}

#[cfg(target_os = "macos")]
impl CustomOp1 for UnitLowerInverse {
    fn name(&self) -> &'static str {
        "unit_lower_inverse"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        Err(Error::msg(
            "unit_lower_inverse: no CPU implementation; the CPU path inverts by block doubling",
        ))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_metal_kernels::metal::ComputeCommandEncoder;
        use candle_metal_kernels::utils::{Input, Output};
        use objc2_metal::MTLSize;

        let dtype = storage.dtype();
        let (size, dims) = (self.size, layout.dims());
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(dtype, self.name()));
        }
        if dims.len() != 3 || dims[1] != size || dims[2] != size {
            return Err(Error::msg(format!(
                "unit_lower_inverse: expected [N, {size}, {size}] systems, got {dims:?}"
            )));
        }
        let (n, s) = (dims[0], dims[1]);
        if !s.is_power_of_two() || s > MAX_SIZE || !layout.is_contiguous() {
            return Err(Error::msg(format!("unit_lower_inverse: expected a contiguous f32 [{n}, {s}, {s}] with a power-of-two S <= {MAX_SIZE}")));
        }

        let device = storage.device().clone();
        let count = n * s * s;
        let buffer = device
            .new_buffer_builder()
            .with_size_for(count, DType::F32)
            .with_label("unit_lower_inverse")
            .build()?;
        // One threadgroup per matrix, one thread per column. `Input`/`Output` register the
        // read-after-write hazard; the guard released below only unlocks candle's encoder, which stays
        // open on the shared command buffer, so `device` can move into the storage afterwards.
        {
            let encoder = device.command_encoder()?;
            let encoder: &ComputeCommandEncoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&self.pipeline);
            let input = Input::with_offset(
                storage.buffer(),
                layout.start_offset() * dtype.size_in_bytes(),
            );
            candle_metal_kernels::set_params!(encoder, (input, Output::new(&buffer), s as u32));
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: 1,
                    height: n,
                    depth: 1,
                },
                MTLSize {
                    width: s,
                    height: 1,
                    depth: 1,
                },
            );
        }
        let out = MetalStorage::new(buffer, device, count, DType::F32);
        Ok((out, layout.shape().clone()))
    }
}

/// Binary op carrying its pipeline, cloned from [`Kernels`].
#[cfg(target_os = "macos")]
struct CausalConv {
    pipeline: ComputePipeline,
}

#[cfg(target_os = "macos")]
impl CustomOp2 for CausalConv {
    fn name(&self) -> &'static str {
        "causal_conv"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(Error::msg(
            "causal_conv: no CPU implementation; keep the candle op sequence",
        ))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder};
        use candle_metal_kernels::utils::{Input, Output};
        use objc2_metal::MTLSize;

        let dtype = s1.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) || dtype != s2.dtype() {
            return Err(Error::msg(format!(
                "causal_conv: expected matching f32/f16/bf16 tensors, got {dtype:?} and {:?}",
                s2.dtype()
            )));
        }
        let &[b, t, channels] = l1.dims() else {
            return Err(Error::msg(format!(
                "causal_conv: expected a [B, T, C] input, got {:?}",
                l1.dims()
            )));
        };
        let &[c2, taps] = l2.dims() else {
            return Err(Error::msg(format!(
                "causal_conv: expected a [C, K] weight tensor, got {:?}",
                l2.dims()
            )));
        };
        let &[stride_b, stride_t, stride_c] = l1.stride() else {
            return Err(Error::msg(format!(
                "causal_conv: expected three input strides, got {:?}",
                l1.stride()
            )));
        };
        if c2 != channels || taps == 0 {
            return Err(Error::msg(format!(
                "causal_conv: weights are {c2} x {taps}, expected {channels} x K with K >= 1"
            )));
        }

        let device = s1.device().clone();
        let count = b * t * channels;
        let buffer: std::sync::Arc<Buffer> = device
            .new_buffer_builder()
            .with_size_for(count, dtype)
            .with_label("causal_conv")
            .build()?;
        {
            let encoder = device.command_encoder()?;
            let encoder: &ComputeCommandEncoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&self.pipeline);
            let xs = Input::with_offset(s1.buffer(), l1.start_offset() * dtype.size_in_bytes());
            let w = Input::with_offset(s2.buffer(), l2.start_offset() * dtype.size_in_bytes());
            candle_metal_kernels::set_params!(
                encoder,
                (
                    xs,
                    w,
                    Output::new(&buffer),
                    b as u32,
                    t as u32,
                    channels as u32,
                    taps as u32,
                    stride_b,
                    stride_t,
                    stride_c
                )
            );
            // One thread per output element, capped at the pipeline's limit.
            let width = self
                .pipeline
                .max_total_threads_per_threadgroup()
                .min(count.max(1));
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: count.div_ceil(width),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
        }
        let out = MetalStorage::new(buffer, device, count, dtype);
        Ok((out, l1.shape().clone()))
    }
}

/// Ternary op carrying its pipeline (cloned from [`Kernels`]) and the head layout.
#[cfg(target_os = "macos")]
struct RecurrentDelta {
    pipeline: ComputePipeline,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
}

#[cfg(target_os = "macos")]
impl CustomOp3 for RecurrentDelta {
    fn name(&self) -> &'static str {
        "recurrent_delta"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(Error::msg(
            "recurrent_delta: no CPU implementation; keep the chunked rule",
        ))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
        s3: &MetalStorage,
        l3: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder};
        use candle_metal_kernels::utils::{Input, Output};
        use objc2_metal::MTLSize;

        let (nk, nv, dk, dv) = (
            self.key_heads,
            self.value_heads,
            self.key_dim,
            self.value_dim,
        );
        let dtype = s1.dtype();
        if s2.dtype() != DType::F32 || s3.dtype() != DType::F32 {
            return Err(Error::msg(format!(
                "recurrent_delta: g and beta must be f32, got {:?} and {:?}",
                s2.dtype(),
                s3.dtype()
            )));
        }
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDTypeForOp(dtype, self.name()));
        }
        let &[b, t, channels] = l1.dims() else {
            return Err(Error::msg(format!(
                "recurrent_delta: expected a [B, T, C] mixed tensor, got {:?}",
                l1.dims()
            )));
        };
        let &[gb, gt, gnv] = l2.dims() else {
            return Err(Error::msg(format!(
                "recurrent_delta: expected a [B, T, nv] g tensor, got {:?}",
                l2.dims()
            )));
        };
        let &[bb, bt, bnv] = l3.dims() else {
            return Err(Error::msg(format!(
                "recurrent_delta: expected a [B, T, nv] beta tensor, got {:?}",
                l3.dims()
            )));
        };
        if (gb, gt, gnv) != (b, t, nv) || (bb, bt, bnv) != (b, t, nv) {
            return Err(Error::msg(format!("recurrent_delta: g {gb} x {gt} x {gnv} and beta {bb} x {bt} x {bnv} must be [{b}, {t}, {nv}]")));
        }
        if !l2.is_contiguous() || !l3.is_contiguous() {
            return Err(Error::msg("recurrent_delta: g and beta must be contiguous"));
        }
        if nk == 0 || !nv.is_multiple_of(nk) || dk == 0 || dv == 0 || dk > MAX_DIM || dv > MAX_DIM {
            return Err(Error::msg(format!(
                "recurrent_delta: unsupported heads nk = {nk}, nv = {nv}, dk = {dk}, dv = {dv}"
            )));
        }
        if channels != 2 * nk * dk + nv * dv {
            return Err(Error::msg(format!("recurrent_delta: mixed has {channels} channels, expected {} for nk = {nk}, nv = {nv}, dk = {dk}, dv = {dv}", 2 * nk * dk + nv * dv)));
        }

        let device = s1.device().clone();
        let count = b * t * nv * dv;
        let buffer: std::sync::Arc<Buffer> = device
            .new_buffer_builder()
            .with_size_for(count, DType::F32)
            .with_label("recurrent_delta")
            .build()?;
        // One threadgroup per (batch, value head); `max(dk, dv)` threads, rounded up to a power of two
        // and to one full SIMD group so `simd_sum` is defined. Each thread owns one value column.
        let width = dk.max(dv).next_power_of_two().max(32);
        {
            let encoder = device.command_encoder()?;
            let encoder: &ComputeCommandEncoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&self.pipeline);
            let mixed = Input::with_offset(s1.buffer(), l1.start_offset() * dtype.size_in_bytes());
            let g = Input::with_offset(s2.buffer(), l2.start_offset() * DType::F32.size_in_bytes());
            let beta =
                Input::with_offset(s3.buffer(), l3.start_offset() * DType::F32.size_in_bytes());
            let strides = l1.stride();
            candle_metal_kernels::set_params!(
                encoder,
                (
                    mixed,
                    g,
                    beta,
                    Output::new(&buffer),
                    t as u32,
                    nk as u32,
                    nv as u32,
                    dk as u32,
                    dv as u32,
                    strides[0],
                    strides[1],
                    strides[2]
                )
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: nv,
                    height: b,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
        }
        let out = MetalStorage::new(buffer, device, count, DType::F32);
        Ok((out, (b, t, nv, dv).into()))
    }
}

/// Ternary op carrying its pipeline (cloned from [`Kernels`]) and the variance epsilon.
#[cfg(target_os = "macos")]
struct GatedNorm {
    pipeline: ComputePipeline,
    eps: f32,
}

#[cfg(target_os = "macos")]
impl CustomOp3 for GatedNorm {
    fn name(&self) -> &'static str {
        "gated_norm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(Error::msg(
            "gated_norm: no CPU implementation; keep the op sequence",
        ))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
        s3: &MetalStorage,
        l3: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder};
        use candle_metal_kernels::utils::{Input, Output};
        use objc2_metal::MTLSize;

        let dtype = s1.dtype();
        if s2.dtype() != dtype
            || s3.dtype() != dtype
            || !matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
        {
            return Err(Error::msg(format!(
                "gated_norm: expected matching f32/f16/bf16 core, z and weight, got {dtype:?}, {:?} and {:?}",
                s2.dtype(),
                s3.dtype()
            )));
        }
        let &[b, t, nv, dv] = l1.dims() else {
            return Err(Error::msg(format!(
                "gated_norm: expected a [B, T, nv, dv] core tensor, got {:?}",
                l1.dims()
            )));
        };
        if l2.dims() != l1.dims() {
            return Err(Error::msg(format!(
                "gated_norm: z is {:?}, expected {:?}",
                l2.dims(),
                l1.dims()
            )));
        }
        let &[weights] = l3.dims() else {
            return Err(Error::msg(format!(
                "gated_norm: expected a [dv] weight tensor, got {:?}",
                l3.dims()
            )));
        };
        if weights != dv || dv == 0 || dv > MAX_DIM {
            return Err(Error::msg(format!(
                "gated_norm: weight is [{weights}] for rows {dv} wide, expected dv in 1..={MAX_DIM}"
            )));
        }
        if !l1.is_contiguous() || !l2.is_contiguous() || !l3.is_contiguous() {
            return Err(Error::msg(
                "gated_norm: core, z and weight must be contiguous",
            ));
        }

        let device = s1.device().clone();
        let rows = b * t * nv;
        let count = rows * dv;
        let buffer: std::sync::Arc<Buffer> = device
            .new_buffer_builder()
            .with_size_for(count, dtype)
            .with_label("gated_norm")
            .build()?;
        // One threadgroup per row, one element per thread: `dv` threads rounded up to a power of two
        // and to one full SIMD group so `simd_sum` is defined.
        let width = dv.next_power_of_two().max(32);
        {
            let encoder = device.command_encoder()?;
            let encoder: &ComputeCommandEncoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&self.pipeline);
            let core = Input::with_offset(s1.buffer(), l1.start_offset() * dtype.size_in_bytes());
            let z = Input::with_offset(s2.buffer(), l2.start_offset() * dtype.size_in_bytes());
            let weight = Input::with_offset(s3.buffer(), l3.start_offset() * dtype.size_in_bytes());
            candle_metal_kernels::set_params!(
                encoder,
                (core, z, weight, Output::new(&buffer), dv as u32, self.eps)
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
        }
        let out = MetalStorage::new(buffer, device, count, dtype);
        Ok((out, l1.shape().clone()))
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// Deterministic values in `[-0.5, 0.5)`.
    fn pseudo(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// Strictly lower `[n, s, s]`, scaled so `(I + A)^-1` stays O(1).
    fn strictly_lower(device: &Device, n: usize, s: usize, seed: u32) -> Tensor {
        let mut values = pseudo(n * s * s, seed);
        for m in 0..n {
            for i in 0..s {
                for j in i..s {
                    values[(m * s + i) * s + j] = 0.0;
                }
            }
        }
        Tensor::from_vec(
            values
                .iter()
                .map(|value| value / s as f32)
                .collect::<Vec<f32>>(),
            (n, s, s),
            device,
        )
        .unwrap()
    }

    /// `(I + A) X == I` to `1e-4`, with a unit diagonal and an exact zero above it.
    fn check_inverse(a: &Tensor, kernels: &Kernels, n: usize, s: usize) {
        let x = unit_lower_inverse(a, s, kernels).unwrap();
        assert_eq!(x.dims3().unwrap(), (n, s, s));
        let eye = Tensor::eye(s, DType::F32, a.device()).unwrap();
        let product = a
            .broadcast_add(&eye)
            .unwrap()
            .matmul(&x)
            .unwrap()
            .broadcast_sub(&eye)
            .unwrap();
        let worst = product
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(worst <= 1e-4, "S = {s}: (I + A) X - I = {worst}");
        for (index, value) in x
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .enumerate()
        {
            let (row, column) = (index / s % s, index % s);
            if row == column {
                assert_eq!(*value, 1.0, "S = {s}: diagonal is {value}");
            } else if column > row {
                assert_eq!(*value, 0.0, "S = {s}: above the diagonal is {value}");
            }
        }
    }

    #[test]
    fn inverse_is_the_unit_lower_inverse() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        }; // no Metal device on this machine
        let kernels = Kernels::new(&device, DType::F32).unwrap();
        let n = 3;
        for s in [1usize, 2, 8, 16, 64] {
            check_inverse(&strictly_lower(&device, n, s, 9), &kernels, n, s);
            // The same system inside a larger buffer, addressed by `narrow`: a contiguous layout with a
            // non-zero start offset, which the kernel has to honour (candle keeps a contiguous view's
            // offset).
            let whole = strictly_lower(&device, n + 2, s, 11);
            let view = whole.narrow(0, 1, n).unwrap();
            assert!(view.is_contiguous());
            check_inverse(&view, &kernels, n, s);
        }
    }

    /// `qwen3_5.rs`'s unfused sequence: left zero pad, one multiply and one add per tap, then `silu`,
    /// all in the tensor's dtype.
    fn unfused(xs: &Tensor, weights: &Tensor) -> Tensor {
        let (_, t, channels) = xs.dims3().unwrap();
        let taps = weights.dim(1).unwrap();
        (0..taps)
            .map(|tap| {
                let weight = weights
                    .narrow(1, tap, 1)
                    .unwrap()
                    .reshape(channels)
                    .unwrap();
                xs.pad_with_zeros(1, taps - 1, 0)
                    .unwrap()
                    .narrow(1, tap, t)
                    .unwrap()
                    .broadcast_mul(&weight)
                    .unwrap()
            })
            .reduce(|sum, term| (sum + term).unwrap())
            .unwrap()
            .silu()
            .unwrap()
    }

    #[test]
    fn conv_matches_the_unfused_path() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        }; // no Metal device on this machine
        let (b, channels) = (2, 3);
        // f32 agrees to well under its own rounding; f16/bf16 can differ by a few units in the last
        // place, which is where `silu`'s `exp` sits. The tap sums themselves round identically (same
        // order, same dtype).
        for (dtype, tolerance) in [
            (DType::F32, 1e-5f32),
            (DType::F16, 5e-3),
            (DType::BF16, 3e-2),
        ] {
            let kernels = Kernels::new(&device, dtype).unwrap();
            // `t < k` is all padding on the left, then `t == k` and `t > k`.
            for (t, taps) in [
                (1usize, 4usize),
                (2, 4),
                (4, 4),
                (5, 4),
                (17, 4),
                (1, 1),
                (5, 1),
            ] {
                let values = pseudo(b * t * channels, 3);
                let xs = Tensor::from_vec(values, (b, t, channels), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let weights =
                    Tensor::from_vec(pseudo(channels * taps, 4), (channels, taps), &device)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap();
                // `[B, C, T]` transposed to `[B, T, C]`: non-contiguous input, read through its strides.
                let strided =
                    Tensor::from_vec(pseudo(b * t * channels, 5), (b, channels, t), &device)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap();
                if t > 1 {
                    assert!(!strided.is_contiguous());
                }
                for input in [&xs, &strided] {
                    let ours = causal_conv(input, &weights, &kernels).unwrap();
                    assert_eq!(ours.dims3().unwrap(), (b, t, channels));
                    let reference = unfused(input, &weights).to_dtype(DType::F32).unwrap();
                    let worst = ours
                        .to_dtype(DType::F32)
                        .unwrap()
                        .sub(&reference)
                        .unwrap()
                        .abs()
                        .unwrap()
                        .max_all()
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap();
                    assert!(
                        worst <= tolerance,
                        "{dtype:?} t = {t}, k = {taps}: fused conv differs by {worst}"
                    );
                }
            }
        }
    }

    /// The per-token recurrence of the reference, whole batch, on the CPU: `q`, `k` and `v` are the
    /// channel blocks of `mixed` and `kh = h / (nv / nk)`.
    #[allow(clippy::too_many_arguments)] // the reference's own shape
    fn cpu_recurrent(
        mixed: &[f32],
        g: &[f32],
        beta: &[f32],
        b: usize,
        t: usize,
        nk: usize,
        nv: usize,
        dk: usize,
        dv: usize,
    ) -> Vec<f32> {
        let (rep, channels) = (nv / nk, 2 * nk * dk + nv * dv);
        let norm = |x: &[f32]| {
            let n = (x.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
            x.iter().map(|a| a / n).collect::<Vec<f32>>()
        };
        let dk_scale = (dk as f32).sqrt();
        let mut out = vec![0f32; b * t * nv * dv];
        for bi in 0..b {
            for h in 0..nv {
                let kh = h / rep;
                let mut state = vec![0f32; dk * dv];
                for i in 0..t {
                    let row = &mixed[(bi * t + i) * channels..(bi * t + i + 1) * channels];
                    let qi: Vec<f32> = norm(&row[kh * dk..kh * dk + dk])
                        .iter()
                        .map(|a| a / dk_scale)
                        .collect();
                    let ki = norm(&row[(nk + kh) * dk..(nk + kh) * dk + dk]);
                    let decay = g[(bi * t + i) * nv + h].exp();
                    state.iter_mut().for_each(|s| *s *= decay);
                    for c in 0..dv {
                        let mem: f32 = (0..dk).map(|r| state[r * dv + c] * ki[r]).sum();
                        let delta =
                            (row[2 * nk * dk + h * dv + c] - mem) * beta[(bi * t + i) * nv + h];
                        for r in 0..dk {
                            state[r * dv + c] += ki[r] * delta;
                        }
                        out[((bi * t + i) * nv + h) * dv + c] =
                            (0..dk).map(|r| state[r * dv + c] * qi[r]).sum();
                    }
                }
            }
        }
        out
    }

    #[test]
    fn recurrent_delta_matches_the_cpu_recurrence() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        }; // no Metal device on this machine
        let (b, nk, nv) = (2, 2, 4);
        for (t, dk, dv) in [
            (1usize, 8usize, 6usize),
            (8, 8, 6),
            (31, 8, 6),
            (64, 8, 6),
            (8, 128, 128),
        ] {
            let channels = 2 * nk * dk + nv * dv;
            let mixed = pseudo(b * t * channels, 21);
            // Decays in (-0.3, 0] and betas in [0, 1), as the model produces them.
            let g: Vec<f32> = pseudo(b * t * nv, 22)
                .iter()
                .map(|x| -0.3 * (x + 0.5))
                .collect();
            let beta: Vec<f32> = pseudo(b * t * nv, 23).iter().map(|x| x + 0.5).collect();
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let kernels = Kernels::new(&device, dtype).unwrap();
                let input = Tensor::from_vec(mixed.clone(), (b, t, channels), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let rounded = input
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let actual = recurrent_delta(
                    &input,
                    &Tensor::from_vec(g.clone(), (b, t, nv), &device).unwrap(),
                    &Tensor::from_vec(beta.clone(), (b, t, nv), &device).unwrap(),
                    nk,
                    nv,
                    dk,
                    dv,
                    &kernels,
                )
                .unwrap();
                assert_eq!(actual.dims4().unwrap(), (b, t, nv, dv));
                let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let expected = cpu_recurrent(&rounded, &g, &beta, b, t, nk, nv, dk, dv);
                let worst = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    worst <= 1e-4,
                    "{dtype:?} T = {t}, dk = {dk}, dv = {dv}: recurrence differs by {worst}"
                );
            }
        }
    }

    /// The gated RMSNorm as the full op sequence of `qwen3_5.rs`, on the same device.
    fn gated_norm_ops(
        core: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        dtype: DType,
        eps: f64,
    ) -> Tensor {
        use candle_core::D;
        let core = core.to_dtype(DType::F32).unwrap();
        let variance = core.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
        let normed = core
            .broadcast_div(&(variance + eps).unwrap().sqrt().unwrap())
            .unwrap();
        let normed = normed
            .to_dtype(dtype)
            .unwrap()
            .broadcast_mul(weight)
            .unwrap();
        let gated = (normed.to_dtype(DType::F32).unwrap()
            * z.to_dtype(DType::F32).unwrap().silu().unwrap())
        .unwrap();
        gated.to_dtype(dtype).unwrap()
    }

    #[test]
    fn gated_norm_matches_the_op_sequence() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        }; // no Metal device on this machine
        let eps = 1e-6f64;
        let (b, t, nv) = (2, 3, 2);
        // f32 agrees to well under its own rounding. For f16/bf16 the variance reduction order differs
        // from candle's (SIMD groups here, blocks there), so a value parked on a rounding boundary can
        // move one unit in the last place of the dtype and the three dtype roundings follow it; the
        // tolerance is a couple of those units.
        for dv in [8usize, 128] {
            let core =
                Tensor::from_vec(pseudo(b * t * nv * dv, 31), (b, t, nv, dv), &device).unwrap();
            let z = Tensor::from_vec(pseudo(b * t * nv * dv, 32), (b, t, nv, dv), &device).unwrap();
            let weight = Tensor::from_vec(pseudo(dv, 33), (dv,), &device).unwrap();
            for (dtype, tolerance) in [
                (DType::F32, 1e-5f32),
                (DType::F16, 5e-3),
                (DType::BF16, 2e-2),
            ] {
                let kernels = Kernels::new(&device, dtype).unwrap();
                let core = core.to_dtype(dtype).unwrap();
                let z = z.to_dtype(dtype).unwrap();
                let weight = weight.to_dtype(dtype).unwrap();
                let ours = gated_norm(&core, &z, &weight, eps, &kernels).unwrap();
                assert_eq!(ours.dims4().unwrap(), (b, t, nv, dv));
                assert_eq!(ours.dtype(), dtype);
                let ours = ours.to_dtype(DType::F32).unwrap();
                let reference = gated_norm_ops(&core, &z, &weight, dtype, eps)
                    .to_dtype(DType::F32)
                    .unwrap();
                let worst = ours
                    .sub(&reference)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(
                    worst <= tolerance,
                    "{dtype:?} dv = {dv}: gated norm differs by {worst}"
                );
            }
        }
    }
}
