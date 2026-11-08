//! Device, dtype and cache-directory selection shared by every engine, plus
//! the lazily loaded, unloadable model slot.

use crate::error::{InlError, Result};
use candle_core::{DType, Device};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;
use tokio::sync::Mutex as AsyncMutex;

/// Which compute device a model should load onto.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceSpec {
    /// Metal on macOS, else CUDA when this build has it and a GPU is present,
    /// else the CPU.
    #[default]
    Auto,
    Cpu,
    Metal,
    Cuda,
}

impl FromStr for DeviceSpec {
    type Err = InlError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            other => Err(InlError::Config(format!(
                "invalid device '{other}': expected auto, cpu, metal, or cuda"
            ))),
        }
    }
}

impl DeviceSpec {
    /// `None` is `auto`.
    pub fn parse(value: Option<&str>) -> Result<Self> {
        value.map_or(Ok(Self::Auto), str::parse)
    }
}

/// A numeric precision a caller can ask for. `None` where an `Option` is taken
/// means "the engine's default for this device".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtypeSpec {
    F32,
    F16,
    Bf16,
}

impl FromStr for DtypeSpec {
    type Err = InlError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::Bf16),
            other => Err(InlError::Config(format!(
                "invalid dtype '{other}': expected auto, f32, f16, or bf16"
            ))),
        }
    }
}

impl DtypeSpec {
    /// `None` and `auto` are `Ok(None)`: the engine picks.
    pub fn parse(value: Option<&str>) -> Result<Option<Self>> {
        match value {
            None | Some("auto") => Ok(None),
            Some(other) => other.parse().map(Some),
        }
    }

    pub fn to_candle(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
        }
    }
}

pub fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        other => other.as_str(),
    }
}

/// Open the device `spec` names.
pub fn resolve_device(spec: DeviceSpec) -> Result<Device> {
    let device = match spec {
        DeviceSpec::Cpu => Device::Cpu,
        DeviceSpec::Auto => auto_device(),
        DeviceSpec::Metal => metal_device()?,
        DeviceSpec::Cuda => cuda_device()?,
    };
    prepare_device(&device);
    Ok(device)
}

/// Environment override for CUDA GEMM accumulation precision. A comma-separated
/// list of `f32`, `f16`, `bf16` selects which of them may run with reduced
/// precision (tensor-core) accumulation; `none` selects the exact CUBLAS
/// compute types. See [`prepare_device`].
pub const CUDA_REDUCED_PRECISION_ENV: &str = "INL_CUDA_REDUCED_PRECISION";

/// What [`CUDA_REDUCED_PRECISION_ENV`] means when it is not set.
///
/// `bf16` and `f16` are half-precision weights, whose GEMMs are already lossy:
/// candle defaults them to `CUBLAS_COMPUTE_32F`, which on a Blackwell part
/// costs the tensor cores the dtype exists for. `f32` is left exact: TF32
/// rounds the operands to 10 mantissa bits, which is a real numeric change to
/// the default path and has to clear the live goldens before it can be on.
pub const CUDA_REDUCED_PRECISION_DEFAULT: &str = "bf16,f16";

/// One-time tuning of a freshly opened device.
///
/// CUDA tensors track their uses with cuda events unless the context opts out,
/// which costs two event creations on every allocation (and the matching
/// destroys) for a guarantee nothing here needs: every engine in this
/// workspace launches its work on the device's single stream, so there is no
/// cross-stream use for an event to order. Turning the tracking off removes
/// that per-allocation driver traffic.
///
/// It then selects the GEMM accumulation precision from
/// [`CUDA_REDUCED_PRECISION_ENV`] (default [`CUDA_REDUCED_PRECISION_DEFAULT`]).
/// CPU and Metal are untouched.
pub fn prepare_device(device: &Device) {
    #[cfg(feature = "cuda")]
    if let Ok(cuda) = device.as_cuda_device() {
        // SAFETY: single stream, as above; no tensor is used or freed from a
        // second stream, so no event ordering is lost.
        unsafe { cuda.disable_event_tracking() };
        let selected = std::env::var(CUDA_REDUCED_PRECISION_ENV)
            .unwrap_or_else(|_| CUDA_REDUCED_PRECISION_DEFAULT.to_string());
        let wants = |name: &str| {
            selected
                .split(',')
                .any(|part| part.trim() == name || part.trim() == "all")
        };
        candle_core::cuda_backend::set_gemm_reduced_precision_f32(wants("f32"));
        candle_core::cuda_backend::set_gemm_reduced_precision_f16(wants("f16"));
        candle_core::cuda_backend::set_gemm_reduced_precision_bf16(wants("bf16"));
    }
    #[cfg(not(feature = "cuda"))]
    let _ = device;
}

/// Free and total bytes of device memory, when the device exposes them. CUDA
/// reports them; CPU and Metal are unified-memory devices whose budget the OS
/// manages, so they answer `None`.
pub fn device_memory_bytes(device: &Device) -> Option<(usize, usize)> {
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        if let Ok(info) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info() {
            return Some(info);
        }
    }
    let _ = device;
    None
}

fn auto_device() -> Device {
    #[cfg(target_os = "macos")]
    if let Ok(device) = Device::new_metal(0) {
        return device;
    }
    #[cfg(feature = "cuda")]
    if let Ok(device) = Device::new_cuda(0) {
        return device;
    }
    Device::Cpu
}

fn metal_device() -> Result<Device> {
    #[cfg(target_os = "macos")]
    {
        Device::new_metal(0)
            .map_err(|error| InlError::Config(format!("failed to open metal device: {error}")))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(InlError::Config(
            "device metal is only available on macOS".to_string(),
        ))
    }
}

fn cuda_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0)
            .map_err(|error| InlError::Config(format!("failed to open cuda device: {error}")))
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(InlError::Config(
            "device cuda requires a build with the `cuda` feature".to_string(),
        ))
    }
}

/// `cpu`, `metal` or `cuda:N`. Metal's `gpu_id` is an IORegistry id, not an
/// ordinal; candle always opens device 0, so its label carries no index.
pub fn device_label(device: &Device) -> String {
    match device.location() {
        candle_core::DeviceLocation::Cpu => "cpu".to_string(),
        candle_core::DeviceLocation::Metal { .. } => "metal".to_string(),
        candle_core::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
    }
}

/// `cpu`, `metal` or `cuda`, from a [`device_label`] such as `cuda:0`.
pub fn device_kind(label: &str) -> &'static str {
    match label.split(':').next() {
        Some("metal") => "metal",
        Some("cuda") => "cuda",
        _ => "cpu",
    }
}

/// Per-pass deadline on CPU when none is set explicitly. A CPU pass is slow but
/// does not hang: a pass of long inputs needs minutes on a mid-range desktop
/// CPU, which a 60 s default meant for a stuck GPU kernel would abort.
pub const CPU_BATCH_TIMEOUT: Duration = Duration::from_secs(900);

/// The deadline a loaded model runs under. An explicit deadline always wins;
/// otherwise CPU gets at least [`CPU_BATCH_TIMEOUT`] and GPUs keep the
/// configured value.
pub fn effective_batch_timeout(configured: Duration, explicit: bool, on_cpu: bool) -> Duration {
    if on_cpu && !explicit {
        configured.max(CPU_BATCH_TIMEOUT)
    } else {
        configured
    }
}

/// Environment override for the model cache directory.
pub const CACHE_DIR_ENV: &str = "INL_CACHE_DIR";

/// Root of im-not-llm's per-user files: `~/.cache/im-not-llm`
/// (`%LOCALAPPDATA%\im-not-llm` on Windows).
pub fn cache_root() -> PathBuf {
    #[cfg(windows)]
    {
        let root = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(root).join("im-not-llm")
    }
    #[cfg(not(windows))]
    {
        let root = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(root).join(".cache").join("im-not-llm")
    }
}

/// Directory model files are cached in: `configured`, else `INL_CACHE_DIR`,
/// else `<cache_root>/models`. Created when missing. `HF_HOME`, when set,
/// still takes precedence inside the hub client (see [`crate::hub`]).
pub fn model_cache_dir(configured: Option<&Path>) -> PathBuf {
    let dir = configured
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os(CACHE_DIR_ENV).map(PathBuf::from))
        .unwrap_or_else(|| cache_root().join("models"));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// A lazily loaded value that can be dropped again and reloaded on demand.
///
/// Concurrent callers share one load; a failed load is not cached.
pub struct LoadSlot<T> {
    value: AsyncMutex<Option<T>>,
}

impl<T: Clone> Default for LoadSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> LoadSlot<T> {
    pub fn new() -> Self {
        Self {
            value: AsyncMutex::new(None),
        }
    }

    pub async fn get_or_load<F, Fut>(&self, load: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut slot = self.value.lock().await;
        if let Some(value) = slot.as_ref() {
            return Ok(value.clone());
        }
        let value = load().await?;
        *slot = Some(value.clone());
        Ok(value)
    }

    /// Drop the held value; true when one was resident. Waits for an
    /// in-progress load first, so the load's value is dropped too.
    pub async fn unload(&self) -> bool {
        self.value.lock().await.take().is_some()
    }

    /// True when a value is resident. A load or unload in progress reports
    /// false.
    pub fn is_loaded(&self) -> bool {
        self.value
            .try_lock()
            .map(|slot| slot.is_some())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn device_parses_each_name_and_defaults_to_auto() {
        assert_eq!(DeviceSpec::parse(None).unwrap(), DeviceSpec::Auto);
        assert_eq!(DeviceSpec::parse(Some("auto")).unwrap(), DeviceSpec::Auto);
        assert_eq!(DeviceSpec::parse(Some("cpu")).unwrap(), DeviceSpec::Cpu);
        assert_eq!(DeviceSpec::parse(Some("metal")).unwrap(), DeviceSpec::Metal);
        assert_eq!(DeviceSpec::parse(Some("cuda")).unwrap(), DeviceSpec::Cuda);
        let error = DeviceSpec::parse(Some("tpu")).unwrap_err();
        assert!(matches!(error, InlError::Config(message) if message.contains("tpu")));
    }

    #[test]
    fn dtype_auto_means_the_engine_picks() {
        assert_eq!(DtypeSpec::parse(None).unwrap(), None);
        assert_eq!(DtypeSpec::parse(Some("auto")).unwrap(), None);
        assert_eq!(
            DtypeSpec::parse(Some("bf16")).unwrap(),
            Some(DtypeSpec::Bf16)
        );
        assert!(DtypeSpec::parse(Some("int8")).is_err());
    }

    #[test]
    fn cpu_always_resolves() {
        let device = resolve_device(DeviceSpec::Cpu).unwrap();
        assert_eq!(device_label(&device), "cpu");
        assert_eq!(device_kind(&device_label(&device)), "cpu");
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_without_the_feature_is_a_config_error_naming_the_feature() {
        let error = resolve_device(DeviceSpec::Cuda).unwrap_err();
        assert!(matches!(error, InlError::Config(message) if message.contains("`cuda` feature")));
    }

    #[test]
    fn device_kind_strips_the_ordinal() {
        assert_eq!(device_kind("cuda:3"), "cuda");
        assert_eq!(device_kind("metal"), "metal");
        assert_eq!(device_kind("anything-else"), "cpu");
    }

    #[test]
    fn cpu_gets_a_long_deadline_unless_one_is_explicit() {
        let short = Duration::from_secs(60);
        assert_eq!(
            effective_batch_timeout(short, false, true),
            CPU_BATCH_TIMEOUT
        );
        assert_eq!(effective_batch_timeout(short, true, true), short);
        assert_eq!(effective_batch_timeout(short, false, false), short);
        let long = Duration::from_secs(2000);
        assert_eq!(effective_batch_timeout(long, false, true), long);
    }

    #[test]
    fn configured_cache_dir_wins_and_is_created() {
        let base = tempfile::tempdir().unwrap();
        let wanted = base.path().join("nested").join("models");
        assert_eq!(model_cache_dir(Some(&wanted)), wanted);
        assert!(wanted.is_dir());
    }

    #[tokio::test]
    async fn load_slot_shares_one_load_and_reloads_after_unload() {
        let slot: LoadSlot<Arc<usize>> = LoadSlot::new();
        let loads = Arc::new(AtomicUsize::new(0));
        let load = |loads: Arc<AtomicUsize>| async move {
            Ok(Arc::new(loads.fetch_add(1, Ordering::SeqCst)))
        };
        assert!(!slot.is_loaded());
        let first = slot.get_or_load(|| load(loads.clone())).await.unwrap();
        let second = slot.get_or_load(|| load(loads.clone())).await.unwrap();
        assert_eq!(*first, 0);
        assert_eq!(*second, 0, "second call reuses the resident value");
        assert!(slot.is_loaded());
        assert!(slot.unload().await);
        assert!(!slot.unload().await, "nothing left to unload");
        let third = slot.get_or_load(|| load(loads.clone())).await.unwrap();
        assert_eq!(*third, 1, "unload forces a reload");
    }

    #[tokio::test]
    async fn load_slot_does_not_cache_a_failed_load() {
        let slot: LoadSlot<u8> = LoadSlot::new();
        let failed = slot
            .get_or_load(|| async { Err(InlError::Inference("boom".into())) })
            .await;
        assert!(failed.is_err());
        assert!(!slot.is_loaded());
        assert_eq!(slot.get_or_load(|| async { Ok(7) }).await.unwrap(), 7);
    }
}
