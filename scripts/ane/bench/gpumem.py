"""GPU memory of one process on an NVIDIA host: NVML at 2 ms when pynvml is installed (the CUDA allocator
frees stream-ordered memory at every sync, so a 250 ms nvidia-smi poll misses the transient peaks),
else nvidia-smi."""
import subprocess

try:
    import pynvml
    pynvml.nvmlInit()
    _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # no NVML: fall back to nvidia-smi
    pynvml = None

INTERVAL = 0.002 if pynvml else 0.25


def gpu_mb(pid):
    if pynvml:
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(_handle):
            if proc.pid == pid and proc.usedGpuMemory is not None:
                return proc.usedGpuMemory / (1 << 20)
        return None
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        got, used = [x.strip() for x in line.split(",")]
        if got == str(pid):
            return float(used)
    return None
