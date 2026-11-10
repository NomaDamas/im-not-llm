# Devices

How to build and run im-not-llm on the machines it is exercised on, with the
exact commands, environment variables and the issues each setup hits.

Verified 2026-11-09 against `main` @ `ce93d66` for
`crates/{inl-core,inl-text,inl-sparse,inl-server}`. `crates/inl-vision` and
`bindings/node` are developed separately; they are workspace members now, so a
`cargo build -p inl-server` pulls `inl-vision` in.

## Env vars, once

| Var | Meaning |
| --- | --- |
| `CUDA_HOME` / `CUDA_PATH` | CUDA toolkit root; `cudarc`/`cudaforge` find `nvcc` and the libs here. |
| `CUDA_COMPUTE_CAP` | Overrides the `nvidia-smi`-detected compute capability for the kernel build. |
| `NVCC_CCBIN` | Host compiler `nvcc` hands its generated code to (Windows). |
| `CUDA_VISIBLE_DEVICES` | Which GPU candle opens as `cuda:0`. |
| `HF_HOME` | Model cache the engines *and* the `hf-hub` client read. Set it to a populated cache to avoid re-downloading. |
| `INL_TEST_DEVICE` | `cpu`/`metal`/`cuda` for the ignored live tests (`gemma2_live`, `qwen3_live`, `sparse_live`, `inl-vision`'s `live`). |
| `INL_CACHE_DIR` | Model cache when `HF_HOME` is not set (`~/.cache/im-not-llm/models` by default). |
| `INL_CUDA_REDUCED_PRECISION` | Which CUDA GEMMs may use reduced-precision (tensor-core) accumulation: a comma-separated list of `f32`, `f16`, `bf16`, or `none`. Default `bf16,f16`; `f32` (TF32) is off because it rounds the default path's operands to 10 mantissa bits. |

Feature flags: `cuda` (NVIDIA), `cudnn` (implies `cuda`; needs cuDNN),
`accelerate` (macOS CPU BLAS), `mkl` (x86 CPU BLAS).

## macOS (Apple Silicon, Metal)

```sh
cargo test   # 147 tests + 7 ignored live, CPU/Metal
HF_HOME=... cargo test -p inl-text --test gemma2_live -- --ignored --nocapture
HF_HOME=... INL_TEST_DEVICE=metal cargo test -p inl-sparse --test sparse_live -- --ignored --nocapture
```

## gpu01 — Linux x86_64, 3x RTX PRO 6000 Blackwell Max-Q (sm_120), driver 580

The preinstalled toolkit was CUDA 12.0 (`/usr/bin/nvcc`), which predates
Blackwell: `sm_120` needs CUDA >= 12.8, and `sm_120a` (what `cudaforge`
generates for compute capability 12.0) needs CUDA >= 12.9. A user-space
toolkit was installed instead of touching the system one:

```sh
# one-time: CUDA 12.9.1 into a writable prefix, no root needed
curl -LO https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
mkdir -p /tmp/cuda_tmp
bash cuda_12.9.1_575.57.08_linux.run --silent --override --toolkit \
     --toolkitpath=/tmp/cuda-12.9 --no-opengl-libs --no-man-page --tmpdir=/tmp/cuda_tmp
```

* Home (`/data`) is **NFS**; `/` (and so `/tmp`) is a **local NVMe**. The
  installer `chgrp -R`s its whole ~10 GB tree, which took >30 min on NFS; put
  `--tmpdir` and `--toolkitpath` on local disk (installed in ~4 min there).
  Copy `~/.cargo` to local disk as well (`CARGO_HOME=/tmp/cargo-home`): builds
  went from >30 min stuck on `cargo fetch` to seconds, because every crate
  read/lock in `~/.cargo` is an NFS round trip. Keep `CARGO_TARGET_DIR` local
  too.
* `cudaforge` picks the arch from `CUDA_COMPUTE_CAP`, else `nvidia-smi`. With
  the 12.9 toolkit, either works: capability 12.0 becomes `sm_120a`.
  Option (b) — PTX for an older arch — is not clean here:
  `CUDA_COMPUTE_CAP=90` yields `sm_90a`, which is arch-specific and cannot be
  JITed forward to `sm_120`, so the kernel build would produce unusable code.
* No cuDNN is installed on this box, so the `cudnn` feature was not tried.

```sh
export CUDA_HOME=/tmp/cuda-12.9
export CUDA_PATH=/tmp/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CARGO_HOME=/tmp/cargo-home
export CARGO_TARGET_DIR=/tmp/inl-target-gpu01cuda
export HF_HOME=/data/home/jeffrey/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0              # pick the idle card, it changes
cargo test -j16 -p inl-core -p inl-text -p inl-sparse -p inl-server --features cuda
INL_TEST_DEVICE=cuda cargo test -j16 --release --features cuda -p inl-sparse -p inl-text -- --ignored --nocapture --test-threads=1
```

Use `-j16` or fewer, and re-check `nvidia-smi` before each run: GPU 1 was idle
when work started and 100%-utilised 30 min later by another user.

### What CUDA does differently from the other devices

`inl-core::device::resolve_device` tunes a CUDA device once, when it is opened:

* **cuda event tracking off.** candle's CUDA tensors create two cuda events on
  every allocation (`cuEventCreate`/`cuEventDestroy`) to order cross-stream use.
  Every engine here runs on the device's single stream, so the events protect
  nothing; a text forward allocated ~1500 tensors, i.e. ~3000 event pairs.
* **reduced-precision GEMMs for half precision** (see
  `INL_CUDA_REDUCED_PRECISION`): candle defaults bf16/f16 GEMMs to
  `CUBLAS_COMPUTE_32F`, which does not use the tensor cores the dtype exists for.
* **the padded-token cap comes from device memory.** `TextOptions`'
  `max_batch_tokens` default (4096) was sized for a unified-memory GPU; on CUDA
  the planner now derives the cap from free device memory (one padded token
  budgeted at 64 KiB, never more than an eighth of what is free) unless the
  caller set the option. On gpu01 that turns `--batch-size 64` from 15-row
  batches into 64-row batches. Metal and CPU keep 4096.

The model side is device-independent but worth knowing on CUDA:

* `candle_nn::rotary_emb::rope` replaces the hand-written
  `x * cos + rotate_half(x) * sin` (measured on gpu01: 294 us -> 20 us per call
  for `[8, 16, 187, 128]` f32).
* the attention mask is built with device ops, not as a host `batch x seq x seq`
  vector. On gpu01 a `[32, 1, 193, 193]` f32 mask cost 5-9 ms to allocate, 3-5 ms
  to fill and 6-8 ms to upload (pageable host->device runs near 0.7 GB/s) per
  forward; the device build is a handful of elementwise kernels.
* `AttentionBackend` has no CUDA fused kernel: candle's `sdpa` implements only
  `cpu_fwd` and `metal_fwd` at this revision, so `Auto` resolves to `Tiled` on
  CUDA. `candle-flash-attn` would be a new dependency with its own kernels.

## windows — Windows 11, TITAN Xp (sm_61), CUDA toolkit 11.8

CPU builds and tests use the default `stable-x86_64-pc-windows-gnu` toolchain:

```bat
set "CARGO_TARGET_DIR=C:\Users\vkehfdl1\im-not-llm\target-win"
set "HF_HOME=C:\Users\vkehfdl1\.cache\huggingface"
cargo test -p inl-core -p inl-text -p inl-sparse -p inl-server
```

### CUDA on Windows

`nvcc` needs MSVC's `cl.exe` as host compiler, and `candle-kernels` passes
`-Xcompiler -fPIC` (which MSVC rejects) for every non-`msvc` Rust target, so
a CUDA build needs both:

1. Visual Studio 2019 Build Tools with the C++ workload (CUDA 11.8 does not
   accept a newer MSVC than VS 2022 17.x; 2019 is the safe pick):
   `vs_buildtools.exe --quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`
   (from `https://aka.ms/vs/16/release/vs_buildtools.exe`, admin shell).
2. `rustup toolchain install stable-x86_64-pc-windows-msvc --profile minimal`.

The TITAN Xp is sm_61 (Pascal). candle's kernels assume sm_70+ in two places
(`sum_f16` uses `atomicAdd(__half*, __half)`, and the MoE WMMA kernels use
`nvcuda::wmma`), so `nvcc` fails on them for `CUDA_COMPUTE_CAP=61`.
`scripts/patch_candle_kernels_pre_volta.py` copies the `candle-kernels` crate
cargo resolved and guards those spots (the MoE kernels trap; im-not-llm never
calls them) and writes a cargo config pointing at the copy:

```bat
python scripts\patch_candle_kernels_pre_volta.py C:\Users\vkehfdl1\ckpv
```

Then build from a shell with the MSVC environment (`inl-cuda.cmd` on the box):

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "CUDA_COMPUTE_CAP=61"
set "CARGO_TARGET_DIR=C:\Users\vkehfdl1\im-not-llm\target-win-cuda"
cargo +stable-x86_64-pc-windows-msvc test --release --features cuda --config C:\Users\vkehfdl1\ckpv.toml ^
  -p inl-text -p inl-sparse --test gemma2_live --test qwen3_live --test sparse_live -- --ignored --test-threads=1
```

GPUs with compute capability 7.0+ do not need the patch script.

dtype on sm_61: `auto` resolves to f32, which is the only sensible choice.
Pascal has no fast f16 path (Qwen3-Embedding-0.6B, `bench_text --count 256
--length 128 --batch-size 32`: f32 10.0 chunks/s, f16 0.55 chunks/s) and no
bf16 kernels (`CUDA_ERROR_NOT_FOUND, named symbol not found`).

Live goldens with `INL_TEST_DEVICE=cuda` (2026-11-10): `gemma2_live` 2/2,
`qwen3_live` 1/1 (cosine CPU vs CUDA 1.0000002 / 0.99999964), `sparse_live`
4/4 (CUDA f32 16.4 chunks/s, f16 1.5, bf16 unavailable), `inl-vision --test
engine` 1/1 (cosines >= 0.9999997).

## Model cache and `INL_TEST_DEVICE`

The engines cache models under `HF_HOME` when set, else
`~/.cache/im-not-llm/models` (the vision engine included). Concurrent first
uses of one cache download each file once: `inl_core::hub` takes a per-file
lock and the waiters read what the holder downloaded. A 404 for
`model.safetensors` is the only thing that sends the loader looking for a
sharded `model.safetensors.index.json`; any other failure is reported as
itself. Pre-populating with `huggingface_hub` saves time on the first run:

```python
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3-Embedding-0.6B", "nlpai-lab/KURE-Sparse-nano", "google/embeddinggemma-2"]:
    snapshot_download(m)
```

(`google/embeddinggemma-2` needs a token with the license accepted; the Rust
loader reads `HF_TOKEN` or `~/.cache/huggingface/token`. A full
`snapshot_download` of it succeeded without a token on the Windows box, but the
gpu01 box had a token in place.)

## Measured throughput (2026-11-09, `main` @ `ce93d66`)

`bench_text --count 128 --length 64 --max-length 512`;
`bench_sparse --n 64 --len 300`. `chunks/s` (`t/s` = tokens/s).

### gpu01, RTX PRO 6000 Blackwell Max-Q (sm_120), CUDA 12.9, `cuda:0`

| model | dtype | batch 8 | batch 32 | batch 64 | batch 128 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | f32 | 30.7 (5751 t/s) | 24.8 (4647) | 21.6 (4058) | 21.3 (3998) |
| Qwen3-Embedding-0.6B | bf16 | 26.3 (4931) | 13.5 (2539) | 10.8 (2028) | 16.2 (3043) |
| EmbeddingGemma 2 | f32 | 15.2 (2875) | 17.7 (3340) | 21.3 (4019) | 17.3 (3266) |
| EmbeddingGemma 2 | bf16 | 13.1 (2467) | — | — | — |

The gpu01 sweep (remaining Gemma2 bf16 sizes, the CPU points and the whole
`bench_sparse` sweep) was still running when this was written; its full output
is at `/tmp/gpu01_bench2.log` on gpu01.

The 64-core CPU measured 0.1 chunks/s on the same corpus (the box is shared
with other users' jobs, so CPU numbers there are not meaningful).

### windows, Ryzen 5 3600 (6 cores), CPU f32

| model | batch 8 | batch 32 | batch 64 | batch 128 |
| --- | --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | 0.95 (178 t/s) | 0.89 (167) | 0.89 (168) | 0.90 (168) |
| EmbeddingGemma 2 | 2.25 (424) | 2.23 (420) | 2.23 (420) | 2.22 (419) |

Sparse (KURE-Sparse-nano, f32, `--max-batch-tokens 4096`, `--n 64 --len 300`):
0.26 chunks/s (296 t/s) at batch 8, 0.26 chunks/s (299 t/s) at batch 128.
The in-crate `sparse_live::throughput_by_device_and_dtype` reports the same
order of magnitude on CPU (1.4 chunks/s for its longer 900-char chunks).

## Known issues

* `qwen3_live` compares CPU against `INL_TEST_DEVICE` (default `auto`); on a
  CPU-only build it compares the CPU with itself.
* sm_6x CUDA builds need `scripts/patch_candle_kernels_pre_volta.py` (see
  windows above).
* gpu01's `/data` is NFS: put cargo state, build dirs and the CUDA toolkit on
  local disk or everything takes an order of magnitude longer.
