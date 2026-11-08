# Devices

How to build and run im-not-llm on the machines it is exercised on, with the
exact commands, environment variables and the issues each setup hits.

Everything below was verified on 2026-11-09 against the workspace at
`crates/{inl-core,inl-text,inl-sparse,inl-server}` (the `inl-vision` crate and
the Node bindings are developed separately).

## Env vars, once

| Var | Meaning |
| --- | --- |
| `CUDA_HOME` / `CUDA_PATH` | CUDA toolkit root; `cudarc`/`cudaforge` find `nvcc` and the libs here. |
| `CUDA_COMPUTE_CAP` | Overrides the `nvidia-smi`-detected compute capability for the kernel build. |
| `NVCC_CCBIN` | Host compiler `nvcc` hands its generated code to (Windows). |
| `CUDA_VISIBLE_DEVICES` | Which GPU candle opens as `cuda:0`. |
| `HF_HOME` | Model cache the engines *and* the `hf-hub` client read. Set it to a populated cache to avoid re-downloading. |
| `INL_TEST_DEVICE` | `cpu`/`metal`/`cuda` for the ignored live tests (`gemma2_live`, `sparse_live`). |
| `INL_CACHE_DIR` | Model cache when `HF_HOME` is not set (`~/.cache/im-not-llm/models` by default). |

Feature flags: `cuda` (NVIDIA), `cudnn` (implies `cuda`; needs cuDNN),
`accelerate` (macOS CPU BLAS), `mkl` (x86 CPU BLAS).

## macOS (Apple Silicon, Metal)

```sh
cargo test                                    # CPU + Metal, 145 tests + 7 ignored live
cargo test -p inl-text --test gemma2_live -- --ignored --nocapture
HF_HOME=... INL_TEST_DEVICE=metal cargo test -p inl-sparse --test sparse_live -- --ignored --nocapture
```

## gpu01 — Linux x86_64, RTX PRO 6000 Blackwell (sm_120), driver 580

The preinstalled toolkit was CUDA 12.0 (`/usr/bin/nvcc`), which predates
Blackwell: `sm_120` needs CUDA >= 12.8, and `sm_120a` (= what `cudaforge`
generates for compute capability 12.0) needs CUDA >= 12.9. A user-space
toolkit was installed instead of using the system one:

```sh
# one-time: install CUDA 12.9.1 into a writable prefix, no root needed
curl -LO https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
mkdir -p /tmp/cuda_tmp
bash cuda_12.9.1_575.57.08_linux.run --silent --override --toolkit \
     --toolkitpath=/tmp/cuda-12.9 --no-opengl-libs --no-man-page --tmpdir=/tmp/cuda_tmp
```

* Home (`/data`) is **NFS**; `/` (and so `/tmp`) is a **local NVMe**. The
  installer `chgrp -R`s its whole ~10 GB tree, which takes tens of minutes on
  NFS, so put `--tmpdir` and `--toolkitpath` on local disk. Cargo is much
  faster with `CARGO_TARGET_DIR` and `CARGO_HOME` on local disk too.
* `cudaforge` picks the arch from `CUDA_COMPUTE_CAP`, else `nvidia-smi`. With
  the 12.9 toolkit, either works: capability 12.0 becomes `sm_120a`.
  (Option (b), PTX for an older arch, is not clean here: `CUDA_COMPUTE_CAP=90`
  yields `sm_90a`, which is arch-specific and cannot be JITed forward to
  `sm_120`.)

```sh
export CUDA_HOME=/tmp/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH          # or use $CUDA_HOME/bin/nvcc explicitly
export CARGO_HOME=/tmp/cargo-home          # local, faster than NFS ~/.cargo
export CARGO_TARGET_DIR=/tmp/inl-target-gpu01cuda
export HF_HOME=/data/home/jeffrey/.cache/huggingface
export CUDA_VISIBLE_DEVICES=1              # other GPUs run other people's jobs
cargo test -p inl-core -p inl-text -p inl-sparse -p inl-server --features cuda
INL_TEST_DEVICE=cuda cargo test --features cuda -p inl-text -p inl-sparse -- --ignored --nocapture --test-threads=1
```

Use `-j16` or fewer: the box is shared.

## windows — Windows 11, TITAN Xp (sm_61), CUDA toolkit 11.8

Default Rust toolchain is `stable-x86_64-pc-windows-gnu`.

```bat
set "CARGO_TARGET_DIR=C:\Users\vkehfdl1\im-not-llm\target-win"
set "HF_HOME=C:\Users\vkehfdl1\.cache\huggingface"
cargo test -p inl-core -p inl-text -p inl-sparse -p inl-server
```

CPU builds and tests fine with the GNU toolchain. **CUDA cannot be built on
this machine**: `nvcc` needs `cl.exe`, and no Visual Studio / Build Tools (no
`vcvars64.bat`, no `cl.exe`, no `vswhere`) is installed; the `2022`/`2019`
directories under `Program Files (x86)\Microsoft Visual Studio` are empty.
`nvcc` fails with `fatal : Cannot find compiler 'cl.exe' in PATH`. And even
with `cl.exe` present, `candle-kernels` passes `-Xcompiler -fPIC` for
non-`msvc` Rust targets, which MSVC rejects, so the CUDA build needs the
`stable-x86_64-pc-windows-msvc` toolchain too.

Live tests: `gemma2_live` and `sparse_live` pass on CPU; `qwen3_live` fails
because it compares CPU against **Metal** (macOS-only) and has no
`INL_TEST_DEVICE` handling.

## Model cache and `INL_TEST_DEVICE`

The engines cache models under `HF_HOME` when set, else
`~/.cache/im-not-llm/models`. **Populate that cache before running the live
tests.** If two live tests start at once on a cold cache they race: one
download wins, the other's `model.safetensors` fetch fails and the loader
falls through to `model.safetensors.index.json`, which does not exist for a
single-file checkpoint, and the test dies with a 404. Set `HF_HOME` to a
populated cache (or pass `--test-threads=1` on a cold one).

Pre-populate with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3-Embedding-0.6B", "nlpai-lab/KURE-Sparse-nano", "google/embeddinggemma-2"]:
    snapshot_download(m)
```

(`google/embeddinggemma-2` needs a token with the license accepted; the Rust
loader reads `HF_TOKEN` or `~/.cache/huggingface/token`.)
