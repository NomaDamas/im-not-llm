# Devices

How to build and run im-not-llm on the machines it is exercised on, with the
exact commands, environment variables and the issues each setup hits.

Registry dependencies use candle 0.11.1 (no workspace git patch). Verified
2026-11-10 on macOS (CPU/Metal), Windows (GNU CPU and MSVC CUDA), and gpu01
(CUDA). `inl-server` enables its CLI/vision dependencies by default;
`default-features = false` keeps only the daemon library and engine traits.

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
| `INL_VISION_ATTENTION_TILE` | positive query tile size for vision tiled attention; default 256, overridden by `MultimodalOptions::with_vision_attention_tile`; Metal SDPA ignores it. |

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

### Registry releases and shared daemon state

`inl-core`, `inl-text`, `inl-sparse` 0.1.0 and `inl-vision`, `inl-server`
0.1.1 are published on crates.io. Each upload ran Cargo's packaged-source
verification. A registry-only Rust consumer ran real Qwen3 inference with
`accelerate` (two finite unit-length 1024-d vectors); `cuda`, `accelerate`
and `mkl` resolve through all four text/daemon crates to candle. The daemon
library's active feature graph excludes `inl-vision` when default features
are disabled.
Linux registry-consumer `cargo check --features cuda` and
`cargo check --features mkl` both passed on gpu01.

MinSync's extraction branch, with git dependencies replaced by registry
versions and its candle patch removed, passed `cargo publish --dry-run`.
Manual shared-state QA found that the CLI omitted the default context from
its instance identity while MinSync included it. `inl-server` 0.1.1 includes
the actual engine default, so `minsync serve` adopts `inl serve`'s PID, URL
and default `~/.cache/im-not-llm/serve/<instance>` directory. The adopted
daemon successfully served a 1024-d embedding request.


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
engine` 1/1 (cosines >= 0.9999997), `inl-vision --test audio` 5/5 (embedding
cosines >= 0.9999998, audio tower max abs diff 5.6e-4 over 44.9).

Audio on gpu01 (`cuda:0` = card 2, `@ 5ae814e`): `inl-vision --test audio`
5/5 (embedding cosines >= 0.9999998), `--test engine` 1/1.

After upgrading to candle 0.11.1, the Windows CPU suite and full CUDA suite
(including live text/sparse/audio/image/video goldens) passed. The macOS
offline suite and all ignored Metal goldens also passed. The optimized vision
path passed 9 Mac unit tests, 8 Windows unit tests, and image/video goldens on
both Metal and Pascal CUDA. Manual Metal HTTP QA embedded an image, interleaved
text/video, and text in one request: three finite 768-d unit vectors; the image
alone versus batched cosine was 0.9999999999996.

## Video

`inl-vision --test video` (four cases: video-only, interleaved text + video,
`video_fps` 2 with 70 tokens/frame, and a 40 s clip resampled to 32 frames)
passes with `INL_TEST_DEVICE` set, cosines vs the reference >= 0.9999998:
Mac CPU and Metal (ffmpeg 9.0), gpu01 CUDA (`INL_FFMPEG=/data/home/jeffrey/bin/ffmpeg`,
ffmpeg 7.0 static) and windows CUDA (`C:\ffmpeg\bin` on `PATH`, 2025 nightly).
Frames are converted with `scale=flags=bitexact+accurate_rnd+full_chroma_int`;
without it ffmpeg 7 and 9 produce different RGB for the same clip (cosine
0.9991 on gpu01). `ffprobe` must sit next to `ffmpeg`.

Memory: the whole golden run on Mac CPU peaks at 4.9 GB RSS (f32 weights plus
the 32-frame, 4226-token sequence batched with two short ones). A 120 s
640x480 30 fps clip through `inl serve --multimodal --device metal` returns in
3.9 s, capped at 32 frames.

## Late interaction (pplx-embed-v2-late)

`inl-late --test late_live -- --ignored` compares every token vector of
`perplexity-ai/pplx-embed-v2-late-0.6b` against Sentence-Transformers f32.
The fixture includes short Korean/English queries, punctuation, and documents
crossing 64-token chunk boundaries. It exercises actual batches of eight,
not only a batch-size limit of eight. CPU, Metal and gpu01 CUDA pass; CUDA's
maximum component difference on the extended fixture is 1.03e-6. The tiny
fixture also honours `INL_TEST_DEVICE` and passes on Metal and CUDA.

The checkpoint's full-attention heads are 256 wide. candle instantiates the
Metal SDPA kernel for f32 at 256, but its tile does not fit the 32 KiB of
threadgroup memory, so creating the pipeline fails at the first forward
("Failed to create pipeline"). f32 therefore uses the tiled attention on Metal
(`/health` `attention: "tiled"`); bf16 keeps the fused kernel (`"fused"`).
The default remains f32. In a longer mixed-length reference probe (up to
1022 input tokens), optimized f32's maximum component difference was 1.33e-6.
bf16's difference from the f32 reference was 0.00676 (original implementation:
0.00816); the earlier 1.3e-3 observation covered only short inputs, not general
retrieval quality. No f32 golden tolerance was relaxed.

`inl serve --model perplexity-ai/pplx-embed-v2-late-0.6b` detects the kind from
`config.json` (`model_type: qwen3_5`) on Metal and gpu01 CUDA; `inl health`,
`inl stop` with `--late` find it. ClawGallery's own VDR library
(`clawgallery-vdr`, `embedding_search` and `late_interaction_score`) works
against it for queries and captions. Its `sync` also sends every image as
`kind: "image"`, which this text-only encoder refuses with 422, so a
ClawGallery index of images still needs an image-capable server.

### Warm inference optimization (2026-11-13)

M4 Max, 64 GiB, release build; one warmup and five measured encodes per case,
median milliseconds. Includes tokenization, inference and host vectors;
excludes download, model load and HTTP/JSON. Before is the original late
encoder from PR #11; after is the optimized encoder, with unchanged dtype,
batch limit (32) and padded-token budget (8192). Document inputs repeat
`Paris is the capital and largest city of France. ` 26 or 102 times; these
synthetic measurements are not a retrieval-quality or mixed-corpus benchmark.

| dtype | input tokens | actual batch | before ms | after ms | speedup |
|---|---:|---:|---:|---:|---:|
| f32 | 8 | 1 | 14.87 | 9.40 | 1.58x |
| f32 | 8 | 8 | 43.88 | 15.66 | 2.80x |
| f32 | 262 | 1 | 54.57 | 39.96 | 1.37x |
| f32 | 262 | 8 | 483.65 | 303.38 | 1.59x |
| f32 | 1022 | 1 | 192.33 | 131.43 | 1.46x |
| f32 | 1022 | 8 | 1977.79 | 1162.89 | 1.70x |
| bf16 | 8 | 1 | 12.95 | 7.87 | 1.65x |
| bf16 | 8 | 8 | 38.52 | 10.24 | 3.76x |
| bf16 | 262 | 1 | 48.07 | 35.64 | 1.35x |
| bf16 | 262 | 8 | 442.89 | 287.25 | 1.54x |
| bf16 | 1022 | 1 | 170.36 | 125.33 | 1.36x |
| bf16 | 1022 | 8 | 1764.54 | 1108.99 | 1.59x |

Reproduce with `cargo run --release -p inl-late --example bench_late --
--device metal --dtype f32 --repeats 5`; substitute `bf16` for the other path.
`--batch-size` sweeps microbatch size; `--document-repeats` and `--only document`
vary document length. Each output JSON row reports the actual token count.

Changes: cache sequence masks and inverse gather indices per model; smaller
power-of-two chunks for short inputs; skip unused single-chunk state work;
use one shared prefix-sum GEMM instead of broadcast copies; remove per-chunk
contiguous copies; gather only the inverse blocks actually used. Metal uses
fused triangular inversion, causal depthwise convolution, short-sequence
delta recurrence, and gated normalization (retaining model-dtype rounding).
Pipelines belong to the model and are released on unload; only the requested
dtype's pipelines are built. Full tiled attention folds grouped query heads
into GEMM rows instead of repeating keys/values. Unpadded bidirectional
inputs need no attention mask. Output projection is a flattened shared GEMM.

Combining input projections was benchmarked and rejected: the extra layout
copies made document inference slower. f32 and batching defaults are unchanged.
Long documents still benefit from microbatch sweeps; no one size wins for both
short queries and long documents. CPU-only Mac deployments should build with
`--features accelerate`: optimized CPU query batch eight took 59.57 ms with
Accelerate versus 122.30 ms without it (three-run medians). CUDA full attention
remains tiled; gpu01 verifies correctness, not a controlled speedup measurement.

Manual Metal HTTP QA: mixed query/document roles and lengths preserve output
order; two concurrent requests return identical vectors; unload followed by
reload reproduces the same vectors and shutdown drains successfully.

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

### gpu01 controlled CUDA A/B (2026-11-10)

Pinned to physical card 2 (`GPU-0ad354e1-d2d1-9aba-d6a0-cdec1e78ec53`),
idle before the first run. Before: `d6da2ef` (parent of `55acc36`, pinned
candle git revision); after: `bb37000` (CUDA optimizations plus registry
candle 0.11.1). This compares complete builds, not an isolated attribution
to one kernel change. Other cards remained busy; card 2 retained about
20–22 GiB of another process's allocations and temperatures ranged 80–92°C.
Results therefore include shared-host/thermal variation, not exclusive-card
laboratory precision.

Three interleaved repeats, reversing variant order on the second repeat.
Text: `--count 1024 --length 128`; sparse: `--n 1024 --len 128
--max-batch-tokens 65536`. Model load is outside reported inference time;
the larger corpus amortizes the text harness's cold first forward. Dense
batch sizes are requested maxima: before uses the old 4096-token cap, after
uses its device-memory cap. Sparse uses the same explicit cap in both builds.
Cells are median chunks/s, **before → after (ratio)**; TF32 rows are separate
opt-in after-build measurements, not the default. Raw samples, including
tokens/s, actual commands and GPU telemetry: `docs/cuda-ab-20261110.jsonl`.

| model | dtype / variant | batch 8 | batch 32 | batch 64 | batch 128 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | f32 | 41.0 → 44.4 (1.08×) | 35.1 → 45.1 (1.29×) | 34.2 → 41.4 (1.21×) | 35.7 → 40.6 (1.14×) |
| Qwen3-Embedding-0.6B | f32 + TF32 | 76.2 | 64.2 | 69.3 | 78.2 |
| Qwen3-Embedding-0.6B | bf16 | 67.7 → 94.2 (1.39×) | 95.9 → 108.3 (1.13×) | 120.9 → 146.2 (1.21×) | 92.7 → 154.4 (1.67×) |
| EmbeddingGemma 2 | f32 | 144.9 → 156.8 (1.08×) | 141.4 → 128.0 (0.91×) | 113.5 → 96.7 (0.85×) | 94.8 → 86.7 (0.91×) |
| EmbeddingGemma 2 | f32 + TF32 | 213.2 | 230.1 | 129.6 | 137.8 |
| EmbeddingGemma 2 | bf16 | 135.8 → 164.5 (1.21×) | 182.8 → 138.4 (0.76×) | 143.7 → 135.5 (0.94×) | 137.9 → 128.3 (0.93×) |
| KURE-Sparse-nano | f32 | 49.9 → 53.1 (1.06×) | 47.7 → 48.8 (1.02×) | 53.0 → 55.2 (1.04×) | 65.3 → 60.9 (0.93×) |
| KURE-Sparse-nano | f32 + TF32 | 71.6 | 59.7 | 65.2 | 71.5 |
| KURE-Sparse-nano | f16 | 94.5 → 111.0 (1.17×) | 82.0 → 72.4 (0.88×) | 94.3 → 100.5 (1.07×) | 70.5 → 88.6 (1.26×) |

Conclusion: Qwen3 improves across these batch sizes; EmbeddingGemma 2 does
**not** have a blanket speedup, and sparse gains are modest/mixed. bf16 is
faster than f32 throughout this after-build text sweep, reversing the earlier
small-corpus result. Keep f32's TF32 accumulation opt-in: throughput alone
does not justify changing the numerical default. No tolerance was relaxed.

TF32 numeric check on the same card: Gemma2's two text goldens and Qwen3's
CPU/CUDA comparison passed (Qwen3 cosines 0.9999989 / 0.9999991). KURE-Sparse
**failed** the existing golden and padding-invariance tests: query weight
error 0.0025709867 exceeds the unchanged 0.002 tolerance, one sparse term
changed, and doc 0's selected indices differed when padded. The normal
`bf16,f16` accumulation setting passed all three sparse correctness tests.
Decision: do not enable TF32 by default; it is not reference-safe for sparse
retrieval. This failed opt-in experiment is expected evidence, not a passing
suite and not a reason to relax tolerances.

### CUDA vision optimization and profile

The before binary uses the pre-optimization vision path at `bb37000`; after
includes the fused axial RoPE and host-retained pooling metadata from
`023ab1d` (same candle 0.11.1). Position lookup and padding masks already run
on-device; invariant RoPE frequencies now upload once per tower. Pool hidden
states stay on-device via `index_add`/`index_select`; the pooler borrows the
preprocessor's positions/padding rather than downloading their device copies.
Metal keeps its fused SDPA path. CUDA retains portable tiled attention, so
Pascal does not gain a flash-attention dependency it cannot run.

Physical card 2, three interleaved repeats (reverse order on repeat 2),
`bench_vision --device cuda --n 16 --size 320x240 --budget 280`, one warmup
image before timing. Median images/s:

| dtype | before, tile 256 | after 128 | after 256 | after 512 | after 1024 | after 2048 |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | 6.569 | 5.355 | 7.065 | 5.181 | 4.999 | 5.372 |
| bf16 | 7.590 | 10.227 | 7.650 | 7.860 | 7.627 | 9.049 |

Keep the 256-row default: larger tiles did not consistently improve f32,
and bf16's best tile differed. The f32 default median improved 7.6%; bf16
was essentially unchanged (+0.8%). Shared-host variation is large (after
f32 tile 256: 4.760–7.729 images/s); do not claim a universal speedup.
The per-tower Rust option and `INL_VISION_ATTENTION_TILE` permit explicit
tuning. Raw samples: `docs/cuda-vision-ab-20261110.jsonl`.

`nsys profile --trace=cuda,nvtx --sample=none` over a warmup plus four f32
images confirmed 192 fused `rope_thd_f32` calls replacing the axial arithmetic
chain: `bmul_f32` launches 585 → 201, `badd_f32` 918 → 726, and `uneg_f32`
192 → 0. H2D copies fell 3507 → 3121 and D2H 493 → 487; remaining copies
include model load, final outputs and the existing multimodal text merge.
GEMMs, broadcast-add and softmax still dominate GPU time. Absolute profiler
times include startup and tracing overhead and are not inference-only
speedups. Reports: `docs/cuda-vision-profile-{before,after}-20261110.csv`;
full `.nsys-rep` and SQLite exports remain under `/tmp/inl-vision-*` on gpu01.

Validation: optimized image and interleaved/video goldens passed on gpu01
(minimum cosine 0.99999982), Windows CUDA (0.99999988), and Metal. Fused axial
RoPE equivalence tests passed on CPU, Metal and CUDA; the default tile and
alternate tile partitioning preserve attention results. Manual Metal HTTP
QA is recorded above. Existing golden tolerances were not changed.


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
