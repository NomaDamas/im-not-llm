# im-not-llm

On-device inference for models that are **not** LLMs: dense text embeddings,
learned-sparse (SPLADE) encoders, late-interaction (multi-vector) text
encoders, and multimodal (image / video / audio) embeddings.
Rust on [candle](https://github.com/huggingface/candle), CPU / Metal / CUDA.

Use it three ways, all backed by the same engines:

| | How |
|---|---|
| **In-process Rust** | depend on `inl-text`, `inl-sparse`, `inl-late`, `inl-vision` |
| **In-process TypeScript** | the Node-API addon in `bindings/node` |
| **Daemon** | `inl serve` — loopback HTTP with `/health`, plus a programmatic lifecycle API |

## Install

```sh
cargo install inl-server --version 0.2.0 --locked
# NVIDIA host with the CUDA toolkit installed:
cargo install inl-server --version 0.2.0 --locked --features cuda
```

For Rust applications:

```toml
[dependencies]
inl-text = "0.1.0"
inl-sparse = "0.1.1"
inl-late = "0.1.0"
inl-vision = "0.1.1"
inl-server = { version = "0.2.0", default-features = false }
```

`inl-server` without default features exposes the daemon library without its
CLI or vision dependency. The crates use the registry release candle 0.11.1;
downstream applications do not need a candle git patch. `cuda`, `accelerate`
and `mkl` forward to the engine dependencies. Pascal CUDA builds need the
kernel compatibility patch described in `docs/DEVICES.md`.

## Models

| Model | Crate | Output | Notes |
|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | `inl-text` | 1024-d dense | f16 on Metal by default, f32 elsewhere |
| `google/embeddinggemma-2` (text) | `inl-text` | 768-d dense | f32 or bf16 only (f16 overflows to NaN); Matryoshka 128/256/512/768 |
| `nlpai-lab/KURE-Sparse-nano` | `inl-sparse` | 36282-term sparse | f32 on every device (f16 changes which terms clear the threshold) |
| `google/embeddinggemma-2` (image, video, audio, interleaved) | `inl-vision` | 768-d dense, same space as the text tower | f32 or bf16 (bf16 on a GPU only); never f16 |
| `perplexity-ai/pplx-embed-v2-late-0.6b` | `inl-late` | one 128-d vector per token | f32 by default; bf16 is faster but less exact |

Every model is checked against its Hugging Face reference with golden tests
(`crates/*/tests`); the tolerances are the ones in the tests.

## Devices

| Device | Status |
|---|---|
| CPU | all models |
| Metal (Apple Silicon) | all models; fused attention for the vision tower |
| CUDA (NVIDIA) | `--features cuda`; Qwen3, EmbeddingGemma 2 text and KURE-Sparse-nano verified on a Blackwell GPU (sm_120). Needs CUDA ≥ 12.9 for Blackwell |

`docs/DEVICES.md` has the exact build commands, environment variables, measured
throughput and the known problems per machine.

## The daemon

```sh
cargo build --release -p inl-server --bin inl

inl serve                         # Qwen3-Embedding-0.6B, dense
inl serve --model nlpai-lab/KURE-Sparse-nano   # sparse: the kind follows the model
inl serve --sparse                # KURE-Sparse-nano, the sparse default
inl serve --multimodal            # EmbeddingGemma 2: text + image + video + audio
inl serve --model perplexity-ai/pplx-embed-v2-late-0.6b   # late interaction: one vector per token
inl serve --multimodal --modalities text,audio   # load only the towers listed
inl serve --model <repo-or-dir> --device cuda --dtype bf16

inl health --multimodal           # prints /health JSON; exit 0 only when status is "ok"
inl unload --multimodal           # drop the weights, keep the process
inl stop   --multimodal           # drain and exit
inl devices                       # cpu / metal / cuda availability of this build
```

Without `--sparse` / `--multimodal` / `--late`, `serve` picks the kind from the
model's `config.json`: a ModernBERT checkpoint is a SPLADE encoder, a Qwen3.5
checkpoint a late-interaction encoder, anything else a dense text encoder.
Dense and sparse both serve OpenAI `POST /v1/embeddings` (sparse rows are
`[{index, value}]`) next to TEI's `/embed` / `/embed_sparse`, so switching
between dense and sparse is only a `--model` change.
Multimodal is always explicit, because EmbeddingGemma 2 also serves dense text.

Late-interaction inference defaults to f32 (including on Metal). Its Metal
delta-rule, convolution and normalization kernels are fused; bf16 is an
explicit speed/precision tradeoff, not the default. Run
`cargo run --release -p inl-late --example bench_late -- --device metal --dtype f32`
for warm latency/throughput; measured before/after results and precision
limits are in [the device guide](docs/DEVICES.md#late-interaction-pplx-embed-v2-late).
For Mac CPU deployments, enable `--features accelerate` at build time.

`health`, `stop` and `unload` find the running server by **model** (a kind
flag only narrows the search); they do not need the device or dtype it was
started with. If several servers match, they refuse and list them — pass
`--state-dir` to pick one.

One model instance is served once: a second `inl serve` for the same instance
adopts the running server instead of loading another copy. State lives in
`~/.cache/im-not-llm/serve/<instance>` (`%LOCALAPPDATA%\im-not-llm\serve` on
Windows). The server binds loopback only: it has no authentication.

### Routes

| Route | Engine | Body |
|---|---|---|
| `GET /health` | all | — |
| `POST /embed` | dense, multimodal (text only) | TEI: `{"inputs": [...], "input_type": "query"\|"passage"}` → `[[f32]]` |
| `POST /embed` | late | `{"model"?, "dimensions"?, "inputs": [{"kind": "text"\|"caption", "role": "query"\|"document", "value"}]}` → `{"model", "dimensions", "embeddings": [[[f32]]]}` (one L2-normalised vector per token; the ClawGallery VDR protocol). A `model`/`dimensions` other than the served one is a 400 |
| `POST /v1/embeddings` | dense, multimodal, sparse | OpenAI: `{"input": [...]}`; multimodal accepts items `{"type": "image", "data": "<base64>"}`; sparse returns each `embedding` as `[{"index", "value"}]` rows (no `base64` encoding) |
| `POST /embed_sparse` | sparse | TEI sparse: → `[[{"index", "value"}]]` |
| `POST /embed_multimodal` | multimodal | `{"inputs": [{"text"?, "images"?, "videos"?, "audios"?}]}` (media base64) → `{"embeddings", "model", "dimensions"}` |
| `POST /admin/unload`, `POST /admin/shutdown` | all | loopback only |

Errors are JSON `{"error", "error_type"}`: 400 for undecodable base64, 413 for
oversized requests, 422 for a modality the engine does not list (for example
one left out with `--modalities`) or an input with no text and no media.

`/health` reports `status` (`loading` · `ok` · `stopping`), `kind`, `model_id`,
`dimension`, `dtype`, `device`, `attention`, `max_length`, `modalities`,
`loaded`, `requests_total`, `uptime_seconds`, `pid`, `url` and
`build.features`.

### Multimodal input

EmbeddingGemma 2 puts text, images, video and audio in one 768-d space, so a
text query can be scored against image or audio embeddings. Media inside a
text input are placed by `<|image|>` / `<|video|>` / `<|audio|>`
placeholders, consumed in order:

```json
{"inputs": [{"text": "A gradient picture: <|image|> with a blue box.", "images": ["<base64 png>"]}]}
```

An input with media and no text is embedded as its items alone (one
placeholder per item, images then videos then audio, as the reference
processor joins them). Video is decoded with `ffmpeg` + `ffprobe` (on `PATH`, or
`INL_FFMPEG`) and sampled as the reference processor does: 1 frame per second
(`video_fps`), at most 32 frames per video (longer clips are resampled
uniformly), fewer if the frames would not fit the 8192-token context. Audio (WAV, MP3, FLAC, ... anything
`ffmpeg` reads) is resampled to 16 kHz mono and cut at 30 s (750 soft tokens,
one per 40 ms). Image token budget (70 / 140 / 280 / 560 / 1120) trades speed
for detail. `--modalities` (Node: `modalities`) skips loading the vision tower
(no `image`, no `video`) or the audio tower (no `audio`) and their memory.

The vision tower keeps position lookup, padding masks and pooled hidden states
on the device. Axial RoPE uses fused rotary kernels; its frequency table is
cached per tower. Tiled attention uses 256 query rows by default; set
`MultimodalOptions::with_vision_attention_tile(n)` or
`INL_VISION_ATTENTION_TILE` to trade score-buffer memory for fewer matmuls.
Metal's automatic fused attention path ignores this tile setting.

## Programmatic lifecycle

Rust (`inl-server`):

```rust
let daemon = Daemon::spawn(Engine::multimodal(Arc::new(embedder)), options).await?;
daemon.health();                       // status, url, device, …
daemon.stop().await?;                  // drains in-flight requests

let client = DaemonClient::discover(&state_dir, &instance).unwrap();
client.wait_ready(Duration::from_secs(60)).await?;
client.unload().await?;
client.stop().await?;
let live = find_instances(&serve_root, "google/embeddinggemma-2", EngineKind::Multimodal).await;
```

TypeScript (`bindings/node`, `npm run build` first):

```js
const { TextEmbedder, SparseEncoder, Daemon, DaemonClient } = require('@nomadamas/im-not-llm')

const embedder = new TextEmbedder({ id: 'Qwen/Qwen3-Embedding-0.6B' })
const vectors = await embedder.embed(['hello'])   // Float32Array[]

const daemon = await Daemon.spawn({ kind: 'multimodal', model: 'google/embeddinggemma-2' }, {})
await fetch(`${daemon.url()}/health`)
await daemon.stop()
```

See `bindings/node/README.md` for every class, `AbortSignal` support and the
`ERR_INL_*` error codes.

## Rust API

```rust
use inl_text::{TextEmbedder, TextOptions};

let embedder = TextEmbedder::new(TextOptions::default().with_id("Qwen/Qwen3-Embedding-0.6B"))?;
let vectors = embedder.embed(&["hello".to_string()]).await?;   // unit-length, input order
```

The model loads on first use and stays resident until `unload()`; a failed load
is not cached. A forward pass that exceeds its deadline cannot be cancelled, so
after a timeout every later call in that process fails fast: restart the process.

## Layout

```
crates/inl-core     device/dtype selection, Hugging Face hub client, attention, deadlines, engine traits
crates/inl-text     Qwen3-Embedding, EmbeddingGemma 2 text
crates/inl-sparse   ModernBERT + SPLADE pooling (KURE-Sparse-nano)
crates/inl-late     Qwen3.5 text tower + multi-vector heads (pplx-embed-v2-late)
crates/inl-vision   EmbeddingGemma 2 vision + audio towers, image/video/audio preprocessing, multimodal merge
crates/inl-server   daemon library, `inl` CLI, lifecycle client
bindings/node       Node-API addon
```

## Environment

| Variable | Meaning |
|---|---|
| `HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN` | Hugging Face cache, mirror and token |
| `INL_CACHE_DIR` | model cache when `HF_HOME` is unset (`~/.cache/im-not-llm/models`) |
| `INL_FFMPEG` | `ffmpeg` binary for video and audio |
| `INL_TEST_DEVICE` | `cpu` / `metal` / `cuda` for the ignored live tests |
| `INL_VISION_ATTENTION_TILE` | positive query tile size for vision tiled attention; default 256, overridden by the Rust option |

## Tests

```sh
cargo test --workspace                                     # offline
HF_HOME=… cargo test --release --workspace -- --ignored    # real checkpoints
```

## License

MIT OR Apache-2.0. `inl-text`'s Qwen3 code derives from
[fastembed-rs](https://github.com/Anush008/fastembed-rs) (Apache-2.0); see
`THIRD_PARTY_NOTICES.md`.
