# @nomadamas/im-not-llm

im-not-llm's on-device engines as an in-process [Node-API](https://nodejs.org/api/n-api.html) addon, on
[candle](https://github.com/huggingface/candle) (Metal on macOS, CUDA when built with the feature, else CPU):

- **dense text embedding** — Qwen3-Embedding, EmbeddingGemma 2 (`TextEmbedder`);
- **learned sparse (SPLADE)** — KURE-Sparse-nano (`SparseEncoder`);
- **loopback daemon** — the same server as `inl serve`, in this process (`Daemon`) or another (`DaemonClient`).

Node-API is ABI-stable: one binary per platform works on every Node/Electron version >= 18, with no per-version rebuild.

| Platform | Package | Device |
|---|---|---|
| macOS arm64 | `@nomadamas/im-not-llm-darwin-arm64` | Metal (CPU fallback) |
| Windows x64 | `@nomadamas/im-not-llm-win32-x64-msvc` | CPU |
| Linux x64 (glibc) | `@nomadamas/im-not-llm-linux-x64-gnu` | CPU |

A `cuda` variant (`@nomadamas/im-not-llm-linux-x64-gnu-cuda`) is documented but not built here; build it with
`cargo build -p inl-node --release --features cuda` after adding the `cuda` feature to `bindings/node/Cargo.toml`.

## Install

```bash
npm install @nomadamas/im-not-llm            # or: bun add @nomadamas/im-not-llm
```

The weights are resolved exactly as the `inl` CLI does, so both share one copy: `cacheDir` / `modelCacheDir`, else
`INL_CACHE_DIR`, else `HF_HOME`, else the per-user cache. `HF_ENDPOINT` and `HF_TOKEN` are honoured. They are read
from the process environment when a load starts, so set them (`process.env`) before the first call.

## Dense text: `TextEmbedder`

```ts
import { TextEmbedder } from '@nomadamas/im-not-llm'

const embedder = new TextEmbedder({
  id: 'Qwen/Qwen3-Embedding-0.6B', // default for the daemon, required here
  // queryPrefix / passagePrefix, batchSize, maxLength, timeoutSeconds,
  // idleUnloadSeconds, ... (see index.d.ts)
})

const identity = await embedder.identity()
// { id: 'Qwen/Qwen3-Embedding-0.6B', dimension: 1024, dtype: 'f16', device: 'metal', maxLength: 2048 }

const passages = await embedder.embed(['first passage', 'second passage']) // Float32Array[]
const query = await embedder.embedQuery('which passage?') // Float32Array

await embedder.dispose() // releases the model; later calls reject
```

`embed` returns unit-length `Float32Array` vectors in input order; `embedQuery` returns one. The model loads on first
use and stays resident until `unload()`, `dispose()` or `idleUnloadSeconds`. A call may take `{ signal }` to abort.

## Learned sparse: `SparseEncoder`

```ts
import { SparseEncoder } from '@nomadamas/im-not-llm'

const encoder = new SparseEncoder({ id: 'nlpai-lab/KURE-Sparse-nano' })

const documents = await encoder.encode(['the quick brown fox', 'a slow green turtle'])
const query = await encoder.encodeQuery('quick fox')
// each: { indices: Uint32Array, values: Float32Array }  // indices strictly ascending, values > 0
```

## Daemon lifecycle

`Daemon.spawn` serves the instance in this process (loading the model) or adopts a server already serving it. The
server is loopback plain HTTP — no authentication, so it only binds `127.0.0.1` / `::1`.

```ts
import { Daemon, DaemonClient } from '@nomadamas/im-not-llm'

const daemon = await Daemon.spawn(
  { kind: 'dense', model: 'Qwen/Qwen3-Embedding-0.6B' }, // or kind: 'sparse'
  { stateDir: '/tmp/inl-serve' },                        // optional; else the per-user default
)
console.log(daemon.url())      // http://127.0.0.1:<port>
daemon.health()                // { status: 'ok', modelId, dimension, device, ... }
await daemon.stop()            // 'signal' | 'idle' | 'wedged'

const client = DaemonClient.discover({ model: 'Qwen/Qwen3-Embedding-0.6B', stateDir: '/tmp/inl-serve' })
if (client) {
  const health = await client.waitReady(60_000) // polls /health until status === 'ok'
  await client.unload()                          // POST /admin/unload
  await client.stop()                            // POST /admin/shutdown
}

const response = await fetch(daemon.url() + '/health') // ordinary HTTP
```

`/embed` (dense) and `/embed_sparse` (sparse) speak the TEI protocol; `/v1/embeddings` speaks the OpenAI one.

## Errors

A rejected call carries `error.code`:

| Code | Meaning |
|---|---|
| `ERR_INL_CONFIG` | A setting is invalid or unsupported on this build / device (also thrown synchronously by a constructor). |
| `ERR_INL_INFERENCE` | Loading a model or running inference failed (e.g. no network and no cached weights). |
| `ERR_INL_TIMEOUT` | A forward pass exceeded its deadline. The stuck thread cannot be cancelled: **every later call fails fast; restart the process**. |
| `ERR_INL_INVALID_INPUT` | The caller's input cannot be processed. |
| `ERR_INL_ABORTED` | The call was cancelled through its `AbortSignal`. |
| `ERR_INL` | Anything else. |

## Threads, abort and deadlines

Inference runs on the tokio blocking pool; the JS event loop is never blocked.

- `signal` (an `AbortSignal`) rejects the call with `signal.reason` and skips forward passes not yet started. A pass
  already running cannot be interrupted and finishes in the background; the embedder stays usable.
- Each forward pass has a deadline (`perBatchTimeoutSeconds`, else `timeoutSeconds`, else the engine default). When it
  expires the call rejects with `ERR_INL_TIMEOUT`.

## Development

```bash
npm install
npm run build                     # napi build --platform --release → binding.js/.d.ts + *.node
node --test "__test__/*.test.mjs" # tiny fixtures, no network

# Real models (downloads into the cache on first run):
INL_LIVE=1 HF_HOME=/path/to/hf node --test --test-concurrency=1 "__test__/live.test.mjs"
```

`binding.js` / `binding.d.ts` are generated by the build; `index.js` / `index.d.ts` are the hand-written public surface
(they add `AbortSignal` support and `error.code`).
