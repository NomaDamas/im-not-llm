/** `error.code` of a rejected call. */
export type InlErrorCode =
  /** A forward pass exceeded its deadline. The stuck thread cannot be cancelled: restart the process. */
  | 'ERR_INL_TIMEOUT'
  /** A setting is invalid or unsupported on this build / device. */
  | 'ERR_INL_CONFIG'
  /** Loading a model or running inference failed. */
  | 'ERR_INL_INFERENCE'
  /** The caller's input cannot be processed. */
  | 'ERR_INL_INVALID_INPUT'
  /** The call was cancelled through its `AbortSignal`. */
  | 'ERR_INL_ABORTED'
  /** Anything else. */
  | 'ERR_INL'

export interface CallOptions {
  /**
   * Abort the call. Forward passes not yet started are skipped and the promise
   * rejects with `signal.reason`; a pass already running finishes in the
   * background because it cannot be interrupted.
   */
  signal?: AbortSignal
}

/** Settings for {@link TextEmbedder}. Omitted fields take the engine's defaults. */
export interface TextEmbedderOptions {
  /** A Hugging Face repo id or a local model directory (required). */
  id?: string
  /** `auto` (default), `cpu`, `metal` or `cuda`. */
  device?: string
  /** `auto` (default), `f32`, `f16` or `bf16`. */
  dtype?: string
  /** `auto` (default), `eager`, `tiled` or `fused`. */
  attention?: string
  /** Tokens kept per input (default 2048). */
  maxLength?: number
  /** Most inputs per forward pass (default 64). */
  batchSize?: number
  /** Cap on padded tokens per forward pass (default 4096). */
  maxBatchTokens?: number
  /** Default deadline for one forward pass, in seconds (default 60). */
  timeoutSeconds?: number
  /** Deadline for one forward pass; overrides `timeoutSeconds`. */
  perBatchTimeoutSeconds?: number
  /** Model cache directory (else the shared `HF_HOME` cache). */
  cacheDir?: string
  /** Prepended by {@link TextEmbedder.embedQuery}. */
  queryPrefix?: string
  /** Prepended by {@link TextEmbedder.embed}. */
  passagePrefix?: string
  /** Matryoshka output dimension (EmbeddingGemma 2 only). */
  outputDim?: number
  /**
   * Release the model after this many idle seconds; it reloads on the next
   * call. Default: stay resident until `unload()` / `dispose()`.
   */
  idleUnloadSeconds?: number
}

/** The dense embedding space the vectors belong to. */
export interface TextIdentity {
  /** The model id as configured: a hub repo id or local directory. */
  id: string
  dimension: number
  /** Resolved precision: `f32`, `f16` or `bf16`. */
  dtype: string
  /** Resolved device: `cpu`, `metal` or `cuda`. */
  device: string
  maxLength: number
}

/**
 * In-process dense text embedder (Qwen3-Embedding, EmbeddingGemma 2). The
 * model loads on first use and stays resident until `unload()`, `dispose()` or
 * `idleUnloadSeconds`.
 */
export class TextEmbedder {
  /** Invalid settings throw synchronously with `error.code === 'ERR_INL_CONFIG'`. */
  constructor(options?: TextEmbedderOptions)
  /** Model id, dimension, dtype and device. Loads the model if it is not resident. */
  identity(): Promise<TextIdentity>
  /** Embed passages (`passagePrefix` applied): unit-length vectors in input order. */
  embed(texts: string[], options?: CallOptions): Promise<Float32Array[]>
  /** Embed a search query (`queryPrefix` applied). */
  embedQuery(text: string, options?: CallOptions): Promise<Float32Array>
  /** True while the model weights are resident. */
  readonly isLoaded: boolean
  /** Release the model; the next call reloads it. Resolves to whether one was resident. */
  unload(): Promise<boolean>
  /** Release the model and reject every later call. */
  dispose(): Promise<void>
}

/** Settings for {@link SparseEncoder}. Omitted fields take the encoder defaults. */
export interface SparseEncoderOptions {
  /** A Hugging Face repo id or a local model directory (required). */
  id?: string
  /** `auto` (default), `cpu`, `metal` or `cuda`. */
  device?: string
  /** `auto` (default), `f32`, `f16` or `bf16`. */
  dtype?: string
  /** Context length in tokens (default 512). */
  maxLength?: number
  /** Inputs per forward pass (default 64). */
  batchSize?: number
  /** Cap on padded tokens per forward pass (default 1024). */
  maxBatchTokens?: number
  /** Default deadline for one forward pass, in seconds (default 60). */
  timeoutSeconds?: number
  /** Deadline for one forward pass; overrides `timeoutSeconds`. */
  perBatchTimeoutSeconds?: number
  /** Model cache directory (else the shared `HF_HOME` cache). */
  cacheDir?: string
  /** Query marker prepended by {@link SparseEncoder.encodeQuery} (default `[Q] `). */
  queryPrefix?: string
  /** Passage marker prepended by {@link SparseEncoder.encode} (default `[D] `). */
  passagePrefix?: string
}

/** The sparse space the encoder produces. */
export interface SparseIdentity {
  /** The model id as configured: a hub repo id or local directory. */
  id: string
  /** Vocabulary size of the sparse space. */
  dimension: number
  dtype: string
  device: string
  attention: string
  maxLength: number
}

/** One sparse term-weight vector: `indices` strictly ascending, `values` positive and parallel. */
export interface SparseVector {
  indices: Uint32Array
  values: Float32Array
}

/**
 * In-process learned-sparse (SPLADE) encoder (`nlpai-lab/KURE-Sparse-nano`).
 * The model loads on first use and stays resident until `unload()`.
 */
export class SparseEncoder {
  /** Invalid settings throw synchronously with `error.code === 'ERR_INL_CONFIG'`. */
  constructor(options?: SparseEncoderOptions)
  /** The sparse space this encoder produces. Loads the model if needed. */
  identity(): Promise<SparseIdentity>
  /** Encode documents (the passage marker is prepended to every input). */
  encode(texts: string[], options?: CallOptions): Promise<SparseVector[]>
  /** Encode one query (the query marker is prepended). */
  encodeQuery(text: string, options?: CallOptions): Promise<SparseVector>
  /** True while the model weights are resident. */
  readonly isLoaded: boolean
  /** Release the model; the next call reloads it. Resolves to whether one was resident. */
  unload(): Promise<boolean>
}

/** Which engine a daemon serves and how it is built. */
export interface EngineOptions {
  /** `dense` (default), `sparse` or `multimodal`. */
  kind?: 'dense' | 'sparse' | 'multimodal'
  /** Hub repo id or local directory; omitted uses the kind's default. */
  model?: string
  /** `auto` (default), `cpu`, `metal` or `cuda`. */
  device?: string
  /** `auto` (default), `f32`, `f16` or `bf16`. */
  dtype?: string
  /** `auto` (default), `eager`, `tiled` or `fused` (dense only). */
  attention?: string
  /** Context length in tokens; omitted uses the engine default. */
  maxLength?: number
  /** Model cache directory (else the shared `HF_HOME` cache). */
  cacheDir?: string
  /** Multimodal only: towers to load; omitted loads every tower the checkpoint has. */
  modalities?: Array<'text' | 'image' | 'video' | 'audio'>
}

/** Loopback server settings for {@link Daemon.spawn}. */
export interface ServerOptions {
  /** Loopback host; defaults to `127.0.0.1`. */
  host?: string
  /** TCP port; `0` (default) picks a free one. */
  port?: number
  /** Stop after this many idle seconds; omitted serves until stopped. */
  idleTimeoutSeconds?: number
  /** Most inputs per request (default 32). */
  maxInputs?: number
  /** Largest request body in bytes (default 8 MiB). */
  maxRequestBytes?: number
  /** Directory for the `lock`/`pid`/`host`/`port` files; omitted uses the per-user default. */
  stateDir?: string
  /** Prefix applied to `input_type: "query"` requests; defaults per kind. */
  queryPrefix?: string
  /** Prefix applied to `input_type: "passage"` requests; defaults per kind. */
  passagePrefix?: string
}

/** Where a {@link DaemonClient} looks for a served instance. */
export interface DiscoverOptions {
  /** Hub repo id or local directory; omitted uses the kind's default. */
  model?: string
  /** `dense` (default), `sparse` or `multimodal`. */
  kind?: 'dense' | 'sparse' | 'multimodal'
  /** `auto` (default), `cpu`, `metal` or `cuda`. */
  device?: string
  /** `auto` (default), `f32`, `f16` or `bf16`. */
  dtype?: string
  /** Directory holding the discovery files; omitted uses the per-user default. */
  stateDir?: string
}

/** A `GET /health` body. */
export interface DaemonHealth {
  /** `loading`, `ok` or `stopping`. */
  status: string
  /** `dense`, `sparse` or `multimodal`. */
  kind: string
  modelId: string
  dimension: number
  dtype: string
  device: string
  attention: string
  maxLength: number
  modalities: string[]
  maxInputs: number
  instance: string
  pid: number
  host: string
  port: number
  url: string
  version: string
  uptimeSeconds: number
  requestsTotal: number
  loaded: boolean
}

/**
 * A running loopback server, in this process or another. `spawn` returns one
 * either way: a server already serving the instance is adopted instead of
 * loading a second copy of the model.
 */
export class Daemon {
  /**
   * Serve `engineOptions` (with `serverOptions`) in this process, or adopt the
   * server already serving that instance. `url()`/`health()` are usable
   * immediately; the model loads in the background.
   */
  static spawn(engineOptions: EngineOptions, serverOptions?: ServerOptions): Promise<Daemon>
  /** The server's base URL. */
  url(): string
  /** The server's current health. */
  health(): DaemonHealth
  /** Stop the server and wait for it to be gone. Resolves to the stop reason. */
  stop(): Promise<'signal' | 'idle' | 'wedged'>
  /** Wait for the server to stop on its own. Resolves to the stop reason. */
  wait(): Promise<'signal' | 'idle' | 'wedged'>
}

/** Out-of-process control of a served instance: discovery files plus HTTP. */
export class DaemonClient {
  /** A client for the server recorded in the instance's state directory, or `null`. */
  static discover(options: DiscoverOptions): DaemonClient | null
  /** The server's base URL. */
  url(): string
  /** The server's health when it answers as this instance; `null` otherwise. */
  health(): Promise<DaemonHealth | null>
  /** Poll `/health` until the server is ready or `timeoutMs` elapses. */
  waitReady(timeoutMs: number): Promise<DaemonHealth>
  /** `POST /admin/shutdown`: ask the server to drain and exit. */
  stop(): Promise<void>
  /** `POST /admin/unload`: release the model weights; true when a model was resident. */
  unload(): Promise<boolean>
}
