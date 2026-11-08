# im-not-llm

On-device inference engine for models that are **not** LLMs: dense text
embeddings, learned-sparse (SPLADE) encoders, and multimodal (image / video)
embeddings. Rust, candle, CPU / Metal / CUDA. Usable as in-process crates
(Rust, TypeScript via Node-API) or as a loopback daemon with a `/health`
endpoint and a programmatic lifecycle API.

## Rules

- Code is the source of truth; docs and comments are not evidence.
- TDD: write the failing test first. Golden tests compare against the Hugging Face
  reference outputs; never loosen a tolerance to make a test pass.
- No dependency on AutoRAG or MinSync. They are clients of this project.
- Never use an LLM released before 2026 (embedding / sparse / vision models are not LLMs and are fine).
- EmbeddingGemma 2 must not run in float16 (overflows to NaN); only f32 / bf16.
