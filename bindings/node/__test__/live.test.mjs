// Runs the real models. Opt in with INL_LIVE=1; the weights come from the
// shared Hugging Face cache (set HF_HOME) and load on first use. On macOS the
// default device is Metal.
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { createRequire } from 'node:module'

const { TextEmbedder, SparseEncoder, Daemon } = createRequire(import.meta.url)('../index.js')
const live = process.env.INL_LIVE === '1'
const device = process.env.INL_TEST_DEVICE
const options = (extra = {}) => ({ ...(device ? { device } : {}), ...extra })
const norm = (vector) => Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0))
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0)

test('Qwen3-Embedding embeds two sentences in-process', { skip: !live }, async (t) => {
  const embedder = new TextEmbedder(options({ id: 'Qwen/Qwen3-Embedding-0.6B' }))
  t.after(() => embedder.dispose())

  const started = performance.now()
  const identity = await embedder.identity()
  const cold = performance.now() - started
  assert.equal(identity.dimension, 1024)
  t.diagnostic(`identity/load ${cold.toFixed(0)} ms on ${identity.device} (${identity.dtype})`)

  const texts = ['The quick brown fox jumps over the lazy dog.', 'A completely different sentence.']
  const warm = performance.now()
  const vectors = await embedder.embed(texts)
  const elapsed = performance.now() - warm
  t.diagnostic(`embed ${texts.length} texts in ${elapsed.toFixed(0)} ms on ${identity.device}`)

  assert.equal(vectors.length, 2)
  for (const vector of vectors) {
    assert.equal(vector.length, 1024)
    assert.ok(Math.abs(norm(vector) - 1) < 1e-3, `norm ${norm(vector)}`)
  }
  assert.ok(dot(vectors[0], vectors[1]) < 0.999)
})

test('KURE-Sparse encodes a query and a document', { skip: !live }, async (t) => {
  const encoder = new SparseEncoder(options({ id: 'nlpai-lab/KURE-Sparse-nano' }))
  t.after(() => encoder.unload())

  const started = performance.now()
  const identity = await encoder.identity()
  t.diagnostic(`identity/load ${(performance.now() - started).toFixed(0)} ms on ${identity.device}`)

  const document = await encoder.encode(['The quick brown fox jumps over the lazy dog.'])
  const query = await encoder.encodeQuery('quick fox')
  assert.ok(document[0].indices.length > 0)
  assert.ok(query.indices.length > 0)
  t.diagnostic(`document terms ${document[0].indices.length}, query terms ${query.indices.length}`)
})

test('Daemon serves Qwen3-Embedding with an ok /health', { skip: !live }, async (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'inl-node-live-daemon-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))

  const engine = { kind: 'dense', model: 'Qwen/Qwen3-Embedding-0.6B', ...(device ? { device } : {}) }
  const started = performance.now()
  const daemon = await Daemon.spawn(engine, { stateDir: dir })
  t.diagnostic(`Daemon.spawn ${(performance.now() - started).toFixed(0)} ms`)
  t.after(async () => {
    try {
      await daemon.stop()
    } catch {
      /* already stopped */
    }
  })

  const url = daemon.url()
  const response = await fetch(`${url}/health`)
  assert.equal(response.status, 200)
  const body = await response.json()
  assert.equal(body.status, 'ok')
  assert.equal(body.model_id, 'Qwen/Qwen3-Embedding-0.6B')
  assert.equal(body.dimension, 1024)
  t.diagnostic(`/health ok on ${body.device} (${body.dtype}), pid ${body.pid}`)

  const embedded = await fetch(`${url}/embed`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ inputs: ['hello world'], input_type: 'query' }),
  })
  assert.equal(embedded.status, 200)
  const payload = await embedded.json()
  const vector = Array.isArray(payload[0]) ? payload[0] : payload[0].embedding
  assert.equal(vector.length, 1024)

  assert.equal(await daemon.stop(), 'signal')
})
