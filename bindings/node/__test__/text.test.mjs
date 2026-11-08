// Behaviour of the dense text engine against the tiny EmbeddingGemma 2
// fixture (a local model directory works as an id). No network, no HF cache.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const { TextEmbedder } = createRequire(import.meta.url)('../index.js')
const FIXTURE = fileURLToPath(
  new URL('../../../crates/inl-text/tests/fixtures/tiny_embeddinggemma2/', import.meta.url),
)
const options = (extra = {}) => ({ id: FIXTURE, device: 'cpu', ...extra })

const norm = (vector) => Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0))
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0)

test('invalid settings throw synchronously with a config code', () => {
  for (const bad of [
    { id: '' },
    { device: 'tpu' },
    { dtype: 'int8' },
    { attention: 'psychic' },
    { timeoutSeconds: 0 },
    { idleUnloadSeconds: 0 },
  ]) {
    assert.throws(
      () => new TextEmbedder({ ...options(), ...bad }),
      (error) => error.code === 'ERR_INL_CONFIG' && !error.message.startsWith('ERR_'),
      JSON.stringify(bad),
    )
  }
})

test('construction is lazy: no model is loaded until a call needs it', () => {
  const embedder = new TextEmbedder(options())
  assert.equal(embedder.isLoaded, false)
})

test('identity reports the embedding space', async (t) => {
  const embedder = new TextEmbedder(options())
  t.after(() => embedder.dispose())
  assert.equal(embedder.isLoaded, false)

  const identity = await embedder.identity()

  assert.equal(identity.id, FIXTURE)
  assert.equal(identity.dimension, 24)
  assert.equal(identity.maxLength, 2048)
  assert.ok(['f32', 'f16', 'bf16'].includes(identity.dtype), identity.dtype)
  assert.equal(identity.device, 'cpu')
  assert.equal(embedder.isLoaded, true)
})

test('embed returns unit vectors in input order', async (t) => {
  const embedder = new TextEmbedder(options())
  t.after(() => embedder.dispose())
  const texts = ['w0 w1 w2', 'w3 w4 w5', 'w0 w1 w2']

  const vectors = await embedder.embed(texts)

  assert.equal(vectors.length, texts.length)
  for (const vector of vectors) {
    assert.ok(vector instanceof Float32Array)
    assert.equal(vector.length, 24)
    assert.ok(Math.abs(norm(vector) - 1) < 1e-4, `norm ${norm(vector)}`)
  }
  assert.ok(dot(vectors[0], vectors[2]) > 0.999, 'identical text, near-identical vector')
  assert.ok(dot(vectors[0], vectors[1]) < dot(vectors[0], vectors[2]))
  assert.deepEqual(await embedder.embed([]), [])
})

test('embedQuery applies queryPrefix, embed applies passagePrefix', async (t) => {
  // The tiny tokenizer maps unknown prefix characters to the same token, so
  // `q: ` and `p: ` are indistinguishable on it; compare against an unprefixed
  // embedder instead.
  const prefixed = new TextEmbedder(options({ queryPrefix: 'q: ', passagePrefix: 'p: ' }))
  const plain = new TextEmbedder(options())
  t.after(() => Promise.all([prefixed.dispose(), plain.dispose()]))

  const query = await prefixed.embedQuery('w0 w1')
  const plainQuery = await plain.embedQuery('w0 w1')
  assert.equal(query.length, 24)
  assert.notDeepEqual(Array.from(query), Array.from(plainQuery))

  const passage = (await prefixed.embed(['w0 w1']))[0]
  const plainPassage = (await plain.embed(['w0 w1']))[0]
  assert.notDeepEqual(Array.from(passage), Array.from(plainPassage))
})

test('an already aborted signal rejects with its reason before any work starts', async (t) => {
  const embedder = new TextEmbedder(options())
  t.after(() => embedder.dispose())
  const reason = new Error('stop')
  const controller = new AbortController()
  controller.abort(reason)

  await assert.rejects(
    embedder.embed(['w0'], { signal: controller.signal }),
    (error) => error === reason,
  )
  await assert.rejects(
    embedder.embedQuery('w0', { signal: controller.signal }),
    (error) => error === reason,
  )
  assert.equal(embedder.isLoaded, false)
})

test('unload releases the model and the next call reloads identical vectors', async (t) => {
  const embedder = new TextEmbedder(options())
  t.after(() => embedder.dispose())

  const before = await embedder.embedQuery('w0 w1')
  assert.equal(embedder.isLoaded, true)

  assert.equal(await embedder.unload(), true)
  assert.equal(embedder.isLoaded, false)
  assert.equal(await embedder.unload(), false)

  const after = await embedder.embedQuery('w0 w1')
  assert.equal(embedder.isLoaded, true)
  assert.deepEqual(Array.from(after), Array.from(before))
})

test('dispose rejects every later call and is idempotent', async (t) => {
  const embedder = new TextEmbedder(options())
  t.after(() => embedder.dispose())

  await embedder.dispose()
  await embedder.dispose()
  for (const call of [
    () => embedder.embed(['w0']),
    () => embedder.embedQuery('w0'),
    () => embedder.identity(),
  ]) {
    await assert.rejects(call(), /disposed/)
  }
  assert.equal(await embedder.unload(), false)
})
