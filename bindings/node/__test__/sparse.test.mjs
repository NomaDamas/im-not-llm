// Behaviour of the learned-sparse engine against the tiny SPLADE fixture.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const { SparseEncoder } = createRequire(import.meta.url)('../index.js')
const FIXTURE = fileURLToPath(
  new URL('../../../crates/inl-sparse/tests/fixtures/tiny_splade/', import.meta.url),
)
const options = (extra = {}) => ({ id: FIXTURE, device: 'cpu', ...extra })

function assertWellFormed(vector) {
  assert.ok(vector.indices instanceof Uint32Array)
  assert.ok(vector.values instanceof Float32Array)
  assert.equal(vector.indices.length, vector.values.length)
  for (let i = 1; i < vector.indices.length; i++) {
    assert.ok(vector.indices[i] > vector.indices[i - 1], 'indices must be strictly ascending')
  }
  for (const weight of vector.values) {
    assert.ok(weight > 0 && Number.isFinite(weight), `weight ${weight}`)
  }
}

test('invalid settings throw synchronously with a config code', () => {
  for (const bad of [{ id: '' }, { device: 'tpu' }, { dtype: 'int8' }, { timeoutSeconds: 0 }]) {
    assert.throws(
      () => new SparseEncoder({ ...options(), ...bad }),
      (error) => error.code === 'ERR_INL_CONFIG' && !error.message.startsWith('ERR_'),
      JSON.stringify(bad),
    )
  }
})

test('construction is lazy: no model is loaded until a call needs it', () => {
  const encoder = new SparseEncoder(options())
  assert.equal(encoder.isLoaded, false)
})

test('encode returns sparse vectors with ascending indices and positive values', async () => {
  const encoder = new SparseEncoder(options())
  const vectors = await encoder.encode(['w1 w2 w3', 'w5 w6 w7 w8'])

  assert.equal(vectors.length, 2)
  for (const vector of vectors) assertWellFormed(vector)
  assert.ok(vectors[0].indices.length > 0)
  assert.equal(encoder.isLoaded, true)
})

test('encodeQuery returns one sparse vector', async () => {
  const encoder = new SparseEncoder(options())
  const query = await encoder.encodeQuery('w1 w2')
  assertWellFormed(query)
  assert.ok(query.indices.length > 0)
})

test('an already aborted signal rejects with its reason', async () => {
  const encoder = new SparseEncoder(options())
  const reason = new Error('stop')
  const controller = new AbortController()
  controller.abort(reason)

  await assert.rejects(
    encoder.encode(['w1'], { signal: controller.signal }),
    (error) => error === reason,
  )
  await assert.rejects(
    encoder.encodeQuery('w1', { signal: controller.signal }),
    (error) => error === reason,
  )
})

test('unload releases the model and the next call reloads it', async () => {
  const encoder = new SparseEncoder(options())
  await encoder.encodeQuery('w1')
  assert.equal(encoder.isLoaded, true)

  assert.equal(await encoder.unload(), true)
  assert.equal(encoder.isLoaded, false)
  assert.equal(await encoder.unload(), false)

  await encoder.encodeQuery('w1')
  assert.equal(encoder.isLoaded, true)
})
