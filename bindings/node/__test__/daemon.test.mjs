// Daemon lifecycle: spawn an in-process loopback server on the tiny dense
// fixture, discover it, wait until ready, hit it over HTTP, shut it down.
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const { Daemon, DaemonClient } = createRequire(import.meta.url)('../index.js')
const FIXTURE = fileURLToPath(
  new URL('../../../crates/inl-text/tests/fixtures/tiny_embeddinggemma2/', import.meta.url),
)
const engine = { kind: 'dense', model: FIXTURE, device: 'cpu' }

function scratch(t) {
  const dir = mkdtempSync(join(tmpdir(), 'inl-node-daemon-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  return dir
}

test('discover returns null where nothing is served', (t) => {
  const dir = scratch(t)
  assert.equal(
    DaemonClient.discover({ model: FIXTURE, kind: 'dense', stateDir: dir }),
    null,
  )
})

test('spawn -> client.waitReady -> fetch /health -> stop', async (t) => {
  const dir = scratch(t)
  const daemon = await Daemon.spawn(engine, { stateDir: dir })
  t.after(async () => {
    try {
      await daemon.stop()
    } catch {
      /* already stopped */
    }
  })

  const url = daemon.url()
  assert.match(url, /^http:\/\/127\.0\.0\.1:\d+$/)
  const health = daemon.health()
  assert.equal(health.status, 'ok')
  assert.equal(health.kind, 'dense')
  assert.equal(health.loaded, true)
  assert.equal(health.modelId, FIXTURE)
  assert.equal(health.dimension, 24)
  assert.deepEqual(health.modalities, ['text'])

  const client = DaemonClient.discover({
    model: FIXTURE,
    kind: 'dense',
    device: 'cpu',
    stateDir: dir,
  })
  assert.ok(client, 'discovery files must name the served instance')
  assert.equal(client.url(), url)

  const ready = await client.waitReady(60_000)
  assert.equal(ready.status, 'ok')
  assert.equal(ready.modelId, FIXTURE)

  const response = await fetch(`${url}/health`)
  assert.equal(response.status, 200)
  const body = await response.json()
  assert.equal(body.status, 'ok')
  assert.equal(body.instance, health.instance)

  assert.equal(await daemon.stop(), 'signal')
})

test('client.stop drains the server and daemon.wait resolves', async (t) => {
  const dir = scratch(t)
  const daemon = await Daemon.spawn(engine, { stateDir: dir })
  const client = DaemonClient.discover({
    model: FIXTURE,
    kind: 'dense',
    device: 'cpu',
    stateDir: dir,
  })
  assert.ok(client)
  await client.waitReady(60_000)

  await client.stop()
  assert.equal(await daemon.wait(), 'signal')
})
