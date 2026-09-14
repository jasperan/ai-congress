import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createSocket } from '../src/lib/useSocket.js'

test('queues until open, receives JSON, and does not replay an interrupted prompt', () => {
  const originalWindow = globalThis.window
  const originalSocket = globalThis.WebSocket
  const instances = []
  class Socket {
    static OPEN = 1
    readyState = 0
    sent = []
    constructor(url) { this.url = url; instances.push(this) }
    send(value) { this.sent.push(value) }
    close() { this.onclose?.() }
  }
  globalThis.window = { location: { href: 'https://congress.local/' } }
  globalThis.WebSocket = Socket
  const received = []
  let connection
  try {
    connection = createSocket('/ws/chat', { onMessage: data => received.push(data) })
    connection.send({ prompt: 'A test question' })
    const socket = instances[0]
    assert.equal(socket.url, 'wss://congress.local/ws/chat')
    assert.equal(socket.sent.length, 0)
    socket.readyState = 1
    socket.onopen()
    assert.equal(socket.sent.length, 1)
    socket.onmessage({ data: 'malformed' })
    socket.onmessage({ data: '{"type":"start"}' })
    assert.equal(received[0].type, 'start')
    socket.onclose()
    assert.equal(received[1].type, 'error')
    assert.equal(instances.length, 1)
    connection.send({ prompt: 'must not send after close' })
    assert.equal(socket.sent.length, 1)
  } finally {
    connection?.close()
    globalThis.window = originalWindow
    globalThis.WebSocket = originalSocket
  }
})
