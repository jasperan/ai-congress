// A chat request is sent once. Never replay a submitted prompt on reconnect.
export function createSocket(path, { onMessage, onStatus = () => {} }) {
  const url = new URL(path, window.location.href)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  let socket
  let timer
  let attempts = 0
  let closed = false
  let submitted = false
  const queue = []

  function fail(message) {
    if (closed) return
    onStatus('disconnected')
    onMessage({ type: 'error', message })
    close()
  }

  function connect() {
    if (closed) return
    onStatus(attempts ? 'reconnecting' : 'connecting')
    attempts++
    socket = new WebSocket(url.href)
    timer = setTimeout(() => fail('The streaming connection timed out. Please try again.'), 15000)
    socket.onopen = () => {
      clearTimeout(timer)
      if (closed) return
      onStatus('open')
      while (queue.length) {
        socket.send(queue.shift())
        submitted = true
      }
    }
    socket.onmessage = event => {
      let data
      try { data = JSON.parse(event.data) } catch { return }
      onMessage(data)
    }
    socket.onerror = () => onStatus('connection error')
    socket.onclose = () => {
      clearTimeout(timer)
      if (closed) return
      if (submitted || attempts >= 3) {
        fail('The streaming connection was interrupted. Please submit again to retry.')
      } else {
        timer = setTimeout(connect, 500 * 2 ** (attempts - 1))
      }
    }
  }

  function close() {
    closed = true
    clearTimeout(timer)
    queue.length = 0
    socket?.close()
  }

  connect()
  return {
    send(data) {
      if (closed) return
      const message = JSON.stringify(data)
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(message)
        submitted = true
      } else {
        queue.push(message)
      }
    },
    close,
  }
}
