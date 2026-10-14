// Thin fetch wrapper: JSON in, JSON out, errors as thrown Error(message).
async function req(method, url, body, isForm = false) {
  const opts = { method, headers: {} }
  if (body !== undefined) {
    if (isForm) opts.body = body
    else {
      opts.headers['Content-Type'] = 'application/json'
      opts.body = JSON.stringify(body)
    }
  }
  const r = await fetch(url, opts)
  const text = await r.text()
  let data = null
  try { data = text ? JSON.parse(text) : null } catch { data = text }
  if (!r.ok) throw new Error((data && (data.error || data.detail)) || `${r.status} ${r.statusText}`)
  return data
}

export const api = {
  get: (u) => req('GET', u),
  post: (u, b = {}) => req('POST', u, b),
  put: (u, b) => req('PUT', u, b),
  patch: (u, b) => req('PATCH', u, b),
  del: (u) => req('DELETE', u),
  form: (u, fd) => req('POST', u, fd, true),
}

export const fileUrl = (id) => `/api/files/${id}`
export const thumbUrl = (id, size = 320) => `/api/files/${id}/thumb?size=${size}`

export function connectEvents(onMessage) {
  let ws
  let stopped = false
  const open = () => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    ws = new WebSocket(`${proto}://${location.host}/ws`)
    ws.onmessage = (e) => onMessage(JSON.parse(e.data))
    ws.onclose = () => { if (!stopped) setTimeout(open, 1500) }
  }
  open()
  return () => { stopped = true; ws && ws.close() }
}
