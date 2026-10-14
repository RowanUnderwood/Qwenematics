// One shared reactive store for the whole UI (Svelte 5 runes).
import { api, connectEvents, fileUrl, thumbUrl } from './api.js'

export const app = $state({
  tab: 'chat',
  status: null,           // /api/status
  project: null,          // {name, path}
  projects: [],
  refs: [],
  shots: [],
  bible: [],
  jobs: [],
  chat: [],
  streaming: '',          // assistant text arriving now
  thinking: '',
  toolLog: [],            // tool calls in the current turn
  busy: false,
  workflows: [],
  psettings: null,
  exports: [],
  context: [],            // pinned items for the next chat message: {type, id, file_id, label, video}
  lightbox: null,         // {file_id, video, label, item}
  seedhunt: {},           // job_id -> {renders, pick, wait_s, until}
  toasts: [],
  shutdownSteps: null,
  impact: null,           // pending ref-version decision {ref_id, version_id, shots}
})

export function toast(msg, kind = 'info') {
  const t = { id: Math.random(), msg, kind }
  app.toasts.push(t)
  setTimeout(() => { app.toasts = app.toasts.filter((x) => x.id !== t.id) }, kind === 'error' ? 8000 : 4000)
}

export async function guard(fn) {
  try { return await fn() } catch (e) { toast(e.message, 'error') }
}

const timers = {}
function soon(key, fn, ms = 250) {
  clearTimeout(timers[key])
  timers[key] = setTimeout(fn, ms)
}

// File ids restart per project, so URLs carry the project to keep browser caches apart.
const pq = () => `p=${encodeURIComponent(app.project?.path || '')}`
export const fileSrc = (id) => `${fileUrl(id)}?${pq()}`
export const thumbSrc = (id, size) => `${thumbUrl(id, size)}&${pq()}`

export const load = {
  async status(full = false) { app.status = await api.get('/api/status' + (full ? '?full=true' : '')); app.project = app.status.project },
  async projects() { app.projects = await api.get('/api/projects') },
  async refs() { if (app.project) app.refs = await api.get('/api/references') },
  async shots() { if (app.project) app.shots = await api.get('/api/shots') },
  async bible() { if (app.project) app.bible = await api.get('/api/bible') },
  async jobs() { if (app.project) app.jobs = await api.get('/api/jobs') },
  async chat() { if (app.project) app.chat = await api.get('/api/chat') },
  async workflows() { app.workflows = await api.get('/api/workflows') },
  async psettings() { if (app.project) app.psettings = await api.get('/api/project/settings') },
  async exports() { if (app.project) app.exports = await api.get('/api/exports') },
  async all() {
    await load.status()
    await Promise.all([load.projects(), load.workflows()])
    if (app.project) {
      await Promise.all([load.refs(), load.shots(), load.bible(), load.jobs(), load.chat(), load.psettings(), load.exports()])
    } else {
      app.refs = []; app.shots = []; app.bible = []; app.jobs = []; app.chat = []
    }
  },
}

// Every bus event is {kind, ...payload}; chat deltas carry stream: "content" | "reasoning".
function wire(e) {
  switch (e.kind) {
    case 'chat_delta':
      app.busy = true
      if (e.stream === 'reasoning') app.thinking += e.text
      else app.streaming += e.text
      break
    case 'tool_call':
      app.busy = true
      app.toolLog.push({ name: e.name, args: e.args, result: null })
      break
    case 'tool_result': {
      const t = [...app.toolLog].reverse().find((x) => x.name === e.name && x.result === null)
      if (t) t.result = e.result
      break
    }
    case 'chat_message':
      if (e.role === 'assistant') { app.busy = false; app.streaming = ''; app.thinking = ''; app.toolLog = [] }
      if (!app.chat.find((c) => c.id === e.id)) app.chat.push({ id: e.id, role: e.role, content: e.content, attachments: e.attachments || [], tool_calls: e.tool_calls || [] })
      break
    case 'chat_error':
      app.busy = false; app.streaming = ''; app.thinking = ''
      toast('Qwen: ' + e.error, 'error')
      break
    case 'job':
      soon('jobs', load.jobs)
      if (e.state === 'failed') toast(`Job ${e.id} failed: ${e.message}`, 'error')
      if (e.state === 'done') { soon('refs', load.refs); soon('shots', load.shots); soon('exports', load.exports) }
      break
    case 'seedhunt':
      app.seedhunt[e.job_id] = { ...e, until: Date.now() + (e.wait_s || 0) * 1000 }
      soon('shots', load.shots)
      break
    case 'reference': case 'ref_version':
      soon('refs', load.refs); soon('shots', load.shots); break
    case 'ref_impact':
      app.impact = { ref_id: e.ref_id, version_id: e.version_id, shots: e.shots }
      soon('refs', load.refs); break
    case 'shot': case 'render':
      soon('shots', load.shots); break
    case 'bible':
      soon('bible', load.bible); break
    case 'project':
      load.all(); break
    case 'shutdown':
      app.shutdownSteps = e.steps; break
    case 'servers':
      load.status(); load.workflows(); break
    case 'export':
      soon('exports', load.exports); toast('Final cut written: ' + e.path); break
    case 'chat_summarized':
      soon('chat', load.chat); break
  }
}

export function start() {
  load.all()
  return connectEvents(wire)
}

export function pin(item) {
  if (!app.context.find((c) => c.type === item.type && c.id === item.id)) app.context.push(item)
  app.tab = 'chat'
}
