<script>
  import { api } from '../lib/api.js'
  import { app, guard } from '../lib/state.svelte.js'
  import Media from './Media.svelte'

  let text = $state('')
  let list
  let now = $state(Date.now())
  setInterval(() => (now = Date.now()), 1000)

  $effect(() => {
    // keep scrolled to the bottom as messages stream in
    app.chat.length; app.streaming; app.toolLog.length
    if (list) queueMicrotask(() => (list.scrollTop = list.scrollHeight))
  })

  async function send() {
    const t = text.trim()
    if (!t || app.busy) return
    const attachments = app.context.map((c) => ({ type: c.type, id: c.id, file_id: c.file_id }))
    app.busy = true
    const ok = await guard(() => api.post('/api/chat', { text: t, attachments }))
    if (ok) { text = ''; app.context = [] } else app.busy = false
  }
  function key(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }

  async function pick(a) { await guard(() => api.post(`/api/jobs/${a.job_id}/pick/${a.id}`)) }
  function waiting(jobId) {
    const s = app.seedhunt[jobId]
    const job = app.jobs.find((j) => j.id === jobId)
    return s && job && job.state === 'running' && (s.wait_s === 0 || s.until > now)
  }
  function secs(jobId) { const s = app.seedhunt[jobId]; return s && s.wait_s ? Math.max(0, Math.round((s.until - now) / 1000)) : null }
  const itemOf = (a) => a.type === 'ref_version' ? { type: 'ref_version', id: a.id } : a.type === 'ref' ? { type: 'ref', id: a.id }
    : { type: 'render', id: a.id }
</script>

<div class="chat">
  <div class="list scroll" bind:this={list}>
    {#each app.chat as msg (msg.id)}
      {#if msg.role === 'system_note'}
        <div class="note muted">Earlier conversation summarised for Qwen's memory: {msg.content.slice(0, 300)}…</div>
      {:else}
        <div class="msg {msg.role}">
          {#if msg.content}<div class="text">{msg.content}</div>{/if}
          {#if msg.attachments?.length}
            <div class="row atts">
              {#each msg.attachments as a}
                <div class="col att">
                  <Media file_id={a.file_id} video={a.video || a.type === 'render' || a.type === 'preview'}
                         label={a.label || ''} item={itemOf(a)} size={msg.role === 'event' ? 220 : 120}
                         selected={a.picked} badge={a.picked ? 'Qwen\'s pick' : ''} />
                  {#if a.type === 'preview' && waiting(a.job_id)}
                    <button class="small" onclick={() => pick(a)}>{a.picked ? `Upscale this${secs(a.job_id) !== null ? ` (${secs(a.job_id)}s)` : ''}` : 'Use this one'}</button>
                  {/if}
                </div>
              {/each}
            </div>
          {/if}
          {#if msg.tool_calls?.length}
            <details class="tools"><summary class="muted">{msg.tool_calls.length} tool call(s)</summary>
              {#each msg.tool_calls as t}<pre>{t.name}({t.arguments})
→ {t.result}</pre>{/each}
            </details>
          {/if}
        </div>
      {/if}
    {/each}

    {#if app.busy}
      <div class="msg assistant live">
        {#if app.thinking}<details><summary class="muted">thinking…</summary><pre>{app.thinking}</pre></details>{/if}
        {#each app.toolLog as t}<div class="muted tool">⚙ {t.name} {t.result ? '✓' : '…'}</div>{/each}
        {#if app.streaming}<div class="text">{app.streaming}</div>{:else}<div class="muted">Qwen is working…</div>{/if}
      </div>
    {/if}
  </div>

  <div class="composer">
    {#if app.context.length}
      <div class="row chips">
        <span class="muted">About:</span>
        {#each app.context as c, i}
          <span class="chip"><Media file_id={c.file_id} video={c.video} size={48} />
            {c.label}<button class="small" onclick={() => app.context.splice(i, 1)}>✕</button></span>
        {/each}
      </div>
    {/if}
    <div class="row">
      <textarea class="grow" rows="3" bind:value={text} onkeydown={key}
        placeholder="Talk to Qwen: e.g. 'Create a 15 second clip of a fisherwoman mending nets at dawn, making the reference images you need first.' (Enter sends, Shift+Enter for a new line)"></textarea>
      <button class="primary" onclick={send} disabled={app.busy || !text.trim()}>Send</button>
    </div>
  </div>
</div>

<style>
  .chat { display: flex; flex-direction: column; height: 100%; max-width: 1100px; width: 100%; margin: 0 auto; }
  .list { flex: 1; padding: 14px; display: flex; flex-direction: column; gap: 10px; }
  .msg { max-width: 85%; padding: 9px 12px; border-radius: 10px; background: var(--panel); border: 1px solid var(--line); }
  .msg.user { align-self: flex-end; background: #1d2a3d; border-color: #2a3d59; }
  .msg.event { align-self: center; max-width: 95%; background: transparent; border-style: dashed; }
  .text { white-space: pre-wrap; }
  .atts { margin-top: 8px; align-items: flex-start; }
  .att { align-items: center; }
  .tools { margin-top: 6px; font-size: 12px; }
  .tool { font-size: 12px; }
  .note { align-self: center; font-size: 12px; max-width: 80%; }
  .composer { border-top: 1px solid var(--line); padding: 10px 14px; display: flex; flex-direction: column; gap: 6px; }
  .chips { font-size: 12px; }
  .chip { display: inline-flex; align-items: center; gap: 6px; background: var(--panel2); padding: 3px 6px; border-radius: 8px; }
</style>
