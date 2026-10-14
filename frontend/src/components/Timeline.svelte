<script>
  import { api } from '../lib/api.js'
  import { app, guard, toast, load, fileSrc } from '../lib/state.svelte.js'
  import Media from './Media.svelte'

  const PX_PER_S = 22
  let dragId = $state(null)
  let hoverRef = $state(null)

  // references used by more than one shot get a colour so reuse is visible at a glance
  let reuse = $derived.by(() => {
    const count = {}
    for (const s of app.shots) for (const sl of s.slots) count[sl.ref_id] = (count[sl.ref_id] || 0) + 1
    const palette = ['#6aa8ff', '#f0b44c', '#4cc38a', '#d77cf0', '#ef6461', '#5fd3d6', '#c9d36a']
    const out = {}; let i = 0
    for (const [id, n] of Object.entries(count)) if (n > 1) out[id] = palette[i++ % palette.length]
    return out
  })
  let total = $derived(app.shots.reduce((a, s) => a + (s.duration_s || 0), 0))

  async function drop(target) {
    if (dragId == null || dragId === target) return
    const ids = app.shots.map((s) => s.id).filter((id) => id !== dragId)
    ids.splice(ids.indexOf(target), 0, dragId)
    await guard(() => api.post('/api/shots/reorder', ids))
    dragId = null; load.shots()
  }
  async function cut() {
    const missing = app.shots.filter((s) => !s.final_render_id).map((s) => s.title)
    if (missing.length && !confirm(`These shots have no final clip and will be skipped:\n${missing.join('\n')}\n\nContinue?`)) return
    await guard(() => api.post('/api/final_cut', { name: app.project.name }))
    toast('Rendering the final cut…')
  }
</script>

<div class="scroll pad col">
  <div class="row">
    <h2 class="grow">Timeline · {app.shots.length} shots · {total.toFixed(1)} s</h2>
    <button class="primary" onclick={cut}>Render final video with audio</button>
  </div>
  <p class="muted">Drag shots to reorder. Click any clip or picture to view it large; 💬 pins it to the chat. Coloured outlines mark references reused across shots.</p>

  <div class="track">
    {#each app.shots as s, i (s.id)}
      {@const fin = s.renders.find((r) => r.id === s.final_render_id)}
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div class="block card" style="width:{Math.max(170, s.duration_s * PX_PER_S)}px" draggable="true"
           ondragstart={() => (dragId = s.id)} ondragover={(e) => e.preventDefault()} ondrop={() => drop(s.id)}>
        <div class="row head"><strong class="grow">{i + 1}. {s.title}</strong><span class="muted">{s.duration_s}s</span></div>
        <Media file_id={fin?.file_id} video size={Math.max(150, s.duration_s * PX_PER_S - 22)}
               item={fin ? { type: 'render', id: fin.id } : null} label={fin ? '' : 'no final clip'} />
        <div class="row">
          {#if s.stale}<span class="badge warn">stale</span>{/if}
          <span class="badge">{s.mode}</span><span class="badge">{s.status}</span>
        </div>
        <div class="row pics">
          {#each s.slots as sl}
            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <div class="pic" style={reuse[sl.ref_id] ? `outline:2px solid ${reuse[sl.ref_id]}` : ''}
                 class:hl={hoverRef === sl.ref_id}
                 onmouseenter={() => (hoverRef = sl.ref_id)} onmouseleave={() => (hoverRef = null)}>
              <Media file_id={sl.file_id} size={46} item={{ type: 'ref_version', id: sl.version_id }} label="" />
            </div>
          {/each}
        </div>
        <div class="muted desc">{s.description}</div>
      </div>
    {:else}<p class="muted">No shots yet.</p>{/each}
  </div>

  {#if app.exports.length}
    <h3>Final cuts</h3>
    <div class="col">
      {#each app.exports as e}
        <div class="row card">
          <span class="grow">{e.path.split(/[\\/]/).pop()}</span>
          <span class="muted">{(e.bytes / 1048576).toFixed(1)} MB</span>
          <button onclick={() => (app.lightbox = { file_id: e.id, video: true, label: 'Final cut' })}>Play</button>
          <button onclick={() => guard(() => api.post(`/api/files/${e.id}/reveal`))}>Show in Explorer</button>
          <a href={fileSrc(e.id)} download><button>Download</button></a>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .track { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 10px; align-items: flex-start; }
  .block { flex: none; display: flex; flex-direction: column; gap: 6px; cursor: grab; }
  .head { font-size: 13px; }
  .pics { gap: 4px; }
  .pic { border-radius: 6px; }
  .pic.hl { outline: 2px solid #fff !important; }
  .desc { font-size: 12px; max-height: 50px; overflow: hidden; }
</style>
