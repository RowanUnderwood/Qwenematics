<script>
  import { untrack } from 'svelte'
  import { api } from '../lib/api.js'
  import { app, guard, toast, pin, load } from '../lib/state.svelte.js'
  import Media from './Media.svelte'

  const MODES = { ref2va: 'References → video', i2va: 'First frame →', fl2va: 'First + last frame', l2va: '→ Last frame', t2va: 'Text only' }
  const ROLES = ['subject', 'first_frame', 'last_frame']
  let sel = $state(null)
  let shot = $derived(app.shots.find((s) => s.id === sel) || app.shots[0])
  let draft = $state(null)
  let change = $state('')
  let addRole = $state('subject')
  let adding = $state(false)
  let dirty = $state(false)          // unsaved edits: don't overwrite the draft when the shot reloads
  let enhance = $state(true)         // off: the video prompt goes to MiniMax verbatim

  // Follow the server copy (e.g. a prompt Qwen just rewrote) unless the user has unsaved edits.
  $effect(() => {
    const s = shot && { ...shot }
    if (!s) return
    untrack(() => {
      if (!draft || draft.id !== s.id) { draft = s; dirty = false; enhance = true }
      else if (!dirty) draft = s
    })
  })

  async function save() {
    const f = ['title', 'description', 'dialogue', 'audio_notes', 'mode', 'duration_s', 'prompt']
    await guard(() => api.patch(`/api/shots/${shot.id}`, Object.fromEntries(f.map((k) => [k, draft[k]]))))
    dirty = false
    toast('Saved')
  }
  async function newShot() {
    const r = await guard(() => api.post('/api/shots', { title: `Shot ${app.shots.length + 1}`, duration_s: 15 }))
    await load.shots(); sel = r?.shot_id
  }
  async function render(rewrite = false) {
    await save()
    await guard(() => api.post(`/api/shots/${shot.id}/render`, { change: (enhance && change) || null, rewrite_prompt: rewrite, enhance: enhance || rewrite }))
    change = ''; toast('Queued on the video GPU')
  }
  async function addRef(r) {
    await guard(() => api.post(`/api/shots/${shot.id}/slots`, { ref_id: r.ref_id, role: addRole }))
    adding = false
  }
  async function delSlot(sl) { await guard(() => api.del(`/api/shots/${shot.id}/slots/${sl.slot_id}`)) }
  async function useLatest(sl) {
    const r = app.refs.find((x) => x.ref_id === sl.ref_id)
    await guard(() => api.post(`/api/references/${sl.ref_id}/apply_version`, { version_id: r.current_version, shot_ids: [shot.id] }))
  }
  async function setFinal(r) { await guard(() => api.post(`/api/shots/${shot.id}/final/${r.id}`)) }
  async function delRender(r) {
    if (!confirm(`Delete this ${r.stage} render and its files?`)) return
    await guard(() => api.del(`/api/renders/${r.id}`))
  }
  async function delShot() {
    const plan = await guard(() => api.get(`/api/shots/${shot.id}/delete_plan?orphan_refs=true`))
    if (!plan) return
    if (!confirm(`Delete shot '${shot.title}'?\n\nRemoves ${plan.files.length} file(s) (${(plan.bytes / 1048576).toFixed(1)} MB): its renders, plus reference images no other shot uses.\n\n${plan.summary.join('\n')}`)) return
    await guard(() => api.del(`/api/shots/${shot.id}?orphan_refs=true`)); sel = null
  }
</script>

<div class="wrap">
  <aside class="scroll">
    <button class="primary" onclick={newShot} style="width:100%">+ New shot</button>
    {#each app.shots as s, i (s.id)}
      {@const fin = s.renders.find((r) => r.id === s.final_render_id)}
      <button class="item" class:on={shot?.id === s.id} onclick={() => (sel = s.id)}>
        <Media file_id={fin?.file_id} video size={90} onclick={() => (sel = s.id)} />
        <div class="col" style="gap:2px;text-align:left">
          <strong>{i + 1}. {s.title}</strong>
          <span class="muted">{s.mode} · {s.duration_s}s · {s.slots.length} pics</span>
          <span>{#if s.stale}<span class="badge warn">stale</span>{/if}<span class="badge">{s.status}</span></span>
        </div>
      </button>
    {/each}
  </aside>

  {#if shot && draft}
    <section class="scroll pad col" oninput={() => (dirty = true)} onchange={() => (dirty = true)}>
      <div class="row">
        <input class="grow title" bind:value={draft.title} />
        <select bind:value={draft.mode}>{#each Object.entries(MODES) as [k, l]}<option value={k}>{l}</option>{/each}</select>
        <label class="row muted">length <input type="number" min="2" max="15" step="0.5" bind:value={draft.duration_s} style="width:70px" />s</label>
        <button onclick={save}>Save</button>
        <button onclick={() => pin({ type: 'shot', id: shot.id, label: shot.title, video: true, file_id: shot.renders.find((r) => r.id === shot.final_render_id)?.file_id })}>💬 Chat about this shot</button>
        <button class="danger" onclick={delShot}>Delete…</button>
      </div>
      {#if shot.stale}<div class="badge warn">Inputs changed since the last render: re-render to update the clip.</div>{/if}

      <div class="cols">
        <div class="col">
          <label class="f">What happens<textarea rows="4" bind:value={draft.description}></textarea></label>
          <label class="f">Dialogue (verbatim, with who says it)<textarea rows="2" bind:value={draft.dialogue}></textarea></label>
          <label class="f">Sound & music notes<textarea rows="2" bind:value={draft.audio_notes}></textarea></label>
        </div>
        <div class="col">
          <h3>Pictures (&lt;Picture N&gt; in the prompt)</h3>
          <div class="row slots">
            {#each shot.slots as sl (sl.slot_id)}
              <div class="col slot">
                <Media file_id={sl.file_id} size={110} label={`P${sl.slot} ${sl.name}${sl.pending ? ' (rendering…)' : ''}`} item={sl.version_id ? { type: 'ref_version', id: sl.version_id } : null}
                       badge={sl.role !== 'subject' ? sl.role.replace('_', ' ') : ''} />
                <div class="row">
                  {#if sl.outdated}<button class="small" title="A newer version exists" onclick={() => useLatest(sl)}>use latest</button>{/if}
                  <button class="small" onclick={() => delSlot(sl)}>remove</button>
                </div>
              </div>
            {/each}
            <button onclick={() => (adding = !adding)}>+ add</button>
          </div>
          {#if adding}
            <div class="card col">
              <div class="row"><span class="muted">as</span><select bind:value={addRole}>{#each ROLES as r}<option>{r}</option>{/each}</select></div>
              <div class="row picker">
                {#each app.refs.filter((r) => r.current_version && !shot.slots.find((s) => s.ref_id === r.ref_id)) as r}
                  <Media file_id={r.versions.find((v) => v.id === r.current_version)?.file_id} size={80} label={r.name} onclick={() => addRef(r)} />
                {/each}
              </div>
            </div>
          {/if}
        </div>
      </div>

      <div class="card col">
        <div class="row">
          <input class="grow" placeholder={enhance ? "Optional change for this render, e.g. 'slower camera, she smiles at the end'" : 'Enhance is off: edit the video prompt below instead'}
                 bind:value={change} disabled={!enhance} />
          <button class="primary" onclick={() => render(false)}>Render</button>
          <button onclick={() => render(true)} disabled={!enhance} title="Have Qwen rewrite the video prompt first">Rewrite prompt + render</button>
          <label class="row muted" title="Off: the video prompt below goes to MiniMax verbatim; Qwen doesn't rewrite it, even after a low review score">
            <input type="checkbox" bind:checked={enhance} /> Enhance prompt</label>
        </div>
        <details><summary class="muted">Video prompt (editable; Qwen writes it on first render; editing it turns Enhance off)</summary>
          <textarea rows="14" bind:value={draft.prompt} oninput={() => (enhance = false)}></textarea>
        </details>
      </div>

      <h3>Renders</h3>
      <div class="row renders">
        {#each [...shot.renders].reverse() as r (r.id)}
          <div class="col">
            <Media file_id={r.file_id} video size={r.stage === 'final' ? 240 : 150} item={{ type: 'render', id: r.id }}
                   label={`${r.stage} · seed ${r.seed}`} selected={r.id === shot.final_render_id}
                   badge={r.id === shot.final_render_id ? 'in timeline' : r.picked ? 'picked' : r.score != null ? `${r.score}/10` : ''} />
            <div class="row">
              {#if r.stage === 'final' && r.id !== shot.final_render_id}<button class="small" onclick={() => setFinal(r)}>use in timeline</button>{/if}
              <button class="small danger" onclick={() => delRender(r)}>✕</button>
            </div>
          </div>
        {:else}<span class="muted">Nothing rendered yet.</span>{/each}
      </div>
    </section>
  {:else}
    <section class="pad muted">No shots yet. Ask Qwen in the chat to plan some, or add one here.</section>
  {/if}
</div>

<style>
  .wrap { display: flex; height: 100%; }
  aside { width: 280px; border-right: 1px solid var(--line); padding: 10px; display: flex; flex-direction: column; gap: 6px; }
  .item { display: flex; gap: 8px; align-items: center; text-align: left; background: transparent; }
  .item.on { background: var(--panel2); border-color: var(--accent); }
  section { flex: 1; min-width: 0; }
  .title { font-size: 16px; font-weight: 600; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .slots, .renders, .picker { align-items: flex-start; }
  .slot { align-items: center; }
</style>
