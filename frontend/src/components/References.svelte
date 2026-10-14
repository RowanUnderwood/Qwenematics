<script>
  import { api } from '../lib/api.js'
  import { app, guard, toast, pin, load } from '../lib/state.svelte.js'
  import Media from './Media.svelte'

  const KINDS = ['character', 'character_closeup', 'location', 'establishing', 'prop', 'first_frame', 'concept']
  let filter = $state('')
  let creating = $state(false)
  let form = $state({ name: '', kind: 'character', description: '', enhance: true })
  let open = $state(null)            // ref_id whose details are expanded
  let editing = $state(null)         // new-version editor: {ref_id, prompt, enhance}
  let upload = $state({ name: '', kind: 'character', file: null })

  let groups = $derived.by(() => {
    const f = filter.toLowerCase()
    const g = {}
    for (const r of app.refs) {
      if (f && !(`${r.name} ${r.caption} ${r.tags.join(' ')}`.toLowerCase().includes(f))) continue
      ;(g[r.kind] ||= []).push(r)
    }
    return g
  })
  const shotTitle = (id) => app.shots.find((s) => s.id === id)?.title || `shot ${id}`
  const cur = (r) => r.versions.find((v) => v.id === r.current_version)

  async function create() {
    if (!form.name || !form.description) return
    await guard(() => api.post('/api/references', form))
    form = { name: '', kind: 'character', description: '', enhance: true }; creating = false
    toast('Queued on the image GPU')
  }
  function startRevise(r) {
    const v = cur(r)
    editing = { ref_id: r.ref_id, prompt: v?.prompt || r.description || '', enhance: false }
  }
  async function revise(r) {
    if (!editing.prompt.trim()) return toast('The prompt is empty', 'error')
    const ok = await guard(() => api.post(`/api/references/${r.ref_id}/revise`, { prompt: editing.prompt, enhance: editing.enhance }))
    if (!ok) return
    editing = null
    toast(r.used_in_shots.length ? 'New version queued - when it is ready you choose which shots switch to it' : 'New version queued')
  }
  async function setCurrent(r, v) {
    if (r.used_in_shots.length) {
      app.impact = { ref_id: r.ref_id, version_id: v.id, shots: (await api.get(`/api/references/${r.ref_id}/impact`)).shots }
    } else await guard(() => api.post(`/api/references/${r.ref_id}/current/${v.id}`))
  }
  async function delRef(r) {
    const plan = await guard(() => api.get(`/api/references/${r.ref_id}/delete_plan?cascade=true`))
    if (!plan) return
    const warn = plan.blocked_by.length ? `\n\nIt is used by: ${plan.blocked_by.join('; ')}.\nThose shots lose this picture (their finished clips are kept but become stale).` : ''
    if (!confirm(`Delete '${r.name}' and ${plan.files.length} file(s)?${warn}`)) return
    await guard(() => api.del(`/api/references/${r.ref_id}?cascade=true`))
    load.refs()
  }
  async function delVersion(r, v) {
    if (!confirm(`Delete version ${v.n} of '${r.name}'?`)) return
    await guard(() => api.del(`/api/ref_versions/${v.id}?cascade=true`))
  }
  async function doUpload() {
    if (!upload.file || !upload.name) return
    const fd = new FormData()
    fd.append('file', upload.file); fd.append('name', upload.name); fd.append('kind', upload.kind)
    await guard(() => api.form('/api/references/upload', fd))
    upload = { name: '', kind: 'character', file: null }
  }
</script>

<div class="scroll pad">
  <div class="row">
    <input placeholder="Filter…" bind:value={filter} />
    <button class="primary" onclick={() => (creating = !creating)}>+ Design a reference</button>
    <span class="muted">or upload your own:</span>
    <input type="file" accept="image/*" onchange={(e) => (upload.file = e.target.files[0])} />
    <input placeholder="name" bind:value={upload.name} style="width:140px" />
    <select bind:value={upload.kind}>{#each KINDS as k}<option>{k}</option>{/each}</select>
    <button onclick={doUpload} disabled={!upload.file || !upload.name}>Upload</button>
  </div>

  {#if creating}
    <div class="card col" style="margin-top:10px;max-width:700px">
      <div class="row"><input class="grow" placeholder="Name, e.g. Mara - full body" bind:value={form.name} />
        <select bind:value={form.kind}>{#each KINDS as k}<option>{k}</option>{/each}</select></div>
      <textarea rows={form.enhance ? 3 : 8} bind:value={form.description}
        placeholder={form.enhance ? 'What it must show. Qwen writes the Krea prompt from this and the story bible.'
                                  : 'Exact Krea prompt (sent as-is; a locked project style still replaces its Style: line)'}></textarea>
      <div class="row"><button class="primary" onclick={create}>Render</button><button onclick={() => (creating = false)}>Cancel</button>
        <label class="row muted" title="Off: your text goes to Krea verbatim and Qwen doesn't revise it after the critic">
          <input type="checkbox" bind:checked={form.enhance} /> Enhance prompt with Qwen</label></div>
    </div>
  {/if}

  {#each Object.entries(groups) as [kind, refs]}
    <h3>{kind} ({refs.length})</h3>
    <div class="grid">
      {#each refs as r (r.ref_id)}
        {@const v = cur(r)}
        <div class="card col">
          <Media file_id={v?.file_id} size={196} label="" item={{ type: 'ref', id: r.ref_id }}
                 badge={v ? `v${v.n}${v.score != null ? ` · ${v.score}/10` : ''}` : ''} />
          <div class="row"><strong class="grow">{r.name}</strong>
            <button class="small" onclick={() => (open = open === r.ref_id ? null : r.ref_id)}>{open === r.ref_id ? 'less' : 'more'}</button></div>
          <div class="muted cap">{r.caption || r.description}</div>
          {#if r.used_in_shots.length}
            <div class="row">{#each r.used_in_shots as sid}<span class="badge">{shotTitle(sid)}</span>{/each}</div>
          {/if}
          {#if open === r.ref_id}
            <div class="col">
              {#if editing?.ref_id === r.ref_id}
                <div class="col">
                  <span class="muted small">Krea prompt for the new version{v?.n ? ` (from v${v.n})` : ''}</span>
                  <textarea rows="10" bind:value={editing.prompt}></textarea>
                  <div class="row">
                    <button class="primary small" onclick={() => revise(r)}>Render new version</button>
                    <button class="small" onclick={() => (editing = null)}>Cancel</button>
                    <label class="row muted" title="Off: this exact text goes to Krea. On: Qwen polishes it, keeping your edits.">
                      <input type="checkbox" bind:checked={editing.enhance} /> Enhance prompt with Qwen</label>
                  </div>
                </div>
              {/if}
              <div class="row">
                {#if editing?.ref_id !== r.ref_id}<button class="primary small" onclick={() => startRevise(r)}>New version…</button>{/if}
                <button class="small" onclick={() => pin({ type: 'ref', id: r.ref_id, file_id: v?.file_id, label: r.name })}>💬 Chat about it</button>
                <button class="small danger" onclick={() => delRef(r)}>Delete…</button>
              </div>
              <div class="row versions">
                {#each r.versions as ver}
                  <div class="col">
                    <Media file_id={ver.file_id} size={70} item={{ type: 'ref_version', id: ver.id }} badge={'v' + ver.n}
                           selected={ver.id === r.current_version} label={`${r.name} v${ver.n}`} />
                    {#if ver.id !== r.current_version}
                      <button class="small" onclick={() => setCurrent(r, ver)}>use</button>
                      <button class="small danger" onclick={() => delVersion(r, ver)}>✕</button>
                    {/if}
                  </div>
                {/each}
              </div>
              {#if v?.prompt}<details><summary class="muted">Krea prompt (v{v.n}, seed {v.seed})</summary><pre>{v.prompt}</pre></details>{/if}
              {#if v?.critic?.issues?.length}<div class="muted">Critic: {v.critic.issues.join('; ')}</div>{/if}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {:else}
    <p class="muted" style="margin-top:20px">No reference images yet. Ask Qwen in the chat, or design one here.</p>
  {/each}
</div>

<style>
  .cap { font-size: 12px; max-height: 54px; overflow: hidden; }
  .versions { align-items: flex-start; }
</style>
