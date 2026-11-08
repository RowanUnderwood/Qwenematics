<script>
  import { api } from '../lib/api.js'
  import { app, guard, toast, load, imageOnly } from '../lib/state.svelte.js'

  let ps = $state(null)
  let psPath = null                      // the project `ps` was copied from: a save must never reach another project
  $effect(() => {
    if (app.psettings && (!ps || psPath !== app.psettingsPath)) {
      ps = JSON.parse(JSON.stringify(app.psettings)); psPath = app.psettingsPath
    }
  })
  let preferred = (id) => app.workflows.filter((w) => w.id === id)

  async function launch(name) {
    const r = await guard(() => api.post(`/api/servers/${name}/launch`))
    if (r) toast(`Started ${r.launcher} - give it a minute, then press ↻`)
  }
  async function saveSettings() {
    await guard(() => api.put('/api/project/settings', ps)); await load.psettings(); toast('Settings saved')
  }
  function setParam(wf, k, v) {
    ps.params[wf] ||= {}
    if (v === '' || v === null || v === undefined) delete ps.params[wf][k]
    else ps.params[wf][k] = v
  }
  async function cleanup() {
    const plan = await guard(() => api.get('/api/cleanup/plan'))
    if (!plan) return
    const what = imageOnly() ? 'image versions and variations you are not using, and cached thumbnails'
      : 'unpicked previews of finished shots, spent latents, unused old reference versions, and cached proxies/thumbnails'
    if (!confirm(`Clean up ${plan.files.length} file(s), ${(plan.bytes / 1048576).toFixed(1)} MB?\n\nRemoves ${what}.`)) return
    const r = await guard(() => api.post('/api/cleanup'))
    if (r) toast(`Removed ${r.removed} file(s)`)
  }
  const wfKey = (w) => w.id

  let gs = $state(null)                  // GPU sharing settings + pool state
  $effect(() => { app.status; api.get('/api/gpu_sharing').then((r) => (gs = r)) })   // refresh with every status update
  async function saveSharing(change) {
    const r = await guard(() => api.put('/api/gpu_sharing', change))
    if (r) { gs = r; toast('GPU sharing saved') }
  }
  async function restoreDefaults() {
    const what = ['GPU sharing (back to settings.yaml)']
    if (app.project) what.push(imageOnly() ? `this project's iteration settings and workflow parameter overrides` : `this project's video aspect, iteration settings and workflow parameter overrides`)
    const kept = imageOnly() ? 'the locked style and your images' : 'the style locked in the Bible tab, the bible, shots, references and renders'
    if (!confirm(`Restore defaults?\n\nResets: ${what.join('; ')}.\n\nKept: ${kept}.`)) return
    const r = await guard(() => api.post('/api/settings/restore_defaults'))
    if (!r) return
    gs = null; ps = null                  // re-read both from the server
    api.get('/api/gpu_sharing').then((x) => (gs = x))
    await load.psettings()
    toast('Defaults restored')
  }
  let llm = $state(null)                 // /api/llm: backend, active, local, remote
  let llmForm = $state({ backend: 'local', url: '', model: '' })
  let llmTest = $state(null)             // last /api/llm/test result
  let testing = $state(false)
  async function loadLlm() {
    const r = await guard(() => api.get('/api/llm'))
    if (!r) return
    llm = r
    llmForm = { backend: r.backend, url: (r.remote.base_url || '').replace(/^https?:\/\//, ''), model: r.remote.model || '' }
  }
  loadLlm()
  async function testLlm() {
    testing = true; llmTest = null
    llmTest = await guard(() => api.post('/api/llm/test', { remote_url: llmForm.url, remote_model: llmForm.model }))
    testing = false
  }
  async function saveLlm() {
    const r = await guard(() => api.put('/api/llm', { backend: llmForm.backend, remote_url: llmForm.url, remote_model: llmForm.model }))
    if (r) { await loadLlm(); toast(`Qwen now uses ${r.active.model} at ${r.active.url}`) }
  }
  const yn = (v) => (v === true ? 'ok' : v === false ? 'bad' : '')

  async function upgrade() {
    if (!confirm('Upgrade this project to a full project?\n\nShots, the timeline, the story bible and video tools become available. Your images stay as they are.\n\nThis cannot be undone.')) return
    if (await guard(() => api.post('/api/project/upgrade'))) toast('This is now a full project')
  }
</script>

<div class="scroll pad col" style="max-width:1100px">
  <div class="row"><h2 class="grow" style="margin:0">Settings</h2>
    <button onclick={restoreDefaults} title="Reset GPU sharing and this project's settings; keeps the locked style and all work">Restore defaults…</button></div>
  <h2>Qwen (LLM)</h2>
  {#if llm}
    <div class="card col">
      <div class="row">
        <label class="row"><input type="radio" value="local" bind:group={llmForm.backend} /> Local: Ninfer on the 4090 ({llm.local.model})</label>
        <label class="row"><input type="radio" value="remote" bind:group={llmForm.backend} /> Remote: llama.cpp on the LAN</label>
      </div>
      <div class="row">
        <label class="f">Address (IP:port)
          <input bind:value={llmForm.url} placeholder="192.168.2.74:8080" style="width:200px" /></label>
        <label class="f">Model
          <input bind:value={llmForm.model} placeholder="qwen36-genesis-hermes" style="width:240px" /></label>
        <button onclick={testLlm} disabled={testing}>{testing ? 'Testing…' : 'Test'}</button>
        <button class="primary" onclick={saveLlm}>{llmForm.backend === llm.backend ? 'Save' : 'Use'}</button>
      </div>
      {#if llmTest}
        <div class="row" style="font-size:12px">
          <span class="badge {llmTest.up ? 'ok' : 'bad'}">{llmTest.up ? 'up' : 'down'}</span>
          {#if llmTest.up}
            <span class="badge {llmTest.model_found ? 'ok' : 'bad'}">model {llmTest.model_found ? 'found' : 'not found'}</span>
            <span class="badge {yn(llmTest.vision_ok ?? llmTest.vision)}">vision {llmTest.vision_ok ? 'works' : llmTest.vision ? 'claimed, check failed' : 'no'}</span>
            <span class="badge {yn(llmTest.video)}">video {llmTest.video ? 'claimed' : 'no'}</span>
            {#if llmTest.n_ctx}<span class="muted">context {llmTest.n_ctx}</span>{/if}
            {#each llmTest.models as id}<button class="small" onclick={() => (llmForm.model = id)} title="Use this model name">{id}</button>{/each}
          {/if}
          {#if llmTest.error}<span class="muted">{llmTest.error}</span>{/if}
        </div>
      {/if}
      <span class="muted" style="font-size:12px">
        Now using {llm.active.model} at {llm.active.url}; clips go to Qwen as {llm.active.media_mode === 'frames' ? 'timestamped stills' : 'video'}.
        The remote always sends stills, because its server rejects video input. Switch while Qwen and the image/video jobs are idle.
      </span>
    </div>
  {/if}

  <h2>Servers</h2>
  {#if app.status}
    {#each Object.entries(app.status.servers) as [name, s]}
      <div class="row card">
        <strong style="width:60px">{name}</strong><span class="badge {s.up ? 'ok' : 'bad'}">{s.up ? 'up' : 'down'}</span>
        <span class="muted grow">{s.url} {s.gpu || ''} {s.error || ''}</span>
        {#if !s.up && !s.remote}<button onclick={() => launch(name)}>Launch</button>{/if}
      </div>
    {/each}
  {/if}

  <h2>GPU sharing for images</h2>
  {#if gs}
    <div class="card col">
      <div class="row">
        <label class="f">Krea images on the video GPU while no video renders
          <select value={gs.image_on_video_gpu} onchange={(e) => saveSharing({ image_on_video_gpu: e.target.value })}>
            <option value="auto">auto (recommended)</option><option value="off">off: image GPU only</option>
          </select></label>
        {#if !imageOnly()}
          <label class="f">Borrow it from this many image jobs
            <input type="number" min="1" max="20" value={gs.min_backlog} style="width:70px"
                   onchange={(e) => saveSharing({ min_backlog: +e.target.value })} /></label>
        {/if}
        <label class="f">Reference jobs at once
          <input type="number" min="1" max="8" value={gs.image_jobs_in_flight} style="width:70px"
                 onchange={(e) => saveSharing({ image_jobs_in_flight: +e.target.value })} /></label>
      </div>
      <span class="muted" style="font-size:12px">
        Now: video GPU has {gs.pool.loaded.video || 'nothing known'} loaded{gs.pool.busy.video ? ', busy' : ''};
        image GPU {gs.pool.busy.image ? 'busy' : 'idle'}; {gs.pool.waiting_images} image(s) waiting.
        Typical seconds per image: image GPU {gs.pool.est_s.image}, video GPU {gs.pool.est_s.video}.
        {#if gs.krea_on_video_problems.length}<br /><span class="badge bad">can't borrow</span> {gs.krea_on_video_problems[0]}{/if}
      </span>
      <span class="muted" style="font-size:12px">A video render always comes first: it waits only for the image already on the video GPU, which is then emptied (switching back to MiniMax costs ~26 s, so in a full project a single image stays on the image GPU unless Krea is already loaded there; an image-only project always uses the video GPU first). The image GPU joins in from about 4 images. "Reference jobs at once" applies the next time a project is opened.</span>
    </div>
  {/if}

  <h2>Workflows</h2>
  {#each app.workflows as w}
    <div class="card col">
      <div class="row"><strong>{w.key}</strong><span class="muted grow">{w.label}</span>
        <span class="badge {w.usable ? 'ok' : 'bad'}">{w.usable ? 'usable' : 'problems'}</span></div>
      {#each w.problems as p}<div class="muted" style="font-size:12px">• {p}</div>{/each}
    </div>
  {/each}
  <p class="muted">To add a new workflow version: put its folder (source_ui.json, API graph(s), manifest.yaml) under <code>workflows/&lt;id&gt;/&lt;version&gt;/</code>. The old version stays available, and every render records which version made it.</p>

  {#if app.project && ps}
    <h2>Project settings</h2>
    <div class="card col">
      <div class="row">
        {#if !imageOnly()}
          <label class="f">Video aspect ratio
            <select bind:value={ps.video_aspect}>
              {#each (app.workflows.find((w) => w.id === 'minimax_h3')?.params.aspect_ratio?.choices || [ps.video_aspect]) as c}<option>{c}</option>{/each}
            </select></label>
        {/if}
        <label class="f">Auto-iterate with the critic
          <select bind:value={ps.iterate.auto}><option value={undefined}>default</option><option value={true}>on</option><option value={false}>off</option></select></label>
        <label class="f">Extra image rounds <input type="number" min="0" max="5" bind:value={ps.iterate.max_image_iterations} style="width:70px" /></label>
        {#if !imageOnly()}<label class="f">Extra video rounds <input type="number" min="0" max="3" bind:value={ps.iterate.max_video_iterations} style="width:70px" /></label>{/if}
        <label class="f">Accept score (/10) <input type="number" min="1" max="10" bind:value={ps.iterate.accept_score} style="width:70px" /></label>
        {#if !imageOnly()}<label class="f">Seed-hunt auto-proceed (s, 0 = wait for me) <input type="number" min="0" max="600" bind:value={ps.iterate.seedhunt_auto_proceed_s} style="width:70px" /></label>{/if}
      </div>
    </div>

    {#each app.workflows.filter((w) => (w.usable || w.problems.length === 0) && !(imageOnly() && w.id === 'minimax_h3')) as w}
      <h3>{w.key} parameters</h3>
      <p class="muted">Blank = workflow default. <span class="badge ok">Qwen</span> marks knobs Qwen may set per item; everything else is yours, applied to every render in this project.</p>
      <div class="pgrid">
        {#each Object.entries(w.params) as [k, spec]}
          {#if !spec.virtual || k === 'preview_seeds'}
            <label class="f">{k}{#if spec.qwen}<span class="badge ok">Qwen</span>{/if}
              {#if spec.choices}
                <select value={ps.params[wfKey(w)]?.[k] ?? ''} onchange={(e) => setParam(wfKey(w), k, e.target.value)}>
                  <option value="">default ({spec.default ?? '-'})</option>
                  {#each spec.choices as c}<option>{c}</option>{/each}
                </select>
              {:else if spec.type === 'int' || spec.type === 'float' || spec.type === 'seed'}
                <input type="number" step={spec.type === 'float' ? 0.05 : 1} min={spec.min} max={spec.max}
                       placeholder={spec.type === 'seed' ? 'random' : String(spec.default ?? '')}
                       value={ps.params[wfKey(w)]?.[k] ?? ''} onchange={(e) => setParam(wfKey(w), k, e.target.value === '' ? '' : Number(e.target.value))} />
              {:else if spec.type === 'pixaroma_loras'}
                <span class="muted">edit in the workflow file</span>
              {:else}
                <input value={ps.params[wfKey(w)]?.[k] ?? ''} placeholder={String(spec.default ?? '')} onchange={(e) => setParam(wfKey(w), k, e.target.value)} />
              {/if}
              {#if spec.help}<span class="muted help">{spec.help}</span>{/if}
            </label>
          {/if}
        {/each}
      </div>
    {/each}
    <div class="row"><button class="primary" onclick={saveSettings}>Save project settings</button></div>

    <h2>Housekeeping</h2>
    <div class="row"><button onclick={cleanup}>Clean up unused files…</button>
      <span class="muted">Project delete (everything, including ComfyUI-side files) is in the project menu.</span></div>
    {#if imageOnly()}
      <h2>Project type</h2>
      <div class="row"><button onclick={upgrade}>Upgrade to full project…</button>
        <span class="muted">Adds shots, the timeline, the story bible and video. This cannot be undone.</span></div>
    {/if}
  {/if}
</div>

<style>
  .pgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 10px; }
  .help { font-size: 11px; }
  code { background: #0f1114; padding: 1px 4px; border-radius: 4px; }
</style>
