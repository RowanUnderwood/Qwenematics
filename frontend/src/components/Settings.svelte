<script>
  import { api } from '../lib/api.js'
  import { app, guard, toast, load } from '../lib/state.svelte.js'

  let ps = $state(null)
  $effect(() => { if (app.psettings && !ps) ps = JSON.parse(JSON.stringify(app.psettings)) })
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
    if (!confirm(`Clean up ${plan.files.length} file(s), ${(plan.bytes / 1048576).toFixed(1)} MB?\n\nRemoves unpicked previews of finished shots, spent latents, unused old reference versions, and cached proxies/thumbnails.`)) return
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
    if (app.project) what.push(`this project's video aspect, iteration settings and workflow parameter overrides`)
    if (!confirm(`Restore defaults?\n\nResets: ${what.join('; ')}.\n\nKept: the style locked in the Bible tab, the bible, shots, references and renders.`)) return
    const r = await guard(() => api.post('/api/settings/restore_defaults'))
    if (!r) return
    gs = null; ps = null                  // re-read both from the server
    api.get('/api/gpu_sharing').then((x) => (gs = x))
    await load.psettings()
    toast('Defaults restored')
  }
</script>

<div class="scroll pad col" style="max-width:1100px">
  <div class="row"><h2 class="grow" style="margin:0">Settings</h2>
    <button onclick={restoreDefaults} title="Reset GPU sharing and this project's settings; keeps the locked style and all work">Restore defaults…</button></div>
  <h2>Servers</h2>
  {#if app.status}
    {#each Object.entries(app.status.servers) as [name, s]}
      <div class="row card">
        <strong style="width:60px">{name}</strong><span class="badge {s.up ? 'ok' : 'bad'}">{s.up ? 'up' : 'down'}</span>
        <span class="muted grow">{s.url} {s.gpu || ''} {s.error || ''}</span>
        {#if !s.up}<button onclick={() => launch(name)}>Launch</button>{/if}
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
        <label class="f">Borrow it from this many image jobs
          <input type="number" min="1" max="20" value={gs.min_backlog} style="width:70px"
                 onchange={(e) => saveSharing({ min_backlog: +e.target.value })} /></label>
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
      <span class="muted" style="font-size:12px">A video render always comes first: it waits only for the image already on the video GPU, which is then emptied (switching back to MiniMax costs ~26 s, so a single image stays on the image GPU unless Krea is already loaded there). "Reference jobs at once" applies the next time a project is opened.</span>
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
        <label class="f">Video aspect ratio
          <select bind:value={ps.video_aspect}>
            {#each (app.workflows.find((w) => w.id === 'minimax_h3')?.params.aspect_ratio?.choices || [ps.video_aspect]) as c}<option>{c}</option>{/each}
          </select></label>
        <label class="f">Auto-iterate with the critic
          <select bind:value={ps.iterate.auto}><option value={undefined}>default</option><option value={true}>on</option><option value={false}>off</option></select></label>
        <label class="f">Extra image rounds <input type="number" min="0" max="5" bind:value={ps.iterate.max_image_iterations} style="width:70px" /></label>
        <label class="f">Extra video rounds <input type="number" min="0" max="3" bind:value={ps.iterate.max_video_iterations} style="width:70px" /></label>
        <label class="f">Accept score (/10) <input type="number" min="1" max="10" bind:value={ps.iterate.accept_score} style="width:70px" /></label>
        <label class="f">Seed-hunt auto-proceed (s, 0 = wait for me) <input type="number" min="0" max="600" bind:value={ps.iterate.seedhunt_auto_proceed_s} style="width:70px" /></label>
      </div>
    </div>

    {#each app.workflows.filter((w) => w.usable || w.problems.length === 0) as w}
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
  {/if}
</div>

<style>
  .pgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 10px; }
  .help { font-size: 11px; }
  code { background: #0f1114; padding: 1px 4px; border-radius: 4px; }
</style>
