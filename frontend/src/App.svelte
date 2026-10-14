<script>
  import { onMount } from 'svelte'
  import { api } from './lib/api.js'
  import { app, start, load, guard, toast } from './lib/state.svelte.js'
  import Chat from './components/Chat.svelte'
  import References from './components/References.svelte'
  import Shots from './components/Shots.svelte'
  import Timeline from './components/Timeline.svelte'
  import Bible from './components/Bible.svelte'
  import Jobs from './components/Jobs.svelte'
  import Settings from './components/Settings.svelte'
  import Lightbox from './components/Lightbox.svelte'
  import ImpactDialog from './components/ImpactDialog.svelte'

  onMount(start)

  const tabs = [
    ['chat', 'Chat'], ['refs', 'Reference images'], ['shots', 'Shots & video'], ['timeline', 'Timeline'],
    ['bible', 'Story bible'], ['jobs', 'Jobs'], ['settings', 'Settings'],
  ]
  let menu = $state(false)
  let off = $state(null)          // shutdown dialog: { comfy, llm }
  let stopped = $state(false)
  let activeJobs = $derived(app.jobs.filter((j) => j.state === 'running' || j.state === 'queued').length)

  async function newProject() {
    const name = prompt('New project name:')
    if (!name) return
    await guard(() => api.post('/api/projects', { name }))
    await load.all(); menu = false
  }
  async function openProject(p) {
    await guard(() => api.post('/api/projects/open', { path: p.path }))
    await load.all(); menu = false
  }
  async function saveAs() {
    const name = prompt('Save a copy of this project as:', app.project.name + ' copy')
    if (!name) return
    const r = await guard(() => api.post('/api/project/snapshot', { name }))
    if (r) toast('Saved copy to ' + r.path)
    load.projects(); menu = false
  }
  async function archive() {
    const r = await guard(() => api.post('/api/project/archive'))
    if (r) toast('Archived to ' + r.path); menu = false
  }
  async function deleteProject() {
    const plan = await guard(() => api.get('/api/project/delete_plan'))
    if (!plan) return
    const mb = (plan.folder_bytes / 1048576).toFixed(1)
    if (!confirm(`Delete project "${app.project.name}" permanently?\n\nFolder: ${plan.folder} (${mb} MB: chat, images, videos, database)\nPlus ${plan.outside_files.length} file(s) left in ComfyUI folders.\n\nThis cannot be undone.`)) return
    const r = await guard(() => api.del('/api/project'))
    if (r && r.failed.length) toast('Some files could not be removed: ' + r.failed.join(', '), 'error')
    await load.all(); menu = false
  }
  async function shutDown() {
    const r = await guard(() => api.post('/api/shutdown', off))
    if (r) { off = null; stopped = true }
  }
</script>

<div class="shell">
  <header class="row">
    <div class="brand row"><img src="/logo.svg" alt="" class="logo" />Qwenematics</div>
    <div class="proj">
      <button onclick={() => (menu = !menu)}>{app.project ? app.project.name : 'No project'} ▾</button>
      {#if menu}
        <div class="menu card col">
          <button onclick={newProject}>New project…</button>
          {#if app.project}
            <button onclick={saveAs}>Save as / snapshot…</button>
            <button onclick={archive}>Archive to .zip</button>
            <button class="danger" onclick={deleteProject}>Delete project and all its files…</button>
          {/if}
          <h3>Open</h3>
          {#each app.projects as p}
            <button onclick={() => openProject(p)}>{p.folder} <span class="muted">{p.modified}</span></button>
          {:else}<span class="muted">No projects yet</span>{/each}
        </div>
      {/if}
    </div>
    <nav class="row">
      {#each tabs as [id, label]}
        <button class:on={app.tab === id} onclick={() => (app.tab = id)}>
          {label}{#if id === 'jobs' && activeJobs}<span class="badge warn">{activeJobs}</span>{/if}
        </button>
      {/each}
    </nav>
    <div class="grow"></div>
    {#if app.status}
      <div class="row servers">
        {#each Object.entries(app.status.servers) as [name, s]}
          <span class="badge {s.up ? 'ok' : 'bad'}" title={`${s.gpu || s.url}${s.loaded ? ' - ' + s.loaded + ' loaded' : ''}`}>{name}{#if name === 'video' && s.loaded === 'krea'} · images{/if}</span>
        {/each}
        <button class="small" onclick={() => load.status(true)} title="Re-check servers, nodes, models and workflow files">↻</button>
      </div>
    {/if}
    <button class="small" onclick={() => (off = { comfy: true, llm: true })} title="Shut down the harness">⏻ Shut down</button>
  </header>

  <main>
    {#if !app.project && app.tab !== 'settings'}
      <div class="pad col" style="max-width:520px">
        <h2>Open or create a project</h2>
        <button class="primary" onclick={newProject}>New project…</button>
        {#each app.projects as p}<button onclick={() => openProject(p)}>{p.folder}</button>{/each}
      </div>
    {:else if app.tab === 'chat'}<Chat />
    {:else if app.tab === 'refs'}<References />
    {:else if app.tab === 'shots'}<Shots />
    {:else if app.tab === 'timeline'}<Timeline />
    {:else if app.tab === 'bible'}<Bible />
    {:else if app.tab === 'jobs'}<Jobs />
    {:else if app.tab === 'settings'}<Settings />
    {/if}
  </main>
</div>

{#if off}
  <div class="veil">
    <div class="card col dlg">
      <h2>Shut down the harness?</h2>
      {#if activeJobs}<div class="badge warn">{activeJobs} job(s) running or queued will be cancelled. Finished images and clips are kept.</div>{/if}
      <label class="row"><input type="checkbox" bind:checked={off.comfy} /> Also close the harness's ComfyUI windows (image :8190, video :8189)</label>
      <label class="row"><input type="checkbox" bind:checked={off.llm} /> Also stop Qwen (Ninfer on the 4090)</label>
      <span class="muted">Everything is saved as you go. Other ComfyUI instances (e.g. Stability Matrix) are not touched.</span>
      <div class="row"><div class="grow"></div><button onclick={() => (off = null)}>Cancel</button><button class="danger" onclick={shutDown}>Shut down</button></div>
    </div>
  </div>
{/if}
{#if stopped}
  <div class="veil">
    <div class="card col dlg">
      <h2>Harness stopped</h2>
      <span>{app.shutdownSteps?.length ? app.shutdownSteps.join(' · ') : 'Shutting down…'}</span>
      <span class="muted">You can close this tab. Start it again with Qwenematics.lnk or start_qwenematics.bat.</span>
    </div>
  </div>
{/if}

<Lightbox />
<ImpactDialog />

<div class="toasts">
  {#each app.toasts as t (t.id)}<div class="toast {t.kind}">{t.msg}</div>{/each}
</div>

<style>
  .shell { height: 100%; display: flex; flex-direction: column; }
  header { padding: 6px 10px; border-bottom: 1px solid var(--line); background: var(--panel); flex-wrap: nowrap; }
  .brand { font-weight: 600; margin-right: 8px; white-space: nowrap; gap: 7px; flex-wrap: nowrap; }
  .logo { width: 22px; height: 22px; display: block; }
  nav button { border-color: transparent; background: transparent; }
  nav button.on { background: var(--panel2); border-color: var(--line); }
  nav .badge { margin-left: 5px; }
  .proj { position: relative; }
  .menu { position: absolute; top: 34px; left: 0; z-index: 30; min-width: 320px; max-height: 70vh; overflow: auto; }
  .menu button { text-align: left; }
  main { flex: 1; min-height: 0; overflow: hidden; display: flex; flex-direction: column; }
  .toasts { position: fixed; right: 14px; bottom: 14px; display: flex; flex-direction: column; gap: 6px; z-index: 60; }
  .toast { background: var(--panel2); border: 1px solid var(--line); padding: 8px 12px; border-radius: 8px; max-width: 460px; }
  .toast.error { border-color: var(--bad); }
  .servers .badge { cursor: default; }
  .veil { position: fixed; inset: 0; background: rgba(0,0,0,.55); display: grid; place-items: center; z-index: 80; }
  .dlg { max-width: 520px; gap: 10px; }
</style>
