<script>
  // The project's visual style, chosen from the Krea 2 style library by its sample images.
  // Locking a style makes every Krea prompt use its exact Style paragraph.
  import { api } from '../lib/api.js'
  import { app, guard, toast, load } from '../lib/state.svelte.js'

  let lib = $state(null)            // { styles, bases, current }
  let open = $state(false)
  let q = $state('')
  let cat = $state('')
  let base = $state(remembered() || 'portrait')
  let focus = $state(null)          // style shown large in the picker

  const img = (s, size) => `/api/styles/${base}/${s.id}/image${size ? `?size=${size}` : ''}`
  let current = $derived(lib && app.psettings?.krea_style ? lib.styles.find((s) => s.name === app.psettings.krea_style) : null)
  let cats = $derived(lib ? [...new Set(lib.styles.flatMap((s) => s.categories))].sort() : [])
  let shown = $derived(!lib ? [] : lib.styles.filter((s) => {
    if (cat && !s.categories.includes(cat)) return false
    if (!q) return true
    const t = q.toLowerCase()
    return s.name.toLowerCase().includes(t) || s.paragraph.toLowerCase().includes(t)
  }))
  let baseInfo = $derived(lib?.bases.find((b) => b.id === base))

  function remembered() { try { return localStorage.getItem('styleBase') } catch { return null } }
  $effect(() => { try { localStorage.setItem('styleBase', base) } catch {} })
  $effect(() => { if (app.project && !lib) api.get('/api/styles').then((r) => (lib = r)) })

  async function lock(s) {
    const r = await guard(() => api.put('/api/project/style', { name: s ? s.name : null }))
    if (!r) return
    await load.psettings()
    toast(s ? `Style locked: ${s.name}` : 'Style unlocked')
    open = false; focus = null
  }
</script>

<div class="card col">
  <div class="row">
    <h3 style="margin:0" class="grow">Visual style</h3>
    <button onclick={() => { open = !open; focus = current }}>{open ? 'Close picker' : current ? 'Change style…' : 'Choose a style…'}</button>
    {#if current}<button class="danger" onclick={() => lock(null)}>Unlock</button>{/if}
  </div>
  {#if current}
    <div class="row cur">
      <img src={img(current, 384)} alt={current.name} />
      <div class="col grow">
        <strong>{current.name} <span class="badge ok">locked</span></strong>
        <span class="muted small">{current.categories.join(' · ')}</span>
        <span class="small">{current.paragraph}</span>
        <span class="muted small">Every new reference image uses this exact Style paragraph, and Qwen keeps the video prompts in it. Existing references keep their look until you regenerate them.</span>
      </div>
    </div>
  {:else if !open}
    <span class="muted">No style locked: Qwen picks or writes a style per image from the bible. Lock one to keep every image consistent.</span>
  {/if}

  {#if open && lib}
    <div class="row">
      <input class="grow" type="search" placeholder="Search names and descriptions (e.g. noir, watercolor, ghibli)…" bind:value={q} />
      <select bind:value={cat}><option value="">All categories</option>{#each cats as c}<option>{c}</option>{/each}</select>
      <label class="row muted">shown on
        <select bind:value={base}>{#each lib.bases as b}<option value={b.id}>{b.label}</option>{/each}</select></label>
    </div>
    <span class="muted small">{shown.length} styles · samples from the <a href="https://matplinta.github.io/t2i-krea-2-style-library/" target="_blank" rel="noreferrer">Krea 2 Style Library</a>, all rendered with the same seed on: “{baseInfo?.prompt}”</span>

    {#if focus}
      <div class="row focus card">
        <img src={img(focus)} alt={focus.name} />
        <div class="col grow">
          <strong>{focus.name}</strong>
          <span class="muted small">{focus.categories.join(' · ')}</span>
          <span class="small">{focus.paragraph}</span>
          <div class="row">
            {#if focus.name !== current?.name}<button class="primary" onclick={() => lock(focus)}>Use this style for the project</button>
            {:else}<span class="badge ok">current style</span>{/if}
            <button onclick={() => (focus = null)}>Back to all</button>
          </div>
        </div>
      </div>
    {/if}

    <div class="sgrid">
      {#each shown as s (s.id)}
        <button class="tile" class:on={s.name === current?.name} class:sel={s.name === focus?.name} onclick={() => (focus = s)} title={s.paragraph}>
          <img src={img(s, 256)} alt={s.name} loading="lazy" style="aspect-ratio:{baseInfo?.width || 2}/{baseInfo?.height || 3}" />
          <span>{s.name}</span>
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .small { font-size: 12px; }
  .cur { align-items: flex-start; flex-wrap: nowrap; }
  .cur img { width: 150px; border-radius: 6px; flex: none; }
  .focus { align-items: flex-start; flex-wrap: nowrap; }
  .focus img { max-width: 45%; max-height: 520px; border-radius: 6px; object-fit: contain; flex: none; }
  .sgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; max-height: 70vh; overflow: auto; }
  .tile { display: flex; flex-direction: column; gap: 4px; padding: 4px; text-align: center; background: var(--panel2); }
  .tile img { width: 100%; object-fit: cover; border-radius: 4px; background: #0f1114; }
  .tile span { font-size: 12px; line-height: 1.2; }
  .tile.on { border-color: var(--ok); }
  .tile.sel { border-color: var(--accent); }
  @media (max-width: 700px) { .cur, .focus { flex-wrap: wrap; } .focus img { max-width: 100%; } }
</style>
