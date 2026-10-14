<script>
  // Shown when a reference that shots depend on gets a new version: the user picks which
  // shots switch (they become stale and re-render on demand) and which keep the old one.
  import { api } from '../lib/api.js'
  import { app, guard, load } from '../lib/state.svelte.js'

  let chosen = $state({})
  $effect(() => {
    if (app.impact) chosen = Object.fromEntries(app.impact.shots.map((s) => [s.shot_id, true]))
  })
  const title = (id) => app.shots.find((s) => s.id === id)?.title || `shot ${id}`
  const refName = () => app.refs.find((r) => r.ref_id === app.impact?.ref_id)?.name || 'this reference'

  async function apply(all) {
    const ids = all ? null : Object.entries(chosen).filter(([, v]) => v).map(([k]) => +k)
    await guard(() => api.post(`/api/references/${app.impact.ref_id}/apply_version`, { version_id: app.impact.version_id, shot_ids: ids }))
    app.impact = null; load.shots(); load.refs()
  }
</script>

{#if app.impact}
  <div class="back">
    <div class="card col box">
      <h2>New version of {refName()}</h2>
      <p>These shots use {refName()}. Shots you switch will use the new version on their <b>next render</b>; their current clips are kept but marked stale. Unticked shots keep the old version.</p>
      {#each app.impact.shots as s}
        <label class="row"><input type="checkbox" bind:checked={chosen[s.shot_id]} />
          {title(s.shot_id)} <span class="muted">slot {s.slot}</span>
          {#if s.final_uses_it}<span class="badge warn">has a finished clip</span>{/if}</label>
      {/each}
      <div class="row">
        <button class="primary" onclick={() => apply(false)}>Switch ticked shots</button>
        <button onclick={async () => { await guard(() => api.post(`/api/references/${app.impact.ref_id}/current/${app.impact.version_id}`)); app.impact = null; load.refs() }}>Only make it the default for new shots</button>
        <button onclick={() => (app.impact = null)}>Decide later</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .back { position: fixed; inset: 0; background: #000a; display: grid; place-items: center; z-index: 55; }
  .box { max-width: 560px; }
</style>
