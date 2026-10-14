<script>
  import { api } from '../lib/api.js'
  import { app, guard, load } from '../lib/state.svelte.js'
  const cls = { done: 'ok', failed: 'bad', running: 'warn', queued: '', cancelled: '' }
</script>

<div class="scroll pad col">
  <div class="row"><h2 class="grow">Jobs</h2><button onclick={load.jobs}>Refresh</button></div>
  {#each app.jobs as j (j.id)}
    <div class="card col">
      <div class="row">
        <strong>#{j.id} {j.kind}</strong><span class="badge">{j.server}</span><span class="badge {cls[j.state]}">{j.state}</span>
        <span class="muted grow">{j.payload?.name || (j.payload?.shot_id ? `shot ${j.payload.shot_id}` : '')} {j.payload?.change ? `- ${j.payload.change}` : ''}</span>
        {#if j.state === 'running' || j.state === 'queued'}<button class="small danger" onclick={() => guard(() => api.post(`/api/jobs/${j.id}/cancel`))}>Cancel</button>{/if}
      </div>
      {#if j.state === 'running'}<div class="progress"><div style="width:{Math.round(j.progress * 100)}%"></div></div>{/if}
      {#if j.message}<div class="muted msg">{j.message}</div>{/if}
    </div>
  {:else}<p class="muted">No jobs yet.</p>{/each}
</div>

<style>.msg { font-size: 12px; white-space: pre-wrap; }</style>
