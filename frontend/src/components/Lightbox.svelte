<script>
  import { api } from '../lib/api.js'
  import { app, pin, guard, toast, fileSrc } from '../lib/state.svelte.js'

  let lb = $derived(app.lightbox)
  let zoom = $state(false)

  function close() { app.lightbox = null; zoom = false }
  function key(e) { if (e.key === 'Escape') close() }
  async function reveal() { await guard(() => api.post(`/api/files/${lb.file_id}/reveal`)) }
  async function saveAs() {
    const dest = prompt('Copy to folder or full path:', 'C:\\Users\\Public\\Videos')
    if (!dest) return
    const r = await guard(() => api.post(`/api/files/${lb.file_id}/save_as`, { dest }))
    if (r) toast('Saved to ' + r.path)
  }
</script>

<svelte:window onkeydown={key} />

{#if lb}
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
  <div class="back" onclick={close}>
    <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
    <div class="box" onclick={(e) => e.stopPropagation()}>
      <div class="bar row">
        <strong class="grow">{lb.label}</strong>
        {#if lb.item}<button onclick={() => { pin({ ...lb.item, file_id: lb.file_id, video: lb.video, label: lb.label }); close() }}>💬 Chat about this</button>{/if}
        <button onclick={reveal}>Show in Explorer</button>
        <button onclick={saveAs}>Save a copy…</button>
        <a href={fileSrc(lb.file_id)} download><button>Download</button></a>
        <button onclick={close}>✕</button>
      </div>
      <div class="stage" class:zoom>
        {#if lb.video}
          <!-- svelte-ignore a11y_media_has_caption -->
          <video src={fileSrc(lb.file_id)} controls autoplay loop></video>
        {:else}
          <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
          <img src={fileSrc(lb.file_id)} alt={lb.label} onclick={() => (zoom = !zoom)} title="Click to toggle 100% zoom" />
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .back { position: fixed; inset: 0; background: #000c; z-index: 50; display: grid; place-items: center; }
  .box { width: 94vw; height: 92vh; display: flex; flex-direction: column; background: var(--panel); border-radius: 10px;
    border: 1px solid var(--line); overflow: hidden; }
  .bar { padding: 8px 10px; border-bottom: 1px solid var(--line); }
  .stage { flex: 1; display: grid; place-items: center; overflow: hidden; background: #000; }
  .stage.zoom { overflow: auto; place-items: start; }
  .stage img, .stage video { max-width: 100%; max-height: 100%; object-fit: contain; }
  .stage.zoom img { max-width: none; max-height: none; cursor: zoom-out; }
  .stage img { cursor: zoom-in; }
</style>
