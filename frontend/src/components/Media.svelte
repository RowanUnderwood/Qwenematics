<script>
  // A clickable thumbnail for any project file. Click opens the lightbox;
  // the small chat button pins the item to the next chat message.
  import { app, pin, thumbSrc } from '../lib/state.svelte.js'

  let { file_id, video = false, label = '', item = null, size = 160, badge = '', selected = false, onclick = null } = $props()
  let failed = $state(false)         // the file was deleted (or is missing on disk)
  $effect(() => { file_id; failed = false })

  function open(e) {
    if (onclick) return onclick(e)
    app.lightbox = { file_id, video, label, item }
  }
</script>

{#if file_id && !failed}
  <div class="m" class:selected style="width:{size}px">
    <button class="img" onclick={open} title={label}>
      <img src={thumbSrc(file_id, Math.max(320, size * 2))} alt={label} loading="lazy" onerror={() => (failed = true)} />
      {#if video}<span class="play">▶</span>{/if}
      {#if badge}<span class="b">{badge}</span>{/if}
    </button>
    {#if item}
      <button class="chat small" title="Chat with Qwen about this" onclick={() => pin({ ...item, file_id, video, label })}>💬</button>
    {/if}
    {#if label}<div class="lbl" title={label}>{label}</div>{/if}
  </div>
{:else}
  <div class="m empty" style="width:{size}px;height:{size * 0.6}px">{failed ? `${label ? label + ': ' : ''}file missing` : label || 'rendering…'}</div>
{/if}

<style>
  .m { position: relative; display: inline-flex; flex-direction: column; gap: 3px; }
  .m.selected .img { outline: 2px solid var(--ok); }
  .img { padding: 0; border: none; background: #000; border-radius: 6px; overflow: hidden; position: relative; line-height: 0; }
  .img img { width: 100%; height: auto; max-height: 260px; object-fit: contain; display: block; }
  .play { position: absolute; inset: 0; display: grid; place-items: center; font-size: 26px; color: #fff;
    text-shadow: 0 0 8px #000; line-height: 1; }
  .b { position: absolute; left: 4px; top: 4px; background: #000a; color: #fff; font-size: 11px; padding: 1px 5px;
    border-radius: 4px; line-height: 1.4; }
  .chat { position: absolute; right: 4px; top: 4px; opacity: .85; padding: 0 4px; line-height: 1.5; }
  .lbl { font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .empty { display: grid; place-items: center; background: var(--panel2); border-radius: 6px; color: var(--muted); font-size: 12px; }
</style>
