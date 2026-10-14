<script>
  import { api } from '../lib/api.js'
  import { app, guard, load } from '../lib/state.svelte.js'
  import StylePicker from './StylePicker.svelte'

  const KINDS = ['plot', 'style', 'character', 'location', 'dialogue', 'audio', 'note']
  let edits = $state({})
  let fresh = $state({ kind: 'character', title: '', body: '' })

  const byKind = (k) => app.bible.filter((e) => e.kind === k)
  async function save(e) {
    await guard(() => api.put('/api/bible', { id: e.id, kind: e.kind, title: e.title, body: edits[e.id] ?? e.body }))
    delete edits[e.id]; load.bible()
  }
  async function del(e) { if (confirm(`Delete '${e.title}'?`)) { await guard(() => api.del(`/api/bible/${e.id}`)); load.bible() } }
  async function add() {
    if (!fresh.title) return
    await guard(() => api.put('/api/bible', fresh)); fresh = { kind: fresh.kind, title: '', body: '' }; load.bible()
  }
</script>

<div class="scroll pad col" style="max-width:1000px">
  <p class="muted">The story bible is the project memory Qwen and its prompt writers read every time: plot, characters' exact looks, locations, the visual style, dialogue and audio. Edit it freely; Qwen keeps it updated too.</p>
  <StylePicker />
  {#each KINDS as k}
    {#if byKind(k).length}
      <h3>{k}</h3>
      {#each byKind(k) as e (e.id)}
        <div class="card col">
          <div class="row"><strong class="grow">{e.title}</strong>
            {#if edits[e.id] !== undefined}<button class="primary small" onclick={() => save(e)}>Save</button>{/if}
            <button class="small danger" onclick={() => del(e)}>✕</button></div>
          <textarea rows={Math.min(14, Math.max(3, e.body.split('\n').length + 1))} value={edits[e.id] ?? e.body}
                    oninput={(ev) => (edits[e.id] = ev.target.value)}></textarea>
        </div>
      {/each}
    {/if}
  {/each}
  <h3>Add an entry</h3>
  <div class="card col">
    <div class="row"><select bind:value={fresh.kind}>{#each KINDS as k}<option>{k}</option>{/each}</select>
      <input class="grow" placeholder="Title" bind:value={fresh.title} /></div>
    <textarea rows="4" bind:value={fresh.body}></textarea>
    <div><button class="primary" onclick={add}>Add</button></div>
  </div>
</div>
