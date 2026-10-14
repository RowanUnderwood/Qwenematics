# Qwenematics

Qwen + cinematics. 
A local GUI where Qwen 3.8 (Ninfer) works as a designer and collaborator. It drives two ComfyUI workflows: **Krea 2** for reference images and **MiniMax H3 SeedHunter** for clips of up to 15 s with audio. It plans the story, makes and reuses reference images, writes the model prompts from the bundled guides, checks its own results, and assembles the final video.

## Running it

1. `Qwenematics.lnk` (or `start_qwenematics.bat`) starts the GUI server and opens http://127.0.0.1:8765.
2. In **Settings**, launch whatever is down:
   - **image**: `launchers\comfy_image_3090.bat`, ComfyUI on the RTX 3090, port 8190 (Krea 2)
   - **video**: `launchers\comfy_video_5090.bat`, ComfyUI on the RTX 5090, port 8189 (MiniMax H3)
   - **llm**: `H:\Ninfer Qwen\launch-ninfer-4090.bat`, Ninfer on the RTX 4090, port 1235. The 4090 must be free; LM Studio holds it when loaded.
3. Create a project and talk to Qwen in **Chat**, for example: *"Create a 15 second video of X, creating all the reference images you think necessary first."*

Ports and GPUs live in `settings.yaml`. The launchers set `CUDA_DEVICE_ORDER=PCI_BUS_ID` (0 = 5090, 1 = 4090, 2 = 3090). Do **not** add `--use-sage-attention` for the 3090: Krea 2 renders black images there.

### GPU sharing for images
While no video is rendering, reference images also use the 5090. Krea takes about 11 s per image there, against about 33 s on the 3090 (`tools/bench_gpu_split.py`, results in `projects/_bench/results.md`).
- **Video comes first.** A shot render waits only for the image already on the 5090, which is then emptied for MiniMax.
- **Switching back to MiniMax costs about 26 s,** so a single image stays on the 3090 unless Krea is already loaded on the 5090.
- **Several reference jobs run at once** (`image_jobs_in_flight`, default 3), so Qwen writes and critiques while the GPUs render. Chat replies can wait behind one or two of those Qwen calls.
- **Settings tab, "GPU sharing for images"** switches this off or tunes it. Changes made there are kept in `projects/.harness_state.json` and override the defaults in `settings.yaml` → `gpu_sharing`.

## How a clip gets made

```
you ──chat──▶ Director (Qwen, tools) ── bible / shots / references (SQLite per project)
                 │ create_reference                │ render_shot
                 ▼                                 ▼
        Krea writer (Krea2 guide)          H3 writer (base + full-reference guides)
                 │                                 │
        Krea 2 on :8190 ──▶ critic ─┐     stage 1: N seeds at low res ──▶ critic picks
                 (auto-iterate)     │     (you can override in chat)  ──▶ stage 2 upscale ──▶ critic
                 ▼                  │                                 ▼
        reference version ◀─────────┘                        final clip ──▶ timeline ──▶ final cut (ffmpeg)
```

- **Reuse first:** Qwen must search existing references before creating one. The tool refuses otherwise.
- **Versions, not overwrites:** editing a reference adds a version. Shots point at a specific version. When a reference that shots use gets a new version, a dialog asks which shots switch. Switched shots are marked *stale* until re-rendered.
- **Context:** each director call gets a freshly built compact project state instead of the whole history. Older turns are text only, media is attached only for the current turn, and history is summarised at 60% of the context window. Writers and the critic are fresh, short calls.
- **Files:** everything lives in `projects/<name>/`. The harness deletes what it leaves in ComfyUI's `input/` and `output/` after each render. "Delete project" removes the folder and any recorded ComfyUI-side files. "Clean up" removes unpicked previews, spent latents and unused versions.

## Layout

| Path | What |
|---|---|
| `harness/comfy/ui_to_api.py` | Editor-workflow → API converter (subgraphs, bypass/mute, Anything Everywhere, Pixaroma state). Fallback for when ComfyUI's own API export fails. |
| `harness/comfy/manifest.py` | Workflow registry: semantic params → node inputs, modes, picture slots, stage hand-off |
| `harness/comfy/runner.py` / `client.py` | Upload → queue → progress → download → clean up ComfyUI's folders |
| `harness/core.py` | Domain logic and jobs (reference renders, seed hunt, final cut) |
| `harness/llm/` | Ninfer client, director agent + tools, writer/critic roles, media proxies |
| `harness/db/` | Project folder, SQLite models, dependency / staleness / delete planning |
| `harness/server.py` + `frontend/` | FastAPI + Svelte UI |
| `prompts/` | System prompts for the director, writers and critic, plus pose/style/H3 knowledge |
| `workflows/<id>/<version>/` | `source_ui.json` + generated API graph(s) + `manifest.yaml` |
| `tools/build_*.py` | Regenerate a workflow's API graphs from its editor JSON |

## Adding or upgrading a workflow

1. Save the new editor workflow as `workflows/<id>/<new_version>/source_ui.json`.
2. Copy the previous version's build script, adjust node ids if they changed, and run it. Or use `python -m harness.cli convert source_ui.json out.api.json` for a plain conversion.
3. Copy the previous `manifest.yaml`, then fix any node ids or titles that moved.
4. Click ↻ in the GUI (or run `python -m harness.cli check`). Versions with missing nodes, models or LoRAs show their problems and are skipped.
5. Set the preferred version in `settings.yaml`. Older versions stay usable for re-rendering old shots.

## Command line

```
python -m harness.cli status | check | snapshot [server]
python -m harness.cli smoke-image "prompt" --kind character
python -m harness.cli smoke-video "prompt" --mode t2va --seeds 2 --seconds 5
python -m pytest
```
