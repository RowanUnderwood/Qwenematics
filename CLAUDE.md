# Qwenematics

A FastAPI + Svelte GUI in which Qwen 3.8 (served by Ninfer) directs two ComfyUI workflows:
- Krea 2 makes reference images.
- MiniMax H3 SeedHunter turns them into clips of up to 15 s with audio.

Projects hold a story bible, shots, references and renders, and end in one final cut. `README.md` covers usage, the folder layout and adding workflows. `notes.txt` is the user's original spec. This file covers what you need when changing the code.

## Machine and services (Windows 11, three GPUs)

All launchers set `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so device 0 = 5090, 1 = 4090, 2 = 3090 (matching nvidia-smi).

| Service | Port | GPU | Started by |
|---|---|---|---|
| Qwenematics (this app) | 8765 | - | `start_qwenematics.bat` / `Qwenematics.lnk` (or `.venv\Scripts\python -m harness.server`) |
| ComfyUI image (Krea) | 8190 | 3090 | `launchers/comfy_image_3090.bat` |
| ComfyUI video (MiniMax) | 8189 | 5090 | `launchers/comfy_video_5090.bat` |
| Ninfer (Qwen) | 1235 | 4090 | `H:/Ninfer Qwen/launch-ninfer-4090.bat`; stop with `stop-ninfer-4090.bat` |
| Stability Matrix ComfyUI | 8188 | 5090 | The user's own. **Never stop it or write to it.** |

- **No `--use-sage-attention` on the 3090 instance.** It renders black Krea images there; it's fine on the 5090.
- **Both harness ComfyUI instances share one ComfyUI install** (`settings.yaml` → `comfy_dir`), with separate user directories.
- **Settings tab:** the GUI can launch every service from there.
- **The ⏻ Shut down button** (`POST /api/shutdown`) stops jobs cleanly, can optionally close the ComfyUI windows and Ninfer, and frees the port.

## Commands

```
.venv\Scripts\python -m pytest                      # offline tests, no GPU/servers needed
cd frontend && npm run build                        # the UI is served from frontend/dist; rebuild after any .svelte/.js change
.venv\Scripts\python -m harness.cli check           # validate workflow versions against the running ComfyUI
.venv\Scripts\python -m harness.cli smoke-image "prompt" --kind character
.venv\Scripts\python -m harness.cli smoke-video "prompt" --mode t2va --seeds 2 --seconds 5
python tools/build_minimax_v251.py                  # regenerate API graphs from source_ui.json
```

**Restarting the server to load backend changes:**
1. Check `/api/jobs` for running jobs first, and don't kill renders the user started.
2. `POST /api/shutdown` with `{"comfy":false,"llm":false}`.
3. Start it again with `.venv\Scripts\python -m harness.server`. Logs go to `projects/_server.log` and `_server.err.log` when started hidden.

Starting a second copy is safe: it detects the running one via `/api/ping` and exits.

Prompt files and `prompts/knowledge/*` are cached (`lru_cache`), so editing them also needs a restart.

## Architecture in one breath

`server.py` (routes, websocket `/ws`) → `core.Harness` (domain logic, jobs) → `comfy/` (manifest registry, runner, client) + `llm/` (Ninfer client, `agent.Director` tool loop, `roles` for the writers, critic and captioner, `tools` for the director's tool schemas and handlers) + `db/` (one SQLite DB per project, `deps` for staleness, impact and delete plans).

- **Job queue:** lanes `image` (several jobs at once), `video` and `local` (one each). Each job runs as its own task, so cancelling also stops a job that is only waiting.
- **Events:** `events.bus` → websocket → `frontend/src/lib/state.svelte.js` `wire()`.

## Conventions and gotchas

- **Workflows are data.** Each `workflows/<id>/<version>/` holds `manifest.yaml` plus generated `*.api.json`.
  - Code addresses semantic params and modes from the manifest, never raw node ids.
  - Never edit a released version in place. Add a new version folder; old renders record `workflow@version` and must stay re-renderable.
  - The MiniMax editor JSON can't be exported from the browser. `comfy/ui_to_api.py` converts it, and the `tools/build_*.py` scripts patch the result.
- **MiniMax rules:**
  - Clip length must be 17n+5 frames at 24 fps (15 s = 362 frames).
  - At most 9 pictures, referenced in prompts as `<Picture k>` in slot order. `core._repack` puts first and last frames first.
  - Stage 1 saves the video and audio latents separately (they're NestedTensors). `LoadLatent` only lists `.latent` files at the top level of ComfyUI's `input/`.
- **ComfyUI's folders are shared.** Every file the harness makes ComfyUI write gets copied into the project and deleted from ComfyUI.
  - `VHS_VideoCombine` writes three files (`x.mp4`, `x-audio.mp4`, `x.png`) but reports one. `runner._comfy_paths` handles this.
  - Only ever delete files the harness created.
- **Projects:** `projects/<name>/project.db` plus `media/`. Track every file in `FileRec`.
  - Deletes go through `db/deps.py` plans; they delete rows first, then files.
  - Reference versions are immutable: edits make a new version, and shots go stale (via `input_hash`) instead of changing.
- **File ids restart at 1 in every project.** Frontend media URLs go through `fileSrc` / `thumbSrc` in `state.svelte.js`, which add the project. Use those, never `/api/files/<id>` directly.
- **Event bus:** `bus.publish(event, **data)` rejects a `kind` key, because the websocket message uses `kind` for the event name. Use `job_kind=`, `stream=` and so on.
- **Ninfer calls:**
  - `reasoning_effort` is none, low, medium or xhigh.
  - Reasoning tokens count against `max_tokens`, so keep generous limits on writer and critic calls or outputs get cut off.
  - Video goes as a `video_url` data URI of a small proxy (`llm/media.py`).
- **Krea styles:**
  - `prompts/knowledge/krea_styles.json` holds 288 tested paragraphs from the Krea 2 Style Library.
  - A project can lock one (`krea_style` project setting, picked in the Bible tab). `styles.enforce_style` then rewrites every Krea prompt's `Style:` line.
  - Sample images are cached in `cache/krea_styles/`.
- **GPU pool (`harness/jobs/gpu_pool.py`):**
  - Krea renders go through `pool.image_slot()`, which yields the server name for `runner.render(server=...)`. The 5090 is borrowed while video is idle.
  - `_shot_job` wraps its whole MiniMax section in `pool.video_slot()` (video first; Krea is emptied out first).
  - Never call `runner.render` for Krea outside the pool, or both models can land on the 5090 (21 + 28.5 GB > 32 GB).
  - Use `wv.problems_on(oi)` to check a workflow against another server; `check_against_server` overwrites `wv.problems`.
  - The image lane runs `image_jobs_in_flight` jobs at once, so job code must be safe to run concurrently.
- **Director context:**
  - `Director.project_state()` regenerates the compact state each turn.
  - Old chat media is never resent.
  - Chat rows with role `event` cost no context.
  - Keep new director-facing text short.
- **Frontend:** Svelte 5 runes. One shared store `app` in `state.svelte.js`. Wrap API calls in `guard()`.
- **Python:** 3.13 venv in `.venv`. Match the surrounding style: compact, type-hinted, comments only where the reason isn't obvious.
- **Generated or local, not source:** `projects/`, `cache/`, `frontend/dist/`. The top-level `*.json` workflows are the user's originals (copies are in `source_workflows/`), so don't modify them.

## Testing live

A real end-to-end test needs all three services up.
- Use a throwaway project, then delete it through `DELETE /api/project` and reopen the project the user had open (`/api/status` shows it).
- Never test in or modify the user's own projects.
- Rough timings: a Krea image takes about 35 s on the 3090. A 15 s MiniMax preview takes about 2.5 min per seed on the 5090.
