# Krea 2 Prompting Guide — Posing, Weighting & Styles

**Derived from:** `Krea2_00207_.json` (ComfyUI workflow, 20 nodes)
**Workflow author credited in-file:** X / @Zuanfilm (pose system)
**Cross-referenced against:** the official `krea-ai/krea-2` prompting docs, Krea's user docs, and the official ComfyUI Krea 2 tutorial.
**Written:** 2026-07-28

---

## 0. TL;DR

| If you want to… | Do this |
|---|---|
| Get a good image with no effort | Write one long, natural-language paragraph. Krea 2 was trained for that. |
| Get *deliberate* control | Use the workflow's labelled block template (§3) |
| Force the model's attention | `(keyword:1.1–3)` — see §4 |
| Remove/invert a concept without CFG | `(keyword:-1)` to `(keyword:-5)` — see §4.3 |
| Control the body precisely | Use the 12-field pose schema (§5) |
| Lock an aesthetic | Paste a full style paragraph from the 289-entry library (§6) |
| Render literal text | Put it in `"quotes"` — official recommendation |

---

## 1. What's actually in the JSON

`Krea2_00207_.json` is a saved **ComfyUI graph**, not a prompt file. Three of its 20 nodes are `Note` nodes containing the documentation this guide is built from:

| Node | Title | Size | Contents |
|---|---|---|---|
| 453 | *(untitled)* | 4.5 KB | Prompt-weighting rules + 7 worked examples |
| 459 | `poses` | 22.9 KB | Pose-descriptor schema, taxonomy, 30 fully-specified poses |
| 461 | `STYLES` | 192 KB | 289 ready-to-paste style paragraphs |

The remaining 17 nodes are the generation pipeline. **The Notes are the payload** — the guide below is essentially those three notes, verified and expanded.

### 1.1 The pipeline as saved

```
SetImageSizeWithScale (1600×1024, scale 1)
        └─> VAEEncodeAdvanced ──> empty_latent ─┐
                                                │
CLIPLoader (qwen3vl_4b_bf16, type: krea2) ──┐   │
UNETLoader (krea2_turbo_fp8)  ──────────────┤   │
                                            ▼   │
                             Krea2PromptWeight (prompt + strength 1)
                                  │ model        │ conditioning
                                  ▼              ├──> positive
                       ComfyUI-Krea2T-Enhancer   └──> negative   ← same tensor
                          (enabled, str 1)       │
                                  │              │
       BasicScheduler(beta, 8, 1) ─┴─> sigmas ─> ClownsharKSampler_Beta  #426
                                                        │ (8 of 9 steps, denoise 1.0)
                                                        ▼
       BasicScheduler(kl_optimal, 2, 0.3) ─> sigmas ─> ClownsharKSampler_Beta #427
                                                        │ (1 step, denoise 0.30 — refine)
                                                        ▼
                                   VAEDecode ×2 ─> ImageCompare ─> SaveImage ("Krea2")
       VAELoader: Wan2_1_VAE_fp32.safetensors
```

### 1.2 Exact settings in the file

| Node | Setting | Value |
|---|---|---|
| `SetImageSizeWithScale` | width × height × scale | **1600 × 1024 × 1** (≈1.56:1) |
| `UNETLoader` | model | `krea2_turbo_fp8.safetensors` |
| `CLIPLoader` | encoder / type | `qwen3vl_4b_bf16.safetensors` / **`krea2`** |
| `VAELoader` | vae | `Wan2_1_VAE_fp32.safetensors` |
| `VAEEncodeAdvanced` | latent channels / resample / fit | `16_channels` / `lanczos` / `fill / crop` |
| `Krea2PromptWeight` | strength | **1.0** |
| `Krea2T-Enhancer` | enabled / strength / advanced | `true` / **1.0** / `false` |
| Sampler #426 | eta, sampler, scheduler | 0.5, `linear/euler`, `beta` |
| Sampler #426 | steps / steps_to_run / denoise / cfg | 9 / **8** / 1.0 / **1.0** |
| Sampler #427 | eta, sampler, scheduler | 1.0, `linear/dormand-prince_6s`, `kl_optimal` |
| Sampler #427 | steps / steps_to_run / denoise / cfg | 9 / **1** / **0.30** / 1.0 |
| `SeedGenerator` | seed / mode | 224836896858758 / `randomize` |

**Two things worth knowing before you copy this workflow:**

1. **Positive and negative conditioning are the same tensor.** `Krea2PromptWeight`'s single `conditioning` output feeds both inputs of sampler #426. This is correct for Turbo — CFG is 1.0, so the negative branch is inert. It also means *there is no negative prompt*; all "negative" steering must come from negative weights (§4.3).
2. **Sampler #427's `model`, `positive` and `negative` inputs are unconnected in the saved graph** (only `latent_image`, `sigmas` and `seed` are wired). If the second refinement pass errors on load, wire the `Krea2T-Enhancer` model output and the `Krea2PromptWeight` conditioning into it exactly as on #426.

### 1.3 Where this deviates from the official ComfyUI setup

| | Official ComfyUI Krea 2 tutorial | This workflow |
|---|---|---|
| Diffusion model | `krea2_turbo_fp8_scaled.safetensors` | `krea2_turbo_fp8.safetensors` |
| Text encoder | `qwen3vl_4b_fp8_scaled.safetensors` | `qwen3vl_4b_bf16.safetensors` (higher precision, more VRAM) |
| VAE | `qwen_image_vae.safetensors` | `Wan2_1_VAE_fp32.safetensors` |
| Sampler | core KSampler, `euler` / `simple` | RES4LYF `ClownsharKSampler_Beta`, two-stage |
| Steps | 8 | 8 + 1 refinement |
| Conditioning | `CLIPTextEncode` | `Krea2PromptWeight` (KJNodes) |

The VAE swap is the notable one. Both are 16-channel latent VAEs, which is why `VAEEncodeAdvanced` is set to `16_channels` and why it runs at all — but the official pairing is the Qwen-Image VAE. If your colours or fine texture look subtly off versus reference images, swap the VAE back first; it's the cheapest variable to eliminate.

The Krea2T-Enhancer is a **third-party experimental node** (capitan01R) that patches the Krea 2 text-fusion stage at runtime to raise prompt adherence. It is not part of the official pipeline. Bypass it when you want a clean baseline — it changes how strongly all of the weighting below bites.

---

## 2. Official baseline: what Krea 2 actually wants

Everything in §3–§6 is a *technique layered on top of* the official model behaviour. Start from the official guidance, because it is short and it is the ground truth:

> "We recommend users to use natural language prompts to generate images. […] Long detailed prompts yield best results, but the model is capable of generating high quality images with minimal prompt engineering. For text rendering, we recommend putting quotes around the words to be rendered."
> — `krea-ai/krea-2`, `docs/prompting.md`

Key official facts:

- **Natural language, not tag soup.** Krea 2 is a single-stream DiT with **Qwen3-VL as its text encoder** — a language model, not CLIP or T5. It parses sentences. Comma-separated keyword lists still work but you leave capability on the table.
- **Long prompts are good.** Every official sample prompt is 30–150 words of continuous description. Several are full paragraphs.
- **Turbo:** 8 steps, **CFG 0.0/1.0 (disabled)**, timestep-shift (mu) 1.15, native **1024–2048 px**. Up to 2K resolution is supported and the official samples are all generated at 2K.
- **RAW/base:** 52 steps, CFG 3.5, up to 1024 px. Undistilled — this is the one to fine-tune or train LoRAs against. LoRAs trained on RAW transfer to Turbo.
- **Quotes for typography:** `a neon sign reading "OPEN LATE"`.
- Krea publishes an official **prompt-expansion system prompt** (`docs/expansion.txt`) you can hand to any LLM to turn a short idea into a Krea-2-shaped paragraph. Its rules: preserve all original subjects, actions, colours and spatial relations; do not invent new objects; use grounded phrasing for pose and spatial layout; respect the stated medium; output one cohesive paragraph.
- Official docs mention **no negative prompt and no weighting syntax**. Weighting is a community/ComfyUI capability (§4), not a model feature.

The hosted product adds Style References (up to 4 images with per-image strength), Moodboards, Krea Styles (LoRAs), a Raw→High **creativity** control, and three **Generative Sliders** (Intensity, Complexity, Movement, each −100…100). None of those exist in this local workflow — §4's weighting and §6's style paragraphs are the local substitute for them.

---

## 3. The workflow's prompt template

The prompt actually saved in node 441 shows the house structure. Blocks are separated by newlines and lightly labelled; the labels are read as ordinary language by Qwen3-VL, so they cost nothing and they keep *you* organised.

```
forced perspective
soft diffused lighting, cinematic light halation
subject: dynamic shot of a french brunette woman with a diamonds ornate royal
        crown and red medieval dress and diamond ornate shoes
detailed and intricate castle in the Black forest and sky filled with billowing
        clouds in the background
Expression: (smirking:3)(one brow raised:2) (looking front:3)
long hair blowing with the wind

Pose: Crossed Arms

leg_position: Stable standing stance with feet shoulder-width apart.
arm_position: Both forearms crossed firmly across the chest.
hand_position: Hands resting beneath the opposite upper arm.
torso_orientation: Upright with shoulders slightly rounded forward.
head_orientation: Chin level or slightly raised.
weight_distribution: Balanced or slightly shifted onto one leg.
center_of_gravity: Centered through the torso.
muscle_tension: Moderate in the shoulders and forearms.
movement_quality: Controlled and reserved.
balance: Stable.
silhouette: Compact upper-body silhouette emphasizing the crossed arms.

Style: luminous photographic aesthetics defined by strong backlighting, radiant
edge illumination, subtle translucency effects, atmospheric depth, graceful tonal
transitions, and a heightened sense of visual separation, creating elegant and
emotionally evocative imagery through carefully controlled exposure, naturalistic
light diffusion, and refined portrait craftsmanship, reminiscent of Peter
Lindbergh and Paolo Roversi, inspired by Vogue editorials
```

### 3.1 Canonical block order

```
1. CAMERA / FRAMING      forced perspective, full body shot, low angle, macro lens
2. LIGHTING              soft diffused lighting, cinematic light halation, rim light
3. subject:              who + wardrobe + material detail
4. ENVIRONMENT           background, architecture, weather, sky
5. Expression:           emotion, gaze direction, brow/mouth detail
6. SECONDARY MOTION      hair blowing, fabric billowing, dust, spray
   (blank line)
7. Pose: <name>          then the 12 descriptor fields (§5)
   (blank line)
8. Style: <paragraph>    one full style entry from §6
```

Why this order works: it front-loads the framing and light (which the DiT resolves early, at high sigma) and back-loads the style paragraph (surface treatment, resolved late). It also mirrors the official expansion rules — subject-with-attributes first, grounded spatial phrasing, medium named explicitly.

### 3.2 Rules of thumb

- **Keep blocks on their own lines.** Newlines are not semantically special to Qwen3-VL, but they stop you from writing a 400-word run-on where every clause dilutes every other clause.
- **One `Style:` paragraph, not three.** Style entries in §6 are already dense compound descriptions. Stacking two fights for the same latent budget and produces mush.
- **`subject:` should be the longest block.** If your style paragraph is longer than your subject description, the image will look great and be about nothing.
- **Don't negate in prose.** "no hat", "without a beard" will not reliably work — there is no CFG to push against. Use `(hat:-2)` instead (§4.3).
- **Text goes in quotes**, per official guidance, and belongs in the environment or subject block: `a weathered wooden sign reading "THE BLACK FOREST INN"`.

---

## 4. Prompt weighting

### 4.1 What it is, mechanically

Weighting is **not** a Krea 2 model feature. It comes from the `Krea2PromptWeight` node (Kijai / ComfyUI-KJNodes), which implements **per-token attention-value scaling** through the Qwen3-VL encoder — as opposed to the older SD-era approach of scaling the embedding vector after the fact.

That distinction matters in practice:

- It's per-token and semantically grounded, so multi-word phrases like `(face close to the lens:3)` behave coherently instead of smearing.
- It works **even though CFG is disabled**. Attention scaling doesn't need a guidance term, which is precisely why negative weights are usable on a Turbo model that has no negative prompt.
- The node outputs both a patched `MODEL` and the `CONDITIONING`. Both must be used — if you feed the conditioning to your sampler but not the patched model, weights silently do nothing.

An alternative implementation exists (`ComfyUI-krea2-negpip`, NegPiP-style negative weighting) if you prefer that route.

### 4.2 Syntax

```
(keyword:number)
```

Parentheses and a colon. Every token starts at an implicit **1.0**.

| Range | Effect | Notes |
|---|---|---|
| `1.1` – `3.0` | Emphasis, increasingly forceful | The workable band. `3.0` is already very strong. |
| `0.1` – `0.9` | De-emphasis without deletion | `(fog:0.4)` = fog stays, but subtle. |
| `> 3.0` (3–6) | **Breaks the renderer** | Artifacts, pixel distortion, extreme contrast, colour oversaturation. |
| `-0.1` – `-5.0` | Inversion — pushes toward the *opposite* | See §4.3. |

**Formatting traps:** a misplaced paren or colon is a parse error, not a silent no-op. Don't nest weights. Adjacent groups need no separator — `(smirking:3)(one brow raised:2)` is valid — but a space is easier to read.

### 4.3 Negative weights

This is the most useful part of the system, because the workflow has no negative prompt.

```
(face close to the lens:-3)   →  pushes the subject into the background
(face close to the lens:-5)   →  pushes it much further back
(dark aesthetics:-3)          →  produces a light, bright scene
```

Negative weighting is *inversion*, not deletion. `(anime:-2)` steers toward the photographic opposite of anime rather than merely removing anime cues — which, per the SwarmUI feature request that tracked this node, is exactly the point: "it allows you to use negative weighting to prompt away from things like anime even when CFG is not used."

Start at `-1`, escalate to `-3`. `-5` is the practical floor before it degrades.

### 4.4 The global `strength` knob

`Krea2PromptWeight` has a `strength` float (set to **1.0** in this workflow) that multiplies the influence of *every* weight in the prompt.

Use it as a master gain: write your prompt with weights tuned relative to each other, then dial `strength` down to `0.6–0.8` if the whole image is over-driven, or up if nothing is biting. Changing `strength` is much faster than re-tuning eight individual weights.

### 4.5 Weighting cookbook

All seven examples below are from Note 453, verbatim in intent. The base prompt is held constant so you can see exactly what each weight does.

**Base:**
```
forced perspective, face close to the lens
soft diffused lighting, cinematic light halation
subject: dynamic close-up shot of a french brunette woman with a diamonds ornate
royal crown and red medieval dress
detailed and intricate castle and sky filled with billowing clouds in the background
Expression: smiling and screaming
long hair blowing with the wind

style: luminous photographic aesthetics defined by strong backlighting, radiant
edge illumination
```

| Goal | Edit |
|---|---|
| **Move camera closer** | `forced perspective, (face close to the lens:3)` |
| **Intensify an emotion** | `Expression: smiling and (screaming:3)` |
| **Change composition** | append `(camera movement:3)(asymmetric composition:3)` |
| **Intensify lighting** | `soft diffused lighting, (cinematic light halation:3)` |
| **Intensify weather** | `…cinematic light halation, (raining:3)` + `(long hair blowing with the wind:2)` + `style: (luminous photographic aesthetics:2) …` |
| **Push subject back** | `forced perspective, (face close to the lens:-5)` |
| **Brighten the whole scene** | `style: (dark aesthetics:-3) defined by strong backlighting…` |

Note the last two combine in Note 453's own example: `(face close to the lens:-5)` **and** `(dark aesthetics:-3)` in one prompt → distant subject, light scene. Weights compose.

### 4.6 What's worth weighting

| Category | Examples that respond well |
|---|---|
| Camera distance | `face close to the lens`, `forced perspective`, `wide establishing shot` |
| Composition | `asymmetric composition`, `camera movement`, `rule of thirds`, `negative space` |
| Expression | `screaming`, `smirking`, `one brow raised`, `looking front`, `tearful` |
| Lighting | `cinematic light halation`, `rim light`, `hard shadows`, `volumetric god rays` |
| Weather / FX | `raining`, `fog`, `blowing dust`, `snow` |
| Motion | `long hair blowing with the wind`, `fabric billowing` |
| Style intensity | the leading phrase of your style paragraph, e.g. `(luminous photographic aesthetics:2)` |
| Pose emphasis | the pose name, e.g. `(Carrying a goat Over Shoulder:1.2)` |

**Weight sparingly.** Three to five weighted terms in a prompt is a lot. Every weight you add reduces the relative influence of everything unweighted, including your subject.

---

## 5. The posing system

This is the workflow's distinctive contribution, credited to **@Zuanfilm**.

### 5.1 The idea

Naming a pose (`"crossed arms"`) gives the model a fuzzy prior. Krea 2's Qwen3-VL encoder can parse structured, anatomically explicit description — so you specify the pose the way a life-drawing instructor or an animator would, across twelve independent axes. The model then has no room to improvise a generic stance.

### 5.2 The 12-field schema

Always in this order, one per line, each a complete sentence:

| Field | What it controls |
|---|---|
| `leg_position` | Foot placement, stance width, knee flexion |
| `arm_position` | Upper-arm and elbow geometry |
| `hand_position` | Finger detail, grip, contact points |
| `torso_orientation` | Spine, chest, shoulder line, rotation |
| `head_orientation` | Head tilt, chin level, gaze direction |
| `weight_distribution` | Which leg/limb bears load, and how much |
| `center_of_gravity` | Where mass sits relative to the support base |
| `muscle_tension` | Minimal → very high; drives visible anatomy |
| `movement_quality` | The adverbial character — "explosive", "elegant", "reserved" |
| `balance` | Static stability vs. dynamic gait |
| `silhouette` | The read at thumbnail size — the highest-leverage field |

(Eleven named fields; the schema is referred to as twelve-part because `Pose:` heads the block.)

**Usage pattern:**

```
Pose: Crossed Arms

leg_position: …
arm_position: …
…
silhouette: …
```

Or, to bias the whole block harder: `(Carrying a goat Over Shoulder:1.2)` followed by the descriptors.

### 5.3 Adapting a pose to your subject

Note 459's worked example takes stock pose **46 Carrying Over Shoulder** and substitutes the object:

> `(Carrying a goat Over Shoulder:1.2)`
> `arm_position: One arm supports **the goat** over the shoulder while the opposite arm balances naturally.`
> `hand_position: Firm grip securing **the goat**.`
> `silhouette: Asymmetrical silhouette emphasizing **the elevated goat**.`

Swap the noun in the two or three fields that touch it and leave the biomechanics untouched. The physics stay correct; only the referent changes.

### 5.4 Which fields matter most

If you're trimming for prompt budget, keep these four:

1. **`silhouette`** — determines whether the pose reads at all.
2. **`weight_distribution`** — the difference between a figure standing and a figure *pasted* standing.
3. **`torso_orientation`** — controls the whole upper-body read.
4. **`movement_quality`** — cheap adverbial steering that colours everything else.

Drop `center_of_gravity` and `balance` first; they're the most redundant with `weight_distribution`.

### 5.5 The taxonomy

Note 459 describes a **30-group** pose library. The workflow ships three of those groups:

| Group | Name | Poses |
|---|---|---|
| 1 of 30 | Fundamental Standing Poses | 1–10 |
| 2 of 30 | Locomotion & Transitional Poses | 11–20 |
| 5 of 30 | Object Interaction & Force Poses | 41–50 |

Groups 3, 4 and 6–30 (poses 21–40, 51–300) are **not present in this file.** Thirty poses are fully specified; the rest are implied by the numbering. When you need one that isn't here, write it yourself against the 12-field schema — that's the whole point of the schema being explicit.

### 5.6 The 30 shipped poses

Full descriptor blocks, verbatim from Note 459. Copy-paste directly.

<details>
<summary><b>Group 1 — Fundamental Standing Poses (1–10)</b></summary>

**1 Standing Neutral**
```
leg_position: Feet parallel at shoulder width, knees naturally extended with a slight relaxed bend.
arm_position: Arms resting comfortably alongside the torso with relaxed shoulders.
hand_position: Fingers naturally curved with palms facing inward toward the thighs.
torso_orientation: Fully upright with a neutral spine, open chest, and level shoulders.
head_orientation: Head centered above the spine with the chin parallel to the ground.
weight_distribution: Evenly distributed between both legs.
center_of_gravity: Centered between both feet.
muscle_tension: Minimal activation maintaining comfortable posture.
movement_quality: Still, relaxed, balanced, and natural.
balance: Highly stable with symmetrical support.
silhouette: Vertical, symmetrical, clean body outline.
```

**2 Contrapposto**
```
leg_position: One leg bears most of the body weight while the opposite leg relaxes with a slightly bent knee.
arm_position: Arms hang naturally with slight asymmetry following the body's weight shift.
hand_position: Relaxed fingers with one hand often slightly closer to the hip.
torso_orientation: Spine forms a gentle S-curve with hips and shoulders counterbalancing each other.
head_orientation: Head aligned naturally with a subtle relaxed tilt.
weight_distribution: Approximately 80% on the supporting leg.
center_of_gravity: Positioned above the supporting foot.
muscle_tension: Moderate in the supporting leg, minimal elsewhere.
movement_quality: Elegant, effortless, relaxed, and classical.
balance: Stable despite asymmetrical stance.
silhouette: Flowing S-shaped silhouette emphasizing natural body curves.
```

**3 Power Stance**
```
leg_position: Feet shoulder-width apart with knees extended but not locked.
arm_position: Arms relaxed beside the body or elbows bent outward when hands rest on the hips.
hand_position: Relaxed open hands or firmly placed on the hips.
torso_orientation: Chest expanded, shoulders back, spine elongated and upright.
head_orientation: Chin slightly elevated with the head centered.
weight_distribution: Evenly shared between both legs.
center_of_gravity: Low and centered between the feet.
muscle_tension: Moderate activation throughout the entire body.
movement_quality: Controlled, grounded, deliberate, and confident.
balance: Extremely stable.
silhouette: Broad triangular silhouette emphasizing shoulders and chest.
```

**4 Attention Stance**
```
leg_position: Feet together with knees fully extended and aligned.
arm_position: Arms straight alongside the torso without movement.
hand_position: Fingers fully extended and aligned with the thighs.
torso_orientation: Perfectly vertical with military posture.
head_orientation: Straight ahead with chin level.
weight_distribution: Equal across both feet.
center_of_gravity: Perfectly centered.
muscle_tension: High postural control throughout the body.
movement_quality: Completely still and disciplined.
balance: Maximum static stability.
silhouette: Rigid, vertical, symmetrical profile.
```

**5 Relaxed Standing**
```
leg_position: One knee slightly relaxed with weight shifted naturally onto one leg.
arm_position: Arms hanging loosely with relaxed shoulders.
hand_position: Loose fingers with natural curvature.
torso_orientation: Upright with slight natural relaxation of the spine.
head_orientation: Neutral with occasional subtle tilt.
weight_distribution: Primarily on one leg while the other remains relaxed.
center_of_gravity: Shifted toward the supporting leg.
muscle_tension: Low overall muscle engagement.
movement_quality: Casual, effortless, and comfortable.
balance: Stable with slight asymmetry.
silhouette: Soft asymmetrical outline.
```

**6 Hands in Pockets**
```
leg_position: Feet comfortably apart with one leg often carrying more weight.
arm_position: Elbows bent naturally with forearms inside pockets.
hand_position: One or both hands inserted into pockets.
torso_orientation: Upright with relaxed shoulders and chest.
head_orientation: Neutral or slightly tilted.
weight_distribution: Usually favors one leg.
center_of_gravity: Shifted toward the supporting hip.
muscle_tension: Very low.
movement_quality: Casual, relaxed, effortless.
balance: Stable.
silhouette: Relaxed silhouette with softened arm definition.
```

**7 Crossed Arms**
```
leg_position: Stable standing stance with feet shoulder-width apart.
arm_position: Both forearms crossed firmly across the chest.
hand_position: Hands resting beneath the opposite upper arm.
torso_orientation: Upright with shoulders slightly rounded forward.
head_orientation: Chin level or slightly raised.
weight_distribution: Balanced or slightly shifted onto one leg.
center_of_gravity: Centered through the torso.
muscle_tension: Moderate in the shoulders and forearms.
movement_quality: Controlled and reserved.
balance: Stable.
silhouette: Compact upper-body silhouette emphasizing the crossed arms.
```

**8 Hands on Hips**
```
leg_position: Feet shoulder-width apart with knees relaxed.
arm_position: Elbows bent outward symmetrically.
hand_position: Palms resting firmly on the hips.
torso_orientation: Chest expanded with shoulders open.
head_orientation: Facing forward with chin slightly elevated.
weight_distribution: Evenly balanced.
center_of_gravity: Centered over both feet.
muscle_tension: Moderate core engagement.
movement_quality: Assertive, controlled, and stable.
balance: Very stable.
silhouette: Wide silhouette emphasizing elbows and shoulders.
```

**9 One Hand on Hip**
```
leg_position: Weight primarily on one leg with the opposite hip slightly elevated.
arm_position: One elbow bent outward while the opposite arm hangs naturally.
hand_position: One hand resting on the hip, the other relaxed.
torso_orientation: Slight asymmetrical torso curve following the hip shift.
head_orientation: Slight tilt toward or away from the supporting side.
weight_distribution: Mostly on one leg.
center_of_gravity: Above the supporting foot.
muscle_tension: Moderate in the supporting leg and core.
movement_quality: Elegant, fashionable, relaxed.
balance: Stable with asymmetrical posture.
silhouette: Dynamic editorial silhouette emphasizing hip line.
```

**10 Arms Behind Back**
```
leg_position: Feet comfortably apart with knees relaxed.
arm_position: Both arms positioned behind the torso with elbows naturally bent.
hand_position: Hands clasped together or one hand holding the opposite wrist.
torso_orientation: Upright with an open chest and shoulders gently pulled back.
head_orientation: Neutral with calm forward gaze.
weight_distribution: Evenly balanced across both legs.
center_of_gravity: Centered over both feet.
muscle_tension: Low to moderate around the shoulders.
movement_quality: Calm, composed, deliberate.
balance: Stable.
silhouette: Clean frontal silhouette with concealed arm profile.
```
</details>

<details>
<summary><b>Group 2 — Locomotion &amp; Transitional Poses (11–20)</b></summary>

**11 Walking Forward**
```
leg_position: One leg stepping forward while the trailing leg pushes off the ground, knees moving naturally through the gait cycle.
arm_position: Arms swing naturally opposite to the legs with relaxed shoulders.
hand_position: Fingers loosely extended with relaxed natural curvature.
torso_orientation: Upright with a slight forward inclination following the direction of travel.
head_orientation: Head facing forward with chin level and eyes directed toward the destination.
weight_distribution: Constantly transferring from the rear leg to the leading leg.
center_of_gravity: Moving forward between the feet during each step.
muscle_tension: Moderate activation in the legs and core with relaxed upper body.
movement_quality: Smooth, rhythmic, purposeful, and continuous.
balance: Dynamic balance maintained throughout each stride.
silhouette: Strong forward-leading silhouette emphasizing motion.
```

**12 Walking Away**
```
leg_position: One leg advancing while viewed from behind, natural stride length maintained.
arm_position: Arms swing freely in opposition to the legs.
hand_position: Hands relaxed with fingers naturally curved.
torso_orientation: Upright with subtle forward momentum.
head_orientation: Facing away from the viewer with neck naturally aligned.
weight_distribution: Continuously shifting between each supporting foot.
center_of_gravity: Progressively moving forward with each stride.
muscle_tension: Moderate lower-body activation with relaxed shoulders.
movement_quality: Natural, fluid, and continuous.
balance: Stable dynamic gait.
silhouette: Tapered silhouette emphasizing distance and departure.
```

**13 Running Sprint**
```
leg_position: One leg drives forward while the opposite extends explosively behind the body.
arm_position: Elbows bent approximately ninety degrees with powerful forward and backward drive.
hand_position: Hands relaxed with lightly curled fingers.
torso_orientation: Leaning forward aggressively from the ankles with stable core engagement.
head_orientation: Facing directly toward the running direction with chin neutral.
weight_distribution: Supported by a single foot during each ground contact.
center_of_gravity: Positioned ahead of the supporting foot to maximize acceleration.
muscle_tension: Very high throughout the lower body and core.
movement_quality: Explosive, rapid, aggressive, and powerful.
balance: Dynamic athletic balance.
silhouette: Elongated forward-leaning silhouette emphasizing speed.
```

**14 Jogging**
```
leg_position: Moderate stride length with relaxed knee lift and soft foot placement.
arm_position: Elbows bent comfortably with relaxed rhythmic swing.
hand_position: Loose hands without gripping tension.
torso_orientation: Upright with slight forward inclination.
head_orientation: Neutral with steady forward gaze.
weight_distribution: Smooth transfer between each step.
center_of_gravity: Stable and continuously moving forward.
muscle_tension: Moderate lower-body activation with relaxed upper body.
movement_quality: Smooth, rhythmic, sustainable, and relaxed.
balance: Stable during continuous movement.
silhouette: Balanced athletic silhouette with flowing motion.
```

**15 Striding Confidently**
```
leg_position: Long deliberate steps with full extension of each stride.
arm_position: Natural arm swing synchronized with confident gait.
hand_position: Relaxed hands with controlled movement.
torso_orientation: Chest open with upright spine and shoulders back.
head_orientation: Chin slightly elevated with direct forward gaze.
weight_distribution: Firm transfer onto each leading foot.
center_of_gravity: Stable and confidently projected forward.
muscle_tension: Moderate overall body activation emphasizing posture.
movement_quality: Deliberate, elegant, powerful, and controlled.
balance: Highly stable while moving.
silhouette: Tall, commanding silhouette with extended stride.
```

**16 Crouching**
```
leg_position: Deep knee flexion with feet planted firmly beneath the body.
arm_position: Arms positioned naturally in front or beside the knees for balance.
hand_position: Hands open and ready for support if needed.
torso_orientation: Leaning slightly forward while maintaining spinal alignment.
head_orientation: Looking forward or slightly upward depending on the objective.
weight_distribution: Evenly shared across both feet.
center_of_gravity: Lowered close to the ground between the feet.
muscle_tension: Moderate to high activation in the legs and core.
movement_quality: Controlled, cautious, ready for movement.
balance: Stable low center of gravity.
silhouette: Compact low-profile silhouette.
```

**17 Deep Squat**
```
leg_position: Knees fully flexed with hips lowered below knee level and feet flat.
arm_position: Arms resting naturally or extended slightly forward for balance.
hand_position: Relaxed open hands.
torso_orientation: Slight forward inclination with neutral spine maintained.
head_orientation: Facing forward with relaxed neck.
weight_distribution: Evenly distributed through both feet.
center_of_gravity: Very low and centered.
muscle_tension: High lower-body activation with stable core.
movement_quality: Stable, grounded, controlled.
balance: Excellent low-position stability.
silhouette: Compact symmetrical silhouette close to the ground.
```

**18 Kneeling One Knee**
```
leg_position: One knee resting on the ground while the opposite foot remains planted with the knee bent.
arm_position: Arms relaxed beside the torso or resting naturally on the raised thigh.
hand_position: Relaxed open hands.
torso_orientation: Upright with open chest and straight spine.
head_orientation: Facing forward or slightly upward.
weight_distribution: Shared between the grounded knee and planted foot.
center_of_gravity: Low and centered between both support points.
muscle_tension: Moderate activation maintaining stability.
movement_quality: Calm, respectful, and controlled.
balance: Stable with three points of support.
silhouette: Strong asymmetrical silhouette emphasizing humility or readiness.
```

**19 Kneeling Both Knees**
```
leg_position: Both knees resting on the ground with lower legs aligned beneath the body.
arm_position: Arms relaxed beside the torso or resting on the thighs.
hand_position: Hands resting naturally on the thighs or joined together.
torso_orientation: Upright with relaxed shoulders and neutral spine.
head_orientation: Neutral or slightly lowered.
weight_distribution: Evenly distributed across both knees and lower legs.
center_of_gravity: Centered directly above the knees.
muscle_tension: Low to moderate postural activation.
movement_quality: Still, composed, and balanced.
balance: Highly stable.
silhouette: Symmetrical vertical silhouette supported by both knees.
```

**20 Sitting Upright**
```
leg_position: Both feet flat on the floor with knees bent approximately ninety degrees.
arm_position: Arms resting comfortably beside the torso or on the thighs.
hand_position: Hands relaxed on the lap or thighs.
torso_orientation: Straight spine with open chest and neutral shoulders.
head_orientation: Head centered with chin level.
weight_distribution: Evenly distributed across the sitting bones.
center_of_gravity: Centered over the pelvis.
muscle_tension: Low to moderate core activation maintaining posture.
movement_quality: Still, attentive, and composed.
balance: Stable seated posture.
silhouette: Clean vertical seated silhouette emphasizing posture.
```
</details>

<details>
<summary><b>Group 5 — Object Interaction &amp; Force Poses (41–50)</b></summary>

**41 Pointing**
```
leg_position: Feet shoulder-width apart with knees naturally relaxed.
arm_position: One arm fully extended toward the target while the opposite arm remains relaxed beside the body.
hand_position: Index finger extended with remaining fingers gently curled.
torso_orientation: Slightly rotated toward the pointing direction with an upright spine.
head_orientation: Eyes aligned with the pointing direction.
weight_distribution: Slightly shifted toward the leading side.
center_of_gravity: Centered with a subtle directional shift.
muscle_tension: Moderate activation through the extended arm and core.
movement_quality: Intentional, direct, and assertive.
balance: Stable.
silhouette: Strong directional silhouette emphasizing the extended arm.
```

**42 Reaching Forward**
```
leg_position: Feet staggered naturally with one foot slightly ahead.
arm_position: One or both arms extended toward an object beyond immediate reach.
hand_position: Fingers fully extended and slightly separated in preparation to grasp.
torso_orientation: Torso leaning forward from the hips while maintaining spinal alignment.
head_orientation: Eyes fixed on the target.
weight_distribution: Shifted toward the front foot.
center_of_gravity: Projected forward beyond the pelvis.
muscle_tension: Moderate activation through shoulders, core, and legs.
movement_quality: Purposeful, controlled, and exploratory.
balance: Stable with forward reach.
silhouette: Elongated forward-reaching silhouette.
```

**43 Reaching Upward**
```
leg_position: Feet shoulder-width apart with knees softly extended.
arm_position: One or both arms fully extended overhead.
hand_position: Fingers reaching upward with maximum extension.
torso_orientation: Spine elongated vertically with chest lifted.
head_orientation: Looking upward toward the reaching hand.
weight_distribution: Balanced over both feet.
center_of_gravity: Slightly elevated through full-body extension.
muscle_tension: Moderate activation throughout the body.
movement_quality: Aspirational, elongated, and graceful.
balance: Stable.
silhouette: Tall vertical silhouette emphasizing upward movement.
```

**44 Reaching Sideways**
```
leg_position: Feet wider than shoulder width for stability.
arm_position: One arm fully extended laterally while the opposite remains relaxed.
hand_position: Open hand with naturally extended fingers.
torso_orientation: Slight lateral rotation or side bend.
head_orientation: Eyes following the reaching hand.
weight_distribution: Shifted toward the supporting leg.
center_of_gravity: Laterally displaced while remaining balanced.
muscle_tension: Moderate activation through the side body and shoulders.
movement_quality: Fluid, exploratory, and expansive.
balance: Stable.
silhouette: Wide asymmetrical silhouette emphasizing horizontal reach.
```

**45 Holding Object**
```
leg_position: Stable stance with feet shoulder-width apart.
arm_position: Arms bent comfortably while supporting the object close to the torso.
hand_position: Hands securely wrapped around the object.
torso_orientation: Upright with slight stabilization from the core.
head_orientation: Neutral or focused on the object.
weight_distribution: Evenly balanced unless compensating for object weight.
center_of_gravity: Centered close to the body's midline.
muscle_tension: Moderate depending on object weight.
movement_quality: Controlled, careful, and steady.
balance: Stable.
silhouette: Compact silhouette centered around the object.
```

**46 Carrying Over Shoulder**
```
leg_position: Stable walking or standing stance.
arm_position: One arm supports the object over the shoulder while the opposite arm balances naturally.
hand_position: Firm grip securing the object.
torso_orientation: Slight lean opposite the carried load.
head_orientation: Neutral with forward gaze.
weight_distribution: Shifted slightly toward the supporting side.
center_of_gravity: Offset toward the load.
muscle_tension: Moderate to high in shoulders, back, and legs.
movement_quality: Strong, deliberate, and efficient.
balance: Stable despite asymmetrical load.
silhouette: Asymmetrical silhouette emphasizing the elevated object.
```

**47 Carrying in Arms**
```
leg_position: Stable stance with feet comfortably apart.
arm_position: Both arms wrapped securely around the carried object.
hand_position: Hands supporting the object's weight from below and the sides.
torso_orientation: Upright with slight backward compensation.
head_orientation: Looking forward or toward the carried subject.
weight_distribution: Evenly distributed with slight backward compensation.
center_of_gravity: Shifted toward the carried load.
muscle_tension: Moderate activation through arms, shoulders, and core.
movement_quality: Careful, protective, and controlled.
balance: Stable.
silhouette: Rounded silhouette centered around the carried object.
```

**48 Lifting**
```
leg_position: Feet shoulder-width apart with knees bent and hips lowered.
arm_position: Arms extended downward before initiating the lift.
hand_position: Secure grip around the object.
torso_orientation: Neutral spine with forward hip hinge.
head_orientation: Looking slightly forward to maintain spinal alignment.
weight_distribution: Even across both feet.
center_of_gravity: Low and centered beneath the load.
muscle_tension: High activation throughout the legs, core, and back.
movement_quality: Powerful, controlled, and coordinated.
balance: Stable lifting posture.
silhouette: Compact, grounded silhouette emphasizing strength.
```

**49 Pushing**
```
leg_position: One foot positioned forward with knees slightly bent.
arm_position: Both arms extended toward the object with elbows nearly straight.
hand_position: Palms pressing firmly against the object.
torso_orientation: Leaning forward from the ankles with strong core engagement.
head_orientation: Eyes directed toward the object being pushed.
weight_distribution: Primarily over the front foot.
center_of_gravity: Shifted forward beyond the pelvis.
muscle_tension: High activation through the entire body.
movement_quality: Forceful, deliberate, and sustained.
balance: Stable through staggered stance.
silhouette: Forward-driving silhouette emphasizing exertion.
```

**50 Pulling**
```
leg_position: One foot positioned behind the other for leverage.
arm_position: Elbows bent while drawing the object toward the torso.
hand_position: Firm gripping position.
torso_orientation: Slight backward lean with engaged core.
head_orientation: Looking toward the object being pulled.
weight_distribution: Shifted toward the rear supporting leg.
center_of_gravity: Positioned behind the object to maximize leverage.
muscle_tension: High activation in arms, back, and legs.
movement_quality: Powerful, controlled, and resistant.
balance: Stable through rear-weighted stance.
silhouette: Dynamic backward-leaning silhouette emphasizing pulling force.
```
</details>

### 5.7 Writing your own pose

Fill all 11 fields. Write each as a full declarative sentence. Two conventions from the shipped set worth copying:

- **Describe state, not instruction.** "Feet parallel at shoulder width" — not "place the feet at shoulder width."
- **Quantify when you can.** "Approximately 80% on the supporting leg", "elbows bent approximately ninety degrees", "hips lowered below knee level."
- **`silhouette` gets an adjective + a focal noun.** "Broad triangular silhouette emphasizing shoulders and chest." That construction is what makes the field work.

---

## 6. The style library (289 entries)

### 6.1 What's in there

Node 461 holds **289 style paragraphs**, one per line, `Name: description`. 285 unique names — `Comic Européen Prestige`, `Bande Dessinée (Hergé/Goscinny Tradition)` and `Beatrix Potter Style` each appear twice with identical text, and `Film Noir` appears twice with two *different* descriptions (pick whichever reads better for your shot).

Each entry is 60–150 words. They are designed to be pasted whole into the `Style:` block — the length is the point. This is exactly the "long detailed prompts yield best results" behaviour the official docs describe, pre-authored.

### 6.2 The two authoring formulas

Every entry follows one of two templates. Learn them and you can write your own.

**Formula A — "built on" (the majority):**

> A `<3 mood adjectives>` `<medium>` style built on `<linework / mark-making>`, `<shading / tonal treatment>`, `<proportion / form logic>`, `<atmosphere or emotional register>`, and `<a summarising "X intensity/charm/gravitas that feels Y">`. **Influenced by:** `<tradition>`, `<technique>`, `<studio or school practice>`, `<method>`.

**Formula B — "characterized by" (used for the artist-anchored entries):**

> `<medium>` aesthetics characterized by `<trait>`, `<trait>`, `<trait>`, `<trait>`, and `<trait>`, blending `<A>` with `<B>` through `<mechanism>`, `<mechanism>`, and `<mechanism>`, **reminiscent of** `<Artist 1>` and `<Artist 2>`, **inspired by** `<Work 1>` and `<Work 2>`.

Both end with attribution clauses. That trailing "Influenced by / reminiscent of / inspired by" list is doing real work — Qwen3-VL knows these names and works, and the clause anchors the abstract adjectives to something concrete.

### 6.3 Four entries in full

**`Backlit Portraiture`** (the one used in the shipped prompt):
> luminous photographic aesthetics defined by strong backlighting, radiant edge illumination, subtle translucency effects, atmospheric depth, graceful tonal transitions, and a heightened sense of visual separation, creating elegant and emotionally evocative imagery through carefully controlled exposure, naturalistic light diffusion, and refined portrait craftsmanship, reminiscent of Peter Lindbergh and Paolo Roversi, inspired by Vogue editorials and In the Mood for Love.

**`Cinematic Photography`:**
> A dramatically composed photographic style built on deliberate, film-like framing that favors wide aspect ratios and purposeful negative space, controlled color-grading logic that unifies highlight and shadow into a single cohesive mood, soft, motivated lighting sources that feel narratively justified rather than purely technical, a patient, unhurried compositional stillness charged with implied story beyond the frame, and a polished, narrative-driven visual gravitas rooted in feature-film visual-language tradition. Influenced by: Roger Deakins cinematography tradition, anamorphic-lens visual technique, narrative film-lighting practice, color-grading post-production method.

**`Ghibli Style`:**
> A gentle, hand-painted animation illustration style built on soft watercolor-inspired shading that gives every plane a breathable, tender texture, meticulous naturalistic detail applied with unhurried patience, rounded, expressive proportions carrying quiet emotional sincerity, a hushed, wondrous stillness pervading even the most fantastical scenes, and a nostalgic, wholesome hand-crafted intensity that feels timeless and deeply human. Influenced by: Hayao Miyazaki, Studio Ghibli production technique, traditional cel-animation painting method, naturalist watercolor illustration tradition.

**`Golden Hour Photography`:**
> A softly directional photographic style built on a low, raking angle of natural illumination that stretches long, gentle shadows across the frame, a diffused, glowing quality softened by increased atmospheric scattering at that time of day, a flattering, low-contrast luminosity that wraps gently around form rather than casting harsh definition, an unhurried, fleeting stillness carried through the brief window of ideal light, and a nostalgic, cinematic tenderness that feels naturally soft and effortlessly atmospheric. Influenced by: natural available-light photography tradition, magic-hour cinematography technique, outdoor portrait-lighting practice, documentary golden-hour shooting method.

### 6.4 Full index

Nine implicit families. The first entry of each family is a **generic base style** — `Photography`, `Painting`, `Drawing`, `3D Render` etc. — usable on its own when you want the medium without a specific voice.

<details>
<summary><b>Anime &amp; Manga (1–42)</b></summary>

Anime Style · Ghibli Style · 70s Anime · Manga Style · Akira Style · Akira Toriyama Style · Atsushi Ohkubo Style · Attack on Titan Style · Chainsaw Man Style · Jujutsu Kaisen Style · My Hero Academia Style · Berserk Manga Style · Bleach Style · Frieren Style · Boichi Style · Cinematic Anime · Classic Shojo Illustration · Euromanga Style · Watercolor Anime Illustration · Granblue Fantasy Pin-Up Style · Katsuya Terada Style · Manhwa Ultradetailed Art · Murata Hyperrealism · MidJourney Niji-Inspired Digital Illustration · Shindol Style · Solo Leveling Art · Pony 2.5D Realism · Pony/Illustrious Checkpoint Style · Yoshitaka Amano Style · Yoshitaka Amano Fine Art · Death Note Style · JoJo's Bizarre Adventure Style · One Piece Style · Sailor Moon Style · Chibi Anime · Anime Key Visual · Junji Ito Style · Anime Illustration · Ultradetailed Manga Illustration · Polished Ultrarealistic Anime Pin-Up · Thin Delicate Outline Romantic Anime-Photographic Hybrid · Cel Shading
</details>

<details>
<summary><b>Cartoon &amp; Western Animation (43–60)</b></summary>

Cartoon Style · Invader Zim Style · Nickelodeon Animation Style · Rick and Morty Style · Simpsons Style · South Park Style · Steven Universe Style · Adventure Time Style · Hanna-Barbera Style · Family Guy Style · Modern Cartoon · Rubber Hose Cartoon · Disney Ultradetailed Illustration · Classic Disney Feature Animation · Cyberpunk · Bruce Timm Noir · Arcane Animation Still · Toon Shader
</details>

<details>
<summary><b>Comics &amp; Graphic Novel (61–93)</b></summary>

Comics Style · Alex Ross Style · Alternative Comics Style · Amanda Conner Illustration · American Comic Realism · Comic Européen Prestige · American Pulp Illustration · Frank Miller Style · Bengal Illustration · Comic Européen Prestige *(dup)* · Crosshatched Graphic Novel · Graphic Novel · Franco-Belgian Ultradetailed Comics · Bande Dessinée (Hergé/Goscinny Tradition) · Frank Cho Style · Enki Bilal-Inspired Comic and Digital Painting · Hellboy Style · Archie Comics Style · Bande Dessinée (Hergé/Goscinny Tradition) *(dup)* · Mark Brooks / Manapul Style · Marvel Comics Style · Marvel Cover Art · Milo Manara Style · Psychological Thriller Pulp Digital Painting · DC Comics Style · Zenescope Illustration · Ryan Benjamin Style · Travis Charest Style · Adi Granov Style · Bold Contour Style · Clean Line Art · Moebius-Inspired Illustration · Vintage Comics
</details>

<details>
<summary><b>Drawing, Sketch &amp; Ink (94–122)</b></summary>

Drawing · Caricature · Charcoal Rendering · Hyperrealistic Colored-Pen Sketch · Jarek Kubicki Style · Playful Hand-Drawn Illustration · Chinese Ink Drawing · Sketch · Beatrix Potter Style · Children's Book Drawing · Vintage Children's Book Drawing · Gris Grimly Style · Tim Burton Style · Fashion Sketch · Loose Crosshatched Character Sketch · Wojak/Meme-Sketch Style · Ink Pen Drawing · Etching · Technical Drawing · Conté Pastel · Crayola Colored-Book Style · Bookplate Illustration · Boulet Graphic Illustration · Collector Storybook Illustration · Saul Steinbeck Art · Whimsical Illustration · Doodle · Very Simplistic Doodle · Beatrix Potter Style *(dup)*
</details>

<details>
<summary><b>Photography (123–176)</b></summary>

Photography · 35mm Photography · Adobe Lightroom Masterpiece · Analog Photography · Annie Leibovitz Portraiture · Atmospheric Matte Painting · Backlit Portraiture · Black And White Documentary · Bleach Bypass Photography · Blue Hour Cinema · Butterfly Lighting · Candid Photography · Cinematic Photography · Commercial Beauty Photography · Editorial Photography · Cross Process Photography · Cross-Processed Film Photography · Cyberpunk Photography · Fish-Eye Photography · Surveillance Camera Photo · Direct-Flash Photography · Dutch Angle Photography · Full Body Photo · GQ Photoshoot · HDR Photography · High-Key Lighting Photography · Hitchcockian Photo · Film Noir · Film Noir *(2nd variant)* · Instagram Model Glamour Photography · Art Photography · Kodachrome · National Geographic Photography · Motion Blur Photography · Natural Lighting Photography · Neon Photography · Low-Key Photography · Night Photography · Peter Lindbergh–inspired photography style · Photoshoot · Playboy / Maxim-Style Photoshoot · Hyper Synthwave · Selfie Photo · Sin City Photography · Soft Beauty-Dish Portrait Rendering · Streamer Glam Selfie Realism · Supermodel Photoshoot · LinkedIn Pro Portrait · Sitcom Still · Rembrandt-Inspired Photography · Stranger Things-Inspired Art · Vogue Magazine · Golden Hour Photography · Cinematic Tron
</details>

<details>
<summary><b>Graphic Design, Flat &amp; Vector (177–195)</b></summary>

Design · Butcher Billy Style · Corporate Memphis · 8-Bit Style · Kintsugi-Inspired Digital Art · Soviet Propaganda Poster · Minecraft Style · Funko Pop Style · Hyper-Stylized Illustration · Graphic Flat-Vector Travel Poster Illustration · Ornamental Patterns Hyperrealism Portrait · Flat Illustration · Vector Art · Bauhaus · Poster Art · Minimalist Line Art · Minimalism · Gradients · Origami
</details>

<details>
<summary><b>3D, Craft &amp; Physical Media (196–218)</b></summary>

3D Render · Pixar Animation · Cinematic Final Fantasy 3D Render · Marble Sculpture · Playful Retro Pop · Low Poly · LEGO Minifigure · Paper Cutout · Collage · Claymation · Mixed Media · Wallace and Gromit Style · Muppets Style · Zelda Wind Waker Style · PS1 Graphics · Woodcut · Contemporary Lithography · Variable-Size Ben-Day Dots Illustration · Contemporary Ukiyo-E · Mosaic Art · Stained Glass Flat Art · Isometric 3D Graphic · Quilling
</details>

<details>
<summary><b>Digital Painting &amp; Concept Art (219–252)</b></summary>

Digital Painting · Castlevania Concept Art · Sakimichan Style · Scribble Fine Art · Gris Grimly-Inspired Hyperrealism · Absolute Painterly Style · Artgerm Style · ArtStation Masterpiece · Bastien Lecouffe-Deharme Fantasy · Bernie Wrightson Gothic · Blacklight Style · X-Ray Style · Brassaï Night Photography · Character Design Illustration · Charlie Bowater Style · Contemporary Fantasy Art · Cosmic Horror Illustration · Contemporary Commercial Illustration · Editorial Illustration · Cover Art · Interplay of Shadow and Light · Kekai Kotaki Style · Robin Eley Style · Dark Fantasy Illustration · GTA V Polished Cover Art · Expressive Character Illustration · Splatterpunk Horror Illustration · James Jean Style · Victo Ngai Style · Kay Nielsen Style · Hyper-Stylized Digital Painting · Movie Digital Art · RossDraws Style · Loish Ultrarealism
</details>

<details>
<summary><b>Painting &amp; Fine Art (253–289)</b></summary>

Painting · Abstract Expressionism · Academic Precision Realism · Acrylic Dreamscape · Adventure Illustration · Agnes Cecile Style · Airbrush Fantasy · Van Gogh Style · Monet Style · Wet-on-Wet Watercolor Ink Painting · Alphonse Mucha Masterwork · Anato Finnstark Style · Rembrandt Painting · Art Deco Gold Highlights · Ashley Wood Illustration · Tenebrism · Chiaroscuro · Bill Sienkiewicz Expressionism · Graffiti Painting · Boris Vallejo Style · Brushwork Emphasis · Pin-Up · 2020s Pin-Up · Hyperrealism · Ultrarealism · Casey Baugh Style · Commercial Illustration · Giallo Poster Art · Grindhouse Poster Illustration · Cubism · Carne Griffiths Style · Jackson Pollock Style · Watercolor Painting · Contemporary Oil Painting · Pop Art · Thick Brushstrokes Painting · Impasto
</details>

### 6.5 Pulling an entry out of the JSON

The Note node is not searchable in the ComfyUI UI once it's long. Extract to a flat file once:

```bash
python - <<'PY'
import json, re
d = json.load(open(r"C:\Users\ac\Documents\Krea2_00207_.json", encoding="utf-8"))
styles = next(n for n in d["nodes"] if n["id"] == 461)["widgets_values"][0]
with open(r"C:\Users\ac\Documents\krea2_styles.txt", "w", encoding="utf-8") as f:
    for line in (l.strip() for l in styles.split("\n")):
        if line:
            f.write(line + "\n\n")
PY
```

Swap `461` for `459` to get the poses and `453` for the weighting notes.

### 6.6 Using styles well

- **One per prompt.** Two style paragraphs = ~200 words of competing surface description.
- **Match the style family to the subject.** A pose block written in anatomical prose plus `8-Bit Style` will fight itself; the pose detail has nowhere to go at that resolution.
- **Weight the style's opening phrase, not the whole paragraph.** `(luminous photographic aesthetics:2)` works. Weighting a 120-word block does not.
- **Trim before you add.** If an entry mentions an element you don't want (e.g. `Backlit Portraiture`'s "subtle translucency effects"), delete that clause rather than negating it.
- **Artist names are load-bearing.** The "reminiscent of / inspired by" clause is often the strongest signal in the paragraph. If a style isn't landing, that clause is the first thing to strengthen.

---

## 7. Worked recipes

### 7.1 Editorial portrait, subject dominant

```
extreme close-up, (face close to the lens:2.5)
hard directional key light from camera left, deep falloff
subject: a woman in her thirties, sharp jawline, close-cropped platinum hair,
matte black high-neck knit, single gold earring
solid deep crimson seamless background
Expression: (neutral:2) (direct eye contact with the lens:3)

Pose: Crossed Arms

arm_position: Both forearms crossed firmly across the chest.
hand_position: Hands resting beneath the opposite upper arm.
torso_orientation: Upright with shoulders slightly rounded forward.
head_orientation: Chin level or slightly raised.
muscle_tension: Moderate in the shoulders and forearms.
movement_quality: Controlled and reserved.
silhouette: Compact upper-body silhouette emphasizing the crossed arms.

Style: A considered, narrative-driven photographic style built on deliberate,
purposeful lighting calibrated to support a story rather than purely flatter,
confident, composed framing designed for immediate publication-ready clarity, a
polished, magazine-grade retouching finish that idealizes without erasing
character…
```

### 7.2 Wide environmental shot — inverting the close-up bias

```
wide establishing shot, (face close to the lens:-4)
overcast diffused daylight, (fog:0.5)
subject: a lone figure in a long charcoal coat, small in frame
vast wet salt flat stretching to a low horizon, distant mountain ridge
(dark aesthetics:-2)

Pose: Walking Away

leg_position: One leg advancing while viewed from behind, natural stride length maintained.
torso_orientation: Upright with subtle forward momentum.
head_orientation: Facing away from the viewer with neck naturally aligned.
weight_distribution: Continuously shifting between each supporting foot.
movement_quality: Natural, fluid, and continuous.
silhouette: Tapered silhouette emphasizing distance and departure.

Style: <Golden Hour Photography, full paragraph>
```

`(face close to the lens:-4)` does the work that "wide shot" alone often won't, and `(dark aesthetics:-2)` lifts the whole exposure without a negative prompt.

### 7.3 High-action, physically grounded

```
low angle, dutch tilt, (camera movement:2.5) (asymmetric composition:2)
harsh backlight, (volumetric dust:2)
subject: a courier in worn leather, satchel strap across the chest
narrow rain-slick alley, neon signage reading "NIGHT MARKET"
Expression: (determined:2)

Pose: (Running Sprint:1.3)

leg_position: One leg drives forward while the opposite extends explosively behind the body.
arm_position: Elbows bent approximately ninety degrees with powerful forward and backward drive.
torso_orientation: Leaning forward aggressively from the ankles with stable core engagement.
weight_distribution: Supported by a single foot during each ground contact.
center_of_gravity: Positioned ahead of the supporting foot to maximize acceleration.
muscle_tension: Very high throughout the lower body and core.
movement_quality: Explosive, rapid, aggressive, and powerful.
silhouette: Elongated forward-leaning silhouette emphasizing speed.

Style: <Cinematic Photography, full paragraph>
```

Note the quoted signage — official guidance for text rendering.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Weights do nothing | Patched `MODEL` output of `Krea2PromptWeight` not wired to the sampler | Wire both outputs; the conditioning alone is not enough |
| Weights do nothing (part 2) | Global `strength` too low | Raise `Krea2PromptWeight.strength` toward 1.0+ |
| Artifacts, blown contrast, oversaturation | A weight above 3.0 | Cap emphasis at 3.0; use 1.5–2.5 as the working band |
| Prompt parse error | Stray/nested paren or missing colon | `(word:1.5)` — one level, colon required |
| Pose ignored | Descriptor block buried mid-prompt, or contradicted by the camera line | Put the pose block after the subject with blank lines around it; check `full body shot` vs. `close-up` |
| Everything is beige and generic | Style paragraph longer than the subject block | Expand the subject; don't add more style |
| "no X" in the prompt does nothing | No CFG, no negative prompt | Use `(X:-2)` |
| Colours/texture subtly off vs. reference | Non-standard VAE | Try `qwen_image_vae.safetensors` |
| Second sampler pass errors | #427's `model`/`positive`/`negative` unconnected in the saved graph | Rewire from the enhancer + prompt-weight nodes |
| Soft, mushy detail | Refinement denoise too high | #427 is at 0.30 — 0.20–0.35 is the useful range; above ~0.45 it starts reinventing |

---

## 9. Quick reference card

```
WEIGHTS          (kw:1.1–3)  emphasize      (kw:0.1–0.9) soften
                 (kw:-1..-5) invert         >3.0 breaks the renderer
                 global multiplier = Krea2PromptWeight.strength

PROMPT ORDER     camera → lighting → subject: → environment → Expression:
                 → secondary motion → Pose: + 11 descriptors → Style:

POSE FIELDS      leg_position, arm_position, hand_position, torso_orientation,
                 head_orientation, weight_distribution, center_of_gravity,
                 muscle_tension, movement_quality, balance, silhouette

OFFICIAL         natural language · long prompts win · "quotes" for text
TURBO            8 steps · CFG 0/1 · mu 1.15 · 1024–2048px
RAW              52 steps · CFG 3.5 · ≤1024px · train LoRAs here

THIS WORKFLOW    1600×1024 · 8+1 steps · CFG 1 · beta → kl_optimal
                 pos == neg conditioning (no negative prompt available)
```

---

## 10. Sources

**Official**
- [krea-ai/krea-2 — docs/prompting.md](https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md) — natural language, long prompts, quoted text, 2K turbo
- [krea-ai/krea-2 — docs/expansion.txt](https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt) — official prompt-expansion system prompt
- [krea-ai/krea-2 — README](https://github.com/krea-ai/krea-2) — RAW 52 steps/CFG 3.5, Turbo 8 steps/CFG 0/mu 1.15
- [Krea docs — Krea 2](https://www.krea.ai/docs/user-guide/features/krea-2) — variants, Srefs, moodboards, creativity, Generative Sliders
- [ComfyUI docs — Krea 2 workflow](https://docs.comfy.org/tutorials/image/krea/krea-2) — official model files, 1K–2K, 8-step default

**Nodes used in this workflow**
- [Krea2PromptWeight (ComfyUI-KJNodes) documentation](https://comfyai.run/documentation/Krea2PromptWeight) — inputs, strength, attention-value scaling
- [SwarmUI issue #1441 — Krea 2 Prompt Weighting](https://github.com/mcmonkeyprojects/SwarmUI/issues/1441) — attention-value vs. embedding-level scaling; negative weighting without CFG
- [capitan01R/ComfyUI-Krea2T-Enhancer](https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer) — text-fusion patch, strength 0.0–2.0, text_scale 1.5–2.0
- [ClownsharkBatwing/RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) — ClownsharKSampler_Beta, eta/steps_to_run/bongmath
- [blue-pen5805/ComfyUI-krea2-negpip](https://github.com/blue-pen5805/ComfyUI-krea2-negpip) — alternative NegPiP-style negative weighting

**In-file**
- `Krea2_00207_.json` nodes 453 (weighting), 459 (poses, credited to X/@Zuanfilm), 461 (289 styles)
