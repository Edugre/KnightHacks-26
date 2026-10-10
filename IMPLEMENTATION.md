# Censor Agent: Implementation Plan

Companion to [PLAN.md](PLAN.md). PLAN.md covers what we're building and why. This file covers how: module boundaries, interfaces, milestones with exit criteria, constraints, and the conditions that tell us we succeeded.

Coding agents: read this file first. The interfaces in sections 3 and 4 are the contracts. Don't change them without bumping the schema version and telling the team.

---

## 1. Constraints

These are fixed. Design inside them.

| Constraint | What it means for the build |
| --- | --- |
| **One Radeon AI PRO R9700 (gfx1201, 32 GB) on AUP Learning Cloud** | Every model runs on that GPU (VSCode Server GPU resource). Detector, VLM and agent LLM share 32 GB; M0 measured 18 GB peak with all of them, see [docs/STACK.md](docs/STACK.md). Laptops run the UI and stubs only. |
| **Sessions last at most 240 minutes** | The server stops after 4 h and has to be relaunched from the hub. Commit often; keep weights and installs in the NFS home; `scripts/setup.sh` re-checks the environment. The VLM takes ~3.5 min to load from NFS, so it must stay resident in one long-running process. |
| **The ROCm PyTorch build is preinstalled and must not be replaced** | PyTorch 2.9.1+rocm7.13 works on the GPU (M0 passed). Install with `pip --user --break-system-packages`, `--no-deps` for anything that pins torch. llama.cpp remains the fallback for the LLM. |
| **Fully local inference** | No cloud API calls in the pipeline: the "nothing leaves the machine" claim has to be literally true. Coding agents during development are fine. |
| **About 30 hours left (as of Sat 01:40)** | The milestone times below assume submission on **Sun around 09:00**. Shift every time once we confirm the real deadline. |
| **The agent never touches pixels** | The agent can only edit the mask timeline through typed tools. Only the renderer writes video. |
| **Fail closed** | Uncertain frames, lost tracks and pipeline errors all mean more blur, never less. |
| **No explicit data** | We never collect, store or train on NSFW content. The NSFW detector is optional, and we demo it only on safe content. |
| **Footage consent** | Benchmark and demo clips are our own footage (with the people in them agreeing) or public-domain video. |
| **Licenses** | Ultralytics/YOLOE is AGPL-3.0. That's fine for an open hackathon repo, but keep the repo public and note it in the README. Check each model card before use. |
| **The judges must be able to run it** | One command (`./run.sh`) brings up the whole app on a clean box. Pin every dependency. |

---

## 2. Stack and repo layout

**Stack (fixed by M0):** Python 3.12 (system, ROCm PyTorch 2.9.1), FastAPI, OpenCV (headless) and ffmpeg for the backend; Ultralytics YOLOE-11s-seg with MobileCLIP text prompts plus BoT-SORT/ByteTrack; **one Qwen3-VL-8B-Instruct process (transformers, bf16) serving both the agent and the VLM checks**, with validated JSON tool calls; React + Vite web UI. Measurements and reasons are in [docs/STACK.md](docs/STACK.md).

```
censor/
  core/
    timeline.py        # pydantic models for the mask timeline, op log, fold/undo
    video.py           # probe, frame iterator, frame<->time conversion
  vision/
    backends.py        # loads each model on the backend chosen in M0
    detect.py          # open-vocab detector (+ face/text fallbacks)
    track.py           # tracker, box smoothing, hold-after-loss
    vlm.py             # yes/no attribute checks on track crops (~1 fps)
  agent/
    tools.py           # typed tool definitions + implementations
    agent.py           # LLM loop: request -> tool calls -> ops
    intents.py         # template parser fallback (no LLM)
  render/
    render.py          # timeline -> ffmpeg filter graph / OpenCV render
    live.py            # (stretch) delay buffer + standing rules
  server/
    app.py             # FastAPI: upload, request, timeline, preview, export
schema/
  timeline.schema.json # generated from core/timeline.py, the shared contract
  examples/            # hand-written timelines for stubs and tests
web/                   # React UI: player, chat, draw-a-box, timeline strip, undo
bench/
  clips/               # gitignored; manifest.yaml lists sources and consent
  labels/              # per-clip box labels for every 10th frame
  requests.yaml        # the 20 written requests + expected targets
  run_bench.py         # writes bench/report.md + report.json
scripts/
  smoke_test.py        # M0 backend checks, writes docs/STACK.md
  setup.sh             # pinned env + model downloads
run.sh                 # one command: start LLM server, API, UI
```

Every module gets a CLI entry point so people can test it alone (for example `python -m censor.vision.detect clip.mp4 "license plate"` dumps boxes as JSON).

---

## 3. The shared contract: mask timeline

The timeline is an **op log** plus a **derived state**. Ops are what the agent and UI produce. State is what the renderer consumes. Undo pops the last op and re-folds the log.

```jsonc
// Op (append-only log)
{
  "op_id": "op_7",
  "request_id": "r_3",              // chat message or UI action that caused it
  "kind": "add_targets",            // add_targets | remove_targets | set_effect | add_rule
  "targets": ["trk_12", "trk_15"],
  "effect": "blur",                 // blur | pixelate | box
  "padding": 0.15,                  // fractional box expansion
  "time_range": [41.0, 63.5]        // optional; seconds
}

// Derived state (what the renderer reads)
{
  "version": 1,
  "video": { "fps": 30, "width": 1920, "height": 1080, "n_frames": 1800 },
  "targets": {
    "trk_12": {
      "label": "license plate",
      "source": "detector",          // detector | user_box | rule
      "effect": "blur",
      "padding": 0.15,
      "keyframes": [
        { "f": 120, "box": [x1, y1, x2, y2], "conf": 0.81, "state": "detected" },
        { "f": 121, "box": [...],           "conf": 0.00, "state": "tracked" },
        { "f": 140, "box": [...],           "conf": 0.00, "state": "held" },
        { "f": 141, "box": [...],           "conf": 0.40, "state": "uncertain" }
      ]
    }
  }
}
```

**Renderer rules (deterministic, all fail-closed):**
- Interpolate boxes between keyframes. Apply `padding`.
- `held`: keep the last box for up to `HOLD_FRAMES` (default 15) after a track is lost, growing it 5% per frame.
- `uncertain`: double the padding.
- An error while processing a segment blurs the whole frame for that segment and logs a warning. We don't silently skip.

**Track store:** the detection/tracking pass runs once per video, for all prompts the agent has asked about. It's cached as `tracks.json` (track id, label, keyframes, thumbnail path). The agent picks from cached tracks, so repeat requests are cheap.

---

## 4. Agent toolbox

About ten typed tools. Arguments are JSON-schema constrained so a small model can't emit malformed calls. Tools return compact JSON (ids, labels, times, counts), never images.

| Tool | Signature | Notes |
| --- | --- | --- |
| `video_info` | `() -> {duration, fps, n_frames}` | |
| `resolve_time` | `(phrase) -> [start, end]` | Deterministic: "halfway" means 50% ±10%, "first minute", "at 1:20" |
| `find_objects` | `(query, time_range?) -> [{track_id, label, start, end, max_conf}]` | Runs/extends the track store |
| `check_attribute` | `(track_ids, question) -> [{track_id, answer, conf}]` | VLM yes/no on 3–5 crops per track |
| `track_box` | `(frame, box) -> track_id` | Draw-a-box path |
| `add_masks` | `(track_ids, effect, padding?, time_range?) -> op_id` | |
| `remove_masks` | `(track_ids) -> op_id` | |
| `set_effect` | `(track_ids, effect) -> op_id` | |
| `add_rule` | `(category, effect) -> op_id` | Live/standing rules; categories are fixed |
| `summarize` | `() -> {n_targets, by_label, uncovered_uncertain}` | The agent reports this back to the user |
| `undo` | `() -> op_id` | |

**Agent loop:** at most 6 tool calls per request. If a call fails or the model gives up, `intents.py` tries a template parse ("blur all X", "blur X in the Y shirt"). If that fails too, the agent asks the user to clarify. It never guesses silently.

---

## 5. Milestones

Each milestone has an exit check that someone runs and posts in the team chat. Times assume a Sun ~09:00 deadline.

### M0: Stack proven (target Sat 10:00), Gate 1
Owner: A (Vision and stack), with C helping.
- `rocminfo` reports gfx1151. Record the ROCm and driver versions.
- `scripts/smoke_test.py` loads one small model on each of PyTorch ROCm, ONNX Runtime and llama.cpp. Results go into `docs/STACK.md`.
- Detector on 640 px frames: fps and peak memory recorded.
- Agent LLM and VLM candidates served; tokens/s recorded.
- Timeline schema v1 committed with two hand-written examples.

**Exit:** `docs/STACK.md` names the backend for each model, and the detector runs at ≥ 10 fps or we've switched to a fallback.
**If it fails:** run the detector on ONNX Runtime CPU with detection every 5th frame plus the tracker, and run the LLM on llama.cpp Vulkan. Don't spend more than 2 hours on any one ROCm problem.

### M1: Walking skeleton (target Sat 14:00)
All four roles, against stubs.
- C: the renderer applies `schema/examples/*.json` to a clip and exports an MP4.
- B: the agent turns "blur all plates" into a valid `add_masks` op against a stubbed `find_objects`.
- D: the UI uploads, plays, sends a chat message, shows the timeline strip and exports, all wired to the API.
- A: `find_objects` returns real tracks for one prompt on one clip.

**Exit:** upload, type a request, export a blurred MP4 through the real API path. Stubs are allowed anywhere except the renderer.

### M2: Real detection and tracking (target Sat 18:00)
Owner: A, plus C for flicker handling.
- Plates, faces, people and screens/text detected and tracked. Track store cached.
- Hold-after-loss and padding are in the renderer.
- Three labeled clips in `bench/`. `run_bench.py` reports recall and over-blur.

**Exit:** a bench report exists for 3 clips. Note per-category recall. If plate recall is under 0.8, start the time-boxed plate fine-tune (2 hours max, see PLAN.md concern 5).

### M3: Agent works for real (target Sat 21:00), Gate 2
Owner: B.
- All tools from section 4 work against the real track store.
- `check_attribute` works for at least color-of-clothing questions.
- `bench/requests.yaml` has 20 requests with expected targets. The scorer runs.

**Exit:** ≥ 14/20 requests resolve correctly with the template fallback enabled.
**If it fails:** demo template intents plus draw-a-box only, and present the LLM path as experimental.

### M4: Feature freeze (target Sun 01:00), Gate 3
All roles.
- Draw-a-box tracking, undo and preview all work in the UI.
- About 10 labeled clips. Full bench report including baselines (fixed-prompt detector without the VLM check; our pipeline with fail-closed off).
- Stage latency and peak memory in the report.

**Exit:** demo steps 1, 2 and 4 from PLAN.md run end to end on the Strix Halo without anyone touching code. After this point, only bug fixes go in.

### M5: Ship (target Sun 06:00)
Owner: D, with everyone helping.
- `./run.sh` works on a clean environment (fresh clone, fresh venv) on the box.
- README: what it is, how to run it, the bench table, limits (including NSFW), and the license note.
- Backup demo video recorded. Devpost draft written.

**Exit:** someone who didn't write the setup script follows only the README and gets the demo running.

### M6 (stretch): Live replay
Owner: C. Starts only after M4 passes.
- Replay a recorded clip through `live.py` with a delay buffer (default 2 s) and one standing rule ("blur any plate").
- Show the buffer delay and the per-frame latency on screen. Label it honestly as a replay.

**Cut line:** if M6 isn't working by Sun 05:00, drop it from the demo.

---

## 6. Success conditions

**Must hit (pass/fail):**
1. `./run.sh` brings up the whole app on the Strix Halo from a fresh clone.
2. The 3-minute demo flow from PLAN.md runs live, with the backup video ready in case it doesn't.
3. `bench/run_bench.py` regenerates the numbers we show, from committed labels.
4. No network calls during inference. Show this by running the demo with outbound network blocked, or by showing the egress log.
5. No labeled target is left unmasked for more than one labeled frame in a row in the demo clips.

**Initial targets** (revise after the first M2 run, and report honestly whatever we get):

| Metric | Target |
| --- | --- |
| Recall: faces | ≥ 0.95 |
| Recall: plates | ≥ 0.90 |
| Recall: screens/text | ≥ 0.85 |
| Over-blur (mean masked non-target area) | ≤ 10% of frame |
| Request success | ≥ 15 / 20 |
| Fail-closed rate on uncertain frames | 100% (it's a rule, not a hope) |
| Offline speed, 1080p | ≥ 0.5× real time |
| Peak memory, all models loaded | ≤ 100 GB (leaves headroom on 128 GB) |

**Judging story:** recall and fail-closed behavior show it's safe. Plain-language requests and draw-a-box show it's useful. Stage latency and memory on a single Strix Halo show it's AMD-native. Fully local operation shows it's private.

---

## 7. Decision points and fallbacks

| When | Signal | Decision |
| --- | --- | --- |
| M0 | GPU backend crashes or is slow | Switch that model to ONNX/CPU or llama.cpp Vulkan. Detect less often and lean on the tracker. |
| M2 | Plate recall < 0.8 | Do the 2-hour plate fine-tune, or switch to a dedicated plate detector. Never ship worse plates silently. |
| M3 | Agent < 14/20 | Make template intents the primary path and the LLM a bonus. |
| M3 | VLM > 1 s per crop | Use HSV color heuristics for clothing color; keep the VLM only for questions color can't answer. |
| M4 | Fewer than 10 clips labeled | Report on what we have, with the clip count stated. |
| Sun 05:00 | Live replay not stable | Cut it. |

---

## 8. Working agreements

- `main` always runs. Work on short branches named `role/feature`, then merge through small PRs or a quick review in chat.
- A schema change means bumping `version`, regenerating `schema/timeline.schema.json`, updating `schema/examples/`, and posting in chat.
- Each module ships with a CLI and at least one test that runs on the stubs in `schema/examples/`.
- Model weights, clips and outputs are gitignored. `scripts/setup.sh` downloads weights; `bench/clips/manifest.yaml` records clip sources and consent.
- Record every number in `bench/report.md` from a script, never by hand.
- Owners from PLAN.md: A Vision and stack · B Agent and toolbox · C Renderer and video I/O · D App, benchmark and demo. Fill in names there.
