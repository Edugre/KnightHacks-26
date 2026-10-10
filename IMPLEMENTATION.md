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
    tracks.py          # track store (detector output cached per video)
    video.py           # probe, frame iterator, frame<->time conversion
  vision/
    detect.py          # YOLOE text-prompt detector + tracker; StubDetector replays ground truth
    track.py           # draw-a-box tracking (OpenCV MIL)
    vlm.py             # yes/no attribute checks on track crops, via the model server
  agent/
    tools.py           # typed tool definitions + implementations
    agent.py           # LLM loop: request -> tool calls -> ops
    intents.py         # template parser fallback (no LLM)
  render/
    render.py          # timeline -> OpenCV masks -> ffmpeg; fail-closed rules
    live.py            # (stretch, not started) delay buffer + standing rules
  server/
    app.py             # FastAPI: upload, request, timeline, preview, export
    model_server.py    # resident Qwen3-VL for the agent and VLM checks (port 8100)
schema/
  timeline.schema.json # generated from core/timeline.py, the shared contract
  examples/            # hand-written timelines for stubs and tests
web/
  index.html           # minimal test UI today; the React UI (player, chat, draw-a-box, timeline strip, undo) is not started
bench/
  clips/               # gitignored; manifest.yaml lists sources and consent
  labels/              # per-clip box labels for every 10th frame
  requests.yaml        # the 20 written requests + expected targets
  run_bench.py         # writes bench/report.md + report.json
  gt.py                # label format; `prelabel` writes detector drafts
  label_server.py      # labeling tool (+ label.html), runs on a laptop
scripts/
  setup.sh             # server env (never replaces the ROCm torch)
  smoke_yoloe.py       # M0 detector check
  smoke_vlm.py         # M0 VLM + agent check (loads its own 18 GB copy of the model)
  make_test_clip.py    # synthetic clip + ground-truth sidecar for stub mode
run.sh                 # one command: model server + API + UI (gpu), stub, test, bench
run.ps1                # Windows laptops: stub, test, bench
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

**Status (Sat 02:50): M0 complete.** Details in [docs/STACK.md](docs/STACK.md).
- [x] `rocminfo` reports gfx1201; ROCm 7.13, PyTorch 2.9.1+rocm7.13.
- [x] PyTorch ROCm works on the GPU, so ONNX Runtime and llama.cpp weren't needed. Smoke tests are `scripts/smoke_yoloe.py` and `scripts/smoke_vlm.py`.
- [x] Detector at 640 px: 191 fps, 0.26 GB. (Re-measure at 1280 px.)
- [x] Qwen3-VL-8B: VLM check 311 ms, agent 28 tok/s, 17.9 GB peak.
- [x] Timeline schema v1 committed (`schema/timeline.schema.json`, generated from `censor/core/timeline.py`) with examples in `schema/examples/`.

**Exit:** `docs/STACK.md` names the backend for each model, and the detector runs at ≥ 10 fps or we've switched to a fallback.
**If it fails:** run the detector on ONNX Runtime CPU with detection every 5th frame plus the tracker, and run the LLM on llama.cpp Vulkan. Don't spend more than 2 hours on any one ROCm problem.

### M1: Walking skeleton (target Sat 14:00)
All four roles, against stubs.
- C: the renderer applies `schema/examples/*.json` to a clip and exports an MP4.
- B: the agent turns "blur all plates" into a valid `add_masks` op against a stubbed `find_objects`.
- D: the UI uploads, plays, sends a chat message, shows the timeline strip and exports, all wired to the API.
- A: `find_objects` returns real tracks for one prompt on one clip.

**Exit:** upload, type a request, export a blurred MP4 through the real API path. Stubs are allowed anywhere except the renderer.

**Status (Sat 03:40): exit met on stubs; real tracks wait on a real clip.**
- [x] C: the renderer applies timelines and exports MP4 (`tests/test_core.py`, `tests/test_e2e_stub.py`).
- [x] B: the agent turns "blur all plates" into `add_masks` against the stub (template fallback, in the tests). On the server the LLM path ran, but the synthetic clip gave it nothing to mask.
- [~] D: `web/index.html` uploads, sends requests, previews masked frames, draws a box, undoes and exports. No timeline strip yet, and the React UI isn't started.
- [ ] A: YOLOE runs on the server (147.5 fps at 640 px, Sat 03:15), but there is no real clip yet. The synthetic clip is drawn shapes, so YOLOE finds nothing in it.

### M2: Real detection and tracking (target Sat 18:00)
Owner: A, plus C for flicker handling.
- Plates, faces, people and screens/text detected and tracked. Track store cached.
- Hold-after-loss and padding are in the renderer.
- Three labeled clips in `bench/`. `run_bench.py` reports recall and over-blur.

**Exit:** a bench report exists for 3 clips. Note per-category recall. If plate recall is under 0.8, start the time-boxed plate fine-tune (2 hours max, see PLAN.md concern 5).

**Status (Sat 03:40): tooling done, clips next.**
- [x] Hold-after-loss, interpolation and padding are in the renderer.
- [x] `bench/run_bench.py` scores recall, over-blur, request success, both baselines, latency and GPU memory (`./run.sh bench`, `.\run.ps1 bench`). Its report on the synthetic clip only checks the plumbing.
- [x] Labeling tool: `python -m bench.label_server bench/clips/<id>.mp4`, plus `python3 -m bench.gt prelabel` for detector drafts on the server. Steps are in README, "Adding a real clip".
- [x] PP4AV image benchmark (`bench/pp4av.py`): 3,447 public driving images with face and plate boxes, so no hand labeling. Data stays in `data/pp4av/` on the server (CC BY-NC-ND 4.0, never committed).
- [ ] **Blocking (found Sat 04:25 on PP4AV): YOLOE can't find faces.** With the shipped settings (prompts "face" and "license plate", conf 0.15, imgsz 1280), face recall is **0.00** over 3,893 faces and plate recall 0.06 over 8,028 plates; over-blur 0.0%, so it barely detects anything. On a 300-image sample with conf 0.05 and extra prompts ("human face", "head", "number plate"), faces only reach 0.04 (0.19 at 32 to 64 px tall) while plates reach 0.95 at 32 to 64 px but 0.17 under 16 px. **Next:** swap in a dedicated face detector (EgoBlur face, Apache 2.0, is the first candidate; EgoBlur also has a plate model) and lower the plate threshold. Compare with `python3 -m bench.pp4av run --limit 300 ...`.
- [x] Three Pexels clips in `bench/clips/manifest.yaml` (`street_talk`, `street_walk`, `street_plates`) for requests and tracking; labels still to do (detector drafts via `bench.gt prelabel` once the face detector works).

### M3: Agent works for real (target Sat 21:00), Gate 2
Owner: B.
- All tools from section 4 work against the real track store.
- `check_attribute` works for at least color-of-clothing questions.
- `bench/requests.yaml` has 20 requests with expected targets. The scorer runs.

**Exit:** ≥ 14/20 requests resolve correctly with the template fallback enabled.
**If it fails:** demo template intents plus draw-a-box only, and present the LLM path as experimental.

**Status (Sat 03:40): in progress.**
- [x] `find_objects` rejects compound queries ("person in a red shirt"); the agent searches for the plain object and then calls `check_attribute`.
- [x] The scorer runs. `bench/requests.yaml` has 7 synthetic requests; the 20 real ones come with the real clips.
- [ ] **Open issue: one request took 75 s on the server** (Sat 03:15, real models, synthetic clip). Likely causes: `YoloeDetector.run` loads a fresh model and makes a full-video pass for every new query, and when the LLM makes no edit the template fallback runs the same searches again. Section 7 sets 3 s per tool call as the line for switching to llama.cpp, so measure where the time goes first.

### M4: Feature freeze (target Sun 01:00), Gate 3
All roles.
- Draw-a-box tracking, undo and preview all work in the UI.
- About 10 labeled clips. Full bench report including baselines (fixed-prompt detector without the VLM check; our pipeline with fail-closed off).
- Stage latency and peak memory in the report.

**Exit:** demo steps 1, 2 and 4 from PLAN.md run end to end on the R9700 server without anyone touching code. After this point, only bug fixes go in.

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
1. `./run.sh` brings up the whole app on the R9700 server from a fresh clone.
2. The 3-minute demo flow from PLAN.md runs live, with the backup video ready in case it doesn't.
3. `bench/run_bench.py` regenerates the numbers we show, from committed labels.
4. No network calls during inference. Show this by running the demo with outbound network blocked, or by showing the egress log.
   *How (Sat 04:00):* `./run.sh` (gpu and bench) puts Hugging Face and Ultralytics in offline mode and sets `CENSOR_OFFLINE=1`, so `censor/netguard.py` refuses and logs every outbound connection or DNS lookup to `logs/egress.log`. Models come from the caches `scripts/setup.sh` fills. Proof for the demo: the model server's `/health` shows `"offline": true` and `logs/egress.log` stays empty. `python3 scripts/check_offline.py [clip]` checks the caches and a guarded detection pass without loading the 18 GB model. `CENSOR_ALLOW_NETWORK=1` turns it off.
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
| Peak GPU memory, all models loaded | ≤ 28 GB (leaves headroom on the 32 GB R9700) |

**Judging story:** recall and fail-closed behavior show it's safe. Plain-language requests and draw-a-box show it's useful. Stage latency and memory on a single Radeon AI PRO R9700 show it's AMD-native. Fully local operation shows it's private.

---

## 7. Decision points and fallbacks

| When | Signal | Decision |
| --- | --- | --- |
| M0 | GPU backend crashes or is slow | Resolved: PyTorch ROCm works for every model (see docs/STACK.md). |
| M3 | Agent tool calls too slow (> 3 s per call) or GPU memory tight | Serve the LLM with llama.cpp (ROCm/HIP build) from a Q4/Q8 GGUF, and keep transformers for the VLM checks. |
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
- Owners from PLAN.md: A Vision and stack · B Agent and toolbox · C Renderer and video I/O · D App, benchmark and demo. Fill in names there. So far (Sat 03:40), Eduardo has built the stack, scaffold and agent; Alex the benchmark, labeling tool and Windows launcher.
- Before opening a PR: `git pull --rebase origin main`, then run the tests (`./run.sh test`, or `.\run.ps1 test` on Windows).
- Shared files need a heads-up in chat before you edit them: `censor/core/timeline.py`, `censor/server/app.py` (the UI's API), `run.sh`, `scripts/setup.sh`, `requirements.txt`.

**Sharing the AUP server.** The team shares one account, so it's one session, one 32 GB GPU and one home directory.
- Each person works in their own clone (for example `~/kh-<name>`), never in someone else's checkout.
- Share the one resident model server on port 8100 (about 18 GB). A second copy won't fit, so don't run `scripts/smoke_vlm.py` while it's up. Run your own app on another port: `PORT=8001 ./run.sh` reuses the model server when it's healthy.
- Set `git config user.name` and `user.email` inside your clone, or your commits go out under whoever configured the account.
- Sessions stop after 240 minutes. Agree who relaunches it, and commit and push before it expires.
