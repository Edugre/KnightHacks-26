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
  netguard.py          # egress guard: refuses and logs outbound connections when CENSOR_OFFLINE=1
  core/
    timeline.py        # pydantic models for the mask timeline, op log, fold/undo
    tracks.py          # track store (detector output cached per video)
    video.py           # probe, frame iterator, frame<->time conversion
  vision/
    detect.py          # YOLOE text-prompt detector + tracker; StubDetector replays ground truth
    track.py           # draw-a-box tracking (OpenCV MIL)
    specialist.py      # router: faces + plates -> EgoBlur (every 3rd frame, IoU tracker), the rest -> YOLOE
    vlm.py             # yes/no attribute checks on track crops, via the model server
  agent/
    tools.py           # typed tool definitions + implementations (+ Session.person_for: face -> its person)
    agent.py           # routing: template for plain requests, one-call planner for the rest, else clarify
    planner.py         # LLM writes the whole plan as JSON once; plain code runs it with the toolbox
    intents.py         # template parser (no LLM): objects, person descriptions -> VLM questions, time phrases
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
  index.html           # the editor (layout 4a): requests with the agent's steps, coverage strips, targets, versions, timeline + player, draw-a-box, undo, export
bench/
  clips/               # gitignored; manifest.yaml lists sources and consent
  labels/              # per-clip box labels for every 10th frame
  requests.yaml        # the 20 written requests + expected targets
  run_bench.py         # writes bench/report.md + report.json
  gt.py                # label format; `prelabel` writes detector drafts
  label_server.py      # labeling tool (+ label.html), runs on a laptop
  pp4av.py             # face + plate recall on the public PP4AV images (data in ~/datasets/pp4av, never committed)
  review.py            # cached EgoBlur on every frame, replayed track rules, crop sheets, per-plate scoring (any CUDA GPU)
  profile_scan.py      # where the scan time goes, GPU vs CPU-only
  egoblur_speed.py     # fp16 / autotuning / input scale / two streams for EgoBlur
scripts/
  setup.sh             # server env (never replaces the ROCm torch)
  smoke_yoloe.py       # M0 detector check
  smoke_vlm.py         # M0 VLM + agent check (loads its own 18 GB copy of the model)
  make_test_clip.py    # synthetic clip + ground-truth sidecar for stub mode
  fetch_clips.sh       # downloads the three Pexels benchmark clips
  check_offline.py     # proves every model loads with the network off
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
  "time_range": [41.0, 63.5]        // optional; seconds. add_targets: mask only this window; remove_targets (schema v2): unmask only this window
}

// Derived state (what the renderer reads)
{
  "version": 2,                     // 2: remove_targets honors time_range
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
| `remove_masks` | `(track_ids, time_range?) -> op_id` | `time_range` (schema v2): unmask only that window. Not in the LLM's tool schemas yet |
| `set_effect` | `(track_ids, effect) -> op_id` | |
| `add_rule` | `(category, effect) -> op_id` | Live/standing rules; categories are fixed |
| `move_box` | `(track_id, frame, box) -> {track, op_ids}` | UI only, not offered to the LLM: re-tracks from a corrected box from `frame` on (`remove_masks` + `add_masks` over `[frame, end]`, one request id) |
| `summarize` | `() -> {n_targets, by_label, uncovered_uncertain}` | The agent reports this back to the user |
| `undo` | `() -> op_id` | |

**Hand edits from the UI** go through the same toolbox, so they land in the op log like the agent's: `POST /api/videos/{id}/box` (`track_box` + `add_masks`), `POST /api/videos/{id}/targets` (`add_masks`, `remove_masks`, `set_effect`, or `move_box` for `"action": "move"`). `GET /api/videos/{id}/masks` gives the UI the renderer's per-frame rectangles for the live preview. Route list: the docstring of `censor/server/app.py`.

**Agent (changed Sat 09:50):** plain requests ("blur all plates and the woman") are parsed by `intents.py` with no model. Anything else ("everyone except the guy in red", "only the woman's face") gets one LLM call that writes the whole plan as JSON: objects, "only" / "except" yes/no questions, one person or a group, time phrase, effect (`planner.py`). Plain code runs the plan with the tools above, so the LLM never handles track ids. An unusable plan is retried once with the reason, then the template is tried, then the agent asks the user to clarify. It never guesses silently. (The first version let the LLM call one tool per step, up to 6; that took over 2 minutes per request.)

---

## 5. Milestones

Each milestone has an exit check that someone runs and posts in the team chat. Times assume a Sun ~09:00 deadline.

**Full demo flow on the R9700 (Sat 06:00, current `main`, app through the AUP proxy, model server `"offline": true`):** on `street_walk`, "blur all plates and the person in the red hoodie around halfway" blurred only the man in the red hoodie, only inside the halfway window (Qwen3-VL attribute check on the template path); drawing a box on a face tracked it through the whole clip (still on her face 6 s later); export worked. **Not a detector miss, a parsing bug:** the car plate at the start looked unblurred because the template applies the request's time phrase ("around halfway") to every object in it, plates included. EgoBlur did find that plate (0 to 0.96 s, conf 0.95), and "blur all faces and plates" blurred all three faces and the plate from the first frame. **Fixed since (07:10):** a time phrase binds to the objects in its own part of the request. **Also seen:** a plate false alarm on a blue parking sign. The test UI now loads sample clips from the server (no file picker; the in-app browser can't open one), reopens a video with `?video=<id>`, and plays the blurred export after Export (the player used to show only the original). **Then fixed (06:00): exports with no masks.** A request waited on the session lock behind an unrelated pre-scan ("person") while the user clicked Export, which rendered zero masks. Scans are now per query (a finished query answers at once), and the UI shows a status line and keeps Export disabled until the request finishes. Verified with "blur all car plates" on `street_plates`: the taxi plate is blurred in the exported video (4,354 masked boxes over 325 frames), but with 232 plate tracks, most of them false alarms. **Then fixed (Sat 07:00): stale, unmasked exports.** AUP's Cloudflare caches responses ending in `.mp4` or `.jpg` for 4 hours when the app sends no `Cache-Control` (`cf-cache-status: HIT`, `max-age=14400`), so the Download link kept serving an early export with no masks after a masked one had been rendered. The app now sends `Cache-Control: no-store` on every response, and each export returns its own URL (`export.mp4?v=<render time>`) for the player and the link. Both parts are needed: with `no-store` Cloudflare stops caching at its edge (`cf-cache-status: BYPASS`), but it still tells the browser `max-age=14400`, so only a new URL per render keeps the browser from replaying an old file. Verified on the R9700 through the proxy: an Export sent 2 s into "blur all car plates" on `street_plates` waited for it and came back masked. Requests, boxes, undo and exports on one video now take turns on the server, so an Export clicked mid-request (another tab, or after the proxy drops a long request at about 100 s) waits for it, up to `CENSOR_EXPORT_WAIT` (80 s), then answers 409 with what it is waiting on. The UI refuses to export with no masks and, when the proxy drops a request, polls `busy` on `GET /api/videos/{id}` until the server finishes.

**Where we are (Sat 09:55): start here.** The demo flow works end to end on the R9700: pick a clip, type a request, Export, and the player shows the blurred video (pick `street_plates` -> "blur all faces and plates" -> finished export in about 15 s). Faces and plates go to EgoBlur (fp16), everything else to YOLOE; the upload scan runs at 0.76x real time at 1080p, 24x the CPU. Plates on the labeled clips are masked 100% / 94.6% of their frames with 3% plate over-blur. Person descriptions become VLM questions with a probability ("blur the woman", "the guy in the black t-shirt": right on every person in focus), and requests with exceptions take one LLM call for the plan (2-7 s per request). Everything runs offline with an egress log. How to run it: README, "Run" and "Using the app". How to check detections on a laptop: README, "Reviewing detections and masks".

**Server down (Sat 19:10):** while the AUP host is down, work runs on laptops, split into four parallel streams in [WORKPLAN_LOCAL.md](WORKPLAN_LOCAL.md) (temporary).

**Open, in order:**
1. **Fragmented person tracks: partly fixed (Sat 22:00, laptop), and mostly not a tracker problem.** A track is one object followed through the video; masks and VLM questions work per track, so a person split into pieces costs a VLM check per piece and can get different answers per piece. What was found on the three clips (YOLOE-11 + BoT-SORT, as in the app):
   - **Most short tracks are different people, not one person split.** On `street_walk` the people in focus are already one track each for the whole clip; the short ones are several distant pedestrians, 25-40 px tall, each seen briefly. On `street_talk` many are not whole people (a bag strap, a car wheel, someone cut off at the frame edge). On `street_plates` it's a crowd walking in and out of view.
   - **New: `censor/vision/stitch.py`, on by default (`CENSOR_STITCH=0` turns it off).** It joins a piece B to a piece A when B starts within 30 frames of A's end (or overlaps it as a duplicate box), where A was heading (within 0.75 box heights), at about A's size (1.4x), with similar colors (hue-saturation histograms, computed during tracking). Joining two different people is the failure that matters, so every join on the three clips was checked by eye (`bench/stitch_review.py` draws them): all correct. Person tracks: `street_walk` 15 -> 11, `street_talk` 14 -> 13, `street_plates` 59 -> 56; no time added. Tests: `tests/test_stitch.py`.
   - **Tried and dropped:** a longer BoT-SORT memory (90 frames instead of 30) changed nothing; BoT-SORT appearance re-identification fails with YOLOE's own features (`model: auto`, a tensor shape error) and made things worse with a separate classifier (`yolo11n-cls.pt`: 19 / 15 / 76 tracks); plain ByteTrack is worse too (23 / 12 / 66 tracks, and fewer person boxes kept: 774 vs 906 on `street_walk`, so fewer moments masked); YOLOE-26 fragments less but misses real people (DETECTOR_OPTIONS.md item 2). Configs in `bench/trackers/`, numbers from `bench/compare_people.py --tracker`.
   - **What would still help:** the remaining cost is many distinct brief people, so the lever left is fewer VLM checks per request (skip or batch tiny, blurry tracks, which an unsure answer blurs anyway) or SAM 3's memory tracker on the server (DETECTOR_OPTIONS.md item 3).
2. **One slow first request (seen once).** On a freshly loaded `street_walk`, the first request ("blur the woman") took 59 s although it only needed the person scan, which takes about 9 s on that clip. Every later run of the same steps took 9-10 s, and it happened before the VLM stall fixes (item 5 below), so it may have been one of those, but that is not proven. Why it matters: the first request is the one a demo audience sees, and past about 100 s the AUP proxy times the browser out. **The fix:** log how long each step of a request takes (waiting for the scan, each tool call, the VLM calls) in the app log, so if it happens again the log shows where the time went.
3. **Label the three Pexels clips and write about 20 requests** in `bench/requests.yaml`, so the request success rate is measured (item 7 below).
4. **Demo polish and M5:** still to do on the R9700: `./run.sh` from a fresh clone there, and the screen recording of the demo (item 8). **Fresh clone on Windows: works (Sat 23:30).** Clone, README setup (8 s), `./run.ps1 test` (99 pass in 26 s), `./run.ps1 stub`, then in the page: load **synthetic**, "blur all plates and the woman" (3 tracks), export (180 frames, 0 failed). **Backup exports: made (Sat 23:30)** with `scripts/make_backup.py` on a laptop GPU, real detectors, real clips; in `data/backup/` and the team drive, not git (docs/DEMO.md, "If live fails"). **UI and docs done (Sat night, WORKPLAN_LOCAL Worker 4, tested in stub mode on `synthetic`):** `web/index.html` is now the conversation-first editor (layout 4a of a design hand-off that isn't in the repo): targets on the left (click to outline that target on the frame), versions on the right (undone ones struck through), and each reply shows the agent's steps in plain words ("Asked "Is this a woman?" about 2 tracks: 0 yes, 0 no, 2 unsure"), a scrubbable masked frame and one coverage bar per target. **Timeline** opens a drawer with the player (original or blurred export) and one labeled row per target (label + effect); clicking a row seeks. Undo removes a whole request's ops (a request can add several), draw-a-box works on the latest frame, there are busy states, inline errors and a "?" list of example requests. Still plain HTML and JS, no build step, no API changes. **Crowd fixes (Sat 22:45, after trying it on the real `street_plates` with real detectors on a laptop GPU):** 127 targets meant 127 coverage bars, which pushed the frame off screen, so above 8 targets the bars are one per kind ("14 license plates · 113 faces") plus the selected target, with a legend; long track-id lists read as a count ("T15, T16, T17 and 110 more"); and "Review 114 uncertain" is now "Check 114 low-confidence spots", explained as what it is (the detector saw it but wasn't sure; it's masked anyway, with extra padding), not "lost track". Demo script: [docs/DEMO.md](docs/DEMO.md); Devpost draft: [docs/DEVPOST.md](docs/DEVPOST.md) (request success rate still a TODO for the benchmark run). **Video preview with layers (Sat night, merged Sun 03:39 in #13/#15, tested in stub mode on `synthetic`):** the editor plays the video at the top of the page with two stackable layers on a canvas, synced to the frame on screen (`requestVideoFrameCallback`): **Boxes**, every detected track, solid if masked, dashed if not, labeled with the VLM's latest answer (kept in the browser from each request's `calls`); and **Masks**, the renderer's own rectangles from the new read-only route `GET /api/videos/{id}/masks` (built from `FrameMasker`, so lead-in, hold and padding match the export), blurred and pixelated the way `apply_effect` does it in plain JS: INTER_AREA shrink to 1/24, OpenCV's Gaussian kernel with its mirrored border, bilinear back up. Against the server's rendered frames, the mean pixel difference inside the masks is 3.1 (plate) to 7.6 (a person), against 22-84 for the unmasked original and 1.7 from JPEG and decoding alone; a first version that let the browser resample left the plate white (45.8). About 10 ms per frame at 720p. **Exact export** plays the rendered file and turns stale after any edit. Tests: `tests/test_preview_routes.py`. **Hand edits (merged in #15):** click a box in the Preview (or a Targets row) to select it, then mask it, unmask it (Delete) or change its effect; a new route `POST /api/videos/{id}/targets` (`add` / `remove` / `effect` on chosen track ids) calls the toolbox's `add_masks`, `remove_masks` and `set_effect`, so each edit is one op that Undo reverts. The Preview has its own play, frame-step and scrub controls (the browser's covered the bottom of the picture, where plates are). **Moving or resizing a mask (merged Sun 03:41 in #16, schema version 2):** with a masked box selected, drag it or a corner in the Preview; the server re-tracks from the corrected box from that frame on (`Session.move_box`, `POST /targets` with `"action": "move"`) and records two ops under one request id: `remove_targets` with `time_range [f, last frame of the new track]` on the old track and `add_targets` with the same window on the new one. **Fail-closed fix (Sun, Alex):** the window first ran to the end of the video, but the box tracker (OpenCV MIL) stops where it loses the object, so after that the object showed unmasked; now the old mask comes back right where the new track ends (`test_move_keeps_the_old_mask_where_the_new_track_stops`). The only contract change: `fold` now honors `time_range` on `remove_targets` (unmask only that window; before, it was ignored); `SCHEMA_VERSION` 2, schema regenerated, example `schema/examples/ops_move.json`. The renderer is unchanged, so the old box is still held for up to 15 frames (0.5 s) after the cut, overlapping the new one: extra blur, never less. Tests: `tests/test_target_routes.py`.
5. **Small plate fixes: done (Sat 22:15, laptop).** (a) **The taxi's sticker** on `street_plates` was a confident "plate" (131x123 px, 0.92) blurred for a second: confident plate tracks must now be wider than tall too (median 1.25:1, `CENSOR_PLATE_CONFIDENT_MIN_ASPECT`), except at the frame edge (cut-off plates) or under 16 px tall, where a box's shape is noise. It also dropped three other square false tracks; every real plate track was 1.69:1 or wider. (b) **An 11x7 px plate** on `street_walk` was there on 4 frames in a row but on only one checked frame (EgoBlur runs every 3rd), so it was dropped as a blip: a dropped tiny blip (under 16 px tall, >= 0.4) now gets the frames around it checked (`recheck_tiny` in `censor/vision/specialist.py`), 0-8 extra model calls per clip here. Doing that for every weak blip tripled the false plates on `street_plates` (tail lights are there in the next frame too), hence tiny only. Result: `street_plates` plates still masked 100%, plate over-blur 3.0% -> 2.3% of the frame; `street_walk` 94.6% -> 98.2% masked, longest a plate stays visible 0.16 s -> 0.04 s. Checked by eye on the rendered frames. Also: YOLOE `half=True` for the person scan was tested Sat evening on a laptop and is slightly worse (more fragments, fewer detections): keep fp32 (DETECTOR_OPTIONS.md item 2, which also drops YOLOE-26).

**What was done today, in order (details and measurements):**
1. **Plate false alarms: fixed (Sat 06:40, shape check 07:00; the rule was tightened in item 4: now 2 confident detections).** Detection stays at conf 0.1, but a plate track only counts if it has one detection at or above 0.5 or appears in at least 3 checked frames (`CENSOR_PLATE_CONFIRM_CONF`, `CENSOR_PLATE_CONFIRM_FRAMES` in `censor/vision/specialist.py`); isolated weak blips are dropped and logged. Measured on the R9700:
   - `street_plates`: 232 -> **53** plate tracks (179 one-frame blips dropped); `street_walk`: 41 -> **12**. Every real plate checked stayed: "UL 7246" (0.96), "FL 320" (1.0), "UK 2189" (1.0) and the car at the start of `street_walk` (0.93). The false alarm on the parking sign (0.17) is gone. Check by eye with `scripts/overlay_tracks.py`.
   - PP4AV plate sweep (300 images, single photos, so only the confident half of the rule applies; a worst case): recall / false alarms per image at conf 0.1: 0.65 / 0.80; 0.2: 0.57 / 0.44; 0.3: 0.54 / 0.26; 0.5: 0.46 / 0.12; 0.7: 0.38 / 0.05; 0.9: 0.30 / 0.01. Raising the global threshold costs too much recall on small plates, which is why the rule keeps 0.1 and relies on frames.
   - **Shape check added (Sat 07:00):** a weak-only plate track that passes the frame count must also have a plate-shaped median box: width/height at least 1.2 and height at most 25% of the frame (`CENSOR_PLATE_MIN_ASPECT`, `CENSOR_PLATE_MAX_HEIGHT`). On PP4AV, 96.7% of labeled plates pass that test, and keeping weak plates only when plate-shaped gives recall 0.643 vs 0.650 with false alarms per image 0.66 vs 0.80 (a 1.5:1 cutoff would cost 2.8 points: 90.7% of real plates pass). Clips with confirmation plus shape: `street_plates` 49 tracks (232 with no rule), `street_walk` 11 (41); the 585x953 px "plate" over a dress is gone and every real plate checked stays.
   - **Still there:** weak detections that persist and are plate-shaped (road markings, sign panels), and a few confident non-plates near bumpers. Good enough for the demo; the labeled clips will put a number on it.
2. **Time phrases: fixed (Sat 07:10).** A time phrase now applies to the objects in its own part of the request (`censor/agent/intents.py`, `Plan.times`): "blur all plates and the person in the red hoodie around halfway" blurs the plates for the whole clip and the person only around halfway. A new part starts at a comma or at "and" followed by a new noun phrase ("and the ...", "and all ..."), so "faces and plates in the first 10 seconds" limits both. A time phrase before every object, or alone after a comma ("..., around halfway"), applies to all of them. The planner's plans carry a time phrase per target too. Tested in `tests/test_core.py` and `tests/test_e2e_stub.py`.
3. **GPU pipeline: done (Sat 07:50), 0.27x -> 0.76x real time at 1080p, 24x the CPU.** Measured first (`bench/profile_scan.py`, `bench/egoblur_speed.py` on the R9700): decoding runs at 1,455 fps (0.2 s of a 48 s scan), and EgoBlur was 87% of the upload scan at 0.147 s per model per 1080p frame, with 5.3 of a 5.4 s face scan spent inside the model calls. So decoding once or overlapping it saves little, and the GPU is the limit. What didn't help: EgoBlur takes no batch (TorchScript error), MIOpen autotuning changed nothing, the face and plate models on two streams took 0.296 s per frame against 0.295 s one after the other, and a 0.75x smaller input lost 14 of 92 faces and 29 of 55 plates. What did: **fp16 autocast** (0.049 s per call, same boxes: mean IoU 0.999 faces and 0.98 plates against fp32, none lost; PP4AV recall 0.736 / 0.645 against 0.739 / 0.647, 300 images) and a **warm-up at app start**, since a freshly loaded model's first calls cost about 6 s each. Frames are also decoded ahead on a CPU thread, and skipped frames aren't converted. Upload scan, face + plate + person:

   | Clip | Before | After, first upload | After, warm | CPU only (Threadripper PRO 9985WX, 64 threads) |
   | --- | --- | --- | --- | --- |
   | `street_plates` 1080p, 13 s | 48.1 s (0.27x) | 17.4 s (0.75x) | 17.2 s (0.76x) | - |
   | `street_walk` 720p, 8.8 s | - | - | 7.9 s (1.12x) | 187.8 s (0.05x), same 4 / 11 / 18 tracks |

   **Export was the next bottleneck (fixed 08:05):** the blur ran a Gaussian with a kernel half the box's size on the full-size box (17 ms per padded face), so exporting `street_plates` with its 148 face tracks took 193 s and the browser got a Cloudflare timeout at about 100 s. The blur now smooths the 24x-downscaled box before scaling it back up: 0.2 ms per box, the same export in **2.8 s** (5,912 masked boxes), and no faint streaks of plate digits left as the old blur had. **End to end through the proxy:** pick `street_plates`, "blur all faces and plates" answered 11.4 s after the upload (the upload scan for faces and plates), export done 2.8 s later, 14.7 s in all.

   Settings: `CENSOR_EGOBLUR_FP16=0` (back to fp32), `CENSOR_WARMUP=0` (no warm-up). Reproduce: `python3 -m bench.profile_scan <clip> [--warm-up] [--cpu-scan]`. **Next here, if needed:** YOLOE `half=True` (0.008 vs 0.011 s per frame, not yet checked for accuracy) for the person scan, now 37% of the total.
4. **Over-blur and missed plates: fixed (Sat 08:25), measured with labels.** New tool `bench/review.py`: EgoBlur runs once per clip on every frame (cached in `data/review/<clip>/raw.json`), the app's track rules are replayed from the cache in milliseconds, crop sheets of every candidate track let a person mark the real ones (`bench/labels/<clip>.plate.json`, done for `street_plates` and `street_walk`), and `score` reports, per real plate, the share of frames masked and the longest stretch it stays visible, plus over-blur (masked area outside labeled plates). It runs on any CUDA GPU, so the review loop runs on a laptop (see README). What it found and what changed:
   - **False plate tracks were single confident blips:** 47 of the 52 false plate tracks kept on `street_plates` had one detection >= 0.5 (tail lights, signs, shoes); every real plate track had 6 or more. A plate track now needs 2 confident detections (`CENSOR_PLATE_CONFIRM_HITS`), and a track without them also needs a mean confidence of 0.3 (`CENSOR_PLATE_WEAK_MEAN`; the "7 7" street sign was blurred all clip at mean 0.26).
   - **False faces were car backs:** 300-900 px "faces" at 0.1-0.68 blurred up to the whole frame; every real face that big scored 1.00. A face track taller than 20% of the frame now needs a detection >= 0.8 (`CENSOR_FACE_BIG_FRAC`, `CENSOR_FACE_BIG_CONF`). Small faint faces stay: in a crowd, many real faces never pass 0.3.
   - **Small fast plates were split into one-frame tracks:** a 13 px plate moving 10 px between checked frames doesn't overlap itself, so linking also accepts a similar-sized box within 0.8 box sizes (`LINK_NEAR`), and a dropped fragment right before or after a kept track is folded into it.
   - **Masks started late and slid off small plates:** detection runs every 3rd frame, so masks now start 3 frames before the first detection, and after a track ends the mask follows the object's speed (measured over 4 keyframes, at most 0.25 box sizes per frame, no resizing) instead of standing still while growing 5% per frame; growth is now 2%, and every box gets at least 3 px of padding.

   | Clip | Plates masked (frames) | Longest a plate stays visible | Plate over-blur, mean / worst frame | All masks (faces + plates), mean / worst frame |
   | --- | --- | --- | --- | --- |
   | `street_plates` before | 98.1% | 0.08 s | 9.7% / 45% | 51% / 100% |
   | `street_plates` after | **100%** | **0 s** | **3.0% / 10%** | **14% / 24%** (mostly real faces in the crowd) |
   | `street_walk` before | 72.3% | 0.28 s | 0.2% / 1% | |
   | `street_walk` after | **94.6%** | 0.16 s (an 11x7 px plate seen once at 0.4-0.5) | 0.1% / 0% | 11% / 18% |

   Still open: the taxi's round sticker on `street_plates` is detected as a plate at 0.97 and stays blurred; `street_talk` has no plate labels yet. Reproduce: `python -m bench.review score bench/clips/street_plates.mp4 --cat plate`.
5. **Who a person is: fixed (Sat 09:00).** "blur the woman" used to blur every person: the template parser turned woman, guy and girl into plain "person" and asked the VLM nothing. Now each person description becomes one VLM question with its subject, adjectives and clothing ("Is this a man wearing red hoodie?", "Is this a blonde girl?"); "the woman and the kid" asks both and keeps either; objects inside a description ("holding a phone") are attributes, not targets; `find_objects("woman")` tells the LLM to search "person" and ask. The model server returns P(yes) from the first answer token (`/vlm/yesno` `p_yes`): above 0.65 is yes, below 0.35 no, between is unknown and masked (fail closed). Crops are the largest box from each third of a track with 15% context, fitted into one 256x512 canvas, three per call. **Two ROCm lessons:** MIOpen can't read its kernel database on the AUP image, so each new image size made it search and compile kernels (~15 s per VLM call when crop sizes varied): fixed crop size plus `MIOPEN_FIND_MODE=FAST` in the model server. And `model(**inp)` returns logits for every position (16 GB with three images): use `logits_to_keep=1`. **Measured on `street_walk` (R9700):** 18 person tracks (3 people in focus, 15 blurry background fragments) checked in 5-7 s per request; on the three main people every answer was right at p 0.00 or 1.00 ("blur the woman", "the man in the red hoodie", "the guy in the black t-shirt", "the men", and nobody for "the blue shirt"). Uncertain answers only came from out-of-focus background figures. Followed by the one-call planner (item 6).
6. **Requests the template can't parse: fixed (Sat 09:50), one plan per request.** The LLM used to pick one tool per step and wait for each result (over 2 minutes for "blur everyone except the guy in the red hoodie"). Now Qwen3-VL writes the whole plan once as JSON (`censor/agent/planner.py`): the objects to find, "only" and "except" yes/no questions, whether the request means one particular person, a time phrase, the effect. Plain code runs the plan with the toolbox, so the LLM never handles track ids. "only" keeps yes and unknown; "except" spares tracks answered yes, and when the request means one person ("but the lady", "but mine"), only the single most likely match; unknown stays masked. One person or a group is read from the words (`intents.one_person`), because the LLM marked "everybody but the lady" as a group. Faces are linked to the person whose head they are (`Session.person_for`), so "the woman's face" or "all faces but mine, I'm the one in the black t-shirt" is asked about the whole person. An unusable plan is retried once with the reason, then the template runs; the template's plans use the same executor. **Measured on `street_walk` (R9700):** the plan takes about 1.8 s; whole requests 2-7 s ("everyone except the guy in the red hoodie" 7.0 s: everyone blurred but him; "only the woman's face" 3.0 s; "all faces except the man in the black t-shirt" 3.2 s; "everybody but the lady" 7.0 s: only she stays visible; "make it look cinematic" 1.7 s: asks what to hide). **Still open:** person tracks are fragmented (18 for about 3 people in focus), which makes the checks slower and the answers on blurry fragments noisy.
7. **Label the three Pexels clips** (`bash scripts/fetch_clips.sh`, then `python -m bench.label_server ...`; `python3 -m bench.gt prelabel` gives EgoBlur drafts) and write about 20 requests in `bench/requests.yaml`.
8. **Demo polish and M5:** timeline strip in the UI, fresh-clone test, backup video, Devpost.

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

**Status (Sat 05:35): exit met, including real tracks on a real clip.**
- [x] C: the renderer applies timelines and exports MP4 (`tests/test_core.py`, `tests/test_e2e_stub.py`).
- [x] B: the agent turns "blur all plates" into `add_masks` against the stub (template fallback, in the tests). On the server the LLM path ran, but the synthetic clip gave it nothing to mask.
- [~] D: `web/index.html` uploads, sends requests, previews masked frames, draws a box, undoes and exports. No timeline strip yet, and the React UI isn't started.
- [x] A: real tracks on the Pexels clip `street_walk` through the router (EgoBlur faces, YOLOE people), rendered with 0 failed frames (see M2).

### M2: Real detection and tracking (target Sat 18:00)
Owner: A, plus C for flicker handling.
- Plates, faces, people and screens/text detected and tracked. Track store cached.
- Hold-after-loss and padding are in the renderer.
- Three labeled clips in `bench/`. `run_bench.py` reports recall and over-blur.

**Exit:** a bench report exists for 3 clips. Note per-category recall. If plate recall is under 0.8, start the time-boxed plate fine-tune (2 hours max, see PLAN.md concern 5).

**Status (Sat 05:35): detection works; labels for the real clips are next.**
- [x] Hold-after-loss, interpolation and padding are in the renderer.
- [x] `bench/run_bench.py` scores recall, over-blur, request success, both baselines, latency and GPU memory (`./run.sh bench`, `.\run.ps1 bench`). Its report on the synthetic clip only checks the plumbing.
- [x] Labeling tool: `python -m bench.label_server bench/clips/<id>.mp4`, plus `python3 -m bench.gt prelabel` for detector drafts on the server. Steps are in README, "Adding a real clip".
- [x] PP4AV image benchmark (`bench/pp4av.py`): 3,447 public driving images with face and plate boxes, so no hand labeling. Data stays in `~/datasets/pp4av/` on the server, shared by every clone (CC BY-NC-ND 4.0, never committed).
- [x] **Resolved by the router below. Found Sat 04:25 on PP4AV: YOLOE can't find faces.** With the shipped settings (prompts "face" and "license plate", conf 0.15, imgsz 1280), face recall is **0.00** over 3,893 faces and plate recall 0.06 over 8,028 plates; over-blur 0.0%, so it barely detects anything. On a 300-image sample with conf 0.05 and extra prompts ("human face", "head", "number plate"), faces only reach 0.04 (0.19 at 32 to 64 px tall) while plates reach 0.95 at 32 to 64 px but 0.17 under 16 px. **Next:** swap in a dedicated face detector (EgoBlur face, Apache 2.0, is the first candidate; EgoBlur also has a plate model) and lower the plate threshold. Compare with `python3 -m bench.pp4av run --limit 300 ...`.
- [x] **EgoBlur tried (Sat 05:10, 300-image PP4AV sample, Gen1 face + plate models in `~/datasets/egoblur`).** Recall, faces / plates: YOLOE as shipped 0.00 / 0.07; EgoBlur conf 0.3: 0.23 / 0.52; conf 0.1: 0.32 / 0.64; conf 0.1 with face padding 0.5: **0.74 / 0.65** at 2.8% over-blur; face padding 1.0: 0.79 / 0.67 at 5.0% over-blur. EgoBlur touches 78% of faces: most "misses" were its tight face boxes against PP4AV's whole-head labels, hence the wider face padding. Plate misses are real and mostly under 16 px. Cost: about 0.2 s per image for both models on the R9700, so on video run it every few frames and let the tracker fill in. **Next:** route faces and plates to EgoBlur (conf 0.1, face padding 0.5) and everything else to YOLOE. Reproduce with `python3 -m bench.pp4av run --detector egoblur --conf 0.1 --face-padding 0.5 --limit 300`. The model files are 420 MB each and Meta's CDN refuses the AUP server, so download them on a laptop and upload them.
- [x] **Router (Sat 05:25):** `censor/vision/specialist.py` sends face and plate queries to EgoBlur (every 3rd frame, IoU-linked, face boxes widened) and everything else to YOLOE; `get_detector()` uses it when the models are in `~/datasets/egoblur`. On `street_walk` (8.8 s): all 3 faces found on nearly every checked frame (224 keyframes), 18 YOLOE person tracks, render 221 frames with 0 failures; 27 s including model loading. Open issue: plates at conf 0.1 give 41 short tracks (81 keyframes), mostly false alarms; try a higher plate threshold or dropping one-keyframe tracks once the clips are labeled.
- [x] Three Pexels clips in `bench/clips/manifest.yaml` (`street_talk`, `street_walk`, `street_plates`; `bash scripts/fetch_clips.sh` downloads them) for requests and tracking.
- [ ] **Next:** label them. `python3 -m bench.gt prelabel` now gives EgoBlur drafts for faces and plates, so labeling is mostly confirming frames.

### M3: Agent works for real (target Sat 21:00), Gate 2
Owner: B.
- All tools from section 4 work against the real track store.
- `check_attribute` works for at least color-of-clothing questions.
- `bench/requests.yaml` has 20 requests with expected targets. The scorer runs.

**Exit:** ≥ 14/20 requests resolve correctly with the template fallback enabled.
**If it fails:** demo template intents plus draw-a-box only, and present the LLM path as experimental.

**Status (Sat 05:35): plain requests are fast; LLM requests are slow.**
- [x] `find_objects` rejects compound queries ("person in a red shirt"); the agent searches for the plain object and then calls `check_attribute`.
- [x] The scorer runs. `bench/requests.yaml` has 7 synthetic requests; the 20 real ones come with the real clips.
- [x] **75 s request fixed (Sat 05:30):** YOLOE stays loaded (a fresh predictor per run instead of a fresh model), uploads start a background scan for face, license plate and person (`CENSOR_PRESCAN`), and requests the template parser understands skip the LLM. "blur every person and any license plate" on `street_walk`: 75 s -> **0.03 s**. Agent results now time every tool call and LLM step (`ms`).
- [x] **Scans are per query (Sat 06:00):** each query scans into its own store and merges under a short lock, so a request for a finished query never waits on another scan; two callers for the same query share one scan.
- [x] `add_masks` takes `labels` (e.g. `{"labels": ["person"]}`), so the LLM no longer spells out dozens of track ids; before, it ran past its token limit at 7 to 9.6 s per step.
- [ ] **Open issue:** requests that need the LLM ("everyone except ...", "only ...") are still slow, over 2 minutes for "blur everyone except the person in the red hoodie" on `street_walk`. See the priorities at the top of this section.

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
- Owners from PLAN.md: A Vision and stack · B Agent and toolbox · C Renderer and video I/O · D App, benchmark and demo. Fill in names there. So far (Sat 05:35), Eduardo has built the stack, scaffold and agent; Alex the benchmark, labeling tool, offline guard, PP4AV benchmark, EgoBlur router, latency fixes and Windows launcher.
- Before opening a PR: `git pull --rebase origin main`, then run the tests (`./run.sh test`, or `.\run.ps1 test` on Windows).
- Shared files need a heads-up in chat before you edit them: `censor/core/timeline.py`, `censor/server/app.py` (the UI's API), `run.sh`, `scripts/setup.sh`, `requirements.txt`.

**Sharing the AUP server.** The team shares one account, so it's one session, one 32 GB GPU and one home directory.
- Each person works in their own clone (for example `~/kh-<name>`), never in someone else's checkout.
- Share the one resident model server on port 8100 (about 18 GB). A second copy won't fit, so don't run `scripts/smoke_vlm.py` while it's up. Run your own app on another port: `PORT=8001 ./run.sh` reuses the model server when it's healthy.
- Set `git config user.name` and `user.email` inside your clone, or your commits go out under whoever configured the account.
- Sessions stop after 240 minutes. Agree who relaunches it, and commit and push before it expires.
- The node has four GPUs shared between sessions; `rocm-smi` shows ours as GPU[0], but `/sys/class/drm/card*/device/gpu_busy_percent` lists all four, so don't read GPU busy % from sysfs (it read 98.5% during a CPU-only run).
- The proxy is Cloudflare: it caches `.mp4` and `.jpg` URLs for 4 hours unless the app says not to (it does now), still tells browsers to keep them for 4 hours (so give every changing file a new URL, like `export.mp4?v=`), and it gives up on a response after about 100 s, so the browser can see a failed request that the server is still running. Check `curl -sI <url> | grep -i cf-cache` when a page shows something older than you expect.
- Loading Qwen3-VL takes 3.5 to 9 minutes depending on how busy the storage is (`tail logs/model_server.log` shows progress). To pick up new app code without reloading the model, restart only the app: `pkill -f 'uvicorn censor.server.app:app --host 0.0.0.0 --port 8000'`, then `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=true CENSOR_OFFLINE=1 CENSOR_MODEL_URL=http://127.0.0.1:8100 PYTHONPATH=$PWD nohup python3 -m uvicorn censor.server.app:app --host 0.0.0.0 --port 8000 > logs/app8000.log 2>&1 &`. The page itself (`web/index.html`) is read on every request, so UI-only changes need no restart.
