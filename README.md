# Censor Agent (Knight Hacks 2026, AMD challenge)

Say what to hide in a video ("blur every license plate and the person in the red shirt around halfway") or draw a box. A local agent turns the request into a **mask timeline**, and a deterministic renderer applies it. Uncertain frames fail closed (more blur, never less). Everything runs on one AMD Radeon AI PRO R9700 with ROCm. No footage leaves the machine.

- What and why: [PLAN.md](PLAN.md)
- How, milestones and contracts: [IMPLEMENTATION.md](IMPLEMENTATION.md). **Current status and next steps are in section 5**; working agreements (branches, sharing the AUP server) in section 8.
- Measured stack: [docs/STACK.md](docs/STACK.md)
- Detector alternatives researched but not yet tried (SAM 3, EgoBlur Gen2, YOLOE-26...): [DETECTOR_OPTIONS.md](DETECTOR_OPTIONS.md)

## Run

**GPU server (AUP Learning Cloud, VSCode Server GPU):**
```bash
git clone https://github.com/Edugre/KnightHacks-26.git && cd KnightHacks-26
bash scripts/setup.sh      # once per fresh environment; never replaces the ROCm torch
bash scripts/fetch_clips.sh   # optional: the three Pexels test clips
./run.sh                   # starts the model server (3.5-9 min to load), then the app on :8000
```
Faces and plates use Meta's EgoBlur models (Apache 2.0), which `setup.sh` can't fetch: get `ego_blur_face_gen1.jit` and `ego_blur_lp_gen1.jit` from https://www.projectaria.com/tools/egoblur/ (email form) on a laptop, since Meta's CDN refuses the AUP server, and upload them to `~/datasets/egoblur/`. Without them the app warns and falls back to YOLOE, which misses faces.
Open `https://dub.aupcloud.io/user/<your-user>/proxy/8000/`.

**Using the app:**
1. Pick a clip on the start screen (the clips in `bench/clips`; Claude's in-app browser can't open a file picker, so this is the reliable way) or upload your own.
2. Type a request in the box at the bottom, e.g. `blur all faces and plates`, `blur the woman`, `blur everyone except the guy in the red hoodie` or `pixelate all faces but mine, I'm the one in the black t-shirt`, and press Enter (Shift+Enter for a new line; **?** lists examples). Plain requests are parsed instantly; the rest take one LLM call for the plan (about 2 s) plus a VLM check per person (a few seconds). The first request on a new clip waits for the background scan (about 30 s for a short clip). Each reply lists what the agent did in plain words (what it looked for, each yes/no question with how many people were yes / no / unsure, what it masked), then shows a masked frame and one coverage bar per target: click the bars to scrub the frame. The raw tool calls are under **Tool calls**.
3. **Targets** (left) lists everything masked; click one to outline it on the frame. **Versions** (right) lists every change; **Undo** reverts the latest request as a whole. **Draw box**, then drag on the latest frame, blurs anything by hand (it gets tracked through the clip).
4. **Timeline** (top bar, or **Open timeline** under a reply) opens the player with one row per target (label and effect); click a row to seek there.
5. Click **Export**. When it's done, the Timeline drawer plays the blurred video and a download link appears in the top bar. Export waits for a request that is still running and says so if there are no masks yet. `?video=<id>` in the URL reopens a video later.

`setup.sh` downloads every model; `./run.sh` then runs fully offline: any outbound connection is refused and logged to `logs/egress.log`, so an empty log after the demo shows nothing left the machine. Set `CENSOR_ALLOW_NETWORK=1` to turn that off.

**Laptop, no GPU (stub detector, no models):**
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate     # or any venv; run.sh uses whatever python3 is first on PATH
uv pip install -r requirements.txt
python3 scripts/make_test_clip.py bench/clips/synthetic.mp4    # clip + ground-truth tracks sidecar
./run.sh stub                                                   # http://localhost:8000
```

**Windows laptop (PowerShell):** `run.sh` needs bash, so use `run.ps1`. It uses `.venv` if present, makes the synthetic clip if missing, and falls back to the `imageio-ffmpeg` binary when ffmpeg isn't installed.
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\run.ps1 stub    # http://localhost:8000; add a port number to change it
powershell -ExecutionPolicy Bypass -File .\run.ps1 test
```

In the app, pick **synthetic** on the start screen (its ground-truth sidecar comes along, so the stub detector finds the shapes), then type a request such as `blur all plates and the woman`. Uploading `synthetic.mp4` and then attaching its `synthetic.mp4.tracks.json` on the start screen also works. With no models, requests the template can't parse get a "clarify" reply, and person questions are always "unsure", so every person is blurred (fail closed).

**Tests:** `./run.sh test` runs the suite in about 15 s with no GPU. It covers the schema, fold/undo, the fail-closed renderer, the request templates, upload → request → export end to end, and the benchmark scoring.

## Benchmark

Every number we report comes from `bench/run_bench.py`, which writes `bench/report.md` and `bench/report.json` (targets from IMPLEMENTATION.md section 6).
```bash
./run.sh bench                # GPU server: YOLOE, plus the agent and VLM through the model server if it's running
```
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 bench    # Windows laptop: stub detector on the synthetic clip
```
Clips are listed in `bench/clips/manifest.yaml` (the videos are gitignored), labels live in `bench/labels/<clip>.json` (every 10th frame) and test requests in `bench/requests.yaml`. Options: `python -m bench.run_bench --help`. The stub detector replays ground truth, so on the synthetic clip the report checks the plumbing, not detection quality.

**Reviewing detections and masks (any CUDA GPU, including a laptop):** `bench/review.py` caches EgoBlur on every frame of a clip, then replays the app's track rules in milliseconds. `python -m bench.review detect <clip>` (3 min for a 13 s 1080p clip on an RTX 3050 Ti), `candidates <clip> --cat plate --min-len 3` writes crop sheets of every candidate track to `data/review/<clip>/`, `label <clip> --cat plate --real 0 36 ...` saves the real ones to `bench/labels/<clip>.plate.json`, `score <clip> --cat plate` gives per-plate masking and over-blur, and `frames <clip> --frames 0 60 120` draws detections and masks. On a Windows laptop with an NVIDIA GPU: `uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu126`, and put the two EgoBlur `.jit` files in `~/datasets/egoblur/`.

**Where the scan time goes (GPU server):** `python3 -m bench.profile_scan bench/clips/street_plates.mp4 [--warm-up] [--cpu-scan]` times decoding, each model per frame on GPU and CPU, and the upload scan; `--cpu-scan` runs the same scan on the CPU alone for the GPU vs CPU comparison. `python3 -m bench.egoblur_speed <clip>` compares fp16, autotuning, smaller inputs and two streams for EgoBlur, with how far each moves the boxes.

**Face and plate recall without labeling:** `bench/pp4av.py` scores the detector on PP4AV, 3,447 public driving images with face and plate boxes (CC BY-NC-ND 4.0, so the data stays in `~/datasets/pp4av/` outside the repo, one copy per machine). On the GPU server: `python3 -m bench.pp4av download` once (4.5 GB), then `python3 -m bench.pp4av run` writes `bench/report_pp4av.md`. Options for comparing detectors: `--limit`, `--conf`, `--face-prompts`, `--plate-prompts`, `--face-padding`, `--detector egoblur`, `--debug N`, and `--plate-conf 0.1 0.3 0.5` for plate recall against false alarms per image at several thresholds (the report also counts false alarms: detections that overlap no label).

**Adding a real clip**
1. Record 20 to 40 s in landscape, 1080p at 30 fps, H.264 (iPhone: Settings > Camera > Formats > Most Compatible). Only film people who agree to it.
2. Copy it to `bench/clips/<id>.mp4` and add an entry to `bench/clips/manifest.yaml` (source, consent). Share the video file with the team directly; git ignores it.
3. Label it in the browser, no GPU needed. Pick an object, drag its box, press `C` to carry boxes to the next frame, fix them, press `Enter` to confirm:
   ```bash
   python -m bench.label_server bench/clips/<id>.mp4      # http://localhost:8020; Windows: .venv\Scripts\python.exe -m ...
   ```
   Optional head start on the GPU server: `python3 -m bench.gt prelabel bench/clips/<id>.mp4` writes detector drafts. They are not scored until you confirm each frame, so check every frame for objects the detector missed.
4. Add requests for the clip to `bench/requests.yaml` (20 across the real clips is the target), then commit the labels, manifest and requests.

## Layout

| Path | What | Owner |
| --- | --- | --- |
| `censor/core/` | `timeline.py` (the contract: ops, fold, undo), `tracks.py`, `video.py` | C (schema), all |
| `censor/vision/` | `detect.py` (YOLOE + stub), `specialist.py` (faces + plates -> EgoBlur router), `track.py` (draw-a-box), `vlm.py` (attribute checks) | A |
| `censor/netguard.py` | egress guard: refuses and logs outbound connections when `CENSOR_OFFLINE=1` | all |
| `censor/agent/` | `tools.py` (the toolbox), `agent.py` (routing), `planner.py` (one LLM call -> JSON plan -> toolbox), `intents.py` (template parser, no LLM) | B |
| `censor/render/` | `render.py` (fail-closed renderer, ffmpeg out) | C |
| `censor/server/` | `app.py` (main API), `model_server.py` (resident Qwen3-VL) | D / B |
| `web/index.html` | the editor (conversation-first layout from `design_handoff_conversation_editor/`): requests with the agent's steps, coverage strips, targets, versions, timeline + player, draw-a-box, undo, export. One file, no build step | D |
| `docs/` | `STACK.md` (measured stack), `DEMO.md` (3-minute demo script and backup plan), `DEVPOST.md` (write-up draft) | all / D |
| `schema/` | generated JSON schema + hand-written examples | C |
| `scripts/` | setup, smoke tests, synthetic clip, `fetch_clips.sh`, `check_offline.py` | A |
| `bench/` | `run_bench.py` (scores + report), `review.py` (detections, masks and per-plate scores on a clip), `profile_scan.py` + `egoblur_speed.py` (GPU timing), `compare_people.py` + `stitch_review.py` + `trackers/` (person tracks per detector and tracker, stitcher joins), `pp4av.py` (face/plate recall), `gt.py`, `label_server.py` + `label.html` (labeling tool), clips manifest, labels, requests | D |
| `tests/` | pytest suite | all |

## Useful CLIs

```bash
python3 -m censor.vision.detect clip.mp4 "license plate" "face"          # tracks JSON (GPU)
CENSOR_STUB=1 python3 -m censor.agent.agent clip.mp4 "blur all plates"    # agent result JSON
python3 -m censor.render.render clip.mp4 schema/examples/plate_moving.json out.mp4
python3 -m censor.core.timeline                                           # regenerate schema/timeline.schema.json
python3 scripts/overlay_tracks.py clip.mp4 out.jpg --frames 0 60 --tracks before.json after.json   # compare two detector runs by eye
```

## License note

YOLOE/Ultralytics is AGPL-3.0, so this repo stays public. Model cards: YOLOE-11s-seg, MobileCLIP, Qwen3-VL-8B-Instruct.
