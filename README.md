# Censor Agent (Knight Hacks 2026, AMD challenge)

Say what to hide in a video ("blur every license plate and the person in the red shirt around halfway") or draw a box. A local agent turns the request into a **mask timeline**, and a deterministic renderer applies it. Uncertain frames fail closed (more blur, never less). Everything runs on one AMD Radeon AI PRO R9700 with ROCm. No footage leaves the machine.

- What and why: [PLAN.md](PLAN.md)
- How, milestones and contracts: [IMPLEMENTATION.md](IMPLEMENTATION.md). **Current status and next steps are in section 5**; working agreements (branches, sharing the AUP server) in section 8.
- Measured stack: [docs/STACK.md](docs/STACK.md)

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
1. Pick a clip in **Load a sample clip** (the clips in `bench/clips`; Claude's in-app browser can't open a file picker, so this is the reliable way) or upload your own.
2. Type a request, e.g. `blur all faces and plates`, and press Enter. The status line says what it's doing; the first request on a new clip waits for the background scan (about 30 s for a short clip), and Export stays disabled until the request finishes.
3. **Preview masked frame** shows the current frame with the masks; drag on that preview to blur anything by hand (it gets tracked through the clip).
4. Click **Export**. When it's done, the player switches to the blurred video and a download link appears. `?video=<id>` in the URL reopens a video later.

`setup.sh` downloads every model; `./run.sh` then runs fully offline: any outbound connection is refused and logged to `logs/egress.log`, so an empty log after the demo shows nothing left the machine. Set `CENSOR_ALLOW_NETWORK=1` to turn that off.

**Laptop, no GPU (stub detector, no models):**
```bash
pip install -r requirements.txt
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

In the test UI, pick **synthetic** in **Load a sample clip** (its ground-truth sidecar comes along, so the stub detector finds the shapes), then type a request such as `blur all plates`. Uploading `synthetic.mp4` plus its `synthetic.mp4.tracks.json` through the two file pickers also works.

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
| `censor/agent/` | `tools.py` (the toolbox), `agent.py` (LLM loop + fallback), `intents.py` (templates) | B |
| `censor/render/` | `render.py` (fail-closed renderer, ffmpeg out) | C |
| `censor/server/` | `app.py` (main API), `model_server.py` (resident Qwen3-VL) | D / B |
| `web/index.html` | test UI: sample clips, request + status, preview, draw-a-box, undo, export playback | D |
| `schema/` | generated JSON schema + hand-written examples | C |
| `scripts/` | setup, smoke tests, synthetic clip, `fetch_clips.sh`, `check_offline.py` | A |
| `bench/` | `run_bench.py` (scores + report), `gt.py` (label format, prelabel drafts), `label_server.py` + `label.html` (labeling tool), clips manifest, labels, requests | D |
| `tests/` | pytest suite | all |

## Useful CLIs

```bash
python3 -m censor.vision.detect clip.mp4 "license plate" "face"          # tracks JSON (GPU)
CENSOR_STUB=1 python3 -m censor.agent.agent clip.mp4 "blur all plates"    # agent result JSON
python3 -m censor.render.render clip.mp4 schema/examples/plate_moving.json out.mp4
python3 -m censor.core.timeline                                           # regenerate schema/timeline.schema.json
```

## License note

YOLOE/Ultralytics is AGPL-3.0, so this repo stays public. Model cards: YOLOE-11s-seg, MobileCLIP, Qwen3-VL-8B-Instruct.
