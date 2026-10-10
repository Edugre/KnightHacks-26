# Censor Agent (Knight Hacks 2026, AMD challenge)

Say what to hide in a video ("blur every license plate and the person in the red shirt around halfway") or draw a box. A local agent turns the request into a **mask timeline**, and a deterministic renderer applies it. Uncertain frames fail closed (more blur, never less). Everything runs on one AMD Radeon AI PRO R9700 with ROCm. No footage leaves the machine.

- What and why: [PLAN.md](PLAN.md)
- How, milestones and contracts: [IMPLEMENTATION.md](IMPLEMENTATION.md)
- Measured stack: [docs/STACK.md](docs/STACK.md)

## Run

**GPU server (AUP Learning Cloud, VSCode Server GPU):**
```bash
git clone https://github.com/Edugre/KnightHacks-26.git && cd KnightHacks-26
bash scripts/setup.sh      # once per fresh environment; never replaces the ROCm torch
./run.sh                   # starts the model server (~3-4 min first load), then the app on :8000
```
Open `https://dub.aupcloud.io/user/<your-user>/proxy/8000/`.

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

In the test UI, upload `synthetic.mp4`, then its `synthetic.mp4.tracks.json` in the second file picker, then type a request.

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
| `censor/vision/` | `detect.py` (YOLOE + stub), `track.py` (draw-a-box), `vlm.py` (attribute checks) | A |
| `censor/agent/` | `tools.py` (the toolbox), `agent.py` (LLM loop + fallback), `intents.py` (templates) | B |
| `censor/render/` | `render.py` (fail-closed renderer, ffmpeg out) | C |
| `censor/server/` | `app.py` (main API), `model_server.py` (resident Qwen3-VL) | D / B |
| `web/index.html` | minimal test UI; replace with the real app | D |
| `schema/` | generated JSON schema + hand-written examples | C |
| `scripts/` | setup, smoke tests, synthetic clip | A |
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
