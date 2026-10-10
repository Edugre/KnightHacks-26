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
In the test UI, upload `synthetic.mp4`, then its `synthetic.mp4.tracks.json` in the second file picker, then type a request.

**Tests:** `./run.sh test` runs 17 tests in about 10 s with no GPU. They cover the schema, fold/undo, the fail-closed renderer, the request templates, and upload → request → export end to end.

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
