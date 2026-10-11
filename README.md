# Censor Agent (Knight Hacks 2026, AMD challenge)

Say what to hide in a video ("blur every license plate and the person in the red shirt around halfway") or draw a box. A local agent turns the request into a **mask timeline**, and a deterministic renderer applies it. Uncertain frames fail closed (more blur, never less). Everything runs on one AMD GPU with ROCm: built and measured on a Radeon AI PRO R9700, now running on a Radeon 8060S (Strix Halo) while the Dublin hub is down. No footage leaves the machine.

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
Faces and plates use Meta's EgoBlur models (Apache 2.0), which `setup.sh` can't fetch: get `ego_blur_face_gen1.jit` and `ego_blur_lp_gen1.jit` from https://www.projectaria.com/tools/egoblur/ (email form) on a laptop, since Meta's CDN refuses the AUP servers, and upload them to `~/datasets/egoblur/`. A fresh server (a new hub, or a new home directory) needs them uploaded again. Without them the app warns and falls back to YOLOE, which misses faces.
Open `https://dub.aupcloud.io/user/<your-user>/proxy/8000/`.

**If the Dublin hub is down:** use the San Jose hub, https://sjc.aupcloud.io. Launch settings we used: resource **VSCode Server GPU**, node **AMD Radeon 8060S (Strix Halo iGPU)** (gfx1151, 64 GB LPDDR5X shared with the CPU), git repository `https://github.com/Edugre/KnightHacks-26.git`, runtime **240**. Don't pick the Radeon 890M: its 4 GB can't hold Qwen3-VL-8B (about 18 GB). The session starts with the clone in `~/KnightHacks-26`; `cd` there and run the commands above from `bash scripts/setup.sh` on, after uploading the EgoBlur files. Open `https://sjc.aupcloud.io/user/<your-user>/proxy/8000/` (model server on :8100, app on :8000). Measurements on this server: [docs/STACK.md](docs/STACK.md), "Strix Halo"; older timings in the docs are R9700 (Dublin) numbers.
- **EgoBlur has to be uploaded by hand** to `~/datasets/egoblur/`: drag the files into the VS Code Explorer and wait for the "Uploading..." status to finish before moving them. They're about 420 MB each, and a half-uploaded file fails to load with "failed finding central directory".
- **A check script that loads the EgoBlur TorchScript files must `import torchvision` first.** Otherwise loading fails with `_assert_has_ops` even though the file is fine.
- If `bash scripts/setup.sh` is cut off (a browser disconnect stopped our first run mid-way through the MobileCLIP download), run it again: it's idempotent.
- **If the app answers "found nothing" for every request on a real clip, it is running in stub mode** (`CENSOR_STUB=1`, which only replays the synthetic clip's ground truth). It happened Sun morning after a UI test. Check with `tr '\0' '\n' < /proc/$(pgrep -f 'port 8000')/environ | grep CENSOR`, then restart it with the real detectors (the model server on :8100 keeps running):
  ```bash
  pkill -f 'uvicorn censor.server.app:app --host 0.0.0.0 --port 8000'
  PYTHONPATH=$PWD CENSOR_MODEL_URL=http://127.0.0.1:8100 CENSOR_OFFLINE=1 nohup python3 -m uvicorn censor.server.app:app --host 0.0.0.0 --port 8000 >> logs/app8000.log 2>&1 &
  ```
  Try a UI change in stub mode on another port (`PORT=8001 ./run.sh stub`), so the demo app on :8000 keeps its detectors.

**Using the app** (one workspace: Request and Export on the left, two monitors in the middle, Agent on the right, Timeline along the bottom):
1. **Home screen:** drop a video on it (or browse files), or pick a sample from the list (the clips in `bench/clips`, with their length and resolution; Claude's in-app browser can't open a file picker, so samples are the reliable way). Either opens the workspace on that clip. **Open clip** in the workspace header comes back here; `?video=<id>` skips it. In stub mode, **Attach .tracks.json** adds ground-truth tracks to the open clip, or to the next upload if none is open.
2. **Request:** type what to hide, e.g. `blur all faces and plates`, `blur the woman`, `blur everyone except the guy in the red hoodie` or `pixelate all faces but mine, I'm the one in the black t-shirt` (the **Try** list fills the box), then press Enter or **Generate masks** (Shift+Enter for a new line). Plain requests are parsed instantly; the rest take one LLM call for the plan (about 2 s) plus a VLM check per person (a few seconds). The first request on a new clip waits for the background scan (about 30 s for a short clip).
3. **Agent** lists what the agent did in plain words, with the yes / no / unsure tally of the vision model's answers and the raw **Tool calls**. Under it, every target with its in and out times (on a crowd, one row per kind, e.g. "14 license plates · 113 faces"); targets the agent spared (a yes in an "except" request) are listed as spared. **Check** steps through the low-confidence spots: moments the detector wasn't sure about, masked anyway with extra padding. **Undo** reverts the latest change as a whole.
4. **The monitors** follow one playhead. Only **Program** shows until the clip first plays; then **Source** appears beside it and stays, unless the **Source** switch next to the timecode is off (Draw box always shows it). **Source** shows the original frame with a labeled box per target: solid if masked, dashed if spared or found but not masked. **Program** shows the frame as Export will write it (the renderer's own rectangles from `GET /api/videos/<id>/masks`, blurred or pixelated the same way); after an Export it plays the exported file itself, until the next edit. Space plays and pauses; ← and → step one frame.
5. **Fixing a target:** click its box on Source, its row, or its timeline lane to select it, then pick **Blur**, **Pixelate** or **Box**, or **Remove** (Delete key); an unmasked one gets **Mask it with** instead. **In the wrong place?** Drag the selected box (or a corner) on Source: from that frame on, a new track follows the corrected box, and before it the old one stays. If the new track loses the object, the old mask comes back from there (fail closed). **Draw box** (above Source), then drag, blurs anything by hand; it gets tracked through the clip. Each fix is one version, so Undo reverts it.
6. **Timeline:** one lane per target (masked spans, spared tracks as a thin line, low-confidence ticks). Click or drag on a lane or the ruler to move the playhead; pressing on a lane also selects that target.
7. **Export:** renders the clip, then shows the frame count and **Download**. Export waits for a request that is still running and refuses when nothing is masked. `?video=<id>` in the URL reopens a video later.

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
Clips are listed in `bench/clips/manifest.yaml` (the videos are gitignored; `bash scripts/fetch_clips.sh` downloads the three Pexels clips), labels live in `bench/labels/<clip>.json` (every 10th frame) and test requests in `bench/requests.yaml`. Options: `python -m bench.run_bench --help`. The stub detector replays ground truth, so on the synthetic clip the report checks the plumbing, not detection quality.

**What's labeled.** All three real clips, every 10th frame, every frame confirmed by a person (only confirmed frames are scored). Each clip's `notes` in the manifest say who each label id is and what isn't labeled; the `desc` of each object in the label file says the same.

| Clip | Frames | Faces | Persons | Plates |
|---|---|---|---|---|
| `street_walk` | 23 | 3 people in focus | the same 3 | 5 |
| `street_plates` | 33 | the 13 largest only, not the whole crowd | not labeled | 6 |
| `street_talk` | 52 | 3 | the same 3 | none (side-on, unreadable) |

Read before quoting a number: on `street_walk` the face labels are face boxes (forehead to chin, ear to ear, redrawn Sun from OpenCV YuNet and checked by eye); on `street_talk` and `street_plates` they are still **head** boxes (hair to chin), while EgoBlur masks the face, so a fully masked face there can land just under the 90% coverage that counts as covered and face recall reads low (IMPLEMENTATION.md open item 3). On `street_plates`, person recall and over-blur mean nothing, because persons aren't labeled there. How the labels were made: IMPLEMENTATION.md open item 7.

**The request success rate (target: at least 15 of 20).** `bench/requests.yaml` has 7 requests on the synthetic clip and 20 on the real clips (7 / 6 / 7 on walk / plates / talk), each naming the label ids that must end up masked (`must_mask`) and must stay visible (`must_not_mask`), with an optional time `window` and `before` requests applied first (for "unblur" and "undo"). Score the 20 on the real clips only, with the model server running, and keep the tracked report files untouched while trying things:
```bash
python -m bench.run_bench --clip street_walk --clip street_plates --clip street_talk --out-dir /tmp/bench
```
Without `--clip`, the synthetic requests are pooled into the same count, and YOLOE finds nothing in the synthetic clip. Without the model server, the two "everyone except ..." requests read as "only ..." and fail. Two more can fail for reasons in the app, not the labels: "unblur the woman" after "blur everyone" (on unblur, an unsure vision answer unmasks that person) and "undo" on `street_plates` if its "blur all plates" step finds nothing.

**Reviewing detections and masks (any CUDA GPU, including a laptop):** `bench/review.py` caches EgoBlur on every frame of a clip, then replays the app's track rules in milliseconds. `python -m bench.review detect <clip>` (3 min for a 13 s 1080p clip on an RTX 3050 Ti), `candidates <clip> --cat plate --min-len 3` writes crop sheets of every candidate track to `data/review/<clip>/`, `label <clip> --cat plate --real 0 36 ...` saves the real ones to `bench/labels/<clip>.plate.json`, `score <clip> --cat plate` gives per-plate masking and over-blur, and `frames <clip> --frames 0 60 120` draws detections and masks. On a Windows laptop with an NVIDIA GPU: `uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu126`, and put the two EgoBlur `.jit` files in `~/datasets/egoblur/`.

**Where the scan time goes (GPU server):** `python3 -m bench.profile_scan bench/clips/street_plates.mp4 [--warm-up] [--cpu-scan]` times decoding, each model per frame on GPU and CPU, and the upload scan; `--cpu-scan` runs the same scan on the CPU alone for the GPU vs CPU comparison. `python3 -m bench.egoblur_speed <clip>` compares fp16, autotuning, smaller inputs and two streams for EgoBlur, with how far each moves the boxes. `python3 -m bench.live_speed <clip> [--heights 720 1080] [--fps 30]` asks whether the detectors could keep up with a livestream: per-frame cost of a face + plate + person pass with its tail (p95/p99), a paced run where the latest frame wins, and the detection stride and delay buffer that follow.

**Face and plate recall without labeling:** `bench/pp4av.py` scores the detector on PP4AV, 3,447 public driving images with face and plate boxes (CC BY-NC-ND 4.0, so the data stays in `~/datasets/pp4av/` outside the repo, one copy per machine). On the GPU server: `python3 -m bench.pp4av download` once (4.5 GB), then `python3 -m bench.pp4av run` writes `bench/report_pp4av.md`. Options for comparing detectors: `--limit`, `--conf`, `--face-prompts`, `--plate-prompts`, `--face-padding`, `--detector egoblur`, `--debug N`, and `--plate-conf 0.1 0.3 0.5` for plate recall against false alarms per image at several thresholds (the report also counts false alarms: detections that overlap no label).

**Adding a real clip**
1. Record 20 to 40 s in landscape, 1080p at 30 fps, H.264 (iPhone: Settings > Camera > Formats > Most Compatible). Only film people who agree to it.
2. Copy it to `bench/clips/<id>.mp4` and add an entry to `bench/clips/manifest.yaml` (source, consent). Share the video file with the team directly; git ignores it.
3. Label it in the browser, no GPU needed. Pick an object, drag its box, press `C` to carry boxes to the next frame, fix them, press `Enter` to confirm:
   ```bash
   python -m bench.label_server bench/clips/<id>.mp4      # http://localhost:8020; Windows: .venv\Scripts\python.exe -m ...
   ```
   Optional head start on the GPU server: `python3 -m bench.gt prelabel bench/clips/<id>.mp4` writes detector drafts. They are not scored until you confirm each frame, so check every frame for objects the detector missed.

   How to label so the scores mean something:
   - **Box what is visible, clamped to the frame.** An object counts as masked when 90% of its box is covered, so a box that includes hidden parts is one a perfect system still misses. For faces, box the face (forehead to chin, ear to ear), not the whole head.
   - **Confirm empty frames too.** A confirmed frame with no boxes says "nothing here"; `Enter` confirms every category on the frame at once.
   - **One id per real object across frames**, and a `desc` that requests can be written from: `man, red hoodie`, `woman, salmon tee`; faces `face, man, red hoodie`; plates `plate, UK 2189, red taxi` or `plate, unreadable, blue car left`; people never named in requests `background, ...`.
   - **Type the description before clicking `+ Add`** (the field is read and cleared on Add). After picking a category, click the picture before using keys: the dropdown keeps focus and swallows them.
   - **`Del` or `Backspace` deletes the selected box** whenever the cursor isn't in a text field. Deleting a whole object (×) and merging two objects take two clicks.
   - **`C` copies every box from the previous labeled frame, whatever its category**: delete any that left the picture before you confirm.
   - **One open tab per clip, and a fresh tab after restarting the server.** The page saves its whole copy of the labels on `Enter` or `S`, so an old tab overwrites newer work (and two clips of the same size can even overwrite each other).
4. Add requests for the clip to `bench/requests.yaml` (20 across the real clips is the target), then commit the labels, manifest and requests. Name ids from the saved file, not from the order you added them (deleting and merging leave gaps). Every id in a request must have a confirmed box (inside the `window`, if there is one), or the whole bench run stops with an error that names it.

## Layout

| Path | What | Owner |
| --- | --- | --- |
| `censor/core/` | `timeline.py` (the contract: ops, fold, undo), `tracks.py`, `video.py` | C (schema), all |
| `censor/vision/` | `detect.py` (YOLOE + stub), `specialist.py` (faces + plates -> EgoBlur router), `track.py` (draw-a-box), `vlm.py` (attribute checks) | A |
| `censor/netguard.py` | egress guard: refuses and logs outbound connections when `CENSOR_OFFLINE=1` | all |
| `censor/agent/` | `tools.py` (the toolbox), `agent.py` (routing), `planner.py` (one LLM call -> JSON plan -> toolbox), `intents.py` (template parser, no LLM) | B |
| `censor/render/` | `render.py` (fail-closed renderer, ffmpeg out) | C |
| `censor/server/` | `app.py` (main API), `model_server.py` (resident Qwen3-VL) | D / B |
| `web/index.html` | the editor (dual-monitor workspace): request, the agent's steps and targets, Source and Program monitors on one playhead, timeline, draw-a-box, hand edits, undo, export. One file, no build step | D |
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
