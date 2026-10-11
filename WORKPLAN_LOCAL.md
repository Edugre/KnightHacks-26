# Local work plan while the GPU server is down (temporary)

*Sat Oct 10, 19:10 ET. Delete this file, or fold what's left into IMPLEMENTATION.md, once the AUP server is back.*

The AUP R9700 host is down, so nobody can run the full app with real models for now. This plan splits the open work into four tasks that each run on a laptop and touch different files, so 3-4 people can work at the same time without merge conflicts. **Each worker takes one task below (Worker 1 to 4) and stays in its files.** With three people, Worker 4's task goes to whoever finishes first. Current state and history: IMPLEMENTATION.md, "Where we are".

## What runs on a laptop and what doesn't

| Works locally | How |
| --- | --- |
| The whole app (upload, request, preview, draw-a-box, undo, export) with the **stub detector**, which replays ground-truth boxes | `powershell -ExecutionPolicy Bypass -File .\run.ps1 stub 8010`, then http://localhost:8010 and pick **synthetic** |
| The test suite (92 tests, ~25 s) | `.\run.ps1 test` (Windows) or `./run.sh test` |
| Request parsing (template), plan checking and plan execution | Unit tests in `tests/test_core.py` and `tests/test_planner.py`, with a fake LLM reply |
| The labeling tool | `python -m bench.label_server bench/clips/<clip>.mp4 --port 8020` |
| **EgoBlur faces and plates** on an NVIDIA GPU, or slowly on CPU (~1.5 s a frame) | `bench/review.py` (README, "Reviewing detections and masks") |
| **YOLOE** (people and other objects): not tried on a laptop yet; Ultralytics runs on CPU or CUDA | `uv pip install --python .venv\Scripts\python.exe ultralytics lap`; weights download on first use (needs internet) |

| Does **not** run locally | Effect |
| --- | --- |
| Qwen3-VL-8B (the model server: needs ~18 GB of GPU memory) | No planner LLM: requests the template can't parse ("everyone except ...") answer "clarify". No VLM attribute checks: every answer is "unknown", so attribute requests blur every candidate (fail closed). Test planner logic with fake LLM replies, as `tests/test_planner.py` does. |
| R9700 timings | Don't quote laptop speeds as the product's speed. |

**Model files:** EgoBlur (Apache 2.0) isn't in git. The zips `ego_blur_face_gen1.zip` and `ego_blur_lp_gen1.zip` (390 MB each) were shared with the team; unzip them into `~/datasets/egoblur/` (`%USERPROFILE%\datasets\egoblur\` on Windows). CUDA PyTorch on Windows: `uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

**Clips:** `bench/clips/` is gitignored. Get the three Pexels clips with `bash scripts/fetch_clips.sh` (needs internet; `street_plates` is trimmed to 13 s, see `bench/clips/manifest.yaml`) or copy them from Alex.

## Ground rules

- One branch per task: `D/<task>` (e.g. `D/person-stitch`). Commit as yourself. Run the tests before every push. Open a PR to `main`; keep `main` green.
- Stay inside your task's files (table below). If you must change a shared contract (the timeline schema, tool names or arguments in `censor/agent/tools.py`, the `/api` routes), say so in the team chat first.
- When a task is done, update its entry in IMPLEMENTATION.md ("Where we are", "Open, in order") with what changed and the numbers, as the entries above it do.

## The four tasks

### Worker 1: stitch fragmented person tracks (vision) — done Sat 22:00 (`censor/vision/stitch.py`; results in IMPLEMENTATION.md open item 1)

**Problem:** on `street_walk` the person detector gives 18 tracks for about 3 people in focus; 15 are short scraps of blurry background pedestrians, because the tracker loses someone for a few frames and starts a "new" person. Every request asks the VLM about every track (~0.3 s each), and scraps get inconsistent answers. (IMPLEMENTATION.md, open item 1.)

**Work:**
1. Get YOLOE running locally (install line above) and cache person detections per frame, the way `bench/review.py` caches EgoBlur (add a `--cat person` path, or a small `bench/review_person.py`). Use **YOLOE-11**: YOLOE-26 was tested and misses real people (DETECTOR_OPTIONS.md item 2). `bench/compare_people.py` already counts tracks and scraps per clip and makes crop sheets.
2. Write the stitcher in a new file `censor/vision/stitch.py`: join a track that ends to one that starts within N frames, near where the first was heading, with a similar box size (and optionally similar colors in the crop). Unit tests in `tests/test_stitch.py`.
3. Call it at the end of `YoloeDetector._run` in `censor/vision/detect.py`.
4. Measure on `street_walk`, `street_talk` and `street_plates`: tracks before and after, and check by eye (crop sheets) that no two different people got merged. Merging two people is worse than leaving a scrap: it would make one VLM answer apply to both.

**Done when:** `street_walk` drops from 18 person tracks to about 3-6 with no wrong merges, the tests pass, and the numbers are in IMPLEMENTATION.md.
**Files:** `censor/vision/stitch.py` (new), `censor/vision/detect.py`, `bench/review*.py`, `tests/test_stitch.py`.
**Needs:** an NVIDIA GPU helps (the EgoBlur zips and CUDA PyTorch); CPU works but is slow (~70 ms a frame for YOLOE).

### Worker 2: request timing logs and template coverage (agent)

**Problem:** one request once took 59 s for no known reason (open item 2), and every request the template can't parse costs an LLM call (~2 s on the server, impossible offline).

**Work:**
1. Log how long each step of a request takes, one line per request in the app log: waiting for the scan, each tool call, each VLM call, the planner call. `censor/agent/agent.py` already times tool calls (`calls[].ms`); add the scan wait in `Session._ensure_scanned` (`censor/agent/tools.py`) and print a summary line in `censor/server/app.py`.
2. Widen the template parser (`censor/agent/intents.py`) so more plain requests skip the LLM: synonyms ("hide", "cover", "mask out"), "cars" and other objects, "everything", plurals, "between 0:05 and 0:10". Add a parametrized test per phrasing in `tests/test_core.py`.
3. Collect 30+ real-sounding requests (team, friends) in `tests/test_planner.py` as expected plans, so the planner's executor can be tested now with fake replies, and the prompt checked against the real LLM when the server is back.

**Done when:** the app log shows per-step timings, the template handles the new phrasings with tests, and the request list exists.
**Files:** `censor/agent/intents.py`, `censor/agent/agent.py`, `censor/agent/planner.py`, `censor/server/app.py` (logging only), `tests/test_core.py`, `tests/test_planner.py`.
**Needs:** no GPU.

### Worker 3: labels and the 20 benchmark requests (bench) — done Sun (Pablo, PR #18): all three clips labeled, 20 real-clip requests; how to score them: README "Benchmark", details and caveats: IMPLEMENTATION.md open items 3 and 7

**Problem:** the request success rate (target: at least 15 of 20) can't be measured: `bench/requests.yaml` has 7 requests, all on the synthetic clip, and the real clips only have plate labels (`bench/labels/street_plates.plate.json`, `street_walk.plate.json`).

**Work:**
1. **Faces first** on `street_walk` and `street_plates`: without face labels nobody can tell whether EgoBlur Gen2 (or SCRFD) finds faces better, only whether they disagree. Then label people (with a short description each: "woman, rust top", "man, red hoodie") and plates, on every 10th frame of `street_walk`, `street_talk` and `street_plates` with `bench/label_server.py` (format: `bench/gt.py`; on `street_plates`, the people in focus are enough).
2. Write 13+ more requests in `bench/requests.yaml` over the real clips, each with `must_mask` / `must_not_mask` by label id: plain ones ("blur all plates"), attribute ones ("blur the woman"), exceptions ("everyone except the guy in the red hoodie"), time windows, unblur, undo.
3. Check the file loads: `python -m bench.run_bench --help`, then a stub run.

**Done when:** 20+ requests over real clips with labels; scoring runs when the server is back (the LLM and VLM parts need it).
**Files:** `bench/labels/*.json`, `bench/requests.yaml`, `bench/clips/manifest.yaml`.
**Needs:** no GPU (plate drafts via `bench/review.py` need EgoBlur).

### Worker 4: demo UI, demo script, Devpost (web + docs) — done (PR #12); follow-ups merged: video preview with box and mask layers, hand edits (mask / unmask / restyle / move) (#13, #15, #16; IMPLEMENTATION.md open item 4)

**Problem:** the UI is a test page (`web/index.html`): no timeline, and the agent's work is only visible as raw JSON in the log.

**Work:**
1. **Timeline strip** under the player: one row per masked target (label + effect) with a bar for when it's masked, from `GET /api/videos/<id>/timeline`; click a bar to seek there.
2. **Show what the agent did** in plain words: the plan, each question asked ("Is this a woman?") with how many tracks said yes / no / unsure, and what got masked. The data is already in the request response (`calls`).
3. Small polish: busy states, error messages, a "what can I ask?" hint with example requests.
4. `docs/DEMO.md`: a 3-minute demo script (which clip, which requests, what to point out: offline egress log, R9700 numbers from IMPLEMENTATION.md) and a backup plan if live fails.
5. `docs/DEVPOST.md`: the write-up draft (problem, how it works, AMD/ROCm story with the measured numbers, what's next).

**Done when:** the timeline and agent panel work in stub mode on the synthetic clip, and both docs exist.
**Files:** `web/index.html`, `docs/DEMO.md` (new), `docs/DEVPOST.md` (new).
**Needs:** no GPU; front-end work.

### Extra, for whoever finishes first

- ~~The taxi's sticker blurred as a plate~~ and ~~the 11x7 px plate left visible~~: fixed Sat 22:15 (IMPLEMENTATION.md, open item 5).
- ~~YOLOE `half=True` for the person scan~~ done Sat evening: slightly worse, keep fp32. ~~YOLOE-26~~ tried: misses real people in the night crowd, dropped (DETECTOR_OPTIONS.md item 2, `bench/compare_people.py`).
- ~~EgoBlur Gen2~~ tried: worse than Gen1 for plates on both labeled clips (raw recall 0.94 / 0.89 vs 0.99 / 1.00), the same for faces. Keep Gen1 (DETECTOR_OPTIONS.md item 1).
- ~~Fresh-clone test~~ done Sat 23:30 on Windows: clone, README setup (8 s), `./run.ps1 test` (99 pass), `./run.ps1 stub`, then load **synthetic**, a request and an export in the page all work.
- ~~Backup exports~~ done Sat 23:30: `scripts/make_backup.py`, files on the team drive (docs/DEMO.md, "If live fails"). The screen recording of the full script still needs the R9700.
- Anyone free: make a Hugging Face account and request access to Meta's SAM 3 weights (gated), so SAM 3 can be tried as soon as the server is back (DETECTOR_OPTIONS.md item 3).

## Who touches what

| Files | Owner |
| --- | --- |
| `censor/vision/*` | Worker 1 |
| `censor/agent/*`, `censor/server/app.py` (logging) | Worker 2 |
| `bench/labels/*`, `bench/requests.yaml`, `bench/clips/manifest.yaml` | Worker 3 |
| `bench/review*.py` | Worker 1 (and Extra) |
| `web/index.html`, `docs/DEMO.md`, `docs/DEVPOST.md` | Worker 4 |
| `IMPLEMENTATION.md` | everyone, own entry only |
| `censor/core/*`, `censor/render/*`, `schema/*` | nobody without asking |

## When the server is back

1. Relaunch (VSCode Server GPU, node "AMD Radeon AI Pro R9700"), then `cd ~/kh-alex && git pull && ./run.sh`.
2. Re-measure Worker 1's stitching on the R9700 (person tracks per clip, request time for "blur the woman").
3. Run the benchmark with the real models: `./run.sh bench`, and record the request success rate.
4. Try Worker 2's request list against the real planner; add any phrasing it gets wrong as a test.
5. Delete this file, or move what's left into IMPLEMENTATION.md.
