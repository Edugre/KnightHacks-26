# Stack decisions (M0 results)

Measured Sat Oct 10, 02:20–02:35 ET. **Gate 1 passed.**

**Sun Oct 11: the Dublin hub went down and we moved to the San Jose hub's Strix Halo (gfx1151).** Every measurement below the Strix Halo section is an R9700 (Dublin) number, kept as history; don't quote it as the Strix Halo's speed.

## Strix Halo (gfx1151) on the San Jose hub

Measured Sun Oct 11, ~00:30-01:00 ET. Anything not listed here is not measured yet; don't copy R9700 numbers across.

| Item | Value |
| --- | --- |
| Platform | AUP Learning Cloud San Jose (sjc.aupcloud.io), resource **VSCode Server GPU**, node **AMD Radeon 8060S (Strix Halo iGPU)**, runtime 240 |
| GPU | Radeon 8060S iGPU, **gfx1151**, **64 GB LPDDR5X** shared with the CPU (`rocm-smi` reports 68.7 GB VRAM) |
| CPU, RAM, disk | 32 cores, 62 GB RAM, ~984 GB disk |
| ROCm | 7.13 (hip 7.13.99004) |
| PyTorch | 2.9.1+rocm7.13.0 and torchvision 0.24.0+rocm7.13.0, preinstalled; `scripts/setup.sh` left torch unchanged |
| Python and libraries | Python 3.12.3, Ultralytics 8.4.175, transformers 5.19.0, OpenCV 5.0.0 (headless), ffmpeg 7.0.2 static |
| Session limit | **240 minutes** per launch |
| Clone and ports | `~/KnightHacks-26` (repo URL in the launch form's git repository field); model server on :8100, app on :8000, `https://sjc.aupcloud.io/user/<user>/proxy/8000/` |
| EgoBlur | Re-upload `ego_blur_face_gen1.jit` and `ego_blur_lp_gen1.jit` to `~/datasets/egoblur/` on every fresh server: Meta's CDN refuses AUP hosts (upload gotchas: README, "If the Dublin hub is down") |
| `scripts/setup.sh` | Ran clean once the MobileCLIP download completed. The first run was cut off mid-download by a browser disconnect; rerunning fixed it (the script is idempotent) |
| `./run.sh test` | 114 passed, 1 failed: `tests/test_e2e_stub.py::test_sample_clips_load_without_a_file_picker` (OSError opening a copied clip under pytest's temp dir). **Fixed Sun (Alex):** reopening a video took the first `source.*` in its folder, and on this Linux filesystem that can be the stub sidecar `source.mp4.tracks.json`; the same bug made loading **synthetic** in the app fail with a 500. Test: `test_reopening_a_video_skips_its_sidecar` |
| Offline | Model server `/health` says `"offline": true`; `logs/egress.log` was never created, so nothing tried to connect out |
| Not an option | Radeon 890M node: 4 GB can't hold Qwen3-VL-8B |

| Measurement (Strix Halo) | Result |
| --- | --- |
| Qwen3-VL-8B model server load | **8 s** (`/health`: `load_s` 8, `gpu_mem_gb` 17.5). Probably storage, not the GPU: don't claim the GPU is faster for this |
| YOLOE + EgoBlur warm-up at app start | **21.3 s** ("detectors ready in 21.3 s") |
| GPU memory, model server + app | ~18.7 GB right after model-server start, ~20.3 GB with EgoBlur loaded and a request run |
| `./run.sh bench`, `street_walk` (1280x720, 221 frames) | detect **105 ms/frame**, render **3.7 ms/frame**, offline speed **0.4x real time** (clean rerun Sun 07:46 UTC, commit 956d0bd, GPU otherwise idle; 104.6 ms, the 07:00 run gave 105.2). Detect is the face + plate + person scan. The earlier 45.4 ms / 0.8x (05:00 run) scanned faces and people only, because `street_walk` had no plate labels yet, so it skipped one EgoBlur pass: not comparable |
| `./run.sh bench`, `synthetic` (1280x720, 180 frames) | detect 140.5 ms/frame, render 3.7 ms/frame, 0.2x; agent 1.07 s mean / 7.52 s max per request. Not the product's speed: detect time here is dominated by test setup |
| `./run.sh bench`, GPU memory | detector (bench process) 1.25 GB, model server 17.6 GB, **18.9 GB peak** with all models loaded (target <= 28 GB: pass) |
| `./run.sh bench`, fail-closed on uncertain keyframes | 100% (pass) |
| YOLOE-11s-seg at 640 px (`scripts/smoke_yoloe.py`) | not measured |
| VLM yes/no latency (`scripts/smoke_vlm.py`) | not measured |
| EgoBlur per model, 1080p frame | **130 ms face, 130 ms plate** (p50, fp16 autocast as the app runs it; `bench.live_speed`, 200 frames of `street_plates`, GPU otherwise idle); 58 ms each at 720p. About 2.7x the R9700's 49 ms |
| EgoBlur fp16 vs fp32, 1080p (`bench.egoblur_speed`, 12 frames of `street_plates`, GPU otherwise idle) | **fp16 helps here too: 0.376 s -> 0.129 s per model per frame (2.9x)**, as on the R9700 (3x). Same boxes: faces mean IoU 0.999, none of 92 lost; plates 0.977, 1 of 55 lost. MIOpen autotuning changes nothing (0.136 s fp16); two streams for face + plate are slower than one after the other (0.83 vs 0.78 s, fp32); smaller inputs lose boxes (0.75x: 14 faces and 29 plates lost) |
| Livestream: one face + plate + person pass per frame (`bench.live_speed --frames 200`, `street_plates`, 30 fps, Sun 07:15 UTC, `rocm-smi --showuse` 0% before the run) | 720p: **134 ms** p50, 136 p95, 500 p99, 743 max (a few single slow frames: face 441 ms, plate 426 ms max); 1080p: **284 ms** p50, 287 p95, 288 p99. EgoBlur is 87-92% of a pass, YOLOE person 18 / 23 ms. Planned: detect every **5 / 9 frames**, delay buffer **0.77 / 0.69 s** (720p / 1080p; 720p's is longer because of its p99 tail). Paced run (latest frame wins, 20 s): 24% / 12% of frames detected, largest gap 5 / 9 frames (within HOLD_FRAMES 15), frame age when detections are ready 161 / 302 ms p50. The two parts agree: the paced run's mean gap implies 141 / 285 ms a pass. The first run (60 frames, 277 / 531 ms p50) was about 2x slower, most likely from sharing the GPU: `run_bench`, a full app on port 8001 and the model server were all on it around then, and it read 96% busy shortly after |
| Upload scan, `street_plates` 1080p (`bench.profile_scan`) | not measured |
| Plate recall and over-blur on the labeled clips (`bench.review`, Sun 07:40 UTC, fresh detection cache) | `street_plates` (1080p): **100%** of plate frames masked, 0 s longest visible, over-blur **2.5%** mean / 10.8% worst frame (R9700: 100%, 0 s, 3.0% / 10%). `street_walk` (720p): **93.7%**, 0.16 s longest visible, 0.06% / 0.2% (R9700: 94.6%, 0.16 s; 98.2% on a laptop GPU after the tiny-plate recheck). The gap is one 11x7 px plate seen on 4 frames (1.00-1.12 s): the recheck replays here but doesn't rescue it on this GPU. Reproduce: `python -m bench.review detect <clip>`, then `score <clip> --cat plate` |
| Request success rate, real clips (`./run.sh bench --clip street_walk --clip street_plates --clip street_talk`, Sun 05:44 UTC, 7 min 49 s) | **14 / 20** after correcting 28 verified `street_plates` label errors (Sun 08:17 UTC, commit d5f0ada + the corrected labels; keyword baseline 7, fail-closed off 11; recall faces 0.947, plates 0.98, people 0.99; peak 19.1 GB; the corrections are listed in `bench/clips/manifest.yaml`). Before that, **13 / 20** on the original labels (Sun 07:00 UTC, commit 62955f4, reproduced exactly at 07:46 on 956d0bd; 5 min 43 s; keyword baseline 6, fail-closed off 11; recall faces 0.92, plates 0.95; over-blur 14.9%), after 11 / 20 at f3409e6 (keyword 5, fail-closed off 9) and 9 / 20 on the first run (keyword 4, fail-closed off 8): the "except the woman" tie fix and the redrawn `street_walk` face labels. Recall faces 0.92, plates 0.95, persons 0.99; over-blur 15.1%; peak GPU memory 19.3 GB; detect 137 / 100 ms per 1080p frame (`street_talk` / `street_plates`), 106 ms at 720p; offline speed 0.3-0.4x. Why the 11 failed: IMPLEMENTATION.md open item 3 |
| gfx1151 stability under load, attention settings (PLAN.md concerns 1-2) | no crash seen in the runs above; not stress-tested |

**The bench report's quality numbers don't measure the product.** The run (bench/report.md generated 2026-10-11T05:00:59Z on the Strix Halo; the report wasn't committed) scored only `synthetic` and `street_walk`: `street_talk` and `street_plates` were skipped because `run_bench` expects `bench/labels/<clip>.json`. All 7 requests it ran are written for the synthetic clip, whose drawn shapes real detectors can't find, so its face recall, plate recall, over-blur and request success are not results. That run predates PR #18, which added labels for all three clips and 20 real-clip requests: rerun with `python -m bench.run_bench --clip street_walk --clip street_plates --clip street_talk`. The R9700 plate numbers (100% / 94.6% masked, 3% over-blur) came from `bench/review.py`, not `run_bench`.

**R9700 vs Strix Halo, figures measured on both:**

| | R9700 (Dublin) | Strix Halo (San Jose) |
| --- | --- | --- |
| Qwen3-VL-8B load | 211 s to about 9 min (NFS storage) | 8 s (likely storage, not GPU) |
| Model server GPU memory (`/health`) | 17.7 GB | 17.5 GB |
| Peak GPU memory, all models loaded | 17.9 GB (M0 smoke test) | 18.9 GB (`./run.sh bench`) |
| Detector warm-up at app start (YOLOE + EgoBlur) | 18.5 s | 21.3 s |
| YOLOE fps, EgoBlur per frame, VLM latency, upload scan, plate recall | measured (below) | not measured |

## Hardware and environment (R9700, Dublin)

| Item | Value |
| --- | --- |
| Platform | AUP Learning Cloud Dublin (dub.aupcloud.io), resource **VSCode Server GPU** |
| GPU | AMD Radeon AI PRO R9700: RDNA 4, **gfx1201**, 64 CUs, **32 GB GDDR6** |
| ROCm | 7.13 (7.13.99004-3309c6114a) |
| PyTorch | 2.9.1+rocm7.13.0, preinstalled, GPU visible (`torch.cuda.is_available() == True`) |
| Python | 3.12.3 (system). No `venv` module, so packages go in with `pip --user` |
| Home | `/home/jovyan` is a per-user NFS claim (1.8 TB volume, ~224 GB free). Persistence across relaunch: confirm with `cat ~/.persist-test` |
| Session limit | **240 minutes max** per launch, about 4 credits/min. Relaunch from the hub when it expires |
| Tools | git and uv present; ffmpeg 7.0.2 static installed into `~/bin`; no conda, no llama.cpp |

**Why the R9700 and not the Strix Halo option (Sat; superseded Sun when Dublin went down):** gfx1201 is officially supported by ROCm and has much faster memory. That avoids the gfx1151 segfault and Flash Attention risks from PLAN.md (concerns 1–2). 32 GB holds every model we need. The hub's Strix Halo option (gfx1151, 64 GB) is still available if judges want it; everything below is plain PyTorch, so it should port.

## Models and measured performance (R9700 (Dublin) numbers)

| Role | Model | Backend | Result |
| --- | --- | --- | --- |
| Detector | YOLOE-11s-seg (`yoloe-11s-seg.pt`), text prompts via MobileCLIP | Ultralytics on PyTorch ROCm | **191 fps / 5.2 ms per frame** at 640 px (1280×720 input), **0.26 GB** GPU memory. Bus test image: 5× person (0.92–0.53) |
| VLM attribute check | Qwen3-VL-8B-Instruct, bf16 | transformers on PyTorch ROCm | **311 ms** per yes/no question on one image. Answer correct |
| Agent LLM | Same Qwen3-VL-8B model (shared) | transformers | **28 tok/s** greedy. Produced valid JSON for a tool call |
| Memory, VLM + agent | | | 17.5 GB loaded, **17.9 GB peak**. About 14 GB headroom on the card |
| Face + plate detector (added Sat 05:25) | EgoBlur Gen1 face and plate (`~/datasets/egoblur`, Apache 2.0) | TorchScript on PyTorch ROCm, fp16 autocast | **0.049 s per model per 1080p frame** in fp16 (0.147 s in fp32, 1.5 s on the 64-core CPU); PP4AV recall 0.74 faces / 0.65 plates (conf 0.1, face padding 0.5) vs YOLOE's 0.00 / 0.07, the same in fp16 |
| Upload scan (Sat 07:50) | EgoBlur face + plate every 3rd frame, YOLOE person every frame | | **17.2 s for a 13 s 1080p clip (0.76x real time)**, 48.1 s before; 720p clip 1.12x real time on the GPU vs 0.05x on the CPU alone (24x). `bench/profile_scan.py` |

## Re-check (Sat Oct 10, ~03:15 ET, commit ccfc82f, R9700)

Run on the shared session while the model server and app were already up (GPU memory at 56% before the checks).

| Check | Result |
| --- | --- |
| `bash scripts/setup.sh` (adds `lap`) | Passed; torch unchanged at 2.9.1+rocm7.13.0 |
| `./run.sh test` | 19 passed in 4.8 s |
| `smoke_yoloe.py` at 640 px | **147.5 fps / 6.8 ms per frame**, 0.17 GB peak. Slower than M0's 191 fps, probably because the GPU was shared with the resident model server |
| Model server `/health` | Qwen3-VL-8B resident, 17.7 GB |
| One request through the API, real models, synthetic clip | **75 s** for "blur every person and any license plate"; found nothing (expected on drawn shapes); export fine. Fixed at 05:30: 0.03 s (IMPLEMENTATION.md M3) |

## Decisions

1. **Detection runs on every frame, at 1280 px for small targets.** At 5 ms/frame there's no need to skip frames. Larger input helps distant plates and faces. Re-measure fps at 1280.
2. **One Qwen3-VL-8B process serves both the agent and the VLM checks.** It loads once (211 s from NFS) and stays resident behind a small local HTTP API. Never load it per request.
3. **No llama.cpp for now.** 28 tok/s is enough: the agent makes one call per request that writes the whole plan as JSON (about 1.8 s), not one call per tool step. Revisit only if latency hurts the demo: llama.cpp with a Q4/Q8 GGUF is the fallback. No cloud LLM either: the point is that nothing leaves the machine.
4. **Constrain the agent's output.** In the smoke test the model called `add_masks` with an empty `track_ids` instead of calling `find_objects` first. Since Sat 09:50 the LLM no longer calls tools at all: it writes one JSON plan, `planner.parse_plan` checks it (actions, effects, object names, a kind of person moved into a question), and plain code runs it. See IMPLEMENTATION.md section 4.
5. **Don't let pip touch torch.** Always install with `--user --break-system-packages`, and use `--no-deps` for anything that pins torch (CLIP). Check with `python3 -c "import torch; print(torch.__version__)"`, which must print `+rocm7.13`.
6. **Download at setup, run offline.** Without flags, Ultralytics looks up two DNS hosts on every import (`is_online()`), sends usage analytics when that succeeds, and downloads missing weights into the working directory; transformers checks the Hugging Face Hub when loading. `scripts/setup.sh` now fetches Qwen3-VL (Hugging Face cache) and YOLOE + MobileCLIP (into `<repo>/weights`: Ultralytics' `weights_dir` is the relative path `weights`, resolved from the working directory, and run.sh runs from the repo root), and `./run.sh` sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `YOLO_OFFLINE` and `CENSOR_OFFLINE` (egress guard, `censor/netguard.py`).
7. **Faces and plates go to EgoBlur, everything else to YOLOE** (`censor/vision/specialist.py`). YOLOE is an open-vocabulary object detector and misses faces entirely; EgoBlur is built for exactly these two categories. EgoBlur runs every 3rd frame and a simple IoU tracker links detections.
8. **Keep models loaded and scan at upload.** YOLOE stays loaded between searches, and each upload starts a background scan for faces, plates and people, so requests read cached tracks.
9. **EgoBlur runs in fp16 and the detectors warm up at startup** (`CENSOR_EGOBLUR_FP16`, `CENSOR_WARMUP`). The scan is GPU-bound (decoding is 0.2 s of it), so the per-call cost is what counts: fp16 cut it 3x with the same boxes. EgoBlur can't batch, two streams don't overlap, and smaller inputs lose plates, so those are out (`bench/egoblur_speed.py`). A freshly loaded model is slow for its first calls, so the app runs each model on noise frames in the background when it starts (18.5 s); the first upload then scans at the warm speed.
12. **The VLM sees one input shape and reports P(yes).** On ROCm each new convolution shape costs a MIOpen kernel search (its database isn't readable on the AUP image), so crops go into a fixed 256x512 canvas and the model server sets `MIOPEN_FIND_MODE=FAST`; a yes/no check on three crops takes about 0.3 s. The answer is the first token's probability of yes vs no, not parsed text.
11. **Track rules are tuned on labeled clips, not guessed** (`bench/review.py`, IMPLEMENTATION.md "Where we are" item 4). A plate track needs two confident detections, a big face needs one at 0.8, small fast objects link by distance, and masks start 3 frames early and follow the object's speed after a track ends.
10. **The blur smooths at 1/24 size.** Rendering is CPU work (OpenCV per box, libx264); blurring at full size cost 17 ms per face and 193 s for a crowd clip, smoothing the downscaled box costs 0.2 ms and the same export takes 2.8 s.

## Known noise (harmless)

- `MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable ... gfx1201_32.HIP.fdb.txt` (R9700; expect the gfx1151 name on Strix Halo): no prebuilt tuning DB, so the first run of a new layer shape is slightly slower.
- `warning: xnack 'Off' was requested for a processor that does not support it!`: compiler noise.
- `WARNING NMS time limit 2.050s exceeded` from Ultralytics on the first frame: warm-up only.

## Setup on a fresh session

```bash
bash scripts/setup.sh          # idempotent; installs into ~/.local and ~/bin
python3 scripts/smoke_yoloe.py # detector check
python3 scripts/smoke_vlm.py   # VLM + agent check (first run downloads ~17 GB)
python3 scripts/check_offline.py [clip.mp4]   # every model loads with the network off (safe while the model server runs)
```
`smoke_vlm.py` loads its own 18 GB copy of the model, so skip it while the resident model server is running; check `curl -s localhost:8100/health` instead.

Model load time from the NFS home varies with load on the storage (R9700, Dublin): 211 s at M0, 311 s and about 9 minutes on two restarts on Sat morning. Plan demo restarts with that in mind.
