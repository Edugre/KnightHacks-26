# Detector options on ROCm: research and recommendation

*Sat Oct 10, 2026. Desk research only: **nothing below has been run on the R9700 or any of our clips.** Every speed or accuracy number for a candidate comes from its paper, docs or a third party on other hardware (sources at the end). Treat this as a list of things to measure, not results.*

**Question asked:** is there a better option than YOLOE + EgoBlur, for example one VLM that draws boxes for both faces/plates and "anything"? **Short answer:** no. A VLM is too slow per frame and weak on small objects. The upgrades worth testing keep the same split and swap or add models inside each role.

## How to use this file (for the next agent)

Each candidate has a **Status** line. Before starting one, check it hasn't been done since this was written:
- `git log --all --oneline | grep -i <name>` and `git branch -a` (branches are `role/feature`, e.g. `A/sam3`).
- `grep -rni <name> IMPLEMENTATION.md docs/ bench/report*.md censor/`.
- IMPLEMENTATION.md "Where we are" is where results get recorded.

When you try one, update its **Status** here (date, branch, numbers or why it was dropped) and record the measured result in IMPLEMENTATION.md as the other entries do. At the time of writing, a search of the repo and every branch found none of these candidates tried; the only related note is "YOLOE `half=True` untested for accuracy" (IMPLEMENTATION.md item 5) and the comment in `censor/vision/track.py` suggesting a SAM2-style tracker.

## Baseline we are comparing against (measured, docs/STACK.md and IMPLEMENTATION.md)

| Role | Model | Measured on the R9700 |
| --- | --- | --- |
| Faces + plates | EgoBlur Gen1 (Apache 2.0, TorchScript `.jit`) | 0.049 s per model per 1080p frame in fp16, every 3rd frame; PP4AV recall **0.74 faces / 0.65 plates** (conf 0.1, face padding 0.5). Plate misses are mostly under 16 px |
| Everything else (people, screens...) | YOLOE-11s-seg + BoT-SORT (Ultralytics, AGPL) | 5-7 ms per frame at 640 px; 0.00 faces / 0.07 plates on PP4AV, which is why the router exists |
| Attribute checks + planner | Qwen3-VL-8B, bf16 | ~0.3 s per yes/no check on 3 crops; 17.9 GB peak, ~14 GB free on the card |
| Draw-a-box | OpenCV MIL tracker, CPU | no numbers |
| Whole upload scan | face + plate + person | 17.2 s for a 13 s 1080p clip (0.76x real time) |

Open problems this research is aimed at: plate recall far below the 0.90 target (IMPLEMENTATION.md section 6), face recall below 0.95, and fragmented person tracks (18 tracks for ~3 people on `street_walk`, open item 1).

Constraints any candidate must meet: runs on PyTorch ROCm 7.13 / gfx1201 without replacing the preinstalled torch (install with `--no-deps`); fits in the ~14 GB left next to the resident Qwen3-VL; works offline once downloaded (`censor/netguard.py`); a license we can ship in a public repo.

## Recommendation, in order

| # | What | Effort | Why this order |
| --- | --- | --- | --- |
| 1 | EgoBlur Gen2 on PP4AV | under an hour | Drop-in file swap, same license and format, attacks the plate-recall gap directly |
| 2 | YOLOE-26 in place of YOLOE-11 | small | Same API, published +3 AP on LVIS at the same size |
| 3 | SAM 3 for people and draw-a-box | a day, ROCm risk | Biggest possible gain (tracks with stable ids, strong open vocabulary), but unproven on gfx1201 and possibly slow |
| 4 | SCRFD unioned with EgoBlur for faces | half a day | Raises face recall in the fail-closed direction; license limits it to non-commercial use |
| - | VLM detectors (Qwen3-VL grounding, Rex-Omni, LocateAnything) for scanning | don't | Too slow per frame, weak on small objects. Keep Qwen3-VL for yes/no checks |
| - | Grounding DINO family | don't, unless SAM 3 fails | Lower LVIS AP than SAM 3, slower than YOLOE, custom CUDA kernel |

## Candidates in detail

### 1. EgoBlur Gen2 (faces + plates)

**Status:** not tried.

**What:** Meta's second-generation EgoBlur face and plate models, `ego_blur_face_gen2.jit` and `ego_blur_lp_gen2.jit`, same Apache 2.0 license and TorchScript format as Gen1, same download page (projectaria.com/tools/egoblur, email form; Meta's CDN refuses the AUP server, so download on a laptop as for Gen1).

**Why first:** it's the cheapest test of the biggest gap. `specialist.py` loads files by name (`EGOBLUR_FILES`), so it is a file swap, and `bench/pp4av.py --detector egoblur` already scores it.

**Why it might not help:** Meta publishes no Gen1 vs Gen2 comparison, and Gen2 is aimed at Aria Gen2 head-mounted cameras. Our footage is street video, and PP4AV is driving images, so it could be worse. That's why the step is "measure on PP4AV", not "switch".

**How to test:** put the Gen2 files next to Gen1, point `EGOBLUR_FILES` at them (or add an env var), run `python3 -m bench.pp4av run --detector egoblur --conf 0.1 --face-padding 0.5 --limit 300`, and compare with 0.74 / 0.65. Also run the `--plate-conf` sweep (false alarms per image) and `bench/review.py score <clip> --cat plate` on `street_plates` / `street_walk`, since recall bought with false alarms is not a win. Check speed with `bench/egoblur_speed.py`.

**Adopt if:** plate recall goes up at the same or fewer false alarms per image, face recall doesn't drop, and per-frame cost stays near 0.05 s. A mixed result (better plates, worse faces) is fine too: the router can use Gen2 for one category only.

### 2. YOLOE-26 (open vocabulary)

**Status:** not tried.

**What:** YOLOE built on the YOLO26 architecture (Ultralytics). Published text-prompted LVIS gains over YOLOE-11 of +3.3 / +2.4 / +2.6 AP at the s / m / l sizes, same prompting modes.

**Why:** same Ultralytics API as `YoloeDetector` in `censor/vision/detect.py`, so it is a weight-file and maybe a package-version change. It does nothing for faces or plates (those stay with EgoBlur); it helps people, screens, text and the long tail.

**Risks:** needs a newer `ultralytics`; install it with `--user --break-system-packages --no-deps` and check `torch.__version__` still ends in `+rocm7.13` (docs/STACK.md decision 5). `scripts/setup.sh` must fetch the new weights for offline runs. Ultralytics is AGPL, the same as now.

**How to test:** person tracks per clip and track fragmentation on `street_walk` (18 today), `scripts/overlay_tracks.py` before/after, `scripts/smoke_yoloe.py` for fps. Coordinate with whoever owns person-track stitching (WORKPLAN_LOCAL.md Worker 1): a better detector and the stitcher should be measured separately.

### 3. SAM 3 / SAM 3.1 (people, open vocabulary, draw-a-box)

**Status:** not tried. No ROCm attempt on gfx1201 known.

**What:** Meta's Segment Anything with Concepts (Nov 2025; 3.1 in Mar 2026). Given a short noun phrase ("person", "laptop screen"), it returns every matching instance with a mask and an id, and in video it tracks them all with stable ids. Published: 47.0 AP zero-shot on LVIS (previous best 38.5), roughly 2x OWLv2 on its SA-Co benchmark. ~0.47-0.85B parameters (sources disagree), 3.45 GB checkpoint. Supported in Ultralytics (`SAM3VideoPredictor`); weights are gated on Hugging Face. SAM 3.1 adds faster multi-object tracking (~7x at 128 objects); Ultralytics doesn't support the 3.1 tracker yet.

**Why it's the most promising:**
- **Fragmented person tracks** (open item 1) are a tracker problem. SAM 3 tracks with memory, not IoU/ReID, so one person should stay one id. That would also cut VLM checks per request (one per track, ~0.3 s each).
- **Draw-a-box:** its tracker can follow a user-drawn box, replacing the CPU MIL tracker (`censor/vision/track.py` already anticipates this swap).
- Much stronger open vocabulary than YOLOE for the long tail.
- License: the SAM License allows commercial use, with restrictions (no military/weapons use, trade-controls compliance, redistribute under the same terms, acknowledge in publications). Fine for this project; read it before shipping weights.

**Why it isn't first:**
- **ROCm is unproven for us.** The one ROCm port found (`harrysocool/sam3-tracker-rocm`) is validated only on gfx1151 (Strix Halo), runs through MIGraphX + ONNX Runtime in Docker with ROCm 7.14, not plain PyTorch, at 504 px. We have ROCm 7.13 and no Docker on the AUP image. Plain PyTorch on gfx1201 is untested by anyone we found.
- **Speed:** ~30 ms per image on an H200 (Meta), ~91 ms per frame at 504 px on gfx1151 via MIGraphX (that port), ~1.1 s per image on a P100 and a 2.9 s figure on an RTX PRO 6000 in Ultralytics' docs. A 13 s clip is ~390 frames: anywhere from ~15 s to minutes. YOLOE does it in ~2 s.
- **Not a face/plate replacement:** a third-party benchmark found SAM 3 trailing fine-tuned YOLO specialists by 12-48% on fixed categories. EgoBlur stays.
- Masks are produced but we'd use boxes (`Keyframe` holds a box only); that's fine and keeps the renderer unchanged.

**How to test:** (a) install SAM 3 into the user site with `--no-deps`, load it in bf16/fp16 on the R9700, and time one 1080p frame and a 13 s clip with "person" (watch MIOpen kernel searches on new shapes: fix the input size, `MIOPEN_FIND_MODE=FAST`, as for the VLM). (b) If it runs under ~100 ms per frame, wrap it as a detector behind `detect.get_detector()` for non-face/plate queries and compare person tracks per clip with YOLOE + BoT-SORT (and with Worker 1's stitcher). (c) Memory with Qwen3-VL resident (~14 GB free). If plain PyTorch fails on gfx1201, record the error here and stop; don't sink time into MIGraphX/Docker during the hackathon.

**Adopt if:** it runs on the R9700, people come out as roughly one track each with no merges of different people, and the scan stays near real time (or SAM 3 runs only on demand for queries outside the prescan).

### 4. SCRFD alongside EgoBlur (faces)

**Status:** not tried.

**What:** InsightFace's efficient face detector. Published WIDER FACE hard-set AP 85.3 (SCRFD-34GF) vs 64.2 for RetinaFace at VGA; the hard set is mostly tiny faces, our weak spot. Small variants are fast.

**Why:** take the union of SCRFD and EgoBlur face boxes. More boxes means more recall, and extra blur is the fail-closed direction. A cheap way toward the 0.95 face target.

**Why it's last:** InsightFace's code is MIT, but as far as we know its pretrained models are for non-commercial research only. OK for a hackathon demo, a blocker afterwards; check the current terms before using. Also more false alarms (over-blur), which `bench/pp4av.py` reports.

**How to test:** add `--detector egoblur+scrfd` to `bench/pp4av.py`, compare face recall and over-blur with 0.74 / 2.8%.

### Not recommended (and why, so nobody re-researches them)

- **Qwen3-VL-8B grounding (already resident).** Strong at referring expressions (89.1 RefCOCO average), but it writes each box as text tokens: ~1.1 boxes per second on an H100 per NVIDIA's comparison, and small objects are a known weakness of VLMs (image patches downsample a 15x6 px plate away). Per-frame scanning would take minutes. Possible later use: ground "the guy in red" on a few keyframes and let a tracker carry it, but our current "detect everyone, ask yes/no per track" already works (IMPLEMENTATION.md item 5).
- **Rex-Omni (3B, IDEA, CVPR 2026).** A detection-specialised MLLM: ~7.6 output tokens per box, zero-shot COCO/LVIS comparable to Grounding DINO. Still autoregressive, still per-frame seconds. Same possible keyframe-only use as above.
- **LocateAnything-3B (NVIDIA, Jun 2026).** Parallel box decoding, ~12.7 boxes/s on an H100, ~10x Qwen3-VL. Still far too slow per frame; license terms not checked.
- **Grounding DINO / MM-Grounding-DINO / LLMDet.** MM-Grounding-DINO: 41.4 AP on LVIS minival, below SAM 3's 47.0, and much slower than YOLOE. The original repo needs a custom CUDA kernel for multi-scale deformable attention (Hugging Face transformers has a slower pure PyTorch path). Only worth a look if SAM 3 won't run on ROCm and YOLOE-26 isn't enough.
- **OWLv2.** Permissive license, but lower accuracy than Grounding DINO and SAM 3 and no tracking.
- **RF-DETR (Apache 2.0).** Not open-vocabulary; listed here as the base for the plate fine-tune PLAN.md concern 5 allows (2 hours max) if EgoBlur Gen2 doesn't close the plate gap.
- **A single VLM for everything.** Combines the speed and small-object problems above, and would replace a measured 0.74 / 0.65 detector with an unmeasured one.

## Sources

- SAM 3: [Ultralytics docs](https://docs.ultralytics.com/models/sam-3), [paper](https://arxiv.org/html/2511.16719v1), [Roboflow overview](https://blog.roboflow.com/what-is-sam3/), [ROCm port (gfx1151)](https://github.com/harrysocool/sam3-tracker-rocm), [SAM License text](https://scancode-licensedb.aboutcode.org/sam-2025-11-19.html), [SAM 3 vs specialist models](https://towardsdatascience.com/sam-3-vs-specialist-models-a-performance-benchmark/)
- YOLOE-26: [paper](https://arxiv.org/html/2602.00168), [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe)
- EgoBlur: [GitHub](https://github.com/facebookresearch/EgoBlur), [download page](https://www.projectaria.com/tools/egoblur/)
- SCRFD: [paper](https://arxiv.org/pdf/2105.04714), [README](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/README.md)
- Grounding DINO family: [MM-Grounding-DINO](https://arxiv.org/html/2401.02361v2), [LLMDet](https://arxiv.org/pdf/2501.18954), [overview incl. OWLv2](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/open-vocabulary-detection-grounding-dino-florence-2-rtdetr-rfdetr)
- VLM grounding: [Qwen3-VL report](https://arxiv.org/pdf/2511.21631), [RefCOCO averages](https://benchmarklist.com/benchmarks/refcoco_avg/), [Rex-Omni](https://arxiv.org/pdf/2510.12798), [LocateAnything-3B](https://comfyui-wiki.com/en/news/2026-06-29-nvidia-locateanything-3b-open-source-grounding-model), [speed comparison](https://themenonlab.blog/blog/locateanything-nvidia-parallel-box-decoding)
- RF-DETR license: [Roboflow](https://blog.roboflow.com/rf-detr-is-free-to-use-commercially/)
