# Stack decisions (M0 results)

Measured Sat Oct 10, 02:20–02:35 ET. **Gate 1 passed.**

## Hardware and environment

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

**Why the R9700 and not the Strix Halo option:** gfx1201 is officially supported by ROCm and has much faster memory. That avoids the gfx1151 segfault and Flash Attention risks from PLAN.md (concerns 1–2). 32 GB holds every model we need. The hub's Strix Halo option (gfx1151, 64 GB) is still available if judges want it; everything below is plain PyTorch, so it should port.

## Models and measured performance

| Role | Model | Backend | Result |
| --- | --- | --- | --- |
| Detector | YOLOE-11s-seg (`yoloe-11s-seg.pt`), text prompts via MobileCLIP | Ultralytics on PyTorch ROCm | **191 fps / 5.2 ms per frame** at 640 px (1280×720 input), **0.26 GB** GPU memory. Bus test image: 5× person (0.92–0.53) |
| VLM attribute check | Qwen3-VL-8B-Instruct, bf16 | transformers on PyTorch ROCm | **311 ms** per yes/no question on one image. Answer correct |
| Agent LLM | Same Qwen3-VL-8B model (shared) | transformers | **28 tok/s** greedy. Produced valid JSON for a tool call |
| Memory, VLM + agent | | | 17.5 GB loaded, **17.9 GB peak**. About 14 GB headroom on the card |

## Re-check (Sat Oct 10, ~03:15 ET, commit ccfc82f)

Run on the shared session while the model server and app were already up (GPU memory at 56% before the checks).

| Check | Result |
| --- | --- |
| `bash scripts/setup.sh` (adds `lap`) | Passed; torch unchanged at 2.9.1+rocm7.13.0 |
| `./run.sh test` | 19 passed in 4.8 s |
| `smoke_yoloe.py` at 640 px | **147.5 fps / 6.8 ms per frame**, 0.17 GB peak. Slower than M0's 191 fps, probably because the GPU was shared with the resident model server |
| Model server `/health` | Qwen3-VL-8B resident, 17.7 GB |
| One request through the API, real models, synthetic clip | **75 s** for "blur every person and any license plate"; found nothing (expected on drawn shapes); export fine. See IMPLEMENTATION.md M3 status |

## Decisions

1. **Detection runs on every frame, at 1280 px for small targets.** At 5 ms/frame there's no need to skip frames. Larger input helps distant plates and faces. Re-measure fps at 1280.
2. **One Qwen3-VL-8B process serves both the agent and the VLM checks.** It loads once (211 s from NFS) and stays resident behind a small local HTTP API. Never load it per request.
3. **No llama.cpp for now.** 28 tok/s is enough for short tool calls. Revisit only if agent latency hurts the demo: llama.cpp with a Q4/Q8 GGUF is the fallback.
4. **Constrain the agent's output.** In the smoke test the model called `add_masks` with an empty `track_ids` instead of calling `find_objects` first. The agent loop has to feed tool results back and validate arguments (for example, reject empty `track_ids`). See IMPLEMENTATION.md section 4.
5. **Don't let pip touch torch.** Always install with `--user --break-system-packages`, and use `--no-deps` for anything that pins torch (CLIP). Check with `python3 -c "import torch; print(torch.__version__)"`, which must print `+rocm7.13`.

## Known noise (harmless)

- `MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable ... gfx1201_32.HIP.fdb.txt`: no prebuilt tuning DB, so the first run of a new layer shape is slightly slower.
- `warning: xnack 'Off' was requested for a processor that does not support it!`: compiler noise.
- `WARNING NMS time limit 2.050s exceeded` from Ultralytics on the first frame: warm-up only.

## Setup on a fresh session

```bash
bash scripts/setup.sh          # idempotent; installs into ~/.local and ~/bin
python3 scripts/smoke_yoloe.py # detector check
python3 scripts/smoke_vlm.py   # VLM + agent check (first run downloads ~17 GB)
```
`smoke_vlm.py` loads its own 18 GB copy of the model, so skip it while the resident model server is running; check `curl -s localhost:8100/health` instead.
