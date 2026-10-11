# Devpost write-up (draft)

*Draft. Numbers are from IMPLEMENTATION.md, measured on the AUP Learning Cloud's Radeon AI PRO R9700. Update them if they change before submitting, and fill in the request success rate once `./run.sh bench` has run with the real models.*

## Censor: redaction you can ask for

**Tagline:** Blur people, faces and license plates in video by asking in plain words. It runs on one AMD GPU, and nothing leaves the machine.

## Inspiration

Anyone who posts street footage, bodycam clips, a vlog or a school event has to hide bystanders' faces and plates first. Today that means one of two things: scrubbing through an editor and keyframing boxes by hand, or uploading the footage to a cloud service, which is the opposite of protecting the people in it. We wanted redaction you can ask for in plain language, check frame by frame, and run without a cloud.

## What it does

- **Ask in plain words.** "Blur all plates and the person in the red hoodie around halfway." "Everyone except the woman." "Pixelate faces in the first 10 seconds." "Unblur the plates." "Undo."
- **Or draw a box** on a frame around anything, and it's tracked through the clip.
- **See what the agent did.** Each reply lists its steps in plain words: what it looked for, the yes/no question it asked about each person ("Is this a man wearing a red hoodie?") with how many said yes, no or unsure, and what it masked. A coverage strip shows when each target is masked; click it to jump to that moment.
- **Fail closed.** If the vision model is unsure about a person, or the tracker loses someone, they stay blurred. Errors mean more blur, never less.
- **Undo and versions.** Every change is a version; undo is one click.
- **Export** renders the blurred video locally.

## How we built it

The agent never touches pixels. It edits a **mask timeline** (which object is masked, when, and with which effect) through a small set of typed tools, and a deterministic renderer applies that timeline to the video.

1. **Scan at upload.** As soon as a clip arrives, a background scan finds faces and plates with Meta's **EgoBlur** models and people (and any other object) with **YOLOE**, and tracks them across frames. Requests then read cached tracks instead of waiting on detection.
2. **Understand the request.** Plain requests are parsed by a template in milliseconds, with no model. Anything else (exceptions, unusual wording) goes to **Qwen3-VL-8B** once: it writes the whole plan as JSON (objects to find, "only" and "except" questions, a time phrase, the effect), and plain code runs it with the tools. The model never handles track ids.
3. **Check who's who.** "The woman" or "the guy in the red hoodie" becomes a yes/no question that the same Qwen3-VL model answers about each tracked person, from crops taken across the track, with a probability. Above 0.65 is yes, below 0.35 no, in between unsure, and unsure stays masked.
4. **Render.** Plain code applies the timeline: blur, pixelate or a solid box, with padding, interpolation between detections and a hold after a track is lost.

Stack: FastAPI backend, PyTorch on ROCm, transformers, Ultralytics, OpenCV, and a single-file web UI.

## The AMD story

Everything runs on **one AMD Radeon AI PRO R9700** (RDNA 4, gfx1201, 32 GB) with ROCm and the preinstalled PyTorch 2.9.1+rocm7.13.

- **Every model on one card:** detection, the 8B vision-language model and the planner share 32 GB, at **17.9 GB peak** with everything loaded. One Qwen3-VL process serves both the planner and the attribute checks, loaded once and kept resident.
- **Upload scan at 0.76x real time at 1080p**, 17.2 s for a 13 s clip, down from 48.1 s. On a 720p clip the GPU scans at 1.12x real time against 0.05x on a 64-thread Threadripper PRO CPU alone: **24x faster**.
- **How we got there:** we profiled first. Decoding was 0.2 s of a 48 s scan, and EgoBlur was 87% of it. **fp16 autocast** cut each EgoBlur call 3x (0.147 s to 0.049 s per 1080p frame) with the same boxes (mean IoU 0.999 on faces against fp32, none lost), and a warm-up at startup removed about 6 s of first-call cost per model. Batching, two streams, MIOpen autotuning and smaller inputs didn't help, and we measured each of them.
- **ROCm lessons:** MIOpen couldn't read its kernel database on the cloud image, so every new crop size triggered a kernel search (~15 s per VLM call). Fixed crop sizes plus `MIOPEN_FIND_MODE=FAST` fixed it. Asking the model for only the last position's logits (`logits_to_keep=1`) avoided a 16 GB spike with three images per call.
- **Fast enough to feel interactive:** the planner call takes about 1.8 s, whole requests 2-7 s, and exporting `street_plates` (5,912 masked boxes) takes 2.8 s. From upload to finished export on that clip: about 15 s.

## Private by construction

Inference never touches the network. `./run.sh` puts Hugging Face and Ultralytics in offline mode, and a small egress guard refuses and logs every outbound connection or DNS lookup to `logs/egress.log`. After the demo the log is empty, and the model server's `/health` reports `"offline": true`.

## Results

| | Measured |
| --- | --- |
| Plates masked (share of plate frames), `street_plates` / `street_walk` | **100% / 94.6%** |
| Longest a plate stays visible | 0 s / 0.16 s (one 11x7 px plate) |
| Plate over-blur, mean | 3.0% / 0.1% of the frame |
| Person questions on the three people in focus on `street_walk` | every answer right, at p 0.00 or 1.00 |
| Upload scan, 1080p | 0.76x real time |
| Peak GPU memory, all models | 17.9 GB of 32 GB |
| Request success rate (20 benchmark requests) | *TODO after `./run.sh bench`* |

## Challenges

- **Plate false alarms.** At a low threshold EgoBlur flags tail lights, signs and shoes. A plate track now needs two confident detections, or a plate shape and enough frames; on `street_plates` that took 232 plate tracks down to 49 while keeping every real plate.
- **Small, fast plates.** A 13 px plate moving 10 px between checked frames doesn't overlap itself, so the tracker split it. Linking by size and distance, and starting masks a few frames early, took `street_walk` from 72% to 94.6% of plate frames masked.
- **Step-by-step agents are slow.** Our first LLM loop picked one tool at a time and took over two minutes for "everyone except the guy in the red hoodie". One JSON plan per request, run by plain code, does it in 7 s.
- **Fragmented person tracks.** The tracker sometimes loses a pedestrian for a few frames and starts a "new" person, so one person gets several tracks and the vision model gets asked several times. A stitcher now joins a track to one that starts nearby moments later, heading the same way, at the same size and with similar colors (`street_walk`: 15 -> 11 person tracks). Every join on our three clips was checked by eye, because merging two different people is worse than leaving a piece: one answer would apply to both. Most of what's left is many different people each seen briefly, not broken tracks.

## What we learned

Keep the model out of the pixels. A typed mask timeline between the agent and the renderer made every result checkable, undoable and testable without a GPU, and let four people build against a stub from the first hour. And measure before you optimize: the GPU, not decoding, was the limit, and fp16 was the win.

## What's next

- Fewer vision-model checks per request in crowds, where most person tracks are different people seen briefly.
- Live mode: a standing rule ("blur any plate that appears") on a stream behind a short delay buffer, then an OBS virtual camera.
- Pixel-accurate masks with SAM 3 instead of boxes.
- More categories: screens, on-screen text, documents.

## Built with

amd, rocm, radeon, pytorch, transformers, qwen3-vl, yoloe, ultralytics, egoblur, opencv, fastapi, python, javascript
