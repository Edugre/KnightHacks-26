# Knight Hacks 2026 – Censor Agent Build Plan

Oct 10, 2026 · @Eduardo Goncalvez

> **Update (Sat 03:40):** we build on the AUP Learning Cloud's Radeon AI PRO R9700 (gfx1201, 32 GB), not the Strix Halo this plan describes, which avoids concerns 1 and 2 below ([docs/STACK.md](docs/STACK.md)). Current status and next steps are in [IMPLEMENTATION.md](IMPLEMENTATION.md) section 5.

## The idea

A local censoring agent: say what to hide ("blur every license plate", "the person in the red shirt around the halfway mark") or draw a box on a frame, and the agent turns it into a timeline of masks that a deterministic renderer applies to the video. The same engine runs offline on files and, as a stretch, live behind a delay buffer. It all runs on one Strix Halo, so footage never leaves the machine.

Pitch line: redaction you can ask for in plain language, check frame by frame, and run without a cloud.

Why it fits what won before: the ACT II champion ran a multi-model video pipeline on a single AMD GPU and returned answers backed by video evidence ([results](https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/live)). We follow that shape. Our evidence is the mask timeline, and uncertain frames fail closed (blur) instead of guessing.

## Scope

Offline editing end to end is the must-have; live mode is a stretch we only attempt once offline works and the numbers are in.

**Must have (by Saturday night)**

- Upload a video, type a request, the agent finds targets (plates, faces, people, on-screen text) and tracks them.
- Preview the mask timeline, undo, and export the blurred video.
- A benchmark harness that reports recall and latency.
- One command that runs the whole app on a clean machine.

**Should have**

- Draw-a-box tool: a user-drawn region becomes a tracked mask.
- Attribute requests ("red shirt", "around halfway") resolved with a vision-language check.
- An NSFW category through a pretrained detector, demoed with non-explicit content only.

**Stretch**

- Live replay mode: a standing rule ("blur any plate that appears") on a recorded stream behind a delay buffer.
- OBS virtual-camera hookup, pixel-accurate segmentation masks, MTP for the agent model, hardware video decode.

**Cut**

- No training or collecting explicit data.
- No OBS plugin and no cross-camera identity tracking.

## Architecture

Diagram (rendered in the original doc): request to output in five layers.

1. Input: chat request, drawn box, or standing live rule.
2. Agent: local LLM that calls typed tools (about 10). It never edits pixels.
3. Vision: open-vocabulary detector (YOLOE; NudeNet secondary), tracker, and a VLM check at about 1 fps.
4. Mask timeline: the shared contract everything writes to.
5. Renderer: deterministic ffmpeg that applies the timeline. Output is a file, or live behind a delay buffer.

Requests and drawn boxes go to the agent, which calls typed tools. Vision models find and track targets, everything lands in one mask timeline, and the renderer applies it. Preview and undo edit the timeline, never the video. Fail closed on uncertainty.

## Concerns and workarounds

The biggest risks are the Strix Halo software stack, detector quality on tricky requests, and shipping something judges can actually run. Items marked "earlier chat" come from our previous brainstorm and were not re-verified here.

| # | Concern | What we know | Workaround |
| --- | --- | --- | --- |
| 1 | ROCm may not run our models reliably on gfx1151 | Reported model-load segfaults in vLLM on gfx1151 ([issue](https://github.com/randomvariable/vllm/issues/14)); one report (earlier chat) that stable ROCm 7.2.2 blocked GPU compute | Hour-1 smoke test of three paths: PyTorch ROCm, ONNX Runtime, llama.cpp Vulkan. Pick a backend per model, pin versions in a script, keep a CPU fallback |
| 2 | Flash Attention missing for gfx1151 in PyTorch builds | A forum thread reports the Linux TheRock builds lacked it ([thread](https://community.frame.work/t/pytorch-w-flash-attention-vllm-for-strix-halo/74736)); status today unchecked | Avoid models that require it; use standard attention and a small vision-language model |
| 3 | Vision-model speed on this chip is unknown | We found no published numbers | Benchmark detector fps and memory in hour 1. Detect on ~640 px frames every few frames and let the tracker fill gaps. Run the VLM at about 1 fps, never on the video path |
| 4 | Open-vocabulary detectors are weak on state or relationship prompts ("person in red shirt in the background") | Earlier chat | Decompose: detect all persons, check color or ask the VLM, then track. Show the preview and fail closed when unsure |
| 5 | No verified open license-plate model | Earlier chat | Test zero-shot detection first. If plate recall is low by Saturday noon, fine-tune a small detector on a public plate dataset, time-boxed to 2 hours |
| 6 | NSFW detector is unproven and we can't test it on explicit data | NudeNet's last release was July 2024 with no published accuracy (earlier chat) | Keep it secondary. Never collect or train on explicit data. Benchmark on plates, faces and screens, and state the NSFW limits in the demo |
| 7 | A small local model may call tools unreliably | n/a | Typed tools with schema-constrained output, about 10 tools, template intents for live rules. Score 20 test requests. The agent never touches pixels |
| 8 | Masks flicker or drop during motion | n/a | Expand boxes, hold a mask for N frames after the track is lost, detect every k frames plus a tracker, smooth over time |
| 9 | Live mode is too much scope | n/a | Offline first. Live = replaying a recorded stream through a delay buffer with one standing rule, labeled honestly as a replay |
| 10 | Packaging failures | Many ACT II submissions never scored because of image pull errors, crashes and accuracy gates ([results](https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/live)) | One-command run script. Test on a clean second machine Saturday night. Record a backup demo video |
| 11 | Crowded space and existing redaction products | Video captioning drew 189 ACT II submissions ([results](https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/live)); earlier chat noted commercial redaction tools | Differentiate on plain-language requests, an auditable mask timeline, fully local operation and measured recall |
| 12 | Overlap with an existing club project | n/a | Read Knight Hacks' rules on pre-existing work. If the same product is also an INIT Build entry, build fresh here and keep git history from the event start |

## Benchmark and demo

We report recall first, because a missed plate or face is the failure that matters. Build a small labeled set of about 10 clips (30 to 60 seconds each) from our own footage or public-domain video, with plates, faces and screens. Label every 10th frame by hand.

| Metric | How we measure it | Target (revise after first run) |
| --- | --- | --- |
| Recall | Share of labeled plates, faces and screens covered by a mask | High enough that misses are rare; report per category |
| Over-blur | Share of frame area masked that was not a target | Low; report average |
| Request success | Out of 20 written requests, how many the agent resolves to the right targets | Report as a count |
| Fail-closed rate | Share of uncertain frames that get blurred | Report; shows honest abstention |
| Stage latency | ms per frame for detect, track, VLM check, render | Report each stage and peak memory on the Strix Halo |
| Offline speed | Video length divided by processing time | Report as a multiple of real time |

Baselines: a fixed-prompt detector with no VLM check, and our pipeline with fail-closed off. Showing the gap is the proof.

**Demo flow (3 minutes)**

1. Offline: "blur all plates and the person in the red shirt around halfway." Show the mask timeline preview, then export.
2. Draw a box on a face and track it through the clip.
3. If ready, live replay with a standing rule behind the delay buffer.
4. Show the benchmark numbers, stage latencies and memory use, and say that nothing left the machine.

Record a full backup video of this flow by Sunday morning.

## Team split

Four roles, decoupled by one shared contract: the mask timeline format (per frame or time range: target id, box or polygon, effect, confidence, source). Agree on it in the first hour so everyone can build against a stub. Assign names below; coding agents can cover gaps.

| Role | Owner | Owns | First deliverable |
| --- | --- | --- | --- |
| A: Vision and stack | | Backend smoke test, detector, tracker, VLM check, plate model, latency numbers | Working detector on the box with fps and memory, plus the pinned environment script |
| B: Agent and toolbox | | Local LLM serving, typed tool schemas, request resolution, the 20-request test | Agent that turns one request into a valid mask-timeline call against a stub |
| C: Renderer and video I/O | | Mask timeline schema, ffmpeg render, flicker handling, delay buffer, live replay | Renderer that applies a hand-written timeline and exports video |
| D: App, benchmark and demo | | Web UI (player, chat, draw-a-box, preview, undo), labeled clip set, benchmark harness, packaging, demo video | Clickable UI on stubs and the labeled clip set |

Everyone: any role that finishes early labels clips, tests the one-command run, or covers the riskiest open item in the concerns table.

## Timeline

Diagram (rendered in the original doc): 4 phases, 3 gates.

Phases are relative until we confirm the submission deadline. Live replay only starts after the benchmark numbers are in.

## First hour and open questions

The first hour decides which backend we build on, so do these before any feature work.

- [ ] Run `rocminfo` on the Learning Cloud box; confirm it reports gfx1151 and note the ROCm version.
- [ ] Smoke-test PyTorch ROCm, ONNX Runtime and llama.cpp Vulkan with one small model each; record which work.
- [ ] Run an open-vocabulary detector on a sample video; record fps and peak memory.
- [ ] Write the mask timeline JSON schema and commit it, with one hand-written example.
- [ ] Start downloading the models we will use (detector, tracker, a small vision-language model, the agent LLM).
- [ ] Record or collect the first 3 test clips.
- [ ] Read the Knight Hacks rules on new work and judging format.

**Open questions**

- What exactly is the AMD challenge asking for? We never found the prompt or scoring criteria.
- When is the submission deadline and the judging slot? The timeline uses relative phases until we know.
- Which agent LLM do we serve, and does it run well on the box? Decide after the smoke test.
- Does the team own footage with consent for the benchmark, or do we use public-domain video?
