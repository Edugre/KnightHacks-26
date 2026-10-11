# Demo script (3 minutes)

Live on the R9700 through the AUP proxy, with `street_walk` as the main clip and `street_plates` as the speed shot. Every number below was measured on the R9700 (IMPLEMENTATION.md, "Where we are"). Don't quote laptop timings.

## Before you go on (10 minutes ahead)

1. Relaunch the server if it's older than about 3 hours (sessions end at 4 h), then `cd ~/kh-alex && git pull && ./run.sh`. Wait for `/health` to say `"offline": true` (the model loads in about 3.5 min).
2. Open the app, pick `street_walk` and `street_plates` once each, so the detectors are warm and the clips are scanned, then reload the page with no `?video=` so you start from the start screen.
3. Open a second tab with a terminal showing `logs/egress.log` (it should be empty) and `curl -s localhost:8100/health`.
4. Have the backup video open in a third tab (see "If live fails").
5. Browser zoom so the conversation column and both rails fit; close the Timeline drawer.

## The script

| Time | Do | Say |
| --- | --- | --- |
| 0:00 | Start screen. | "Blurring people out of footage today means scrubbing an editor frame by frame, or uploading the footage to someone's cloud. Censor does it on one AMD GPU, from plain requests, and nothing leaves the machine." |
| 0:15 | Pick **street_walk**. | "The upload starts a background scan for faces, plates and people. A 13 s 1080p clip scans in 17 s on the Radeon AI PRO R9700: 0.76x real time, 24x the CPU." |
| 0:30 | Type **blur all plates and the person in the red hoodie around halfway**, press Enter. | While it works: "Plain requests are parsed instantly. 'The person in the red hoodie' becomes a yes/no question for Qwen3-VL, asked about every person the detector tracked." |
| 0:45 | Point at the reply: the steps list, the yes / no / unsure counts, the coverage strip. | "Here's what it did, in words: it found the plates, asked 'Is this a man wearing a red hoodie?', and blurred only the yes. The plates are blurred the whole clip; he's blurred only around halfway, because the time phrase belongs to him. Anything the model is unsure about stays blurred: we fail closed." |
| 1:05 | Click his row in **Targets**, then click along the coverage strip. | "Select a target and its box lights up; click the strip and the frame jumps there." |
| 1:20 | Type **blur everyone except the woman**. | "Exceptions go to the local LLM: one call writes the whole plan as JSON in about 2 s, and plain code runs it. The model never touches pixels or track ids; it only edits a mask timeline through typed tools." |
| 1:40 | Click **Draw box**, drag around something the detectors don't know (a sign, a shop window). | "And for anything else, draw a box: it's tracked through the clip." |
| 1:55 | Click **Undo** once, show the version marked undone on the right. | "Every change is a version, and undo is one click." |
| 2:05 | Click **Export**. The Timeline drawer opens with the blurred video; play a few seconds. | "The renderer is plain code applying that timeline." |
| 2:20 | Switch to the terminal tab: empty `logs/egress.log`, `/health` with `"offline": true`. | "Every outbound connection is refused and logged during inference. The log is empty." |
| 2:35 | Back to the app, or a slide with the numbers. | "On our labeled clips, 100% and 94.6% of plate frames are masked, with 3% over-blur. Detection, the 8B vision-language model and the planner all share one 32 GB R9700, at 18 GB peak." |
| 2:50 | Close. | "Redaction you ask for in plain language, check frame by frame, and run without a cloud." |

**If there's time left, the speed shot:** pick `street_plates`, type **blur all faces and plates**, then Export. On the R9700 that's about 15 s from upload to finished export.

## What to point out

- **The agent panel:** every step in plain words ("Asked 'Is this a woman?' about 3 tracks: 1 yes, 1 no, 1 unsure"). The raw tool calls are under "Tool calls" if a judge asks.
- **Fail closed:** unsure answers stay masked; low-confidence moments show as the bright marks on the strip and are masked anyway, with extra padding (**Check N low-confidence spots** steps through them). On a crowd like `street_plates` the strip groups targets into one bar per kind, so the frame stays in view.
- **Offline:** the empty egress log and `"offline": true`.
- **AMD numbers** (R9700, IMPLEMENTATION.md): upload scan 0.76x real time at 1080p (24x the CPU); EgoBlur fp16 cut the per-frame cost 3x with the same boxes; export of `street_plates` in 2.8 s; planner call about 1.8 s; requests 2-7 s; 18 GB peak with every model loaded.

## If live fails

| What breaks | Do |
| --- | --- |
| The proxy times out on a request (no answer after ~100 s) | The page keeps waiting for the server and shows the result when it finishes. Keep talking over it, or switch to the backup video. |
| The model server is down (requests answer "clarify", or "unknown" for every person) | Stick to plain requests ("blur all faces and plates" on `street_plates`) and draw-a-box: neither needs the LLM. Say the attribute step is shown in the backup video. |
| The whole server is down | Play the backup video (the full script above, recorded on the R9700). Then show stub mode on a laptop (`./run.sh stub`, pick **synthetic**) to prove the UI is real: requests, coverage strips, draw-a-box, undo and export all work without a GPU. |
| A wrong blur on stage | Say it: "the detector missed / over-blurred that one", click **Undo**, and point out that fail-closed means a miss is rare and visible. |

**Backup video:** record the whole script on the R9700 the night before, screen plus voice, and keep it in two places (a laptop's disk and the team drive).

**Backup exports (already made, Sat 23:30):** `python scripts/make_backup.py` runs the real detectors on any CUDA GPU (or the R9700) and writes blurred exports of the real clips, a before/after still for each, and a `README.md` with the agent's message and timings, into `data/backup/`. The set: `street_plates` "blur all faces and plates" (113 faces, 14 plates), `street_walk` "blur all faces and plates" and "pixelate everyone", `street_talk` "blur all faces". `data/` is gitignored, so they're shared on the team drive, not git. They use the template path (no language model), so they show detection and rendering, not the attribute step ("blur the woman"); that still needs the screen recording on the R9700. Checked by eye: one small false face blur on a shirt at the right edge of `street_talk`.
