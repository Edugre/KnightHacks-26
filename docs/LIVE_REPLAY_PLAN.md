# Live replay (M6): plan

*Sun 2026-10-11, started 04:05 EDT. Deadline 11:00. Branch `feat/live-replay`, off `main` at 365aac6.*

## 1. What we're asked for, and why

Show that Blurr's engine also works on a stream, not only on a finished file: play a recorded clip as if it were a
live feed, apply one standing rule ("blur every face and license plate") with no request per object, and show the
viewer the masked stream behind a short delay buffer, with the delay and per-frame latency on screen. Labeled
honestly as a **replay** (IMPLEMENTATION.md M6). It must keep the project's one promise: **fail closed**: a frame
the detector hasn't vouched for goes out blurred, never bare.

Why now: the clean re-run (docs/STACK.md:40, 200 frames, idle GPU) puts a face + plate + person pass at 134 ms p50
at 720p, so detecting every 5th frame with a 0.77 s buffer fits. Nothing in the demo shows the "and live" half of
the pitch yet (PLAN.md line 11).

Not in scope: real ingest (RTMP, webcam, OBS), requests typed during the stream, VLM checks, person rules, audio,
recording the live output to a file, merging into `main` unless the go/no-go at 08:30 passes.

## 2. Architecture

```
 clip.mp4 ──► Source thread ──► FrameRing (f, frame@720p, t_arrive)   ── paced at the clip's fps, loops
                                   │  newest frame (latest wins)
                                   ▼
                             Detector thread ── detect(f, frame) -> {face: [(box, score)], plate: [...]}
                                   │            real: EgoBlur fp16, the scan's own infer fns; stub: tracks.json replay
                                   ▼
 Emitter thread: at t_arrive(f) + DELAY, mask frame f from the ready detections around f
   a = newest ready detection <= f, b = oldest ready detection >= f
   no a, or no b  ──► "blind" frame: re-send the last vouched frame with a DETECTOR LATE badge   <- fail closed on time
                      (no new pixels go out; before the first vouched frame: full_frame_blur)
   else: boxes for f from the track cache (rebuilt only when the ready set changes), apply_effect() each;
         any exception ──► full_frame_blur(f)
   ──► downscale to the display size ──► JPEG ──► LatestFrame slot (+ stats)
                                   │
 GET /api/live/frame?run=<id>&after=n ◄┘ (long-poll, 2 in flight; web/live.html draws it into an <img>)
 GET /api/live/stats                      (delay, pass p50/p95, stride, blind frames, max gap, emit p95)
 POST /api/live/start {clip, height}, POST /api/live/stop
 python -m censor.live run <clip> --seconds 30 --out X.mp4 --report X.json   (headless, real time, same engine)
 python -m censor.live score X.json bench/labels/<clip>.json                  (the automatic go/no-go gate)
```

Design decisions:

- **Reuse the offline masking, with one new knob.** Inside the delay window we know the detections *after* frame f,
  so frame f is masked as the export would mask it from those keyframes: `link()` (censor/vision/specialist.py:188)
  then `expand_target()` (censor/render/render.py:73): interpolation, hold-after-loss, padding, MIN_PAD_PX.
  `apply_effect()` (render.py:123) writes the pixels. The one change to render.py: `expand_target(target, n_frames,
  lead_frames=LEAD_FRAMES)`. Offline keeps the default (3, sized for `EGOBLUR_EVERY=3`); live passes the real gap to
  the previous detection, because at stride 4-12 a 3-frame lead-in leaves an object that walked in just after the
  previous detection bare for `gap - 4` frames (R1 finding 2). For lead frames beyond LEAD_FRAMES the box is the
  **union** of the speed-extrapolated box and the static first box, grown by HOLD_GROWTH per frame: we don't know
  where a new object was, so we cover both guesses.
- **Fail closed on time, not just on error.** The emitter never waits for the detector. Frame f is maskable only
  when a ready detection exists at or before f **and** one at or after f (R1 finding 1: with only "one after f", a
  gap of 19+ frames between detections leaves frames `a+16..b-4` with no box, and a 743 ms pass has been measured).
  Otherwise the frame is fully blurred. A late detector shows up as blurred frames, not bare faces. What this can't
  cover: an object that appears *and* disappears between two consecutive detections is never seen at all; the
  offline scan has the same blind spot at stride 3.
- **Shape rules yes, history rules no.** The scan's `confirmed()` (specialist.py:222) has two kinds of rule: ones
  that need the whole clip (min frames, min confident hits) and ones that judge the boxes themselves: a big faint
  "face" is a car back (`big_frac` / `big_min_conf`, specialist.py:76-80), a wide faint face is a plate, a plate
  must be plate-shaped and not huge. Live applies only the shape kind, by calling `confirmed(kfs, rule)` with a
  rule dict stripped of `min_conf` / `min_frames` (so it returns after the shape checks). Without this, on
  `street_plates` EgoBlur's 300-900 px false faces at 0.1-0.68 would blur half the frame in flickering held blobs
  (R1). Everything else at `EGOBLUR_CONF` or above is masked: more blur, never less (D2).
- **Blind frames freeze, they don't flash.** A full-frame blur is a 53x30 mush; a stall would look like the app
  broke. A blind frame re-sends the last vouched frame (already masked, no new pixels) with a grey "DETECTOR LATE"
  badge, and counts as blind in the stats. Before the first vouched frame the output is a full-frame blur.
- **Boxes are cached, not recomputed per frame.** `link` + `expand_target` over 25-40 detections is 10-20 ms of
  pure Python per call (R1 timed it) and would fight the detector thread for the GIL at 30 fps. The emitter rebuilds
  the per-frame box table only when the ready set changes (a new detection landed or the window dropped one),
  about every 4-5 frames, and looks frame f up in it.
- **Two deliverables from one engine.** The browser page is the demo; the headless CLI writes an MP4 of the emitted
  frames (badge and frame age overlaid) plus a JSON of per-frame rects, scored against `bench/labels/<clip>.json`.
  The CLI is the go/no-go gate and the fallback if the proxy can't show enough fps: "ran in real time on the GPU,
  here is its output" is still honest.
- **Faces and plates only.** YOLOE person costs 18 ms a pass and needs a VLM to be useful; the standing rule doesn't.
  Two EgoBlur calls at 720p are ~116 ms p50 (STACK.md:38), so stride ≈ 4.
- **Its own process and port (8001), its own page, behind a flag.** The router is included only when
  `CENSOR_LIVE=1`, so the :8000 demo app never exposes `/api/live/start` (one click there would put ~100% GPU load
  under the main demo). `web/index.html` (under active change: #26, #29) is untouched. Live loads its own EgoBlur
  copy (~1.3 GB with activations, STACK.md:34).
- **Long-poll JPEG, not MJPEG.** The AUP proxy goes through Cloudflare; `multipart/x-mixed-replace` is often
  buffered by proxies. One small JPEG per request works through anything, with three guards: the URL has **no
  `.jpg` extension** and carries a `run` id (IMPLEMENTATION.md:381: Cloudflare caches `.jpg` for 4 h whatever the
  origin says, and after a restart `after=n` restarts at 0), the fetch uses `cache: 'no-store'`, and the server
  answers with the newest frame when `after` is ahead of it or `run` doesn't match. One request per frame caps the
  on-screen rate at ~1/RTT (~5-10 fps through Cloudflare + the hub), so the page keeps two polls in flight, frames
  are downscaled to `CENSOR_LIVE_OUT_W` (default 960) at JPEG quality 75 (~50 KB), and the badge states the display
  rate. Masking always happens at the 720p live size; only the shown copy is smaller.

## 3. Implementation

### 3.1 Commits on `feat/live-replay` (in order; each leaves tests green)

**C0. `censor/render/render.py`: `lead_frames` keyword.** `expand_target(target, n_frames, lead_frames=LEAD_FRAMES)`.
In the lead loop (render.py:93-96), frames `k <= LEAD_FRAMES` behave exactly as now; frames `LEAD_FRAMES < k <=
lead_frames` get the union of `_moved(first.box, v0, -k)` and `first.box`, padding `pad_for(first.state) +
HOLD_GROWTH * k`. Test in `tests/test_core.py` (or wherever expand_target is tested; check): default output is
byte-identical to before; `lead_frames=10` covers `first.f - 10 .. first.f - 1`, each box containing `first.box`.

**C1. `censor/live.py`: the engine (target ~250 lines).** No FastAPI; one class, pure helpers.

```python
LIVE_DELAY_S = float(os.environ.get("CENSOR_LIVE_DELAY", "1.0"))   # >= STACK.md's 0.77 s at 720p, with margin
LIVE_HEIGHT = int(os.environ.get("CENSOR_LIVE_HEIGHT", "720"))
LIVE_CATS = ("face", "license plate")
JPEG_QUALITY = 80
WINDOW_BACK = 2 * HOLD_FRAMES   # detections this far before f take part in linking (hold + LINK_GAP fit inside)

Dets = dict[str, list[tuple[Box, float]]]           # category -> [(xyxy, score)]; faces already FACE_EXPANDed
Detect = Callable[[int, np.ndarray], Dets]          # (frame index in the clip, BGR frame at live size) -> Dets

@dataclass(frozen=True)
class Detection:
    f: int            # stream frame number (keeps counting across loops)
    dets: Dets
    took_s: float

def mask_boxes(f: int, ready: list[Detection], hold: int = HOLD_FRAMES) -> list[tuple[Box, float, str]] | None:
    """Boxes (box, padding, effect) to mask on frame f, or None when f can't be vouched for (fail closed)."""

class LiveReplay:
    def __init__(self, clip: str, detect: Detect, height=LIVE_HEIGHT, delay_s=LIVE_DELAY_S,
                 on_emit: Callable[[int, list, bool], None] | None = None,
                 clock=time.monotonic, sleep=time.sleep): ...
    def start(self) -> None          # three daemon threads; idempotent
    def stop(self) -> None           # sets an Event, joins with timeout
    def latest(self, after: int, timeout: float) -> tuple[int, bytes] | None   # Condition wait for a newer JPEG
    def stats(self) -> dict          # see 3.3

def egoblur_detect(infer: dict[str, Infer]) -> Detect      # real: infer[cat](frame), score >= EGOBLUR_CONF, faces _expand(FACE_EXPAND)
def sidecar_detect(tracks_json: Path, src_h: int, live_h: int, n_frames: int) -> Detect   # stub: tracks[f % n], boxes scaled
```

- *Source thread:* `probe(clip)` for fps/size; `frames(clip)` (censor/core/video.py:28) looped; resize to `height`
  (aspect kept, even width, INTER_AREA, as bench/live_speed.py `frame_pool`); sleep until `t0 + f / fps`; put
  `(f, frame, t_arrive)` in a dict ring of the last `ceil((delay_s + 1) * fps)` frames under `self._lock`; set
  `newest = f` and notify. `f` counts across loops; `f % n_frames` is the clip frame.
- *Detector thread:* waits for `newest > last_done`, takes `newest` (latest wins), calls `detect(f % n_frames,
  frame)`, appends `Detection` to `self._dets` under `self._lock` (pruned to `f >= oldest ring frame - WINDOW_BACK`).
  Any exception: count `detector_errors`, log once per exception type to stderr, continue; the emitter blurs frames
  it can't vouch for anyway.
- *Emitter thread:* for f = 0, 1, 2, ...: sleep until `t_arrive(f) + delay_s`; under the lock take `frame` and
  `ready = [d for d in dets if f - WINDOW_BACK <= d.f <= f + hold]` (a list copy; the dict is never iterated outside
  the lock); then **inside one `try`:** `boxes = self._cache.boxes(f, ready)` (below); `None` → blind: publish
  the last vouched JPEG again with the badge (or `full_frame_blur(frame)` if there is none yet), `blind += 1`;
  else `apply_effect(frame, _padded(box, pad, w, h), effect)` per box (render.py:114); `except Exception` →
  `full_frame_blur`, `emit_errors += 1`. Then downscale to the display width, `cv2.imencode(".jpg", quality)`,
  publish `(f, jpeg)` with `notify_all`, record emit time; `on_emit(f, rects, blind)` if set (tests, CLI). If the
  emitter is more than one frame late it skips to the newest due frame and counts `dropped_out`; it never
  publishes an unmasked frame.
- `mask_boxes(f, ready, frame_h, hold)` (pure, the unit-test seam): `a = max(d.f <= f)`, `b = min(d.f >= f)`; if
  either is missing → `None`. Else per category: `seq = [(d.f, d.dets[cat]) for d in ready sorted]`; `tracks =
  link(seq, uncertain_below=0.35)`; drop a track when `not confirmed(kfs, SHAPE_RULES[cat], frame_h, frame_w)`
  where `SHAPE_RULES` = `CONFIRM[cat]` minus `min_conf`, `min_frames` (so only the shape checks run); for each
  kept `kfs`: `prev = max(d.f for d in ready if d.f < kfs[0].f, default=None)`; `lead = kfs[0].f - prev - 1 if prev
  is not None else 0`; `Target(label=cat, source="rule", effect="blur", padding=0.15, keyframes=kfs)`
  (timeline.py:54: `label` and `source` are required, there is no `track_id`); `expand_target(t, n_frames=f + hold +
  1, lead_frames=max(LEAD_FRAMES, lead))`; collect the entry for `f`.
- `BoxCache`: remembers `key = tuple(d.f for d in ready)`; on a key change it recomputes the whole per-frame table
  once (`expand_target` already returns every frame, so build `{frame: [(box, pad, effect)]}` for the window) and
  answers `boxes(f)` from it. `mask_boxes` stays the single-frame reference the cache is tested against (same
  answer for every f in the window).

**C2. `tests/test_live.py`** (no GPU; scripted `detect`; `synthetic.mp4` generated by a fixture the way
`tests/test_e2e_stub.py:19-23` does, since `bench/clips/*` is gitignored):
1. `mask_boxes`: detections only before f → `None`; only after f → `None` (fail closed both ways).
2. Detections at 0 and 22, box at 0 only: with the fix frames 1..21 are all covered (hold 1..15, then the frame-22
   track's lead would be empty because that track has no keyframe; so the test asserts: box seen at 0 **and** 22 →
   linked? no: gap 22 > LINK_GAP 9 → two tracks; frames 16..21 covered by the second track's lead-in of 21 frames).
3. Object first seen at b=5 with the previous detection at a=0 (empty): frames 1..4 carry a box containing b's box.
4. Detections at 0 and 5 with a moving box: frame 2's box equals `expand_target` on the same Target (lerp).
5. Object detected at 0 only, next detection at 5 empty, f=10 → held box; f=16 → no box (but frame still maskable).
6. Faces get FACE_EXPAND in `egoblur_detect`; `sidecar_detect` scales boxes by `live_h / src_h`.
7. End to end: `LiveReplay(synthetic, sidecar_detect(...) wrapped in a 0.15 s sleep, delay_s=0.6, on_emit=record)`,
   run 2.5 s, stop. Assert the invariant only: every emitted frame is blind **or** every ground-truth box present on
   that clip frame is ≥ 90% covered by the masked rects (the coverage rule from bench/run_bench.py). No counts.
   The slow fake detector gives a stride of ~5, so this test would have caught R1's two SEV-1s.
8. A `detect` that raises → frames blind, `stats()["detector"]["errors"] > 0`, engine keeps running.
9. `stop()` returns within 2 s and wakes a pending `latest()`; `latest(after=n, timeout=0.2)` returns `None` when
   nothing newer; `latest(after=10**9)` (ahead of the stream) returns the newest frame.
10. `stats()` before any pass returns nulls, not a 500 (`stats_ms([])` raises: bench/live_speed.py:43).
11. `BoxCache.boxes(f, ready) == mask_boxes(f, ready, ...)` for every f in a scripted window.
12. Shape rule: a 0.3-confidence face box 40% of the frame height is dropped; the same box at 0.95 is kept; a
    plate box 3:1 is kept, a square 0.2 one is dropped.
13. `score`: a report whose rects cover the labels passes; one bare labeled box fails with the frame named.

**C2b. Headless CLI + scorer (in `censor/live.py`, `__main__`).** `run <clip> --seconds N --height 720 --out X.mp4
--report X.json [--stub]`: same engine, `on_emit` writes each emitted frame (badge + "age 312 ms · blind 3/900"
overlay) through the ffmpeg pipe `render_video` uses (render.py:176) and appends `{f, clip_f, blind, rects}` to the
report. `score X.json bench/labels/<clip>.json`: for every labeled, confirmed frame that was emitted, every labeled
face/plate box must be ≥ 90% covered by the rects **or** the frame blind; prints coverage per category, blind
share, and the first failing frame. Exit 1 on any bare labeled box. This is the gate at 08:30.

**C3. `censor/server/live_routes.py` + one line in app.py.** `APIRouter(prefix="/api/live")`. It must **not** import
`censor.server.app` (circular: app imports the router); it redefines `SAMPLES_DIR` and `SAMPLE_NAME` (two lines,
with a comment pointing at app.py:60-61).
- `POST /start {"clip": "<sample id>", "height": 720}`: validate the name; 404 if the clip is missing; 409 if a
  replay is running; **503 with a plain message if `egoblur_available()` is false** (never fall back to YOLOE live).
  Real mode: `det = EgoBlurDetector()`; `det.warm_up(sizes=((h, w),))` **before** the source clock starts (cold
  TorchScript calls take ~6 s per model; without this the first ~60 frames are blind and the go/no-go fails on its
  own; stats say `"warming": true` meanwhile); `detect = egoblur_detect(det.infer)`. Stub mode (`CENSOR_STUB=1`):
  `sidecar_detect` from `<clip>.tracks.json`, 400 if there is none. Returns `stats()` including a fresh `run` id.
- `POST /stop` (joins, then `notify_all` so pending polls return), `GET /stats`.
- `GET /frame?run=<id>&after=n`: `latest(after, timeout=2.0)`; 204 on timeout; JPEG with headers `X-Live-Frame: f`
  and `X-Live-Run`. Wrong or missing `run`, or `after` ahead of the stream → newest frame now. Plain `def`
  (FastAPI threadpool), so a waiting poll doesn't block the event loop.
- `GET /live` → `web/live.html`.
- app.py: after the app is created,
  `if os.environ.get("CENSOR_LIVE") == "1": from censor.server.live_routes import router as live_router;
  app.include_router(live_router)`. Nothing else in app.py changes.

**C4. `web/live.html`** (self-contained, no build step, styles copied from index.html's tokens, ~150 lines):
clip picker (from `GET /api/samples`), Start / Stop, one `<img>` fed by a fetch loop with **two polls in flight**
(`fetch('api/live/frame?run=R&after='+n, {cache: 'no-store'})` → `URL.createObjectURL(blob)`, revoke the previous
one, drop out-of-order frames; on a non-2xx/204 answer back off 0.5 → 1 → 2 → 4 s so a restart doesn't hammer
502s through Cloudflare), a permanent badge **"REPLAY of a recorded clip · delayed 1.0 s · showing N fps"** (N
measured client-side), a **"GPU contended"** flag when pass p95 exceeds 2× the post-warm-up baseline, and a stats
strip refreshed every 500 ms: delay, detector pass p50 / p95, detected share, largest gap vs HOLD_FRAMES, blind
frames, emit p95, frame age. Relative URLs only (the AUP proxy path prefix, as index.html does at line 398). No
link from index.html (avoids touching that file).

**C5. Server run + numbers.** Two steps on the Strix Halo, with nothing else on the GPU (`rocm-smi --showuse`
first; check the session has > 90 min left, relaunch if not).
1. **Transport probe (stub mode, no GPU, as soon as C3/C4 land):**
   `PORT=8001 CENSOR_LIVE=1 ./run.sh stub` and open `https://sjc.aupcloud.io/user/<user>/proxy/8001/api/live/live`,
   start `synthetic`. Read the page's measured display fps. **≥ 12 fps:** the browser page is the demo. **< 12:**
   the headless MP4 is the demo and the page is a bonus; don't spend time on transport alternatives.
2. **Gate run (GPU):**
   ```bash
   git fetch && git checkout feat/live-replay
   PYTHONPATH=$PWD CENSOR_OFFLINE=1 python3 -m censor.live run bench/clips/street_plates.mp4 --seconds 30 \
     --out logs/live_street_plates.mp4 --report logs/live_street_plates.json
   python3 -m censor.live score logs/live_street_plates.json bench/labels/street_plates.json
   ```
   then the same for `street_walk`. Then the browser page: `PORT=8001 CENSOR_LIVE=1 CENSOR_WARMUP=0 PYTHONPATH=$PWD
   CENSOR_OFFLINE=1 HF_HUB_OFFLINE=1 YOLO_OFFLINE=true nohup python3 -m uvicorn censor.server.app:app --host 0.0.0.0
   --port 8001 >> logs/app8001.log 2>&1 &`. Record in STACK.md: pass p50/p95, detected share, max gap, blind share,
   emit p95, frame age, display fps, and the `score` output. If blind > 1% or `score` fails, raise
   `CENSOR_LIVE_DELAY` to 1.5 and repeat once; if it still fails, no-go.

**C6. Docs.** README "Using the app" gets a short "Live replay" paragraph; IMPLEMENTATION.md M6 → done (or "built,
not demoed" with the numbers); DEVPOST one line.

### 3.2 Timebox

| By (EDT) | Done |
|---|---|
| 04:20 | branch, this plan reviewed |
| 05:45 | C0 + C1 + C2 + C2b green on the laptop |
| 06:30 | C3 + C4 working in stub mode on the laptop (`PORT=8001 CENSOR_LIVE=1 ./run.sh stub`); transport probe on the server |
| 07:30 | C5 gate run on the Strix Halo; numbers in STACK.md |
| **08:30** | **Go / no-go.** Go: `score` passes on `street_plates` and `street_walk` (no bare labeled box; blind < 1% after warm-up) and a human watched the MP4 once. Then PR into `main` (new files + `lead_frames` kwarg + the flag-gated app.py lines). No-go: keep the branch, write the numbers up as future work in DEVPOST. |
| 08:30–11:00 | not this feature: demo recording, fresh-clone run. **Record the main demo before any live GPU run**, or on a different session. |

Note on the project's own cut line: IMPLEMENTATION.md:319/360 says to drop M6 if it isn't working by Sun 05:00.
This plan goes past it on purpose, because the engine is now measured (STACK.md:40) and the work is on a branch
that can't touch the demo; the 08:30 gate is the replacement cut line, and the owner has been told.

### 3.3 Stats (`GET /api/live/stats`)

`{"running", "warming", "run", "clip", "size", "fps", "delay_s", "frames_out", "blind", "blind_share",
"dropped_out", "emit_errors", "detector": {"passes", "p50_ms", "p95_ms", "baseline_p95_ms", "errors"},
"emit": {"p50_ms", "p95_ms"}, "detected_share", "gap_frames": {"mean", "max"}, "hold_frames": 15,
"frame_age_ms": {"p50"}}`; percentiles via `bench.live_speed.stats_ms` (import it, don't copy; it's light and
bench is a package), guarded: nulls when there are no samples yet. `baseline_p95_ms` is the p95 of the first 50
passes after warm-up; the page flags "GPU contended" when the rolling p95 exceeds 2× it.

### 3.4 Computation budget (720p, 30 fps; STACK.md:38-40)

- Detector: 2 × 58 ms ≈ 116 ms a pass; latest-wins means it runs back to back, so **~100% GPU duty** while live is
  on. That is the contention rule, not a side note.
- Emitter CPU per frame: box lookup (cached), `apply_effect` per box (cheap, render.py comment), downscale +
  JPEG ~5 ms, full-frame blur ~7 ms when it happens. The track rebuild (10-20 ms) runs once per new detection.
  Emit p95 is in the stats so the GIL fight, if any, is visible.
- Delay 1.0 s = 30 frames ≥ stride (4-5) + p99 pass (500 ms ≈ 15 frames) + encode ≈ 21 frames. With the two
  fixes above, a slow pass shows as blind (frozen) frames, never bare ones.
- Memory: ~1.3 GB for EgoBlur in a second process (STACK.md:34); 64 GB shared, 18.9 GB peak for the app.
- STACK's 116 ms was measured with frames decoded ahead, a sync after each model, and an idle CPU; the iGPU shares
  memory bandwidth with the CPU, so the real pass under decode + encode load is what C5 measures, not a quote.
- **Contention rule:** never run live while the :8000 app is scanning or the demo is being recorded; the router is
  off on :8000 (`CENSOR_LIVE` unset), DEMO.md gets one line: no sample loads on :8000 within 2 min of a live run.
  The page's "GPU contended" flag makes it visible when it happens anyway.

## 4. Decisions

| ID | Question | Options | Default |
|---|---|---|---|
| D1 | Frame transport | long-poll JPEG (2 in flight) / MJPEG / WebSocket | long-poll JPEG (proxy-safe) |
| D2 | Confirmation rules live | shape rules only (`confirmed()` minus history fields) / none / full CONFIRM | shape rules only: car-back "faces" would otherwise flicker over half the frame |
| D3 | Resolution | 720p / 1080p | 720p. 1080p: stride 9 = LINK_GAP 9, any jitter splits tracks; not this morning |
| D4 | Blind frame look | freeze last vouched frame + badge / full-frame blur | freeze: no new pixels either way, and it doesn't look broken |
| D5 | Demo vehicle if the proxy shows < 12 fps | headless MP4 from `censor.live run` / keep tuning transport | MP4: measured on the server, scored, no proxy in the loop |

## 5. Adversarial review

- **R1 correctness (folded):** two SEV-1s changed the masking design: (1) maskable needs a ready detection on both
  sides of f; (2) lead-in sized to the real gap (`lead_frames` kwarg, union box). Also folded: Target's real fields;
  emitter try/except; `detect(f, frame)` interface so the stub replay works; warm-up inside `/start`; no circular
  import from live_routes; test 7 made meaningful (slow fake detector, `on_emit` hook, invariant only) and portable
  (fixture-generated clip); `stats_ms([])` guard; lock discipline on `dets`; GPU duty ~100%; 2 polls in flight;
  1080p ruled out.
- **R1 computation / failure modes (folded):** independently found SEV-1 (2) above. Added: `CENSOR_LIVE=1` gate so
  :8000 never exposes live; blind = freeze + badge (D4); `BoxCache` so link/expand runs per detection, not per
  frame; shape rules live (D2); `.jpg`-less URL + `run` id + `no-store` against Cloudflare's 4 h cache; display
  downscale; client backoff; 503 when EgoBlur is missing; `stop()` wakes polls; headless CLI + label scorer as the
  automatic gate and the fallback demo vehicle (D5); transport probe in stub mode before the GPU run; "GPU
  contended" flag; emit p95; honest memory and measurement caveats; the 05:00 cut-line override made explicit.
  Rejected: WebSocket transport (unmeasured too, and a second transport to debug; the MP4 path covers the risk).
