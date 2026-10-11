# Live replay (M6): plan

*Sun 2026-10-11, started 04:05 EDT. Deadline 11:00. Branch `feat/live-replay`, off `main` at 365aac6.*

## 1. What we're asked for, and why

Show that Blurr's engine also works on a stream, not only on a finished file: play a recorded clip as if it were a
live feed, apply one standing rule ("blur every face and license plate") with no request per object, and show the
viewer the masked stream behind a short delay buffer, with the delay and per-frame latency on screen. Labeled
honestly as a **replay** (IMPLEMENTATION.md M6). It must keep the project's one promise: **fail closed**: a frame
the detector hasn't vouched for goes out frozen or blurred, never bare.

Why now: the clean re-run (docs/STACK.md:40, 200 frames, idle GPU) puts a face + plate + person pass at 134 ms p50
at 720p, so detecting every 5th frame with a 0.77 s buffer fits. Nothing in the demo shows the "and live" half of
the pitch yet (PLAN.md line 11).

Not in scope: real ingest (RTMP, webcam, OBS), requests typed during the stream, VLM checks, person rules, audio,
merging into `main` unless the go/no-go at 08:30 passes.

## 2. Architecture

```
 clip.mp4 ──► Source thread ──► FrameRing (f, frame@720p, t_arrive)   ── paced at the clip's fps, loops
                                   │  newest frame (latest wins)
                                   ▼
                             Detector thread ── detect(clip_f, frame) -> {face: [(box, score)], "license plate": [...]}
                                   │            real: EgoBlur fp16, the scan's own infer fns; stub: tracks.json replay
                                   ▼
 Emitter thread: at t_arrive(f) + DELAY, mask frame f from the ready detections around f
   a = newest ready detection <= f, b = oldest ready detection >= f
   no a, no b, or b.f - a.f > HOLD_FRAMES + 1 ──► "blind": re-send the last vouched frame + DETECTOR LATE badge
                      (no new pixels go out; before the first vouched frame: full_frame_blur)       <- fail closed on time
   else: boxes for f from BoxCache (rebuilt only when the ready set changes), apply_effect() each;
         any exception ──► full_frame_blur(f)
   ──► downscale to the display size ──► JPEG ──► LatestFrame slot (+ stats)
                                   │
 GET /api/live/frame?run=<id>&after=n ◄┘ (long-poll, 2 in flight; web/live.html at /api/live/live draws it)
 GET /api/live/stats                      (delay, pass p50/p95, blind, max gap, emit p95, frame age)
 POST /api/live/start {clip, height}, POST /api/live/stop
 python -m censor.live_cli run <clip> --seconds 30 --out X.mp4 --report X.json   (headless, real time, same engine)
 python -m censor.live_cli score X.json bench/labels/<clip>.json                  (the automatic go/no-go gate)
```

Design decisions:

- **Reuse the offline masking, with one new knob (C0, done).** Inside the delay window we know the detections
  *after* frame f, so frame f is masked as the export would mask it from those keyframes: `link()`
  (censor/vision/specialist.py:188) then `expand_target()` (censor/render/render.py): interpolation, hold-after-loss,
  padding, MIN_PAD_PX. `apply_effect()` writes the pixels. `expand_target(target, n_frames, lead_frames=LEAD_FRAMES)`
  is the one change: offline keeps 3 (sized for `EGOBLUR_EVERY=3`); live passes the real gap to the previous
  detection, because at stride 4-12 a 3-frame lead-in leaves an object that walked in just after the previous
  detection bare for `gap - 4` frames (R1). Past 3 frames the lead box is the union of the speed-extrapolated box
  and the static first box (`_lead_box`): we don't know where a new object was, so we cover both guesses.
- **Fail closed on time, not just on error.** The emitter never waits for the detector. Frame f is *vouched* only
  when a ready detection `a` exists at or before f, one `b` at or after f, **and `b.f - a.f <= HOLD_FRAMES + 1`**
  (R2: with a 22-frame gap and a box seen only at `a`, hold covers a+1..a+15 and frames a+16..b-4 have no box
  although both a and b exist). Otherwise the frame is blind. A late detector shows up as frozen frames, not bare
  faces. What this can't cover: an object that appears *and* disappears between two consecutive detections is never
  seen; the offline scan has the same blind spot at stride 3.
- **Shape rules yes, history rules no.** The scan's `confirmed()` (specialist.py:222) mixes rules that need the
  whole clip (min frames, confident hits) with rules that judge the boxes themselves. Live keeps only the shape kind:
  **faces** via `confirmed(kfs, CONFIRM["face"], frame_h, frame_w)` as is (that rule has no `min_conf`, so it
  returns after the big-faint-face and wide-faint-face checks, specialist.py:228-237; verified); **plates** via a
  small `plate_ok(kfs, frame_w, frame_h)` in live.py, because `confirmed()` returns True before any plate check
  once `min_conf` is stripped (specialist.py:237; R2): keep if any keyframe touches the frame edge (`x1<3 or y1<3 or
  x2>w-3 or y2>h-3`) or the median height < `CONFIRM["license plate"]["shape_min_height"]`; otherwise
  `plate_shaped(median_box, frame_h)` with the rule's defaults. Without these, EgoBlur's 300-900 px false faces on
  car backs (0.1-0.68) would blur half of `street_plates` in flickering held blobs. Everything else at
  `EGOBLUR_CONF` or above is masked: more blur, never less (D2).
- **Blind frames freeze, they don't flash.** A full-frame blur is a 53x30 mush; a stall would look like the app
  broke. A blind frame re-sends the last vouched frame (already masked; kept as an ndarray at display size so the
  grey "DETECTOR LATE" badge can be drawn on it) and counts as blind. Before the first vouched frame the output is a
  full-frame blur.
- **Boxes are cached, not recomputed per frame.** `link` + `expand_target` over 25-40 detections is 10-20 ms of
  pure Python per call and would fight the detector thread for the GIL at 30 fps. `BoxCache` rebuilds the per-frame
  table only when the ready set changes (every 4-5 frames) and looks f up in it; the vouch check runs per f.
- **Two deliverables from one engine.** The browser page is the demo; the headless CLI writes an MP4 of the emitted
  frames (badge and frame age overlaid) plus a JSON report scored against `bench/labels/<clip>.json`. The CLI is
  the go/no-go gate and the fallback if the proxy can't show enough fps: "ran in real time on the GPU, here is its
  output" is still honest.
- **Its own process and port (8001), its own page, behind a flag.** The router is included only when
  `CENSOR_LIVE=1`, so the :8000 demo app never exposes `/api/live/start` (one click there would put ~100% GPU load
  under the main demo). `web/index.html` (under active change: #26, #29) is untouched. Live loads its own EgoBlur
  copy (~1.3 GB with activations, STACK.md:34).
- **Long-poll JPEG, not MJPEG.** The AUP proxy goes through Cloudflare; `multipart/x-mixed-replace` is often
  buffered by proxies. One small JPEG per request works through anything, with guards: the URL has no `.jpg`
  extension and carries a `run` id (IMPLEMENTATION.md:381: Cloudflare caches `.jpg` for 4 h; after a restart
  `after=n` restarts at 0), the fetch uses `cache: 'no-store'` (the `no_store` middleware, app.py:89, covers the
  server side), and the server answers with the newest frame when `after` is ahead or `run` doesn't match. One
  request per frame caps the on-screen rate at ~1/RTT (~5-10 fps through Cloudflare + the hub), so the page keeps
  two polls in flight, frames are downscaled to `CENSOR_LIVE_OUT_W` (default 960) at JPEG quality 75 (~50 KB), and
  the badge states the measured display rate. Masking always happens at the 720p live size.

## 3. Implementation

### 3.0 Interface contract (C1, C2b, C3 and C4 are written in parallel against this; do not deviate)

**`censor/live.py`** (engine; no FastAPI, no `bench` imports except `bench.live_speed.stats_ms`):

```python
Box = tuple[float, float, float, float]
Dets = dict[str, list[tuple[Box, float]]]        # category -> [(xyxy, score)]; faces already FACE_EXPANDed
Detect = Callable[[int, np.ndarray], Dets]       # (clip frame index, BGR frame at live size) -> Dets
LIVE_CATS = ("face", "license plate")
LIVE_DELAY_S = float(os.environ.get("CENSOR_LIVE_DELAY", "1.0"))
LIVE_HEIGHT = int(os.environ.get("CENSOR_LIVE_HEIGHT", "720"))
LIVE_OUT_W = int(os.environ.get("CENSOR_LIVE_OUT_W", "960"))
JPEG_QUALITY = 75
WINDOW_BACK = 2 * HOLD_FRAMES

@dataclass(frozen=True)
class Detection:
    f: int           # stream frame number (keeps counting across loops)
    dets: Dets       # may lack a category: use dets.get(cat, []); an empty Detection still vouches
    took_s: float

def plate_ok(kfs: list[Keyframe], frame_w: int, frame_h: int) -> bool
def mask_boxes(f: int, ready: list[Detection], frame_w: int, frame_h: int,
               hold: int = HOLD_FRAMES) -> list[tuple[Box, float, str]] | None
    # ready: already windowed by the caller to [f - WINDOW_BACK, f + hold], sorted by f; not filtered here.
    # None unless a (max d.f <= f) and b (min d.f >= f) exist and b.f - a.f <= hold + 1.
    # Per cat: seq = [(d.f, d.dets.get(cat, [])) for d in ready]; tracks = link(seq, uncertain_below=0.35);
    # keep a track if (cat == "face" and confirmed(kfs, CONFIRM["face"], frame_h, frame_w)) or
    #                 (cat == "license plate" and plate_ok(kfs, frame_w, frame_h));
    # prev = max((d.f for d in ready if d.f < kfs[0].f), default=None); lead = kfs[0].f - prev - 1 if prev is not None else 0
    # Target(label=cat, source="rule", effect="blur", padding=0.15, keyframes=kfs)
    # expand_target(t, n_frames=f + hold + 1, lead_frames=max(LEAD_FRAMES, lead)); collect entry for f.
class BoxCache:
    def boxes(self, f, ready, frame_w, frame_h, hold=HOLD_FRAMES) -> list | None   # == mask_boxes for every f; table rebuilt when tuple(d.f for d in ready) changes; the vouch check is repeated per f
def egoblur_detect(infer: dict[str, Infer]) -> Detect       # infer[cat](frame), keep score >= EGOBLUR_CONF, faces _expand(b, FACE_EXPAND)
def sidecar_detect(tracks_json: Path, src_h: int, live_h: int, n_frames: int) -> Detect
    # TrackStore dump: {"queries": [], "tracks": {"trk_N": {"track_id", "label", "source", "keyframes": [{"f", "box": [x1,y1,x2,y2], "conf", "state"}]}}}
    # boxes in source pixels, f = clip frame from 0; keep labels in LIVE_CATS only; scale boxes by live_h / src_h;
    # score = conf; faces get FACE_EXPAND too (sidecar faces are raw GT)

class LiveReplay:
    def __init__(self, clip: str, detect: Detect, height: int = LIVE_HEIGHT, delay_s: float = LIVE_DELAY_S,
                 out_w: int = LIVE_OUT_W, on_emit: Callable[[int, int, bool, list[tuple[int,int,int,int]]], None] | None = None)
        # on_emit(f, clip_f, blind, rects): rects are the _padded int rects actually masked (full frame on the error path, [] when blind)
    run: str          # uuid4().hex[:8], new per instance
    warming: bool     # set by the caller (routes) while models warm up; default False
    fps: float; size: tuple[int, int]  # live (w, h); n_frames: int
    def start(self) -> None     # idempotent; three daemon threads
    def stop(self) -> None      # set Event; notify_all under the lock; join (never holding the lock, timeout 2 s each); notify_all again
    def latest(self, after: int, timeout: float) -> tuple[int, bytes] | None
        # first emitted frame with f > after (clients start at after=-1); after > newest -> newest now; None on timeout or stopped.
        # f is the emitted stream frame number; a blind re-send gets its own (new) f.
    def stats(self) -> dict     # exact keys in 3.3
```

Threading rules: one `threading.Condition` (`self._cv`) guards the ring, `dets`, `newest`, the latest JPEG and
counters; every wait is `cv.wait_for(pred, timeout)`; every sleep is `self._stop.wait(dt)` (so `stop()` returns
promptly even during the first `delay_s`); the emitter masks `frame.copy()` (the ring frame is the detector's
input and `apply_effect` mutates in place); if f was evicted from the ring the emitter skips to the oldest ring frame
and counts `dropped_out`; tests run in real time (no fake clock; `Condition.wait` uses real time).

**`censor/live_cli.py`** (imports the engine; `python -m censor.live_cli`):
- `run <clip> [--seconds 30] [--height 720] [--delay 1.0] --out X.mp4 --report X.json [--stub]`. `--stub`:
  `sidecar_detect` from `<clip>.tracks.json` (`src_h` from `probe`), no GPU, no warm-up. Without `--stub`: exit 2 with
  a plain message if `egoblur_available()` is false; else `det = EgoBlurDetector(); det.warm_up(sizes=((h, w),))`
  before `start()`; `detect = egoblur_detect(det.infer)`. Ends when `f / fps >= seconds`. Frames go to
  `ffmpeg -y -loglevel error -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - -c:v libx264 -preset veryfast -crf 20
  -pix_fmt yuv420p out.mp4` (**no** audio map, **no** `-shortest`; `{w}x{h}` is the live size, fps the clip's) via
  `on_emit`, one frame per emitted f, repeating the previous frame for `dropped_out` slots; overlay drawn after the
  rects are recorded: filled dark rectangle + `cv2.putText(img, txt, (12, 28), FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)`
  with `txt = "REPLAY · age 312 ms · blind 3/900"` (+ " · DETECTOR LATE" on blind frames). Report JSON:
  `{"clip", "size": [w, h], "fps", "n_frames", "delay_s", "stats": {...}, "frames": [{"f", "clip_f", "blind", "rects": [[x1,y1,x2,y2], ...]}]}`.
- `score X.json bench/labels/<clip>.json [--min-face 0.89] [--min-plate 0.97]`. Labels (bench/gt.py): `{"clip",
  "video": {fps, width, height, n_frames}, "every": 10, "objects": {gt_id: {category, desc}}, "frames": [{"f", "boxes":
  [{"gt_id", "box": [x1,y1,x2,y2]}], "reviewed"?}]}`; `reviewed` omitted means confirmed, `false` means an unscored
  draft; use `LabelSet.load(path).frames_in()` from bench/gt.py. Scale label boxes by `live_h / video.height`. Score
  every emitted frame whose `clip_f` is a labeled, confirmed frame; a labeled face/plate box is covered when
  `coverage(label_box, rects) >= COVER_MIN` (reuse `coverage`, `_clip_box`, `COVER_MIN` from bench/run_bench.py:57,
  161-175: share of the clipped label box's area inside the union of the rects); a blind frame counts as covered.
  Print per category: boxes, covered, recall, longest run of uncovered frames per object (in seconds), blind share;
  list each failing object with its first bare frame. **Exit 1** if face recall < `--min-face` or plate recall <
  `--min-plate` (defaults = offline recall on these clips minus 0.03: faces 0.92, plates 0.95 → 0.97 is for
  `street_plates` whose plates are 100% offline; pass `--min-plate 0.90` for `street_walk`, 93.7% offline). A gate
  of "zero bare boxes" is unreachable because live has no `recheck_tiny` / `absorb` (R2).

**`censor/server/live_routes.py`** (`APIRouter(prefix="/api/live")`; never imports `censor.server.app`; redefines
`SAMPLES_DIR` and `SAMPLE_NAME` with a comment pointing at app.py:60-61; module state `{"phase": idle|warming|running,
"replay": LiveReplay|None, "last_poll": float}` under a lock; `make_replay(clip, height) -> LiveReplay` is a
module-level factory that tests monkeypatch with a fake engine):
- `POST /start` body `{"clip": str, "height": int = 720}` (height 360..1080 else 422). 404 unknown clip; 409 if phase
  != idle; 503 `{"detail": "EgoBlur models not found in <dir>; live replay needs them"}` when not stub mode and
  `egoblur_available()` is false (never fall back to YOLOE). Sets phase=warming, returns **202** with `stats()` at
  once; a background thread does (real mode) `det = EgoBlurDetector(); det.warm_up(sizes=((h, w),))`, builds the
  replay with `egoblur_detect(det.infer)` (stub: `sidecar_detect`, 400 if no sidecar, checked before 202), `start()`,
  phase=running. Stub mode = `CENSOR_STUB=1`.
- `POST /stop` → `stop()`, phase=idle, 200 with the final `stats()`. `GET /stats` → `stats()` (phase and
  `"warming": true` while warming; `{"phase": "idle", "running": false}` when idle).
- `GET /frame?run=<id>&after=<int>` → `latest(after, timeout=2.0)`; **204** empty on timeout/idle; otherwise
  `image/jpeg` with headers `X-Live-Frame: <f>` and `X-Live-Run: <run>`; wrong/missing `run` → newest frame now.
  Updates `last_poll`. Plain `def` (threadpool).
- **Auto-stop** after 30 s with no `/frame` poll (a closed tab must not hold ~100% GPU): a daemon thread checks
  every 5 s.
- `GET /live` → `web/live.html` (HTMLResponse). Errors are `{"detail": str}` via HTTPException, as app.py.
- app.py, at the very end: `if os.environ.get("CENSOR_LIVE") == "1":\n    from censor.server.live_routes import
  router as live_router\n    app.include_router(live_router)`.

**`web/live.html`** is served at `/api/live/live`, so every URL is relative to `/api/live/`: `start`, `stop`,
`stats`, `frame?run=R&after=N`, and `../samples` for the clip list (R2: `api/live/frame` would resolve to
`/api/live/api/live/frame`).

### 3.1 Commits on `feat/live-replay` (each leaves `PYTHONPATH=$PWD .venv/bin/python -m pytest -q` green)

**C0 (done, 666fb4e).** `lead_frames` kwarg + `_lead_box` + 3 tests.

**C1. `censor/live.py` + C2 `tests/test_live.py`** (TDD; no GPU; `synthetic.mp4` built by a fixture as
tests/test_e2e_stub.py:19-23 does, since `bench/clips/*` is gitignored; the fixture clip is 1280x720, 30 fps, 90
frames; its sidecar has trk_1 "license plate", trk_2 "face", trk_3/4 "person", trk_5 "screen"; one keyframe per frame,
raw GT boxes):
1. `mask_boxes`: detections only before f → `None`; only after f → `None`.
2. Gap rule: (a) box at 0 and 22 → two tracks (22 > LINK_GAP 9); frames 1..21 all covered (second track's lead of
   21); (b) detections at 0 and 22, box at 0 only → frames 16..21 `None` (gap 22 > HOLD+1); (c) detections at 0
   and 16, box at 0 only → frames 1..15 vouched and covered by the held box.
3. Object first seen at b=5, previous detection a=0 empty → frames 1..4 carry a box containing b's box.
4. Detections at 0 and 5 with a moving box → frame 2's box equals `expand_target` on the same Target (lerp).
5. Object detected at 0 only, detection at 5 empty, f=10 → held box present; f=16 → vouched, no box.
6. `egoblur_detect` applies FACE_EXPAND and drops scores < EGOBLUR_CONF; `sidecar_detect(src_h=1080, live_h=720)`
   scales boxes by 2/3 and keeps only LIVE_CATS.
7. End to end on the fixture clip: `detect` = sidecar replay wrapped in a 0.15 s sleep (stride ≈ 5), `delay_s=0.6`,
   `on_emit` records; run 2.5 s; stop. Invariant only: every emitted frame is blind **or** every sidecar face/plate
   box on that `clip_f` is ≥ 90% covered by `rects` (coverage from bench/run_bench.py). This would have caught R1's
   SEV-1s.
8. `detect` that raises → frames blind, `stats()["detector"]["errors"] > 0`, engine still emitting.
9. `stop()` returns within 2 s even right after `start()`, and wakes a pending `latest()`; `latest(after=n, 0.2)` →
   `None` when nothing newer; `latest(after=10**9, 0.2)` → newest.
10. `stats()` before any pass → nulls for percentiles, not an exception.
11. `BoxCache.boxes(...) == mask_boxes(...)` for every f in a scripted window, including `None`s.
12. Shape rules: face box 40% of frame height at 0.3 dropped, at 0.95 kept; plate 3:1 kept; square plate at 0.2
    dropped; square plate touching the frame edge kept.

**C2b. `censor/live_cli.py` + `tests/test_live_cli.py`:** `run --stub` on the fixture clip for 1.5 s writes a
playable MP4 (probe it: n_frames > 0, size == live size) and a report with the documented shape; `score` passes on a
report whose rects cover the sidecar-derived labels (build a tiny label file in the test) and exits 1, naming the
object and frame, when one labeled box is bare on a vouched frame; a blind frame is exempt.

**C3. `censor/server/live_routes.py` + `tests/test_live_routes.py` + app.py gate.** Tests use `TestClient` on a
fresh `FastAPI()` with the router and `make_replay` monkeypatched to a fake engine (no GPU, no clip): 202 then
`stats`; 409 on a second start; 204 on `frame` when idle; JPEG + headers when the fake has a frame; wrong `run` →
newest; `stop` → idle; height 2000 → 422; app without `CENSOR_LIVE` has no `/api/live/*` route, with it it does.

**C4. `web/live.html`** (self-contained, no build step, tokens copied from index.html's `:root`, ~150-200 lines):
clip picker from `../samples`, Start / Stop, one `<img>` fed by a fetch loop with two polls in flight
(`fetch('frame?run='+R+'&after='+n, {cache: 'no-store'})` → blob URL, revoke the previous one, drop out-of-order
frames; on non-2xx/204 back off 0.5 → 1 → 2 → 4 s), a permanent badge **"REPLAY of a recorded clip · delayed 1.0 s ·
showing N fps"** (N measured client-side over the last 2 s), a **"GPU contended"** flag when `detector.p95_ms` >
2 × `detector.baseline_p95_ms`, "warming up the detectors…" while `warming`, and a stats strip refreshed every
500 ms from `stats`: delay, pass p50 / p95, detected share, largest gap vs `hold_frames`, blind share, emit p95,
frame age p50. Relative URLs only. No link from index.html.

**C5. Server run + numbers.** Two steps on the Strix Halo, nothing else on the GPU (`rocm-smi --showuse`; session
> 90 min left).
1. **Transport probe (stub, no GPU, as soon as C3/C4 land):** `PORT=8001 CENSOR_LIVE=1 ./run.sh stub`, open
   `https://sjc.aupcloud.io/user/<user>/proxy/8001/api/live/live`, start `synthetic`, read the measured display
   fps. **≥ 12:** the page is the demo. **< 12:** the headless MP4 is the demo, the page a bonus; no transport work.
2. **Gate run (GPU):**
   ```bash
   git fetch && git checkout feat/live-replay
   PYTHONPATH=$PWD CENSOR_OFFLINE=1 python3 -m censor.live_cli run bench/clips/street_plates.mp4 --seconds 30 \
     --out logs/live_street_plates.mp4 --report logs/live_street_plates.json
   python3 -m censor.live_cli score logs/live_street_plates.json bench/labels/street_plates.json --min-plate 0.97
   PYTHONPATH=$PWD CENSOR_OFFLINE=1 python3 -m censor.live_cli run bench/clips/street_walk.mp4 --seconds 30 \
     --out logs/live_street_walk.mp4 --report logs/live_street_walk.json
   python3 -m censor.live_cli score logs/live_street_walk.json bench/labels/street_walk.json --min-plate 0.90
   ```
   then the page: `PORT=8001 CENSOR_LIVE=1 CENSOR_WARMUP=0 PYTHONPATH=$PWD CENSOR_OFFLINE=1 HF_HUB_OFFLINE=1
   YOLO_OFFLINE=true nohup python3 -m uvicorn censor.server.app:app --host 0.0.0.0 --port 8001 >> logs/app8001.log 2>&1 &`.
   Record in STACK.md: pass p50/p95, detected share, max gap, blind share, emit p95, frame age, display fps, `score`
   output. If `score` fails or blind > 1%: `--delay 1.5` once more; still failing → no-go.

**C6. Docs.** README "Using the app": a "Live replay" paragraph; IMPLEMENTATION.md M6 → done or "built, not demoed"
with numbers; DEVPOST one line; DEMO.md: no sample loads on :8000 within 2 min of a live run.

### 3.2 Timebox

| By (EDT) | Done |
|---|---|
| 04:35 | plan reviewed twice, C0 committed, C1-C4 dispatched in parallel |
| 05:45 | C1 + C2 + C2b green on the laptop |
| 06:30 | C3 + C4 integrated, stub mode on the laptop works; transport probe on the server |
| 07:30 | C5 gate run on the Strix Halo; numbers in STACK.md |
| **08:30** | **Go / no-go.** Go: `score` passes on both clips, blind < 1% after warm-up, a human watched each MP4 once. Then PR into `main`. No-go: keep the branch, write the numbers up as future work in DEVPOST. |
| 08:30–11:00 | not this feature: demo recording, fresh-clone run. **Record the main demo before any live GPU run**, or on another session. |

IMPLEMENTATION.md:319/360 says to drop M6 if it isn't working by Sun 05:00. This plan goes past it on purpose: the
engine is measured (STACK.md:40) and the work is on a branch that can't touch the demo; 08:30 is the replacement cut
line.

### 3.3 Stats (`stats()` and `GET /api/live/stats`): exact keys

`{"phase": "idle"|"warming"|"running", "running": bool, "warming": bool, "run": str, "clip": str, "size": [w, h],
"fps": float, "n_frames": int, "delay_s": float, "frames_out": int, "blind": int, "blind_share": float,
"dropped_out": int, "emit_errors": int, "detector": {"passes": int, "p50_ms", "p95_ms", "baseline_p95_ms", "errors": int},
"emit": {"p50_ms", "p95_ms"}, "detected_share": float, "gap_frames": {"mean", "max"}, "hold_frames": 15,
"frame_age_ms": {"p50"}}`. Percentiles via `bench.live_speed.stats_ms` guarded: `null` when there are no samples.
`baseline_p95_ms` = p95 of the first 50 passes; `p50_ms`/`p95_ms` are over the last 100 passes. `detected_share` =
passes / frames arrived at the source (not frames sent, which lag by the delay); `gap_frames` over consecutive
detection frame numbers; `frame_age_ms` = emit time − t_arrive.

### 3.4 Computation budget (720p; the street clips are 25 fps, synthetic 30; STACK.md:38-40)

- Detector: 2 × 58 ms ≈ 116 ms a pass; latest-wins means it runs back to back: **~100% GPU duty** while live is on.
- Emitter CPU per frame: cached box lookup, `apply_effect` per box, downscale + JPEG ~5 ms, full-frame blur ~7 ms
  when it happens; the track rebuild (10-20 ms) once per new detection. Emit p95 is in the stats.
- Delay 1.0 s ≥ stride (4-5) + p99 pass (500 ms) + encode. With the vouch rule, a slow pass shows as frozen frames.
- Memory: ~1.3 GB for EgoBlur in a second process (STACK.md:34); 64 GB shared, 18.9 GB peak for the app.
- STACK's 116 ms was measured with frames decoded ahead, a sync per model and an idle CPU; the iGPU shares memory
  bandwidth with the CPU, so C5 measures the real pass under decode + encode load.
- **Contention rule:** never run live while the :8000 app is scanning or the demo is being recorded; the router is
  off on :8000; DEMO.md gets the 2-minute rule; the page's "GPU contended" flag shows it when it happens anyway.

## 4. Decisions

| ID | Question | Options | Default |
|---|---|---|---|
| D1 | Frame transport | long-poll JPEG (2 in flight) / MJPEG / WebSocket | long-poll JPEG (proxy-safe) |
| D2 | Confirmation rules live | shape rules only / none / full CONFIRM | shape rules only |
| D3 | Resolution | 720p / 1080p | 720p. 1080p: stride 9 = LINK_GAP 9, any jitter splits tracks |
| D4 | Blind frame look | freeze last vouched frame + badge / full-frame blur | freeze |
| D5 | Demo vehicle if the proxy shows < 12 fps | headless MP4 / keep tuning transport | MP4 |
| D6 | Gate | recall ≥ offline − 0.03 per category / zero bare boxes | recall-based (zero-bare is unreachable without recheck_tiny/absorb) |

**Open edge (code review, not fixed):** the face shape rules are judged on the keyframes inside the window
(WINDOW_BACK + HOLD_FRAMES frames), so a big face whose only confident detections are older than the window could be
dropped by the big-faint-face rule while still on screen. On the labeled clips every real big face scored 1.00 on
every detection, so it has not been seen; a fix would carry each track's max confidence across rebuilds.

## 6. Camera mode (added 04:50, owner's request: "blur a live video so I can use my camera")

**What changes:** the *source*, nothing else. The engine, the vouch rule, the shape rules, the routes' state machine
and the page's frame loop stay as they are. The camera is on the laptop and the GPU is on the AUP server, so frames
go **up** (browser → server) and masked frames come **back** over the existing `/frame` long-poll. Glass-to-glass
latency ≈ upload RTT + 1.0 s delay buffer + download RTT ≈ 1.5-2 s through Cloudflare; the badge says so.

```
 browser: getUserMedia(1280x720) ──► <video> ──► canvas (CAM_W wide) ──► toBlob(jpeg, 0.6) every CAM_INTERVAL ms
          ──► POST /api/live/push (image/jpeg, ≤ 2 uploads in flight; a tick with both busy is skipped: latest wins)
 server:  PushSource.push(jpeg) ──► imdecode ──► resize to live height ──► ring (f, f, img, t_arrive=now)
          detector / emitter / BoxCache / badge / latest(): unchanged
 browser: Source monitor = the raw <video> (what the camera sees now); Program monitor = /frame (masked, delayed)
```

**Engine (`censor/live.py`):** `LiveReplay(clip=None, detect, height, delay_s, out_w, on_emit, push=True)`. In push mode
the source thread is not started; `push(jpeg: bytes) -> int` decodes, resizes to `height` (aspect kept, even width),
sets `self.size` from the first frame (under the lock; `size == (0, 0)` before it, `stats()["size"]` shows that),
appends `_Slot(f, f, img, time.monotonic())`, bumps `frames_in`, notifies, returns `f`. `fps` is nominal 30 for the
ring depth; `stats()` adds `"input_fps"` (frames pushed in the last 2 s / 2). `n_frames = 0`, `clip = "camera"`.
A bad JPEG raises `ValueError` (the route answers 400) and is not counted. Everything downstream paces on
`t_arrive + delay_s`, which already works per slot (no constant-fps assumption). **Frame numbers follow elapsed
time, not arrival count:** `f = max(last_f + 1, round((now - t0) * 30))`. The vouch rule and HOLD_FRAMES are in frame
numbers; if a 2 s Wi-Fi stall were followed by a burst numbered consecutively, two detections 2 s apart in time
would be 1 frame apart in numbering and a box would be interpolated across the stall, i.e. masked in the wrong
place. With time-based numbers the stall is a ~60-frame gap, which the vouch rule refuses to bridge (blind frames,
frozen picture). The emitter takes the first ring slot with `f >= next_f`; never-arrived numbers count as `gaps_in`.

**Routes (`censor/server/live_routes.py`):** `POST /start {"clip": "camera", "height": 540}` → push mode (no sample
file lookup, no sidecar; real mode warms EgoBlur at `(height, round(height * 16/9 / 2) * 2)`; stub mode uses
`yunet_detect` if `CENSOR_YUNET` points at the ONNX file, else a detector that returns no boxes **and** the start
answer and `/stats` carry `"detector_kind": "none"` (`"egoblur"` / `"yunet"` otherwise; the key `detector` is already
the engine's pass-timing dict) so the page can show "no detector loaded: nothing is masked"). `POST /push`:
body = JPEG bytes (`content-type: image/jpeg`), `X-Live-Run` header must match (else 409), 413 above 2 MB, 409 when
not running or not push mode, 400 on a bad JPEG; returns `{"f": n}`. Counts as a poll for the auto-stop watchdog.

**Page (`web/live.html`):** first entry of the picker is **"Camera (this device)"**; picking it asks for the camera
on Start (`{video: {width: {ideal: 1280}, height: {ideal: 720}, facingMode: "user"}, audio: false}`), shows the raw
feed in a second, smaller monitor labeled **Source (now)** next to the masked one labeled **Program (delayed)**, and
runs the upload loop: a canvas `CAM_W` wide (`960` default, selectable 640 / 960 / 1280; height follows the video's
aspect), `toBlob("image/jpeg", 0.6)` every `CAM_INTERVAL = 80` ms (≈12 fps), at most two uploads in flight, counts
of pushed / skipped frames in the stats strip, and the measured upload fps. The badge reads **"LIVE camera · delayed
1.0 s · showing N fps"** (the word REPLAY only for clips). Stop releases the camera (`track.stop()`). Camera access
needs a secure context: `https://` on the proxy, or `http://localhost` on the laptop. A denied permission shows an
inline error. Bandwidth: 960-wide q0.6 ≈ 30-40 KB × 12 fps ≈ 3-4 Mbit/s upstream; 640 for weak Wi-Fi.

**Laptop detector (stub mode) `yunet_detect(onnx_path, live_w, live_h) -> Detect`:** OpenCV's `FaceDetectorYN`
(CPU, ~15-25 ms at 720p), faces only, scores ≥ 0.6, boxes widened by `FACE_EXPAND`; no plates. Model file:
`face_detection_yunet_2023mar.onnx` from opencv_zoo (~230 KB; the laptop may fetch it, the server never needs it).
This exists so the camera demo works on a laptop with no GPU; on the server EgoBlur is mandatory as before.

**Camera-mode review (05:10, folded):** (SEV-1) the big-faint-face rule (car backs) would drop every webcam face
scoring < 0.8, since all of them are "big" at 540p → push mode runs with `face_rule=None` (every face at or above
EGOBLUR_CONF is masked); the clip path keeps the rule. (SEV-2) uploads get an 800 ms timeout each and 3 in flight
(a half-dead link must not hold the slots; frames that arrive late get a late `f` from arrival time, and a long-held
frame would mis-time the hold speed); `push()` reads the JPEG's SOF size before decoding and rejects > 4096×2304
(a 600-byte JPEG with patched dimensions made `imdecode` allocate 1.2 GB); the badge says "{delay} s buffer +
network" and the page measures camera→screen from the `/push` answer to the shown frame (the server's
`frame_age_ms` omits both network legs; expect ~1.3-1.8 s); privacy wording: frames leave the laptop **unmasked**
over HTTPS through the hub to the server, which stores nothing (footer, README, DEVPOST; never say "nothing leaves
the machine" in the camera segment of the demo). Camera acceptance: ≥ 8 fps shown and ≤ 1.8 s camera→screen (the 12
fps upload loop is a ceiling, 2-3 uploads in flight at ~150-200 ms each). Bind the live server to **127.0.0.1** on
the GPU box (the hub proxy reaches localhost; 0.0.0.0 would expose unauthenticated `/push` to the cluster). Check
`getUserMedia` once in the console on the real proxy URL before the demo (a cross-origin iframe or a
Permissions-Policy header would show as NotAllowedError). Keep the camera page in its own window: a hidden tab's
timers throttle and the 30 s watchdog would stop the run. Deferred: warm EgoBlur at the page's real canvas size (a
4:3 camera gives 720×540, only the first passes are slow); `gaps_in` is noisy at 12 fps against the nominal 30.

**Gate for camera mode:** visual. The owner sits in front of the camera on the server page and must never see
their own face bare on the Program monitor while moving, turning and walking in from the side; blind frames freeze
with the badge as before. The clip gate (`score`) is unchanged and still decides the go/no-go for the feature.

**Out of scope still:** audio, plates on the laptop detector, recording camera output, more than one viewer.

**Commits:** C7 engine `push()` + `yunet_detect` + tests (`push` end to end with scripted JPEGs from the fixture
clip; bad JPEG; size set on first frame; input_fps); C8 routes `camera` start + `/push` + tests (409/413/400/run
mismatch; a push refreshes the watchdog); C9 page camera mode. Then: laptop test with a real camera at
`http://localhost:8001/api/live/live` (stub + YuNet), then the server over the proxy with EgoBlur.

## 5. Adversarial review

- **R1 correctness (folded):** maskable needs a ready detection on both sides of f; lead-in sized to the real gap
  (`lead_frames`, union box). Target's real fields; emitter try/except; `detect(f, frame)` so the stub replay works;
  warm-up before the clock; no circular import; e2e test made meaningful and portable; `stats_ms([])` guard; lock
  discipline; GPU duty ~100%; 2 polls in flight; 1080p ruled out.
- **R1 computation / failure modes (folded):** `CENSOR_LIVE=1` gate; blind = freeze + badge (D4); `BoxCache`; shape
  rules live (D2); `.jpg`-less URL + `run` id + `no-store`; display downscale; client backoff; 503 when EgoBlur is
  missing; `stop()` wakes polls; headless CLI + label scorer as gate and fallback (D5); transport probe first; "GPU
  contended" flag; emit p95; measurement caveats; the 05:00 cut-line override made explicit. Rejected: WebSocket.
- **R2 implementation readiness / fold validation (folded):** SEV-1s: plate shape checks are unreachable through
  `confirmed()` without `min_conf` → `plate_ok` in live.py; a/b rule still left a+16..b-4 bare on gaps > 16 →
  `b.f - a.f <= HOLD_FRAMES + 1`; page URLs under the `/api/live` prefix; the zero-bare gate was unreachable → D6.
  Pinned: one `mask_boxes` signature, `dets.get`, route state machine (202 + background warm-up, auto-stop on no
  polls, stub skips the 503), `latest()` semantics, response codes, stats keys, threading rules (`stop_event.wait`,
  `frame.copy()`, eviction), ffmpeg command without audio/-shortest, report and label formats, sidecar shape, CLI in
  its own module so two agents don't share a file, commit prefixes. Contradictions fixed (freeze vs blur, JPEG 75,
  `LIVE_OUT_W`, 25 fps clips).
