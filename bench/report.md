# Benchmark report

Generated 2026-10-10T07:26:14+00:00 by `bench/run_bench.py` at commit `682dcb0`. Do not edit by hand.

- Clips: 1 (synthetic)
- Detector: StubDetector. Agent: template only (CENSOR_MODEL_URL unset). VLM: stub, every answer is 'unknown' (fail closed).
- A labeled box is covered when at least 90% of its area is masked. A request passes when each must-mask object is covered on at least 90% of its labeled frames and each must-stay-visible object on at most 10%.
- Keyword baseline: request nouns go straight to the detector and every hit is masked (no attribute check, no times). Fail-closed off: no hold-after-loss, no extra padding when uncertain, VLM 'unknown' read as 'no'.
- Stub detector: it replays ground truth, so recall here checks the plumbing, not detection quality.

## Targets (IMPLEMENTATION.md section 6)

| Metric | Target | Measured | Pass |
| --- | --- | --- | --- |
| Recall: faces | >= 0.95 | 1.00 | pass |
| Recall: plates | >= 0.90 | 1.00 | pass |
| Recall: screens/text | >= 0.85 | 1.00 | pass |
| Over-blur (mean masked non-target area) | <= 10% of frame | 5.2% | pass |
| Request success | >= 15 / 20 (75%) | 6 / 7 (only 7 requests so far) | pass |
| Fail-closed rate on uncertain keyframes | 100% | n/a | n/a |
| Offline speed (target is for 1080p) | >= 0.5x real time | 2.4x | pass |
| Peak GPU memory, all models loaded | <= 28 GB | n/a | n/a |
| Longest unmasked run of a labeled target | <= 1 labeled frame | 0 | pass |

## Recall by category

| Category | Labeled boxes | Ours | Fail-closed off |
| --- | --- | --- | --- |
| face | 18 | 1.00 | 1.00 |
| license plate | 18 | 1.00 | 1.00 |
| person | 36 | 1.00 | 1.00 |
| screen | 9 | 1.00 | 1.00 |

## Objects

| Clip | Object | Category | Covered frames | Longest miss (labeled frames) |
| --- | --- | --- | --- | --- |
| synthetic | license_plate_1 | license plate | 100% | 0 |
| synthetic | face_1 | face | 100% | 0 |
| synthetic | person_1 | person | 100% | 0 |
| synthetic | person_2 | person | 100% | 0 |
| synthetic | screen_1 | screen | 100% | 0 |

## Over-blur and fail-closed

| Clip | Over-blur | Over-blur, fail-closed off | Uncertain keyframes | Doubled padding |
| --- | --- | --- | --- | --- |
| synthetic | 5.2% | 5.2% | 0 | n/a |

## Requests

Ours: 6 / 7, Keyword baseline: 5 / 7, Fail-closed off: 6 / 7.

| Clip | Request | Path | Ours | Keyword baseline | Fail-closed off | Seconds | Why ours failed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | blur all plates | template | pass | pass | pass | 0.00 |  |
| synthetic | pixelate every face and screen | template | pass | pass | pass | 0.01 |  |
| synthetic | blur the person in the red shirt | template | FAIL | FAIL | FAIL | 0.00 | masked person_2 |
| synthetic | black box the screen in the last 2 seconds | template | pass | pass | pass | 0.00 |  |
| synthetic | blur both people | template | pass | pass | pass | 0.00 |  |
| synthetic | unblur the faces (after: blur all plates and faces) | template | pass | FAIL | pass | 0.00 |  |
| synthetic | make it look cinematic | clarify | pass | pass | pass | 0.00 |  |

## Latency

| Clip | Resolution | Frames | Detect ms/frame | Render ms/frame | Offline speed | Agent s, mean / max |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic | 1280x720 | 180 | 0.1 | 14.0 | 2.4x | 0.00 / 0.01 |

GPU memory: bench process (detector) n/a, model server n/a.
