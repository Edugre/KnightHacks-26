# Benchmark report

Generated 2026-10-11T10:51:37+00:00 by `bench/run_bench.py` at commit `5f38685`. Do not edit by hand.

- Clips: 3 (street_talk, street_walk, street_plates)
- Detector: RoutedDetector. Agent: LLM via CENSOR_MODEL_URL, template fallback. VLM: Qwen3-VL via the model server.
- A labeled box is covered when at least 90% of its area is masked. A request passes when each must-mask object is covered on at least 90% of its labeled frames and each must-stay-visible object on at most 10%.
- Keyword baseline: request nouns go straight to the detector and every hit is masked (no attribute check, no times). Fail-closed off: no hold-after-loss, no extra padding when uncertain, VLM 'unknown' read as 'no'.

## Targets (IMPLEMENTATION.md section 6)

| Metric | Target | Measured | Pass |
| --- | --- | --- | --- |
| Recall: faces | >= 0.95 | 0.95 | FAIL |
| Recall: plates | >= 0.90 | 0.98 | pass |
| Recall: screens/text | >= 0.85 | n/a | n/a |
| Over-blur (mean masked non-target area) | <= 10% of frame | 14.9% | FAIL |
| Request success | >= 15 / 20 (75%) | 14 / 20 | FAIL |
| Fail-closed rate on uncertain keyframes | 100% | 100.0% | pass |
| Offline speed (target is for 1080p) | >= 0.5x real time | 0.4x | FAIL |
| Peak GPU memory, all models loaded | <= 28 GB | 19.1 GB | pass |
| Longest unmasked run of a labeled target | <= 1 labeled frame | 4 | FAIL |

## Recall by category

| Category | Labeled boxes | Ours | Fail-closed off |
| --- | --- | --- | --- |
| face | 337 | 0.95 | 0.93 |
| license plate | 42 | 0.98 | 0.95 |
| person | 225 | 0.99 | 0.97 |

## Objects

| Clip | Object | Category | Covered frames | Longest miss (labeled frames) |
| --- | --- | --- | --- | --- |
| street_talk | face_1 | face | 100% | 0 |
| street_talk | face_2 | face | 100% | 0 |
| street_talk | face_3 | face | 100% | 0 |
| street_talk | person_1 | person | 100% | 0 |
| street_talk | person_2 | person | 100% | 0 |
| street_talk | person_3 | person | 96% | 1 |
| street_walk | face_1 | face | 100% | 0 |
| street_walk | face_2 | face | 100% | 0 |
| street_walk | face_3 | face | 100% | 0 |
| street_walk | person_1 | person | 100% | 0 |
| street_walk | person_2 | person | 100% | 0 |
| street_walk | person_3 | person | 100% | 0 |
| street_walk | license_plate_1 | license plate | 100% | 0 |
| street_walk | license_plate_3 | license plate | 100% | 0 |
| street_walk | license_plate_4 | license plate | 100% | 0 |
| street_walk | license_plate_6 | license plate | 100% | 0 |
| street_walk | license_plate_7 | license plate | 100% | 0 |
| street_plates | license_plate_1 | license plate | 100% | 0 |
| street_plates | face_1 | face | 100% | 0 |
| street_plates | face_2 | face | 100% | 0 |
| street_plates | face_3 | face | 75% | 3 |
| street_plates | license_plate_2 | license plate | 91% | 1 |
| street_plates | face_4 | face | 100% | 0 |
| street_plates | face_5 | face | 100% | 0 |
| street_plates | face_6 | face | 83% | 1 |
| street_plates | face_9 | face | 78% | 1 |
| street_plates | face_11 | face | 56% | 4 |
| street_plates | license_plate_3 | license plate | 100% | 0 |
| street_plates | face_12 | face | 20% | 2 |
| street_plates | face_8 | face | 20% | 4 |
| street_plates | face_7 | face | 100% | 0 |
| street_plates | face_10 | face | 100% | 0 |
| street_plates | license_plate_5 | license plate | 100% | 0 |
| street_plates | license_plate_6 | license plate | 100% | 0 |
| street_plates | license_plate_4 | license plate | 100% | 0 |
| street_plates | face_13 | face | 100% | 0 |

## Over-blur and fail-closed

| Clip | Over-blur | Over-blur, fail-closed off | Uncertain keyframes | Doubled padding |
| --- | --- | --- | --- | --- |
| street_talk | 15.7% | 15.7% | 255 | 100% |
| street_walk | 19.1% | 19.0% | 140 | 100% |
| street_plates | 9.9% | 5.7% | 232 | 100% |

## Requests

Ours: 14 / 20, Keyword baseline: 7 / 20, Fail-closed off: 11 / 20.

| Clip | Request | Path | Ours | Keyword baseline | Fail-closed off | Seconds | Why ours failed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| street_talk | blur the woman in the salmon tee | template | FAIL | FAIL | FAIL | 14.24 | missed person_3 |
| street_talk | blur the man in the denim shirt | template | FAIL | FAIL | FAIL | 14.85 | missed person_2 |
| street_talk | blur the man in the navy tee | template | pass | FAIL | pass | 14.49 |  |
| street_talk | pixelate everyone | template | FAIL | FAIL | FAIL | 0.00 | missed person_3 |
| street_talk | blur everyone except the woman | llm | pass | FAIL | pass | 23.04 |  |
| street_talk | blur faces in the middle | template | pass | pass | pass | 0.00 |  |
| street_talk | unblur the woman (after: blur everyone) | template | pass | FAIL | pass | 14.25 |  |
| street_talk | blur the faces in the first 5 seconds | template | pass | pass | FAIL | 0.00 |  |
| street_walk | blur all faces | template | pass | pass | pass | 0.00 |  |
| street_walk | blur the plates | template | pass | pass | pass | 0.00 |  |
| street_walk | blur the guy in the red hoodie | template | pass | FAIL | pass | 9.18 |  |
| street_walk | blur the woman | template | pass | FAIL | pass | 9.85 |  |
| street_walk | blur everyone except the guy in the red hoodie | llm | pass | FAIL | pass | 30.08 |  |
| street_walk | blur faces in the first 2 seconds | template | pass | pass | FAIL | 0.00 |  |
| street_walk | unblur the plates (after: blur all plates and faces) | template | pass | FAIL | pass | 0.00 |  |
| street_plates | blur all plates | template | pass | pass | FAIL | 0.00 |  |
| street_plates | blur faces and plates | template | FAIL | FAIL | FAIL | 0.00 | missed face_3, face_6, face_8, face_9, face_11, face_12 |
| street_plates | blur the plates in the last 3 seconds | template | pass | pass | pass | 0.00 |  |
| street_plates | undo (after: blur all faces; blur all plates) | template | FAIL | FAIL | FAIL | 0.00 | missed face_3, face_6, face_8, face_9, face_11, face_12 |
| street_plates | black box every face | template | FAIL | FAIL | FAIL | 0.00 | missed face_3, face_6, face_8, face_9, face_11, face_12 |

## Latency

| Clip | Resolution | Frames | Detect ms/frame | Render ms/frame | Offline speed | Agent s, mean / max |
| --- | --- | --- | --- | --- | --- | --- |
| street_talk | 1920x1080 | 514 | 82.3 | 10.2 | 0.4x | 10.11 / 23.04 |
| street_walk | 1280x720 | 221 | 105.4 | 3.5 | 0.4x | 7.01 / 30.08 |
| street_plates | 1920x1080 | 325 | 99.7 | 4.8 | 0.4x | 0.00 / 0.00 |

GPU memory: bench process (detector) 1.52 GB, model server 17.6 GB.
