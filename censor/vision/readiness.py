"""Read-only detector readiness, separate from the locks and work that load models.

Configuration says what would run; only a successful warm-up or detection run says
it is ready. Observations survive the lightweight detector wrappers made per session.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path

_lock = threading.Lock()
_selection: dict | None = None
_observed: dict[str, dict[str, str]] = {}
_IDS = ("objects", "face", "plate")
_LABELS = {"objects": "People and objects", "face": "Faces", "plate": "License plates"}


def ultralytics_available() -> bool:
    """Check the package without importing it or constructing a model."""
    try:
        return find_spec("ultralytics") is not None
    except (ImportError, ValueError):
        return False


def _configuration() -> dict:
    if os.environ.get("CENSOR_STUB") == "1":
        return {"mode": "stub", "explicit": True, "reason": "Stub mode was selected."}
    if not ultralytics_available():
        return {"mode": "stub", "explicit": False, "reason": "Ultralytics is unavailable; using the stub fallback."}
    # Use the same resolved settings as the router when it has been imported. Before
    # that, reading its environment defaults avoids importing any vision dependency.
    specialist = sys.modules.get("censor.vision.specialist")
    model_dir = getattr(specialist, "EGOBLUR_DIR", Path(os.environ.get(
        "CENSOR_EGOBLUR_DIR", Path.home() / "datasets" / "egoblur")))
    files = getattr(specialist, "EGOBLUR_FILES", {
        "face": os.environ.get("CENSOR_EGOBLUR_FACE", "ego_blur_face_gen1.jit"),
        "license plate": os.environ.get("CENSOR_EGOBLUR_LP", "ego_blur_lp_gen1.jit"),
    })
    return {"mode": "real", "specialist": all((Path(model_dir) / name).exists() for name in files.values())}


def reset() -> None:
    """Forget observations (for isolated tests); the next snapshot reads configuration."""
    global _selection
    with _lock:
        _selection = None
        _observed.clear()


def select_real(specialist: bool) -> None:
    global _selection
    with _lock:
        _selection = {"mode": "real", "specialist": specialist}


def select_stub(reason: str, explicit: bool = False) -> None:
    global _selection
    with _lock:
        _selection = {"mode": "stub", "explicit": explicit, "reason": reason}


def failed(ids, action: str, exc: Exception) -> None:
    message = f"Last detector {action} failed: {type(exc).__name__}: {exc}"[:300]
    with _lock:
        for key in ids:
            _observed[key] = {"status": "unavailable", "message": message}


@contextmanager
def operation(ids, action: str, verified: set[str] | None = None):
    """Observe existing work; never hold our lock while model or video code runs."""
    ids = tuple(ids)
    with _lock:
        previous = {key: _observed.get(key) for key in ids}
        for key in ids:
            if _observed.get(key, {}).get("status") != "ready":
                _observed[key] = {"status": "loading", "message": f"Running detector {action}."}
    try:
        yield
    except Exception as exc:
        failed(ids, action, exc)
        raise
    else:
        with _lock:
            for key in ids:
                if verified is None or key in verified:
                    _observed[key] = {"status": "ready", "message": f"Detector {action} completed successfully."}
                elif previous[key] is not None:
                    _observed[key] = previous[key]
                else:
                    _observed.pop(key, None)


def snapshot() -> dict:
    """Return capabilities and observed state without loading or waiting for detectors."""
    with _lock:
        selection = dict(_selection) if _selection is not None else None
        observed = {key: dict(value) for key, value in _observed.items()}
    selection = selection or _configuration()
    if selection["mode"] == "stub":
        message = f"{selection['reason']} Only supplied tracks sidecars are replayed; no real detection runs."
        return {"mode": "stub", "status": "degraded" if selection["explicit"] else "unavailable",
                "message": message, "detectors": [
                    {"id": key, "label": _LABELS[key], "model": "StubDetector",
                     "status": "unavailable", "message": "Sidecar tracks only; no real detector."}
                    for key in _IDS]}
    rows = []
    for key in _IDS:
        fallback = key != "objects" and not selection["specialist"]
        state = observed.get("objects" if fallback else key, {
            "status": "not_loaded", "message": "Configured; waiting for the first successful detector run."})
        row = {"id": key, "label": _LABELS[key],
               "model": "YOLOE" if key == "objects" or fallback else "EgoBlur", **state}
        if fallback:
            # The router requires both specialist files; a partial installation uses
            # YOLOE for both categories too. Its success is not specialist readiness.
            row["status"] = "unavailable" if state["status"] == "unavailable" else "fallback"
            reliability = "Face detection is unreliable" if key == "face" else "Plate detection may miss targets"
            row["message"] = (f"Both EgoBlur model files are required. Using YOLOE fallback. "
                              f"{reliability}. {state['message']}")
        rows.append(row)
    statuses = {row["status"] for row in rows}
    if statuses == {"ready"}:
        status, message = "ready", "All detectors have completed a successful warm-up or detection run."
    elif statuses == {"unavailable"}:
        status, message = "unavailable", "Detector work failed. See each detector for details; a successful retry will update this status."
    elif "unavailable" in statuses or "fallback" in statuses:
        status, message = "degraded", "Some detector capabilities are limited or failed. Review the details before relying on automatic masks."
    elif "loading" in statuses:
        status, message = "loading", "Detectors are loading or completing their first successful run."
    else:
        status, message = "not_loaded", "Detectors are configured but have not all completed a successful run."
    return {"mode": "real", "status": status, "message": message, "detectors": rows}
