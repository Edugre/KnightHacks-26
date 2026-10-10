"""Video probing and frame iteration (OpenCV; no ffprobe needed)."""
from __future__ import annotations

import queue
import threading
from typing import Iterator

import cv2
import numpy as np

from censor.core.timeline import VideoInfo


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    info = VideoInfo(
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def frames(path: str, every: int = 1, prefetch: int = 0) -> Iterator[tuple[int, np.ndarray]]:
    """Frames (index, BGR) in order; with every=n only every n-th frame. Skipped frames are grabbed but not
    converted, which is cheaper. With prefetch > 0 a CPU thread decodes up to that many frames ahead, so the
    GPU doesn't wait for the decoder (OpenCV releases the GIL while it decodes)."""
    if prefetch > 0:
        yield from _prefetched(path, every, prefetch)
        return
    cap = cv2.VideoCapture(str(path))
    i = 0
    try:
        while True:
            if i % every:
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, frame
            i += 1
    finally:
        cap.release()


def _prefetched(path: str, every: int, depth: int) -> Iterator[tuple[int, np.ndarray]]:
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    end = object()

    def put(item) -> bool:  # False once the consumer has gone away
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def decode():
        try:
            for item in frames(path, every):
                if not put(item):
                    return
        except Exception as e:  # noqa: BLE001  handed to the consumer below
            put(e)
        put(end)

    t = threading.Thread(target=decode, daemon=True, name="decode")
    t.start()
    try:
        while (item := q.get()) is not end:
            if isinstance(item, Exception):
                raise item
            yield item
    finally:  # the consumer stopped early or failed: let the decoder exit
        stop.set()
        t.join(timeout=5)


def read_frame(path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IndexError(f"frame {index} not readable in {path}")
    return frame
