"""Check that inference runs with the network off (IMPLEMENTATION.md success condition 4). GPU server.

  python3 scripts/check_offline.py [clip.mp4]

With the egress guard on, confirms the Qwen3-VL cache is complete, YOLOE and MobileCLIP load without
downloading (from the repo root, like run.sh) and, given a clip, that a detection pass makes no network
call. It never loads the 18 GB VLM, so it's safe to run while the model server is up. Exit code 1 on failure.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

LOG = Path(tempfile.gettempdir()) / f"egress_check_{os.getpid()}.log"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", YOLO_OFFLINE="true", CENSOR_OFFLINE="1",
                  CENSOR_EGRESS_LOG=str(LOG))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from censor import netguard  # noqa: E402

netguard.install_from_env()
MID = os.environ.get("CENSOR_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
failed = False


def check(name: str, fn) -> None:
    global failed
    try:
        print(f"ok    {name}: {fn()}")
    except Exception as e:  # report every check, then fail at the end
        failed = True
        print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:300]}")


def vlm_cache() -> str:
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(MID, local_files_only=True))
    shards = set(json.loads((path / "model.safetensors.index.json").read_text())["weight_map"].values())
    missing = sorted(s for s in shards if not (path / s).exists())
    if missing:
        raise FileNotFoundError(f"missing shards {missing}; run scripts/setup.sh")
    from transformers import AutoProcessor
    AutoProcessor.from_pretrained(MID)
    return f"{len(shards)} shards + processor in {path}"


def yoloe_weights() -> str:
    os.chdir(ROOT)  # run.sh runs from the repo root, and Ultralytics resolves its weights_dir ("weights") from here
    from ultralytics import YOLOE
    m = YOLOE("yoloe-11s-seg.pt")
    m.set_classes(["person"], m.get_text_pe(["person"]))  # loads MobileCLIP
    return "yoloe-11s-seg.pt and mobileclip_blt.ts load with downloads disabled"


def detect(clip: str) -> str:
    from censor.core.tracks import TrackStore
    from censor.vision.detect import YoloeDetector
    store = TrackStore()
    YoloeDetector().run(clip, ["person", "face", "license plate"], store)
    return f"{len(store.tracks)} tracks"


check("Qwen3-VL cache", vlm_cache)
check("YOLOE + MobileCLIP weights", yoloe_weights)
if len(sys.argv) > 1:
    clip = str(Path(sys.argv[1]).resolve())
    check(f"detection on {Path(clip).name}", lambda: detect(clip))
blocked = LOG.read_text().strip() if LOG.exists() else ""
print("egress log: empty, nothing tried to leave the machine" if not blocked else f"egress log:\n{blocked}")
sys.exit(1 if failed or blocked else 0)
