"""M0 detector smoke test: YOLOE with text prompts on the GPU. Usage: python3 scripts/smoke_yoloe.py [imgsz]"""
import sys, time
import numpy as np
import torch
from ultralytics import YOLOE

imgsz = int(sys.argv[1]) if len(sys.argv) > 1 else 640
names = ["license plate", "face", "person", "screen"]
m = YOLOE("yoloe-11s-seg.pt")
m.set_classes(names, m.get_text_pe(names))

r = m.predict("https://ultralytics.com/images/bus.jpg", device=0, imgsz=imgsz, verbose=False)[0]
print("detections:", [(names[int(c)], round(float(s), 2)) for c, s in zip(r.boxes.cls, r.boxes.conf)])

frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
for _ in range(5):
    m.predict(frame, device=0, imgsz=imgsz, verbose=False)
torch.cuda.synchronize(); t = time.time(); n = 50
for _ in range(n):
    m.predict(frame, device=0, imgsz=imgsz, verbose=False)
torch.cuda.synchronize(); dt = time.time() - t
print(f"imgsz {imgsz}: fps {n/dt:.1f}  ms/frame {1000*dt/n:.1f}  peak GPU mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
