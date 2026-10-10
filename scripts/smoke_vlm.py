"""M0 VLM + agent smoke test (one shared Qwen3-VL model). Usage: python3 scripts/smoke_vlm.py"""
import json, time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MID = "Qwen/Qwen3-VL-8B-Instruct"  # fallback: Qwen/Qwen2.5-VL-7B-Instruct
t = time.time()
proc = AutoProcessor.from_pretrained(MID)
model = AutoModelForImageTextToText.from_pretrained(MID, torch_dtype=torch.bfloat16, device_map="cuda")
print(f"load: {time.time()-t:.0f}s  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")


def ask(content, max_new=64):
    msgs = [{"role": "user", "content": content}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to("cuda")
    torch.cuda.synchronize(); t = time.time()
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    torch.cuda.synchronize(); dt = time.time() - t
    new = out[0, inp["input_ids"].shape[1]:]
    return proc.decode(new, skip_special_tokens=True), dt, len(new)


img = Image.open("bus.jpg")  # downloaded by smoke_yoloe.py
ask([{"type": "image", "image": img}, {"type": "text", "text": "hi"}], 4)  # warmup
ans, dt, n = ask([{"type": "image", "image": img},
                  {"type": "text", "text": "Is anyone in this image wearing a dark jacket? Answer only yes or no."}], 8)
print(f"VLM check: {ans!r}  {dt*1000:.0f} ms")

tools = ("Tools: find_objects(query:str), add_masks(track_ids:list[str], effect:'blur'|'pixelate'). "
         "Reply ONLY with JSON: {\"tool\": ..., \"args\": {...}}")
ans, dt, n = ask([{"type": "text", "text": tools + "\nUser: blur every license plate in the video"}], 96)
print(f"agent: {ans.strip()}\n  {n} tokens in {dt:.2f}s = {n/dt:.1f} tok/s")
try:
    json.loads(ans.strip().strip('`').removeprefix('json'))
    print("  valid JSON")
except Exception as e:
    print("  NOT valid JSON:", e)
print(f"peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
