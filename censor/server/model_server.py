"""Resident model server: loads Qwen3-VL-8B ONCE (~3.5 min from NFS) and serves both jobs.

  POST /vlm/yesno  {"images": [b64 jpg...], "question": str}         -> {"answer": "yes"|"no"|"unknown", "p_yes": float, "conf": float, "ms": int}
  POST /llm/chat   {"messages": [{"role","content"}...], "max_new_tokens": int} -> {"text": str, "tokens": int, "ms": int}
  GET  /health

Run on the GPU server:  python3 -m censor.server.model_server   (port 8100; set CENSOR_MODEL_URL=http://127.0.0.1:8100)
"""
from __future__ import annotations

import base64
import io
import os
import threading
import time

from censor import netguard

netguard.install_from_env()  # before torch/transformers import, so model loading is covered too
# MIOpen's kernel database isn't readable on the AUP image, so by default every new convolution shape (the VLM's
# vision encoder sees one per image size) triggers a kernel search and compile of several seconds. FAST uses what's
# tuned already and falls back to a default kernel right away.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import torch  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

MID = os.environ.get("CENSOR_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

app = FastAPI(title="censor model server")
_lock = threading.Lock()  # one generate() at a time on the GPU
_t0 = time.time()
proc = AutoProcessor.from_pretrained(MID)
model = AutoModelForImageTextToText.from_pretrained(MID, torch_dtype=torch.bfloat16, device_map="cuda")
LOAD_S = time.time() - _t0


def _generate(messages: list[dict], max_new_tokens: int) -> tuple[str, int, int]:
    inp = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to("cuda")
    with _lock, torch.inference_mode():
        t = time.time()
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        ms = int((time.time() - t) * 1000)
    new = out[0, inp["input_ids"].shape[1]:]
    return proc.decode(new, skip_special_tokens=True).strip(), len(new), ms


class YesNo(BaseModel):
    images: list[str]
    question: str


class Chat(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 256


@app.get("/health")
def health():
    return {"model": MID, "load_s": round(LOAD_S), "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1e9, 1),
            "offline": netguard.active()}


# Token ids that start a yes or a no answer, found once from the tokenizer.
_YES_NO = {w: proc.tokenizer.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", "no", "No", "YES", "NO")}
# p_yes between these is "unknown", which the agent treats as a match (fail closed). Outside them the model is sure
# enough: a clear "no" leaves the person unmasked instead of blurring everyone it hesitated on.
UNSURE = (float(os.environ.get("CENSOR_VLM_NO_BELOW", "0.35")), float(os.environ.get("CENSOR_VLM_YES_ABOVE", "0.65")))


@app.post("/vlm/yesno")
def yesno(req: YesNo):
    """P(yes) from the model's first answer token, rather than parsing generated text: the text gave only yes/no
    with no way to tell a sure answer from a guess."""
    content = [{"type": "image", "image": Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")}
               for b in req.images]
    content.append({"type": "text", "text": req.question + "\nThe images are crops of the same object over time. "
                    "Answer only yes or no."})
    inp = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to("cuda")
    with _lock, torch.inference_mode():
        t = time.time()
        logits = model(**inp).logits[0, -1].float()
        ms = int((time.time() - t) * 1000)
    probs = torch.softmax(logits, -1)
    py = float(sum(probs[i] for w, i in _YES_NO.items() if w.lower() == "yes"))
    pn = float(sum(probs[i] for w, i in _YES_NO.items() if w.lower() == "no"))
    p_yes = py / (py + pn) if py + pn > 0 else 0.5
    answer = "yes" if p_yes >= UNSURE[1] else "no" if p_yes <= UNSURE[0] else "unknown"
    return {"answer": answer, "p_yes": round(p_yes, 3), "conf": round(abs(p_yes - 0.5) * 2, 3), "ms": ms,
            "raw": proc.tokenizer.decode([int(logits.argmax())]).strip(), "yes_no_mass": round(py + pn, 3)}


@app.post("/llm/chat")
def chat(req: Chat):
    msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            if isinstance(m.get("content"), str) else m for m in req.messages]
    text, n, ms = _generate(msgs, req.max_new_tokens)
    return {"text": text, "tokens": n, "ms": ms}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CENSOR_MODEL_PORT", "8100")))
