"""Serve YOUR fine-tuned model (Qwen2.5-Coder-7B + LoRA adapter) as an OpenAI-compatible API,
so the backend can use it with no code changes. Run this on a GPU machine (Kaggle / Colab / cluster).

  pip install fastapi uvicorn transformers peft bitsandbytes accelerate
  BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct ADAPTER_PATH=/kaggle/working/v2_lmm \
      uvicorn serve_model:app --host 0.0.0.0 --port 8000

Then expose it (e.g. `cloudflared tunnel --url http://localhost:8000`) and in the backend .env:
  LLM_BASE_URL=https://<tunnel>.trycloudflare.com/v1
  LLM_API_KEY=anything
  LLM_MODEL=brepline
  LLM_REASONING_EFFORT=          (leave empty)
  N_CAND=2                       (one GPU: candidates run one after another)
Set ADAPTER_PATH empty to serve the plain base model (useful for comparing).
"""
import os, time, uuid, threading
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
ADAPTER = os.getenv("ADAPTER_PATH", "")
NAME = os.getenv("SERVED_NAME", "brepline")

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16))
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
LOCK = threading.Lock()  # one generation at a time on a single GPU
app = FastAPI()


class Msg(BaseModel):
    role: str
    content: str


class Req(BaseModel):
    model: str | None = None
    messages: list[Msg]
    temperature: float = 0.2
    max_tokens: int = 1500


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat(req: Req):
    text = tok.apply_chat_template([m.model_dump() for m in req.messages],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with LOCK, torch.no_grad():
        out = model.generate(**ids, max_new_tokens=min(req.max_tokens, 1500),
                             do_sample=req.temperature > 0,
                             temperature=max(req.temperature, 0.01), top_p=0.95,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][ids["input_ids"].shape[1]:]
    content = tok.decode(gen, skip_special_tokens=True)
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:10], "object": "chat.completion",
            "created": int(time.time()), "model": NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ids["input_ids"].shape[1], "completion_tokens": len(gen),
                      "total_tokens": ids["input_ids"].shape[1] + len(gen)}}