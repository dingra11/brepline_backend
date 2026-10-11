import os, time, uuid, threading, queue, contextlib, torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList)

BASE = os.environ["BASE_MODEL"]
ADAPTER = os.getenv("ADAPTER_PATH", "")
NAME = "brepline"
MAX_NEW = 1500

tok = AutoTokenizer.from_pretrained(BASE)
TOK_LOCK = threading.Lock()
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
EOS_IDS = sorted({i for i in (IM_END, tok.eos_token_id) if isinstance(i, int) and i >= 0})


def load(gpu):
    m = AutoModelForCausalLM.from_pretrained(
        BASE, device_map={"": gpu},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16))
    if ADAPTER:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, ADAPTER)
    return m.eval()


N_GPU = max(1, torch.cuda.device_count())
MODELS = []
for g in range(N_GPU):
    print(f"loading copy on GPU {g}", flush=True)
    MODELS.append(load(g))
FREE = queue.Queue()
for g in range(N_GPU):
    FREE.put(g)
print(f"ready: {N_GPU} model copies", flush=True)

app = FastAPI()


def finished(t):
    if t.count("```") >= 2:
        return True
    s = t.strip()
    return s.startswith("{") and s.count("{") > 0 and s.count("{") == s.count("}")


def trim(t):
    if t.count("```") >= 2:
        i = t.index("```")
        j = t.index("```", i + 3)
        return t[: j + 3]
    s = t.strip()
    if s.startswith("{") and "}" in s:
        return s[: s.rfind("}") + 1]
    return s


class DoneCode(StoppingCriteria):
    def __init__(self, start, prefill=""):
        self.start = start
        self.prefill = prefill

    def __call__(self, input_ids, scores, **kw):
        n = input_ids.shape[1] - self.start
        if n < 6 or n % 3:
            return False
        with TOK_LOCK:
            t = tok.decode(input_ids[0][self.start:], skip_special_tokens=True)
        return finished(self.prefill + t)


class Msg(BaseModel):
    role: str
    content: str


class Req(BaseModel):
    model: str | None = None
    messages: list[Msg]
    temperature: float = 0.2
    max_tokens: int | None = None
    prefill: str = ""
    adapter: bool = True


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat(req: Req):
    t0 = time.time()
    with TOK_LOCK:
        text = tok.apply_chat_template([m.model_dump() for m in req.messages],
                                       tokenize=False, add_generation_prompt=True)
        text += req.prefill
        enc = tok(text, return_tensors="pt")
    start = enc["input_ids"].shape[1]
    limit = min(req.max_tokens or MAX_NEW, MAX_NEW)
    gpu = FREE.get()
    try:
        print(f"[req] gpu={gpu} prompt_tokens={start} max_new={limit}", flush=True)
        m = MODELS[gpu]
        ids = enc.to(m.device)
        ctx = m.disable_adapter() if (not req.adapter and hasattr(m, "disable_adapter")) else contextlib.nullcontext()
        with ctx, torch.inference_mode():
            out = m.generate(
                **ids,
                max_new_tokens=limit,
                do_sample=req.temperature > 0,
                temperature=max(req.temperature, 0.01),
                top_p=0.95,
                repetition_penalty=1.05,
                eos_token_id=EOS_IDS,
                pad_token_id=tok.eos_token_id,
                stopping_criteria=StoppingCriteriaList([DoneCode(start, req.prefill)]),
            )
    finally:
        FREE.put(gpu)
    gen = out[0][start:]
    with TOK_LOCK:
        content = trim(req.prefill + tok.decode(gen, skip_special_tokens=True))
    reason = "stop" if len(gen) < limit else "length"
    print(f"[done] gpu={gpu} new_tokens={len(gen)} finish={reason} {time.time()-t0:.1f}s", flush=True)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:10],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": NAME,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": reason}],
        "usage": {"prompt_tokens": start,
                  "completion_tokens": len(gen),
                  "total_tokens": start + len(gen)},
    }
