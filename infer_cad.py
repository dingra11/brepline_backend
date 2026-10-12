"""Load base model + saved LoRA adapter, generate CadQuery, and (optionally) score it by EXECUTING it.

  python infer_cad.py --prompt "Create a CadQuery model of a rectangular plate 80 x 50 x 8 mm ..."
  python infer_cad.py --eval /kaggle/working/cad_synth/val.jsonl --n 100     # needs `pip install cadquery`

Metrics printed (these, not training loss, tell you if the model got better on complex parts):
  exec_ok      generated code ran and produced one valid solid
  bbox_ok      bounding box within 2% (or 0.1 mm) of the target
  vol_ok       volume within 5% of the target
  broken by tier / family, so you see where it fails.
"""
import argparse, json, math, re, collections, signal, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-Coder-7B"
ADAPTER = "/kaggle/working/cad_lora_adapter"
SYSTEM = "You are a Large Mechanical Model. Generate Python CadQuery code for the requested geometry."

def load(adapter):
    cap = torch.cuda.get_device_capability(0)
    dt = torch.bfloat16 if cap[0] >= 8 else torch.float16
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt)
    tok = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=q, device_map={"": 0}, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, adapter)      # same base model name as in adapter_config.json
    model.eval()
    return tok, model

@torch.no_grad()
def generate(tok, model, prompt, max_new=2048):
    text = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

def extract(s):
    m = re.search(r"```python\n(.*?)(```|$)", s, re.S)
    return (m.group(1) if m else s).strip()

class TO(Exception): pass
def run(code):
    import cadquery as cq
    ns = {"cq": cq, "math": math}
    alarm = hasattr(signal, "SIGALRM")
    if alarm: signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TO())); signal.alarm(30)
    try: exec(code, ns)
    finally:
        if alarm: signal.alarm(0)
    res = ns["result"]
    if len(res.solids().vals()) != 1: raise ValueError("not a single solid")
    s = res.val()
    if not s.isValid(): raise ValueError("invalid")
    bb = s.BoundingBox()
    return [bb.xlen, bb.ylen, bb.zlen], s.Volume()

def close(a, b, rel, ab=0.1): return abs(a - b) <= max(ab, rel * abs(b))

def evaluate(tok, model, path, n):
    rows = [json.loads(l) for l in open(path)][:n]
    tot, ok = collections.Counter(), collections.defaultdict(collections.Counter)
    for i, r in enumerate(rows):
        m = r["meta"]; key = (f"ops{min(m['n_ops'], 6)}", m["family"], m.get("class", "major"))
        try:
            code = extract(generate(tok, model, m["prompt"]))
            bb, vol = run(code)
            e = True
            bbok = all(close(x, y, .02) for x, y in zip(sorted(bb), sorted(m["bbox"])))
            vok = close(vol, m["volume"], .05, 1.0)
        except Exception as ex:
            e = bbok = vok = False
        for k in ("all",) + key:
            tot[k] += 1; ok[k]["exec_ok"] += e; ok[k]["bbox_ok"] += bbok; ok[k]["vol_ok"] += vok
        if (i + 1) % 10 == 0: print(f"{i+1}/{len(rows)}", flush=True)
    for k in sorted(tot):
        print(f"{k:14s} n={tot[k]:4d} " + " ".join(f"{m}={100*ok[k][m]/tot[k]:5.1f}%" for m in ("exec_ok", "bbox_ok", "vol_ok")))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER); ap.add_argument("--prompt"); ap.add_argument("--eval"); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    tok, model = load(a.adapter)
    if a.prompt: print(extract(generate(tok, model, a.prompt)))
    if a.eval: evaluate(tok, model, a.eval, a.n)
