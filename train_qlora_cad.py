"""QLoRA fine-tuning of Qwen2.5-Coder-7B on a ChatML CAD dataset (Kaggle).

Design choices (all deliberate, see the notes in the chat reply):
  * Plain transformers.Trainer, no TRL: SFTTrainer/SFTConfig argument names changed across
    versions (tokenizer->processing_class, max_seq_length->max_length, ...). Plain Trainer is stable
    and lets us mask the prompt so loss is computed ONLY on the assistant (CadQuery) tokens.
  * Samples longer than MAX_LEN are DROPPED, never truncated (truncated code is invalid code and
    teaches the model to stop mid-statement). The script prints how many are dropped.
  * Every trainable parameter is forced to fp32 and verified, then a 1-sample forward/backward
    smoke test checks grad dtypes BEFORE the real run.
Run in a Kaggle cell:   !python train_qlora_cad.py
Env overrides: DATA_PATH, MAX_LEN, MAX_STEPS, MAX_SAMPLES, PRECISION (auto|fp16|bf16), RESUME=1
"""
import os, sys, json, math, time, glob, inspect, random
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")          # one GPU: avoids DataParallel surprises on T4x2
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch, transformers, peft, accelerate, datasets
from packaging import version
from collections import Counter
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments, TrainerCallback, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------------------------------------------------------------------- config
BASE_MODEL   = "Qwen/Qwen2.5-Coder-7B"
DATA_PATH    = os.environ.get("DATA_PATH", "/kaggle/input/cadjsonl/formatted_cad_data.jsonl")
OUT_DIR      = "/kaggle/working/qlora_cad"                 # checkpoints
ADAPTER_DIR  = "/kaggle/working/cad_lora_adapter"          # final adapter + tokenizer
MAX_LEN      = int(os.environ.get("MAX_LEN", 2048))        # complex parts need >1024; lower to 1536 on OOM
MAX_SAMPLES  = int(os.environ.get("MAX_SAMPLES", 0))       # 0 = all (Kaggle sessions are time-limited!)
MAX_STEPS    = int(os.environ.get("MAX_STEPS", -1))        # -1 = 1 epoch
EVAL_SAMPLES = 100
BATCH, GRAD_ACC, LR = 1, 8, 2e-4
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
SAVE_STEPS = 200
SEED = 42
ASSIST = "<|im_start|>assistant\n"

def hr(t): print("\n" + "=" * 8, t, "=" * 8, flush=True)

# ----------------------------------------------------------------------------- 1. environment
hr("ENVIRONMENT")
assert torch.cuda.is_available(), "No CUDA GPU. Kaggle: Settings -> Accelerator -> GPU T4 x2 / P100."
props = torch.cuda.get_device_properties(0)
cap = torch.cuda.get_device_capability(0)
print(f"GPU            : {props.name}  | compute capability {cap[0]}.{cap[1]} | {props.total_memory/2**30:.1f} GiB")
for n, m in [("python", sys), ("torch", torch), ("transformers", transformers), ("peft", peft),
             ("accelerate", accelerate), ("datasets", datasets)]:
    print(f"{n:15s}: {getattr(m, '__version__', sys.version.split()[0])}")
try:
    import bitsandbytes as bnb; print(f"{'bitsandbytes':15s}: {bnb.__version__}")
except Exception as e:
    sys.exit(f"bitsandbytes import failed: {e}\n  fix: !pip install -U bitsandbytes   (then restart the session)")

# Native bf16 needs Ampere (cc >= 8.0). T4 (7.5) and P100 (6.0) do NOT: torch.cuda.is_bf16_supported()
# can still say True on T4 because it is emulated, which is slow and is not what we want.
native_bf16 = cap[0] >= 8
prec = os.environ.get("PRECISION", "auto").lower()
USE_BF16 = (prec == "bf16") or (prec == "auto" and native_bf16)
if USE_BF16 and not native_bf16:
    sys.exit("PRECISION=bf16 requested but this GPU has no native bf16. Use fp16.")
COMPUTE = torch.bfloat16 if USE_BF16 else torch.float16
print(f"precision      : {'bf16' if USE_BF16 else 'fp16'} (compute dtype {COMPUTE}); trainable params will be fp32")
set_seed(SEED)

# ----------------------------------------------------------------------------- 2. dataset checks
hr("DATASET")
if not os.path.isfile(DATA_PATH):
    print(f"NOT FOUND: {DATA_PATH}\nFiles under /kaggle/input:")
    for p in glob.glob("/kaggle/input/**/*", recursive=True)[:60]: print("  ", p)
    sys.exit("Fix DATA_PATH (Add Data -> your dataset) and rerun.")
raw = datasets.load_dataset("json", data_files=DATA_PATH, split="train")
print("rows:", len(raw), "| columns:", raw.column_names)
if "text" not in raw.column_names:
    sys.exit(f"No 'text' column. Found {raw.column_names}. Rename/convert the field to 'text'.")
bad = sum(1 for t in raw["text"] if not isinstance(t, str) or ASSIST not in t)
print(f"rows without a '{ASSIST.strip()}' marker or non-string text: {bad}")
if MAX_SAMPLES: raw = raw.shuffle(seed=SEED).select(range(min(MAX_SAMPLES, len(raw))))

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "right"
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
assert IM_END is not None and IM_END != tok.unk_token_id, "<|im_end|> missing from tokenizer"

def encode(ex):
    t = ex["text"]
    i = t.find(ASSIST) if isinstance(t, str) else -1
    if i < 0: return {"input_ids": [], "labels": []}
    i += len(ASSIST)
    p = tok(t[:i], add_special_tokens=False)["input_ids"]
    r = tok(t[i:].rstrip("\n"), add_special_tokens=False)["input_ids"]   # ends with <|im_end|>
    return {"input_ids": p + r, "labels": [-100] * len(p) + r}          # loss on assistant tokens only

enc = raw.map(encode, remove_columns=raw.column_names, desc="tokenizing")
lens = sorted(len(x) for x in enc["input_ids"])
pct = lambda q: lens[min(len(lens) - 1, int(q * len(lens)))]
print(f"token lengths: p50={pct(.5)} p90={pct(.9)} p99={pct(.99)} max={lens[-1]}")
over = sum(l > MAX_LEN for l in lens)
print(f"samples > MAX_LEN({MAX_LEN}): {over} ({100*over/len(lens):.1f}%)  -> dropped, not truncated")
if over / len(lens) > 0.05:
    print("WARNING: >5% dropped. These are your COMPLEX parts, so the model never sees them. "
          "Raise MAX_LEN (watch VRAM) and/or make the target code shorter (canonical, rounded coordinates).")
enc = enc.filter(lambda e: 0 < len(e["input_ids"]) <= MAX_LEN)
print("usable samples:", len(enc))
split = enc.train_test_split(test_size=min(EVAL_SAMPLES, max(1, len(enc) // 50)), seed=SEED)
train_ds, eval_ds = split["train"], split["test"]

class Collator:
    def __init__(s, pad): s.pad = pad
    def __call__(s, feats):
        m = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), m), s.pad, dtype=torch.long)
        lab = torch.full((len(feats), m), -100, dtype=torch.long)
        att = torch.zeros((len(feats), m), dtype=torch.long)
        for i, f in enumerate(feats):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"]); lab[i, :n] = torch.tensor(f["labels"]); att[i, :n] = 1
        return {"input_ids": ids, "attention_mask": att, "labels": lab}
collator = Collator(tok.pad_token_id)

# ----------------------------------------------------------------------------- 3. model
hr("MODEL")
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=COMPUTE)
kw = dict(quantization_config=bnb_cfg, device_map={"": 0}, attn_implementation="sdpa", low_cpu_mem_usage=True)
# `torch_dtype` was renamed `dtype` in transformers 4.56 -> pick the non-deprecated name
kw["dtype" if version.parse(transformers.__version__) >= version.parse("4.56.0") else "torch_dtype"] = COMPUTE
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kw)
model.config.use_cache = False
GC_KW = {"use_reentrant": False}
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=GC_KW)
# (If you hit OOM, replace the line above with:  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=GC_KW);
#  model.enable_input_require_grads()   -- skips the fp32 upcast of embeddings/lm_head, saving ~2 GiB.)
lcfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg)

def to_fp32_trainables(m):
    n = 0
    for _, p in m.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float(); n += 1
    return n

def dtype_report(m, tag):
    c = Counter()
    for _, p in m.named_parameters(): c[("trainable" if p.requires_grad else "frozen", str(p.dtype))] += p.numel()
    print(f"[{tag}] parameter dtypes:")
    for (k, d), n in sorted(c.items()): print(f"    {k:9s} {d:16s} {n/1e6:10.2f} M")
    return not any(k == "trainable" and d != "torch.float32" for (k, d) in c)

print("recast to fp32:", to_fp32_trainables(model), "tensors")
assert dtype_report(model, "after get_peft_model"), "non-fp32 trainable params remain"
model.print_trainable_parameters()

# ----------------------------------------------------------------------------- 4. smoke test (grad dtypes)
hr("SMOKE TEST (1 sample fwd/bwd)")
model.train()
b = {k: v.to("cuda") for k, v in collator([train_ds[0]]).items()}
with torch.autocast("cuda", dtype=COMPUTE):
    loss = model(**b).loss
loss.backward()
gdt = Counter(str(p.grad.dtype) for _, p in model.named_parameters() if p.requires_grad and p.grad is not None)
nonfinite = sum(1 for _, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all())
print(f"loss={loss.item():.4f} | grad dtypes={dict(gdt)} | tensors with non-finite grads={nonfinite}")
print(f"peak VRAM after 1 step: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
model.zero_grad(set_to_none=True)
# A parameter's .grad always has the parameter's dtype. So the "_amp_foreach_non_finite_check_and_unscale"
# BFloat16 error can only occur if some trainable PARAMETER is bf16 when Trainer runs.
assert set(gdt) <= {"torch.float32"}, f"Non-fp32 grads: {gdt}"
assert math.isfinite(loss.item()), "loss is NaN/inf on the first sample (see notes on fp16 overflow)"

# ----------------------------------------------------------------------------- 5. trainer
hr("TRAIN")
steps_per_epoch = math.ceil(len(train_ds) / (BATCH * GRAD_ACC))
total_steps = MAX_STEPS if MAX_STEPS > 0 else steps_per_epoch
ta = dict(output_dir=OUT_DIR, per_device_train_batch_size=BATCH, per_device_eval_batch_size=1,
          gradient_accumulation_steps=GRAD_ACC, num_train_epochs=1, max_steps=MAX_STEPS,
          learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * total_steps)),
          weight_decay=0.0, max_grad_norm=1.0, fp16=not USE_BF16, bf16=USE_BF16,
          optim="paged_adamw_8bit", gradient_checkpointing=True, gradient_checkpointing_kwargs=GC_KW,
          logging_steps=10, save_strategy="steps", save_steps=SAVE_STEPS, save_total_limit=2,
          eval_steps=SAVE_STEPS, report_to="none", remove_unused_columns=False, seed=SEED,
          dataloader_num_workers=2)
ta["eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
   else "evaluation_strategy"] = "steps"
args = TrainingArguments(**ta)

class StopOnNaN(TrainerCallback):
    def on_log(self, a, state, control, logs=None, **k):
        l = (logs or {}).get("loss")
        if l is not None and not math.isfinite(l):
            print("!! NaN/inf loss -> stopping (fp16 overflow? lower LR to 1e-4 / MAX_LEN)"); control.should_training_stop = True

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                  data_collator=collator, callbacks=[StopOnNaN()])
to_fp32_trainables(trainer.model)                       # guard: nothing may have re-cast the adapters
assert dtype_report(trainer.model, "after Trainer init"), "Trainer init produced non-fp32 trainables"

resume = get_last_checkpoint(OUT_DIR) if (os.path.isdir(OUT_DIR) and os.environ.get("RESUME") == "1") else None
t0 = time.time()
trainer.train(resume_from_checkpoint=resume)
print(f"training time: {(time.time()-t0)/3600:.2f} h")

# ----------------------------------------------------------------------------- 6. save
hr("SAVE")
trainer.model.save_pretrained(ADAPTER_DIR)
tok.save_pretrained(ADAPTER_DIR)
with open(os.path.join(ADAPTER_DIR, "training_meta.json"), "w") as f:
    json.dump({"base_model": BASE_MODEL, "max_len": MAX_LEN, "compute_dtype": str(COMPUTE), "lora_r": LORA_R,
               "lora_alpha": LORA_ALPHA, "n_train": len(train_ds), "versions": {
                   "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}}, f, indent=2)
with open(os.path.join(ADAPTER_DIR, "log_history.json"), "w") as f: json.dump(trainer.state.log_history, f)
need = ["adapter_config.json"]
have = os.listdir(ADAPTER_DIR)
assert all(n in have for n in need) and any(n.startswith("adapter_model") for n in have), have
print("saved:", sorted(have))
print("base model recorded in adapter_config:", json.load(open(f"{ADAPTER_DIR}/adapter_config.json"))["base_model_name_or_path"])
