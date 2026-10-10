"""Turn saved generations into fine-tuning files.

  python export_training.py --source local     --format all --out export/
  python export_training.py --source firebase  --format messages --out export/
  python export_training.py --with-plan --include-edits --min-score 10

Keeps only: verified, non-edit (unless --include-edits), not down-voted,
de-duplicated by code hash. Down-votes are read from feedback (same source).
Formats (one JSONL each):
  chatml            {"text": "<|im_start|>system ... <|im_end|>"}     (Qwen / ChatML, SFTTrainer text field)
  messages          {"messages": [{role, content}, ...]}              (TRL / OpenAI style)
  prompt_completion {"prompt": "...", "completion": "..."}            (completion-only loss)
  alpaca            {"instruction": "...", "input": "", "output": "..."}
  repair            {"messages": [...broken code + error -> fixed code]} from failures + final fixes
"""
import os, sys, json, argparse, hashlib, collections
import storage

TRAIN_SYS = ("You write CadQuery (Python) code. Units mm, Z up. Assign the final solid to `result`. "
             "Helpers available: radial(shape,n,r=0), bolt_circle(body,n,r,d,face), spur_gear(m,z,t,bore).")
FENCE = "`" * 3


def load_rows(source="local"):
    if source == "firebase":
        return (storage.read_remote("generations"), storage.read_remote("feedback"),
                storage.read_remote("failures"))
    return (storage.read_local("generations"), storage.read_local("feedback"),
            storage.read_local("failures"))


def _user(r, with_plan):
    u = r["prompt"]
    if with_plan and r.get("plan"):
        u += "\n\nPlan: " + json.dumps(r["plan"], ensure_ascii=False)
    if r.get("is_edit") and r.get("parent_code"):
        u += f"\n\nExisting code:\n{FENCE}python\n{r['parent_code']}{FENCE}"
    return u


def _asst(code):
    return f"{FENCE}python\n{code.strip()}\n{FENCE}"


def select(rows, min_score=10, include_edits=False):
    gens, votes, _ = rows if isinstance(rows, tuple) else (rows, [], [])
    down = collections.Counter()
    for v in votes:
        if v.get("vote") == "down":
            down[v.get("job")] += 1
        elif v.get("vote") == "up":
            down[v.get("job")] -= 1
    seen, keep, st = set(), [], collections.Counter()
    for r in gens:
        st["total"] += 1
        if not r.get("code") or not r.get("ok"):
            st["not_ok"] += 1; continue
        if r.get("score", 0) < min_score:
            st["low_score"] += 1; continue
        if r.get("is_edit") and not include_edits:
            st["edit_skipped"] += 1; continue
        if down[r.get("job")] > 0:
            st["downvoted"] += 1; continue
        h = hashlib.sha1(r["code"].encode()).hexdigest()
        if h in seen:
            st["duplicate"] += 1; continue
        seen.add(h)
        keep.append(r)
    st["kept"] = len(keep)
    return keep, st


def build(rows, fmt="messages", min_score=10, include_edits=False, with_plan=False):
    """Return (jsonl_text, stats). fmt may be 'all' only via the CLI."""
    keep, st = select(rows, min_score, include_edits)
    lines = []
    if fmt == "repair":
        gens = {r["job"]: r for r in keep}
        for f in rows[2]:
            g = gens.get(f.get("job"))
            if g and f.get("code") and f["code"].strip() != g["code"].strip():
                err = "; ".join(f.get("issues") or [])[:600]
                lines.append({"messages": [
                    {"role": "system", "content": TRAIN_SYS},
                    {"role": "user", "content": f"Request: {g['prompt']}\n\nThis code has a problem: {err}\n{FENCE}python\n{f['code']}{FENCE}\nReturn the corrected code."},
                    {"role": "assistant", "content": _asst(g["code"])}]})
        st["repair_pairs"] = len(lines)
    else:
        for r in keep:
            u, a = _user(r, with_plan), _asst(r["code"])
            if fmt == "chatml":
                lines.append({"text": f"<|im_start|>system\n{TRAIN_SYS}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"})
            elif fmt == "messages":
                lines.append({"messages": [{"role": "system", "content": TRAIN_SYS},
                                           {"role": "user", "content": u},
                                           {"role": "assistant", "content": a}]})
            elif fmt == "prompt_completion":
                lines.append({"prompt": u, "completion": a})
            elif fmt == "alpaca":
                lines.append({"instruction": u, "input": "", "output": a})
            else:
                raise ValueError(f"unknown format {fmt}")
    return "\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + ("\n" if lines else ""), dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="local", choices=["local", "firebase"])
    ap.add_argument("--format", default="all")
    ap.add_argument("--out", default="export")
    ap.add_argument("--min-score", type=int, default=10)
    ap.add_argument("--include-edits", action="store_true")
    ap.add_argument("--with-plan", action="store_true")
    a = ap.parse_args()
    rows = load_rows(a.source)
    os.makedirs(a.out, exist_ok=True)
    fmts = ["chatml", "messages", "prompt_completion", "alpaca", "repair"] if a.format == "all" else [a.format]
    for f in fmts:
        text, st = build(rows, f, a.min_score, a.include_edits, a.with_plan)
        p = os.path.join(a.out, f"{f}.jsonl")
        open(p, "w").write(text)
        print(f"{p}: {text.count(chr(10))} rows  {st}")


if __name__ == "__main__":
    main()
