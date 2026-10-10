"""No-LLM smoke test: runs every few-shot example through validation + sandbox,
checks storage + export. Usage: python smoke_test.py"""
import os, tempfile, json
tmp = tempfile.mkdtemp()
os.environ["OUT_DIR"] = tmp + "/out"
os.environ["DATA_DIR"] = tmp + "/data"
os.environ["FIREBASE_ENABLED"] = "0"

import cadpipe as cp, storage, export_training as et
from examples_lib import EXAMPLES

bad = 0
for tags, req, code in EXAMPLES:
    ok, err = cp.validate_code(code)
    meta = cp.run_sandbox(code, os.path.join(cp.OUT_DIR, "t_" + tags[0])) if ok else {"ok": False, "error": err}
    score, issues, _ = cp.evaluate(meta, {})
    flag = "OK " if meta.get("ok") and score == 10 else "BAD"
    bad += flag == "BAD"
    print(flag, f"{req[:48]:48}", [round(x, 1) for x in meta.get("bbox", [])], issues or "")

# safety validator
for evil in ["import os\nresult=1", "open('x')\nresult=1", "x=().__class__\nresult=1", "a=1"]:
    assert not cp.validate_code(evil)[0], evil
print("validator blocks unsafe code: OK")

# storage + export round trip
storage.save_generation({"job": "aaaaaaaaaaaa", "prompt": "washer", "code": EXAMPLES[2][2], "ok": True,
                         "verified": True, "score": 10, "is_edit": False, "plan": {}})
storage.save_feedback("aaaaaaaaaaaa", "up")
rows = et.load_rows("local")
for f in ("chatml", "messages", "prompt_completion", "alpaca"):
    text, st = et.build(rows, f)
    assert text.count("\n") == 1, (f, st)
print("storage + export round trip: OK")
raise SystemExit(1 if bad else 0)
