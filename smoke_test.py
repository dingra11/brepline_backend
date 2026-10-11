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

# polygon / revolve builders (exact volume + size are checked by evaluate)
import profiles
for spec in [
    {"type": "polygon", "points": [[0, 0], [60, 0], [60, 10], [10, 10], [10, 40], [0, 40]], "thickness": 5,
     "holes": [{"x": 50, "y": 5, "d": 5}, {"x": 5, "y": 30, "d": 5}]},
    {"type": "polygon", "points": [[0, 0], [50, 0], [50, 30], [0, 30]], "thickness": 3,
     "cutouts": [[[10, 10], [40, 10], [40, 20], [10, 20]]]},
    {"type": "revolve", "profile": [[0, 0], [20, 0], [20, 50], [0, 50]]},
    {"type": "revolve", "profile": [[0, 0], [9, 0], [12, 12], [20, 22], [20, 140], [19.2, 152], [17.2, 166],
                                    [13.8, 180], [8.8, 192], [0, 200]],
     "radial": [{"count": 4, "polygon": [[20, 60], [20, 15], [42, 0], [42, 22]], "thickness": 2}]},
]:
    code, plan, _ = profiles.build(spec)
    meta = cp.run_sandbox(code, os.path.join(cp.OUT_DIR, "p_" + spec["type"] + str(len(spec.get("holes", [])))))
    score, issues, _ = cp.evaluate(meta, plan)
    flag = "OK " if meta.get("ok") and score == 10 else "BAD"
    bad += flag == "BAD"
    print(flag, f"{spec['type']:8}", [round(x, 1) for x in meta.get("bbox", [])], issues or meta.get("error", ""))

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
