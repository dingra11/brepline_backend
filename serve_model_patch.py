"""Run once in the notebook that holds serve_model.py.
Adds two request fields: `prefill` (text appended to the prompt) and `adapter` (False = run the
base model without the LoRA, used for the planning step). Safe to run again."""
s = open("serve_model.py").read()


def sub(a, b):
    global s
    assert a in s, "not found: " + a[:60]
    s = s.replace(a, b, 1)


if "prefill" not in s:
    sub("    max_tokens: int | None = None", "    max_tokens: int | None = None\n    prefill: str = \"\"")
    sub("    def __init__(self, start):\n        self.start = start",
        "    def __init__(self, start, prefill=\"\"):\n        self.start = start\n        self.prefill = prefill")
    sub("        return finished(t)", "        return finished(self.prefill + t)")
    sub("        enc = tok(text, return_tensors=\"pt\")", "        text += req.prefill\n        enc = tok(text, return_tensors=\"pt\")")
    sub("StoppingCriteriaList([DoneCode(start)])", "StoppingCriteriaList([DoneCode(start, req.prefill)])")
    sub("content = trim(tok.decode(gen, skip_special_tokens=True))",
        "content = trim(req.prefill + tok.decode(gen, skip_special_tokens=True))")
if "adapter: bool" not in s:
    sub("    prefill: str = \"\"", "    prefill: str = \"\"\n    adapter: bool = True")
    sub("import os, time, uuid, threading, queue, torch", "import os, time, uuid, threading, queue, contextlib, torch")
    sub("        with torch.inference_mode():",
        "        ctx = m.disable_adapter() if (not req.adapter and hasattr(m, \"disable_adapter\")) else contextlib.nullcontext()\n"
        "        with ctx, torch.inference_mode():")
open("serve_model.py", "w").write(s)
print("patched")
