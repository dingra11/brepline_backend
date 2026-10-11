"""Text -> CAD pipeline.

prompt -> planner (JSON) -> retrieved few-shot examples -> N parallel candidates
-> validation + sandboxed CadQuery run -> scoring -> repair loop -> best result.
Every job is written to storage (generations / failures) for later training.
"""
import os, re, ast, json, time, uuid, shutil, hashlib, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover
    resource = None

import storage
from examples_lib import EXAMPLES

PIPELINE_VERSION = "1.0"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "runner.py")
OUT_DIR = os.getenv("OUT_DIR", "/tmp/cad_out")
N_CAND = int(os.getenv("N_CAND", "3"))
REPAIRS = int(os.getenv("REPAIRS", "2"))
CAD_WORKERS = int(os.getenv("CAD_WORKERS", "2"))
CAD_TIMEOUT = int(os.getenv("CAD_TIMEOUT", "90"))
CAD_MEM_MB = int(os.getenv("CAD_MEM_MB", "4096"))
LLM_MODEL = os.getenv("LLM_MODEL", "")
# "chat" = general instruct model (default). "lmm" = the fine-tuned brepline model, which was
# trained on one fixed prompt format (see LMM_SYS / SPEC_SHOTS below).
LLM_MODE = os.getenv("LLM_MODE", "chat").strip().lower()
FENCE = "`" * 3
PREFILL = FENCE + "python\n"

os.makedirs(OUT_DIR, exist_ok=True)
_SEM = threading.Semaphore(CAD_WORKERS)

DENSITY = {  # g/cm3
    "Al 6061": 2.70, "Steel": 7.85, "Stainless 304": 8.00, "Titanium": 4.43,
    "Brass": 8.50, "Copper": 8.96, "ABS": 1.04, "PLA": 1.24, "Nylon": 1.14,
}

# ------------------------------------------------------------------ prompts

CQ_SYS = f"""You are an expert CAD engineer who writes CadQuery (Python, OpenCascade) code.
Reply with ONE python code block ({FENCE}python ... {FENCE}) and nothing else.

Rules:
- Units are millimetres. Z is up. Import only cadquery as cq (math / numpy allowed).
- The final object MUST be assigned to a variable named `result` (a cq.Workplane).
  Only for genuine multi-part assemblies you may define `parts = {{"name": Workplane, ...}}` instead.
- These helpers already exist, do NOT redefine or import them:
  radial(shape, n, r=0)   copies a shape (modelled along +X) n times around Z and unions them (fins, spokes, ribs)
  bolt_circle(body, n, r, d, face=">Z")   n through-holes of diameter d on a circle of radius r
  spur_gear(m, z, t, bore=0)   spur gear solid, module m, z teeth, thickness t
- Use the exact dimensions the request gives. When a dimension is missing choose realistic engineering values.
- The result must be ONE connected solid: make features overlap or touch so unions fuse.
- Prefer simple robust operations: extrude, revolve, loft, union, cut, hole, shell, polarArray.
  Avoid fillet/chamfer on complex edge selections, and keep fillet radii well below wall thickness.
- No file I/O, no show_object, no print, no exports.
"""

PLAN_SYS = """You are a mechanical design planner. Turn the request into a compact JSON object:
{"object": "short name",
 "assembly": false,
 "overall_size_mm": [x, y, z] or null,   (only if the request states or clearly implies overall dimensions)
 "features": ["ordered build steps with numbers"],
 "notes": "anything ambiguous and the assumption taken"}
Output JSON only."""

# ---- lmm mode: the exact system prompt and user format the adapter was trained on ----
LMM_SYS = "You are a Large Mechanical Model. Generate Python CadQuery code for the requested geometry."

SPEC_SYS = """You turn a CAD request into a precise build spec. Reply with ONLY the spec, in exactly the format of the examples.
Rules: units are millimetres. Sketch planes: XY plane (normal +z) extrudes up; XZ plane (normal -y).
The first step is "new body"; later steps are "join" or "cut". A hole is a circle in the same sketch as the profile.
The bounding box must be the true overall size. Use the exact dimensions the request gives; choose realistic ones when missing."""

_H = "Create a CadQuery model with a bounding box of {bb} mm, built from {n} extrusion(s) ({ops}) using {kinds} sketch profiles. Units are millimeters.\nSteps:\n"
SPEC_SHOTS = [
    ("washer, 16 mm outer diameter, 8 mm inner diameter, 2 mm thick",
     _H.format(bb="16.0 x 16.0 x 2.0", n=1, ops="new body", kinds="circle") +
     "1) new body: sketch on XY plane (normal +z) at z=0.0: circle r=8.0 at (0.00, 0.00, 0.00); "
     "circle r=4.0 at (0.00, 0.00, 0.00); extrude 2.0 mm"),
    ("rectangular plate 60 x 40 x 5 mm with two 6 mm holes, 40 mm apart",
     _H.format(bb="60.0 x 40.0 x 5.0", n=1, ops="new body", kinds="circle and poly") +
     "1) new body: sketch on XY plane (normal +z) at z=0.0: 4-segment profile, x -30.00..30.00, "
     "y -20.00..20.00, z=0.00; circle r=3.0 at (-20.00, 0.00, 0.00); circle r=3.0 at (20.00, 0.00, 0.00); "
     "extrude 5.0 mm"),
    ("L bracket 60 mm wide, 40 mm base, 40 mm tall, 5 mm thick, two 6 mm mounting holes in the base",
     _H.format(bb="60.0 x 40.0 x 40.0", n=2, ops="new body, join", kinds="circle and poly") +
     "1) new body: sketch on XY plane (normal +z) at z=0.0: 4-segment profile, x -30.00..30.00, "
     "y -20.00..20.00, z=0.00; circle r=3.0 at (-20.00, -5.00, 0.00); circle r=3.0 at (20.00, -5.00, 0.00); "
     "extrude 5.0 mm\n"
     "2) join: sketch on XZ plane (normal -y) at y=20.0: 4-segment profile, x -30.00..30.00, "
     "y=20.00, z 0.00..40.00; extrude 5.0 mm"),
    ("pipe 30 mm outer diameter, 24 mm inner diameter, 100 mm long",
     _H.format(bb="30.0 x 30.0 x 100.0", n=1, ops="new body", kinds="circle") +
     "1) new body: sketch on XY plane (normal +z) at z=0.0: circle r=15.0 at (0.00, 0.00, 0.00); "
     "circle r=12.0 at (0.00, 0.00, 0.00); extrude 100.0 mm"),
    ("block 50 x 30 x 20 mm",
     _H.format(bb="50.0 x 30.0 x 20.0", n=1, ops="new body", kinds="poly") +
     "1) new body: sketch on XY plane (normal +z) at z=0.0: 4-segment profile, x -25.00..25.00, "
     "y -15.00..15.00, z=0.00; extrude 20.0 mm"),
    ("flange 80 mm diameter, 8 mm thick, with a 30 mm diameter hub 20 mm tall and a 12 mm through bore",
     _H.format(bb="80.0 x 80.0 x 28.0", n=2, ops="new body, join", kinds="circle") +
     "1) new body: sketch on XY plane (normal +z) at z=0.0: circle r=40.0 at (0.00, 0.00, 0.00); "
     "circle r=6.0 at (0.00, 0.00, 0.00); extrude 8.0 mm\n"
     "2) join: sketch on XY plane (normal +z) at z=8.0: circle r=15.0 at (0.00, 0.00, 8.00); "
     "circle r=6.0 at (0.00, 0.00, 8.00); extrude 20.0 mm"),
]


# ------------------------------------------------------------------ LLM

_clients = {}


def llm(messages, temperature=0.2, max_tokens=2500, extra_body=None, planner=False):
    """One chat completion. planner=True uses PLAN_BASE_URL/PLAN_API_KEY/PLAN_MODEL when set
    (so the planning step can run on a different endpoint than the code model)."""
    from openai import OpenAI
    own_planner = planner and bool(os.getenv("PLAN_BASE_URL"))
    pre = "PLAN_" if own_planner else "LLM_"
    base = os.getenv(pre + "BASE_URL")
    key = os.getenv(pre + "API_KEY") or os.getenv("LLM_API_KEY")
    model = os.getenv(pre + "MODEL") or os.getenv("LLM_MODEL")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set (load .env before starting the server)")
    if pre not in _clients:
        _clients[pre] = OpenAI(base_url=base, api_key=key,
                               timeout=int(os.getenv("LLM_TIMEOUT", "120")))
    client = _clients[pre]
    eb = dict(extra_body or {})
    if LLM_MODE == "lmm":
        # the fine-tuned model has a fixed output size; reasoning-model padding does not apply
        if planner and not own_planner:
            eb["adapter"] = False        # planning is not what the adapter was trained for
    else:
        # reasoning models (gpt-oss, qwen3...) spend part of max_tokens on hidden thinking
        max_tokens = max(max_tokens, int(os.getenv("LLM_MIN_TOKENS", "4000")))
        if os.getenv("LLM_REASONING_EFFORT"):
            eb["reasoning_effort"] = os.getenv("LLM_REASONING_EFFORT")
    extra = {"extra_body": eb} if eb else {}
    last = None
    tries = max(1, int(os.getenv("LLM_RETRIES", "3")))
    for i in range(tries):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens, **extra)
            return r.choices[0].message.content or ""
        except Exception as e:
            last = e
            if i < tries - 1:
                time.sleep(2 ** i)
    raise RuntimeError(f"LLM call failed: {last}")


def extract_code(text):
    m = re.search(FENCE + r"(?:python|py)?\s*\n(.*?)" + FENCE, text, re.S)
    if m:
        return m.group(1).strip() + "\n"
    if "import cadquery" in text or "result" in text:
        return text.strip() + "\n"
    return ""


def extract_spec(text):
    """Pull the 'Create a CadQuery model ... Steps: ...' spec out of a planner reply."""
    i = (text or "").find("Create a CadQuery model")
    if i < 0:
        return ""
    s = text[i:].split("<|im_end|>")[0].replace(FENCE, "").strip()
    return s if "Steps:" in s and "extrude" in s else ""


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


# ------------------------------------------------------------------ validation

_BAN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "getattr",
              "setattr", "delattr", "globals", "locals", "vars", "os", "sys",
              "subprocess", "socket", "shutil", "breakpoint", "exit", "quit"}
_OK_IMPORTS = {"cadquery", "math", "numpy"}


def validate_code(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    assigns = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in _OK_IMPORTS:
                    return False, f"import not allowed: {a.name}"
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] not in _OK_IMPORTS:
                return False, f"import not allowed: {n.module}"
        elif isinstance(n, ast.Name):
            if n.id in _BAN_NAMES:
                return False, f"name not allowed: {n.id}"
            if isinstance(n.ctx, ast.Store) and n.id in ("result", "parts"):
                assigns = True
        elif isinstance(n, ast.Attribute):
            if n.attr.startswith("__"):
                return False, f"attribute not allowed: {n.attr}"
    if not assigns:
        return False, "script never assigns `result`"
    return True, ""


# ------------------------------------------------------------------ sandbox

def _make_preexec(timeout, mem):
    """Resource limits for the child. Linux only; returns None elsewhere (e.g. macOS),
    where setrlimit(RLIMIT_AS) fails and breaks the subprocess launch."""
    if resource is None or not sys.platform.startswith("linux"):
        return None

    def _set(kind, value):
        try:
            resource.setrlimit(kind, (value, value))
        except (ValueError, OSError):
            pass  # best-effort: never let a limit failure kill the launch

    def limits():
        _set(resource.RLIMIT_AS, mem)
        _set(resource.RLIMIT_CPU, timeout + 10)

    return limits


def run_sandbox(code, wd, timeout=None):
    """Run code in a resource-limited subprocess. Returns meta dict (meta['ok'])."""
    timeout = timeout or CAD_TIMEOUT
    os.makedirs(wd, exist_ok=True)
    cf = os.path.join(wd, "model.py")
    with open(cf, "w") as f:
        f.write(code)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("LLM_", "ACCESS_", "ADMIN_", "FIREBASE_"))}
    mem = CAD_MEM_MB * 1024 * 1024
    preexec = _make_preexec(timeout, mem)

    kwargs = dict(capture_output=True, text=True, timeout=timeout, env=env, cwd=wd,
                  start_new_session=True)  # replaces os.setsid() inside preexec_fn
    if preexec is not None:
        kwargs["preexec_fn"] = preexec

    with _SEM:
        try:
            p = subprocess.run([sys.executable, RUNNER, cf, wd], **kwargs)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout}s (model too complex)"}
        except (subprocess.SubprocessError, OSError) as e:
            return {"ok": False, "error": f"could not start sandbox: {e}"}
    mp = os.path.join(wd, "meta.json")
    if os.path.exists(mp):
        try:
            return json.load(open(mp))
        except Exception:
            pass
    tail = (p.stderr or "")[-600:]
    return {"ok": False, "error": f"runner crashed (rc={p.returncode}; out of memory?) {tail}"}


# ------------------------------------------------------------------ scoring

HINTS = [
    ("StdFail_NotDone", "A fillet/chamfer/shell failed. Remove it or use a much smaller value."),
    ("No pending wires", "Draw a closed 2D profile before extrude/revolve."),
    ("Cannot find a solid", "An operation received an empty stack; build the solid first."),
    ("is not defined", "Use only `cq`, `math`, and the provided helpers; define every variable."),
    ("has no attribute", "That CadQuery method does not exist; use standard methods only."),
    ("timeout", "Too complex. Simplify: fewer features, no fine patterns."),
    ("out of memory", "Too complex. Simplify the geometry."),
    ("did not define `result`", "Assign the final Workplane to `result`."),
    ("BRep_API", "Boolean failed. Make overlapping parts clearly overlap, avoid coincident faces."),
]


def hint(error):
    e = error or ""
    for k, v in HINTS:
        if k.lower() in e.lower():
            return v
    return ""


def evaluate(meta, plan):
    """Return (score 0-10, issues list, size_checked)."""
    if not meta.get("ok"):
        return 0, [meta.get("error", "failed")[-500:]], False
    issues, score, size_checked = [], 10, False
    if not meta.get("valid", True):
        score -= 4
        issues.append("solid is geometrically invalid")
    if meta.get("volume", 0) <= 0:
        return 2, ["zero volume"], False
    ns = meta.get("n_solids", 1)
    if ns > 1 and not plan.get("assembly"):
        score -= 4
        issues.append(f"result has {ns} disconnected solids; make features overlap so they fuse into one")
    exp = plan.get("overall_size_mm")
    if isinstance(exp, list) and len(exp) == 3 and all(isinstance(x, (int, float)) and x > 0 for x in exp):
        size_checked = True
        got = sorted(meta["bbox"])
        want = sorted(float(x) for x in exp)
        tol = 0.05 if LLM_MODE == "lmm" else 0.25
        if any(abs(g - w) / w > tol for g, w in zip(got, want)):
            score -= 3
            issues.append(f"overall size {[round(x, 1) for x in got]} mm differs from expected {want} mm")
    return max(score, 0), issues, size_checked


# ------------------------------------------------------------------ planning / prompting

def make_plan(prompt, parent_code=None):
    msg = prompt if not parent_code else f"Modify the existing design. Change request: {prompt}"
    try:
        return extract_json(llm([{"role": "system", "content": PLAN_SYS},
                                 {"role": "user", "content": msg}], 0.1, 600))
    except Exception:
        return {}


def make_spec_plan(prompt, parent_prompt=None):
    """lmm mode: turn the plain-English request into the build spec the adapter was trained on."""
    req = prompt if not parent_prompt else f"{parent_prompt}. Change: {prompt}"
    msgs = [{"role": "system", "content": SPEC_SYS}]
    for q, spec in SPEC_SHOTS:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": spec}]
    msgs.append({"role": "user", "content": req})
    err = ""
    for temp in (0.1, 0.4):
        try:
            spec = extract_spec(llm(msgs, temp, 500, planner=True))
        except Exception as e:
            err, spec = str(e), ""
        if spec:
            m = re.search(r"bounding box of ([\d.]+) x ([\d.]+) x ([\d.]+) mm", spec)
            size = [float(g) for g in m.groups()] if m else None
            return {"object": req[:60], "assembly": False, "overall_size_mm": size, "spec": spec}
    return {"error": err or "planner produced no valid build spec"}


def lmm_msgs(spec):
    return [{"role": "system", "content": LMM_SYS}, {"role": "user", "content": spec}]


def pick_examples(prompt, k=3):
    p = prompt.lower()
    scored = sorted(((sum(t in p for t in tags), i) for i, (tags, _, _) in enumerate(EXAMPLES)),
                    reverse=True)
    idx = [i for s, i in scored if s > 0][:k] or [3, 4]
    return [EXAMPLES[i] for i in idx]


def base_msgs(prompt, plan, examples, parent_code=None):
    msgs = [{"role": "system", "content": CQ_SYS}]
    for _, req, code in examples:
        msgs.append({"role": "user", "content": f"Request: {req}"})
        msgs.append({"role": "assistant", "content": f"{FENCE}python\n{code}{FENCE}"})
    u = f"Request: {prompt}\n"
    if plan:
        u += f"Design plan: {json.dumps(plan)}\n"
    if parent_code:
        u += f"\nExisting code to modify (keep everything not mentioned):\n{FENCE}python\n{parent_code}{FENCE}\n"
    msgs.append({"role": "user", "content": u})
    return msgs


# ------------------------------------------------------------------ job storage helpers

def job_dir(job):
    return os.path.join(OUT_DIR, job)


def valid_job(job):
    return bool(re.fullmatch(r"[0-9a-f]{12}", job or ""))


def load_out(job):
    p = os.path.join(job_dir(job), "out.json")
    if not valid_job(job) or not os.path.exists(p):
        return None
    out = json.load(open(p))
    mp = os.path.join(job_dir(job), "mesh.json")
    if os.path.exists(mp):
        out["mesh"] = json.load(open(mp))
    return out


def _attempt(msgs, wd, plan, temp):
    res = {"code": "", "score": 0, "issues": [], "meta": {}, "stage": "llm", "size_checked": False}
    if msgs is None:
        res.update(issues=[plan.get("error") or "planner produced no valid build spec"], stage="plan")
        return res
    try:
        if LLM_MODE == "lmm":
            # the code fence is part of the prompt in the training data, so it is pre-filled
            text = llm(msgs, temp, 1200, extra_body={"prefill": PREFILL})
        else:
            text = llm(msgs, temp, 2500)
    except Exception as e:
        res["issues"] = [str(e)]
        return res
    code = extract_code(text)
    res["code"] = code
    if not code:
        res["issues"] = ["no code block in reply"]
        res["stage"] = "extract"
        return res
    ok, err = validate_code(code)
    if not ok:
        res.update(issues=[err], stage="validate")
        return res
    meta = run_sandbox(code, wd)
    score, issues, sc = evaluate(meta, plan)
    res.update(meta=meta, score=score, issues=issues, size_checked=sc,
               stage="run" if not meta.get("ok") else "scored")
    return res


# ------------------------------------------------------------------ main entry

def generate(prompt, parent=None, material="Al 6061", log=None, job=None,
             fresh=False, source="user"):
    """Returns (out, record). `out` is what the API returns; `record` was saved for training."""
    log = log or (lambda m: None)
    t0 = time.time()
    job = job or uuid.uuid4().hex[:12]
    parent_code = None
    if parent and valid_job(parent):
        pp = os.path.join(job_dir(parent), "model.py")
        if os.path.exists(pp):
            parent_code = open(pp).read()
    material = material if material in DENSITY else "Al 6061"

    key = hashlib.sha256(f"{prompt}|{material}|{hash_str(parent_code)}".encode()).hexdigest()[:24]
    cpath = os.path.join(OUT_DIR, f"cache_{key}.txt")
    if not fresh and os.path.exists(cpath):
        hit = load_out(open(cpath).read().strip())
        if hit:
            log("served from cache")
            hit["cached"] = True
            return hit, None

    log("planning")
    if LLM_MODE == "lmm":
        parent_prompt = None
        if parent_code:
            try:
                parent_prompt = json.load(open(os.path.join(job_dir(parent), "out.json"))).get("prompt")
            except Exception:
                pass
        plan = make_spec_plan(prompt, parent_prompt)
        examples = []
        msgs = lmm_msgs(plan["spec"]) if plan.get("spec") else None
    else:
        plan = make_plan(prompt, parent_code)
        examples = pick_examples(prompt)
        msgs = base_msgs(prompt, plan, examples, parent_code)

    log(f"generating {N_CAND} candidates")
    wd = job_dir(job)
    os.makedirs(wd, exist_ok=True)
    temps = ([0.1] + [0.5] * (N_CAND - 1)) if LLM_MODE == "lmm" else ([0.2] + [0.7] * (N_CAND - 1))
    with ThreadPoolExecutor(N_CAND) as ex:
        futs = [ex.submit(_attempt, msgs, os.path.join(wd, f"c{i}"), plan, temps[i])
                for i in range(N_CAND)]
        cands = [f.result() for f in futs]
    attempts = N_CAND
    best = max(cands, key=lambda c: c["score"])
    log(f"best candidate score {best['score']}/10")

    failures = [dict(c, repaired=False) for c in cands if c["score"] < 10 and c["code"]]
    for r in range(REPAIRS):
        if best["score"] >= 10 or msgs is None:
            break
        log(f"repair round {r + 1}")
        problem = "; ".join(best["issues"]) or "unknown problem"
        if LLM_MODE == "lmm":
            # the adapter never saw repair turns, so retry the same spec with fresh sampling
            fix, ftemp = msgs, 0.5
        else:
            fix = [*msgs,
                   {"role": "assistant", "content": f"{FENCE}python\n{best['code']}{FENCE}"},
                   {"role": "user", "content": f"That code has a problem: {problem}\n"
                                               f"{hint(problem)}\nReturn the full corrected code."}]
            ftemp = 0.2
        c = _attempt(fix, os.path.join(wd, f"r{r}"), plan, ftemp)
        attempts += 1
        if c["score"] < 10 and c["code"]:
            failures.append(dict(c, repaired=False))
        if c["score"] > best["score"]:
            best = dict(c, repaired_from=problem)

    meta = best["meta"]
    ok = bool(meta.get("ok"))
    out = {"job": job, "ok": ok, "cached": False, "prompt": prompt, "material": material,
           "plan": plan, "code": best["code"], "score": best["score"],
           "issues": best["issues"], "attempts": attempts,
           "verified": ok and best["score"] >= 10,
           "elapsed_s": round(time.time() - t0, 1)}
    if ok:
        vol = meta["volume"]
        out.update(bbox=[round(x, 2) for x in meta["bbox"]], volume_mm3=round(vol, 1),
                   mass_g=round(vol / 1000.0 * DENSITY[material], 2),
                   n_solids=meta["n_solids"], files={"step": True, "stl": True, "py": True},
                   bbox_min=[round(x, 2) for x in meta.get("bbox_min", [])] or None,
                   area_mm2=round(meta["area"], 1) if meta.get("area") else None,
                   com=[round(x, 2) for x in meta["com"]] if meta.get("com") else None,
                   counts=meta.get("counts"), density_g_cm3=DENSITY[material])
    # copy winning artifacts to the job root
    win_dir = _find_dir(wd, meta) if ok else None
    if win_dir:
        for fn in ("model.step", "model.stl", "mesh.json"):
            shutil.copy(os.path.join(win_dir, fn), os.path.join(wd, fn))
    with open(os.path.join(wd, "model.py"), "w") as f:
        f.write(best["code"])
    for d in os.listdir(wd):
        if re.fullmatch(r"[cr]\d", d):
            shutil.rmtree(os.path.join(wd, d), ignore_errors=True)
    if not ok:
        out["error"] = (best["issues"] or ["generation failed"])[0][-400:]
    elif best["issues"]:
        out["warning"] = "; ".join(best["issues"])
    with open(os.path.join(wd, "out.json"), "w") as f:
        json.dump(out, f)
    if ok and not fresh:
        open(cpath, "w").write(job)

    # ---- persist for training
    rec = {"job": job, "source": source, "prompt": prompt, "is_edit": bool(parent_code),
           "parent_job": parent, "parent_code": parent_code, "material": material,
           "plan": plan, "code": best["code"], "ok": ok, "verified": out["verified"],
           "size_checked": bool(best.get("size_checked")), "score": best["score"],
           "issues": best["issues"], "bbox": out.get("bbox"), "volume_mm3": out.get("volume_mm3"),
           "mass_g": out.get("mass_g"), "n_solids": out.get("n_solids"), "attempts": attempts,
           "repaired_from": best.get("repaired_from"), "llm_model": LLM_MODEL or os.getenv("LLM_MODEL", ""),
           "pipeline_version": PIPELINE_VERSION, "elapsed_s": out["elapsed_s"],
           "retrieved_examples": [e[1] for e in examples]}
    storage.save_generation(rec)
    for f in failures[:6]:
        storage.save_failure({"job": job, "prompt": prompt, "code": f["code"], "stage": f["stage"],
                              "score": f["score"], "issues": f["issues"],
                              "pipeline_version": PIPELINE_VERSION})
    log("done" if ok else "failed")
    return load_out(job) or out, rec


def hash_str(s):
    return hashlib.sha1((s or "").encode()).hexdigest()[:10]


def _find_dir(wd, meta):
    """Directory (c*/r*) whose meta.json equals the winning meta."""
    for d in sorted(os.listdir(wd)):
        mp = os.path.join(wd, d, "meta.json")
        if re.fullmatch(r"[cr]\d", d) and os.path.exists(mp):
            try:
                if json.load(open(mp)) == meta and os.path.exists(os.path.join(wd, d, "mesh.json")):
                    return os.path.join(wd, d)
            except Exception:
                pass
    return None
