"""Text -> CAD pipeline.

prompt -> planner (JSON) -> retrieved few-shot examples -> N parallel candidates
-> validation + sandboxed CadQuery run -> scoring -> repair loop -> best result.
Every job is written to storage (generations / failures) for later training.
"""
import os, re, ast, json, math, time, uuid, shutil, hashlib, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover
    resource = None

import storage
import profiles
from examples_lib import EXAMPLES

PIPELINE_VERSION = "1.1"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "runner.py")
OUT_DIR = os.getenv("OUT_DIR", "/tmp/cad_out")
N_CAND = int(os.getenv("N_CAND", "3"))
ROUTE = os.getenv("ROUTE", "1") != "0"   # lmm mode: send round / polygon parts to the profile builders
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

# v3 prompt grammar (see make_cad_dataset.py)
SPEC_SYS = """You convert a CAD request into a precise build spec for a fine-tuned CadQuery model.
Reply with ONLY the spec, in exactly the format of the examples:

Create a CadQuery model. Units are millimeters, Z up. Overall size: X x Y x Z mm.
Steps:
1) Base: <base part>.
2) <feature>.   (zero or more feature steps, in build order)

Rules
- Millimetres. Parts sit with their base on z=0, centred on the XY origin, axis / height along Z.
- "Overall size" is the TRUE bounding box in mm (base size plus any boss / hub height on top). Compute it.
- Use the exact numbers the request gives; when a number is missing choose a realistic one. Keep every
  hole inside the part (hole edge at least 3 mm from any edge) and keep hole patterns clear of bosses and bores.
- Write numbers with one decimal in steps (6.0), and use ONLY these phrasings (fill the <>):

Base:
  a rectangular plate <W> x <D> x <H> mm (X x Y x Z), base on z=0, centred on the XY origin
  a rectangular block <W> x <D> x <H> mm (X x Y x Z), base on z=0, centred on the XY origin
  a round disc of diameter <D> mm and thickness <T> mm, base on z=0, axis along Z
  a regular <N>-sided prism with a circumscribed diameter of <D> mm and height <H> mm, base on z=0
  a flat washer: outer diameter <D> mm, inner diameter <d> mm, thickness <T> mm, base on z=0
  a hex nut: regular hexagon with <D> mm across corners (corners on the X axis), <H> mm tall, base on z=0; a <B> mm bore through the centre (no thread modelled)
  a hex-head bolt along Z: shank diameter <d> mm, shank length <L> mm from z=0, hex head <D> mm across corners and <K> mm tall on top of the shank (no thread modelled)
  a dowel pin: diameter <d> mm, length <L> mm, base on z=0, both ends chamfered <C> mm
  a stepped shaft along the Z axis starting at z=0, made by revolving a half-section (section 1: diameter <d> mm, length <L> mm; section 2: ...)
  a V-groove pulley along Z: outer diameter <D> mm, width <W> mm, bore diameter <d> mm, V-groove <G> mm deep centred on the width with <F> mm flat lands on each side
  a pipe flange: disc diameter <D> mm, thickness <T> mm, base on z=0   (then its three steps, see example)
  a bearing block <W> x <D> x <H> mm (X x Y x Z), base on z=0, centred on the XY origin
  a rectangular nameplate <W> x <D> x <T> mm, base on z=0, centred on the XY origin
  the single character "<c>" in the default font, font size <S> mm, extruded <H> mm upward from z=0, centred on the origin
Feature steps (plates / blocks):
  Round the vertical edges with a <R> mm fillet.
  Chamfer the top face edges by <C> mm.
  Drill <N> through-holes of diameter <d> mm in a <nx> x <ny> grid (spacing <sx> x <sy> mm) centred on the top face.
  Make <N> counterbored holes (hole diameter <d> mm, counterbore diameter <cb> mm, depth <h> mm) in a <nx> x <ny> grid (spacing <sx> x <sy> mm) centred on the top face.
  Make <N> 90-degree countersunk holes (hole diameter <d> mm, countersink diameter <c> mm) in a <nx> x <ny> grid (spacing <sx> x <sy> mm) centred on the top face.
  Cut a centred rectangular pocket <pw> x <pd> mm, <h> mm deep, into the top face.
  Cut a centred through-slot <L> mm long and <W> mm wide along the X axis.
  Drill a <d> mm through-hole at the centre of the top face.
  Add a centred cylindrical boss of diameter <d> mm and height <h> mm on the top face.
  Drill a <d> mm centred bore through the whole part.
  Drill a <d> mm hole along the X axis through the part at mid-height.
Feature steps (discs / prisms):
  Chamfer the top face edges by <C> mm.   Fillet the top face edges with a <R> mm radius.
  Drill <N> through-holes of diameter <d> mm evenly spaced on a <PCD> mm bolt circle.
  Cut a ring groove (outer diameter <D> mm, inner diameter <d> mm, <h> mm deep) in the top face.
  Add a centred cylindrical boss of diameter <d> mm and height <h> mm on the top face.
  Drill a <d> mm centred bore through the whole part.
Nameplate text: Engrave the text "<TXT>" centred on the top face, font size <S> mm, <h> mm deep.
                Add raised text "<TXT>" centred on the top face, font size <S> mm, <h> mm high.
                (text: capital letters A-Z and digits only, 2-7 characters)
If the request is not one of these, describe the closest base part plus feature steps in the same phrasing."""

_S = "Create a CadQuery model. Units are millimeters, Z up. Overall size: {bb} mm.\nSteps:\n"
SPEC_SHOTS = [
    ("washer, 16 mm outer diameter, 8 mm inner diameter, 2 mm thick",
     _S.format(bb="16 x 16 x 2") +
     "1) Base: a flat washer: outer diameter 16.0 mm, inner diameter 8.0 mm, thickness 2.0 mm, base on z=0."),
    ("rectangular plate 60 x 40 x 5 mm with four 6 mm holes, 40 mm apart in X and 20 mm apart in Y",
     _S.format(bb="60 x 40 x 5") +
     "1) Base: a rectangular plate 60.0 x 40.0 x 5.0 mm (X x Y x Z), base on z=0, centred on the XY origin.\n"
     "2) Drill 4 through-holes of diameter 6.0 mm in a 2 x 2 grid (spacing 40.0 x 20.0 mm) centred on the top face."),
    ("flange 80 mm diameter, 8 mm thick, four 6 mm bolt holes on a 60 mm bolt circle, 30 mm hub 20 mm tall, 12 mm bore",
     _S.format(bb="80 x 80 x 28") +
     "1) Base: a pipe flange: disc diameter 80.0 mm, thickness 8.0 mm, base on z=0.\n"
     "2) Drill 4 through-holes of diameter 6.0 mm evenly spaced on a 60.0 mm bolt circle.\n"
     "3) Add a centred hub of diameter 30.0 mm and height 20.0 mm on the top face.\n"
     "4) Drill a 12.0 mm bore through the whole flange and hub."),
    ("stepped shaft: 20 mm diameter for 50 mm, then 12 mm diameter for 30 mm",
     _S.format(bb="20 x 20 x 80") +
     "1) Base: a stepped shaft along the Z axis starting at z=0, made by revolving a half-section "
     "(section 1: diameter 20.0 mm, length 50.0 mm; section 2: diameter 12.0 mm, length 30.0 mm)."),
    ("L bracket, base 80 x 60 mm, 50 mm tall leg, 5 mm thick, two 6 mm holes in the base",
     _S.format(bb="80 x 60 x 50") +
     "1) Base: an L-bracket: a base plate 80.0 x 60.0 x 5.0 mm (X x Y x Z) on z=0 centred on the origin, "
     "joined to a vertical leg 5.0 mm thick, 60.0 mm wide, 50.0 mm tall at the -X end.\n"
     "2) Drill 2 through-holes of diameter 6.0 mm in the base plate on a 2 x 1 grid (spacing 50.0 x 1.0 mm), clear of the leg."),
    ("nameplate 100 x 30 x 4 mm with AB12 engraved",
     _S.format(bb="100 x 30 x 4") +
     "1) Base: a rectangular nameplate 100.0 x 30.0 x 4.0 mm, base on z=0, centred on the XY origin.\n"
     '2) Engrave the text "AB12" centred on the top face, font size 8.0 mm, 1.0 mm deep.'),
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
    if LLM_MODE == "lmm" and not own_planner:
        # the fine-tuned model has a fixed output size; reasoning-model padding does not apply
        if planner:
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
    """Pull the v3 'Create a CadQuery model ... Steps: 1) Base: ...' spec out of a planner reply."""
    i = (text or "").find("Create a CadQuery model")
    if i < 0:
        return ""
    s = text[i:].split("<|im_end|>")[0].replace(FENCE, "").strip()
    return s if "Steps:" in s and "Base:" in s else ""


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


BOX_WORDS = re.compile(r"\b(block|cube|box|brick|slab|plate|bar|cuboid|square|rectang\w*|prism|beam|tile|"
                       r"panel|sheet|stock|plank|board|dice|die|crate|case|enclosure|housing|spacer|pad|shim|billet)\b", re.I)


def is_plain_box(meta):
    c = meta.get("counts") or {}
    bb = meta.get("bbox") or [0, 0, 0]
    bv = bb[0] * bb[1] * bb[2]
    return c.get("faces") == 6 and bv > 0 and meta.get("volume", 0) / bv > 0.99


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
    ev = plan.get("expect_volume_mm3")
    if isinstance(ev, (int, float)) and ev > 0:      # exact for polygon / revolve builds
        size_checked = True
        if abs(meta["volume"] - ev) / ev > 0.02:
            score -= 3
            issues.append(f"volume {meta['volume']:.0f} mm3 differs from the expected {ev:.0f} mm3")
    eh = plan.get("expect_height_mm")
    if isinstance(eh, (int, float)) and eh > 0 and len(meta.get("bbox", [])) == 3:
        size_checked = True
        if abs(meta["bbox"][2] - eh) / eh > 0.05:
            score -= 3
            issues.append(f"height {meta['bbox'][2]:.1f} mm differs from the expected {eh:.1f} mm")
    if plan.get("allow_box") is False and is_plain_box(meta):
        score -= 4
        issues.append("result is a plain box, but the request does not describe a box; "
                      "the shape was probably not understood")
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
            m = re.search(r"Overall size:\s*([\d.]+) x ([\d.]+) x ([\d.]+) mm", spec)
            size = [float(g) for g in m.groups()] if m else None
            return {"object": req[:60], "assembly": False, "overall_size_mm": size, "spec": spec}
    return {"error": err or "planner produced no valid build spec"}


# ---- routing: round parts -> revolve builder, outline parts -> polygon builder
ROUTE_SYS = """Classify a CAD request by how the part is best built. Answer with exactly one word.
REVOLVE = a round part whose radius changes along its axis (lathe-turned / spun shape): rocket, vase, bottle, goblet, glass, bell, cone, funnel, chess piece, knob, pulley with a groove, bullet, dome, lamp base, candle holder.
POLYGON = a flat part cut from an outline that is not a plain rectangle or circle: star, hexagon or pentagon plate, hex nut, arrow, L / T / U / Z shaped bracket or plate, gusset, triangle, or any shape given as a list of points.
PRISM = everything else: block, plate with holes, washer, ring, tube, flange, spacer, plain cylinder, stepped shaft, box with pockets."""
ROUTE_SHOTS = [("make a rocket", "REVOLVE"), ("washer, 16 mm outer, 8 mm inner, 2 mm thick", "PRISM"),
               ("L bracket 60 x 40 mm", "POLYGON"), ("plate 100 x 60 x 8 with four corner holes", "PRISM"),
               ("a vase 120 mm tall", "REVOLVE"), ("6 point star, 50 mm", "POLYGON")]


def route_kind(req):
    msgs = [{"role": "system", "content": ROUTE_SYS}]
    for q, a in ROUTE_SHOTS:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": req})
    try:
        t = llm(msgs, 0.0, 6, planner=True).upper()
    except Exception:
        return "PRISM"
    m = re.search(r"REVOLVE|POLYGON|PRISM", t)
    return m.group(0) if m else "PRISM"


def _star(n, ro, ri):
    pts = []
    for i in range(2 * n):
        r, a = (ro if i % 2 == 0 else ri), math.pi / 2 + i * math.pi / n
        pts.append([round(r * math.cos(a), 1), round(r * math.sin(a), 1)])
    return pts


def _jd(o):
    return json.dumps(o, separators=(",", ":"))


_PROFILE_RULES = """Reply with ONE JSON object and nothing else. Millimetres. Pick realistic dimensions when the request gives none.
"""
PROFILE_SYS = {
    "REVOLVE": _PROFILE_RULES + """The part is spun 360 degrees around the Z axis. Give its half-section as
"profile": a CLOSED outline of [radius, height] points, radius >= 0, going up the outside and back down the inside
(a solid part comes back along the axis at radius 0; a hollow part has an inner wall). The outline must not cross itself.
Use 8-20 points; shape curves (nose cones, bowls, bells) with several points.
Optional "radial": fins / ribs, each {"count": n, "polygon": [[radius, height], ...], "thickness": t}; the polygon is the
side view of one fin, its inner edge may sit on the body surface (it is sunk into the body automatically).""",
    "POLYGON": _PROFILE_RULES + """The part is a flat outline extruded upward. Give "points": the outline as [x, y] points in order
(any direction, no repeated first point, must not cross itself), centred or starting at the origin, plus "thickness".
Optional "holes": [{"x":..,"y":..,"d":..}] round through-holes (inside the outline, at least d/2 + 1 mm from every edge) and
"cutouts": [[[x,y],...],...] polygonal windows fully inside the outline.""",
}
PROFILE_SHOTS = {
    "REVOLVE": [
        ("make a rocket", _jd({"type": "revolve",
         "profile": [[0, 0], [9, 0], [12, 12], [20, 22], [20, 140], [19.2, 152], [17.2, 166], [13.8, 180], [8.8, 192], [0, 200]],
         "radial": [{"count": 4, "polygon": [[20, 60], [20, 15], [42, 0], [42, 22]], "thickness": 2}]})),
        ("wine glass 70 mm wide and 150 mm tall", _jd({"type": "revolve",
         "profile": [[0, 0], [30, 0], [30, 3], [4, 6], [4, 70], [12, 76], [30, 90], [35, 115], [33, 150],
                     [31, 150], [32, 115], [27, 92], [10, 80], [0, 78]]})),
        ("pulley 60 mm diameter, 20 mm wide, V groove, 10 mm bore", _jd({"type": "revolve",
         "profile": [[5, 0], [30, 0], [30, 6], [26, 10], [30, 14], [30, 20], [5, 20]]})),
    ],
    "POLYGON": [
        ("L bracket 60 x 40 mm, 10 mm wide legs, 5 mm thick, two 5 mm holes", _jd({"type": "polygon",
         "points": [[0, 0], [60, 0], [60, 10], [10, 10], [10, 40], [0, 40]], "thickness": 5,
         "holes": [{"x": 50, "y": 5, "d": 5}, {"x": 5, "y": 30, "d": 5}]})),
        ("hex nut M8, 13 mm across flats, 6.5 mm thick", _jd({"type": "polygon",
         "points": [[7.5, 0], [3.8, 6.5], [-3.8, 6.5], [-7.5, 0], [-3.8, -6.5], [3.8, -6.5]], "thickness": 6.5,
         "holes": [{"x": 0, "y": 0, "d": 6.8}]})),
        ("5 point star plate, 60 mm across, 4 mm thick", _jd({"type": "polygon", "points": _star(5, 30, 12),
                                                                "thickness": 4})),
    ],
}


def make_profile_plan(req, kind, parent_spec=None):
    """Planner -> validated profile spec -> (code, plan) for the deterministic builders, or (None, error)."""
    msgs = [{"role": "system", "content": PROFILE_SYS[kind]}]
    for q, a in PROFILE_SHOTS[kind]:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": req if not parent_spec else
                 f"Current shape: {_jd(parent_spec)}\nChange: {req}\nReturn the complete new shape."})
    err = ""
    for temp in (0.1, 0.2, 0.4):
        try:
            raw = llm(msgs, temp, 900, planner=True)
            spec = extract_json(raw)
            spec["type"] = kind.lower()
            code, plan, _ = profiles.build(spec)
            return code, plan
        except profiles.SpecError as e:
            err = str(e)
            msgs = msgs[:len(msgs)] + [{"role": "assistant", "content": raw},
                                       {"role": "user", "content": f"That shape is invalid: {err}. Return the corrected JSON."}]
        except Exception as e:
            err = str(e)
    return None, err


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
    return _run_code(code, wd, plan, res)


def _run_code(code, wd, plan, res=None):
    res = res or {"code": code, "score": 0, "issues": [], "meta": {}, "stage": "run", "size_checked": False}
    res["code"] = code
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
             fresh=False, source="user", spec=None):
    """Returns (out, record). `out` is what the API returns; `record` was saved for training.
    spec = a polygon / revolve profile from the editor: built directly, no LLM involved."""
    log = log or (lambda m: None)
    t0 = time.time()
    job = job or uuid.uuid4().hex[:12]
    parent_code = None
    if parent and valid_job(parent):
        pp = os.path.join(job_dir(parent), "model.py")
        if os.path.exists(pp):
            parent_code = open(pp).read()
    material = material if material in DENSITY else "Al 6061"

    key = hashlib.sha256(f"{prompt}|{material}|{hash_str(parent_code)}|"
                         f"{json.dumps(spec, sort_keys=True) if spec else ''}".encode()).hexdigest()[:24]
    cpath = os.path.join(OUT_DIR, f"cache_{key}.txt")
    if not fresh and os.path.exists(cpath):
        hit = load_out(open(cpath).read().strip())
        if hit:
            log("served from cache")
            hit["cached"] = True
            return hit, None

    direct, route, examples, msgs, plan = None, "prism", [], None, {}
    if spec is not None:                                   # user-drawn profile: no LLM at all
        route = str(spec.get("type", "polygon")) if isinstance(spec, dict) else "polygon"
        try:
            code, plan, _ = profiles.build(spec)
            direct = {"code": code}
        except profiles.SpecError as e:
            plan = {"error": str(e)}
    else:
        log("planning")
        if LLM_MODE == "lmm":
            parent_prompt, parent_plan = None, {}
            if parent_code:
                try:
                    po = json.load(open(os.path.join(job_dir(parent), "out.json")))
                    parent_prompt, parent_plan = po.get("prompt"), po.get("plan") or {}
                except Exception:
                    pass
            req = prompt if not parent_prompt else f"{parent_prompt}. Change: {prompt}"
            pspec = parent_plan.get("profile_spec") if isinstance(parent_plan, dict) else None
            kind = (pspec or {}).get("type", "").upper() if pspec else ("PRISM" if parent_code else
                                                                      (route_kind(req) if ROUTE else "PRISM"))
            if kind in ("REVOLVE", "POLYGON"):
                log(f"building {kind.lower()} profile")
                code, p2 = make_profile_plan(prompt if pspec else req, kind, pspec)
                if code:
                    direct, plan, route = {"code": code}, p2, kind.lower()
                else:
                    log(f"profile failed ({str(p2)[:80]}), using the extrusion model")
            if not direct:
                plan = make_spec_plan(prompt, parent_prompt)
                msgs = lmm_msgs(plan["spec"]) if plan.get("spec") else None
        else:
            plan = make_plan(prompt, parent_code)
            examples = pick_examples(prompt)
            msgs = base_msgs(prompt, plan, examples, parent_code)
        if isinstance(plan, dict) and not direct:
            plan["allow_box"] = bool(BOX_WORDS.search(prompt))

    wd = job_dir(job)
    os.makedirs(wd, exist_ok=True)
    if direct:
        log("building shape")
        cands = [_run_code(direct["code"], os.path.join(wd, "c0"), plan)]
        attempts = 1
    else:
        log(f"generating {N_CAND} candidates")
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
           "verified": ok and best["score"] >= 10, "route": route,
           "elapsed_s": round(time.time() - t0, 1)}
    if out["verified"]:
        out["check_detail"] = ("valid single solid; size / volume match the " +
                               ("shape you drew" if spec is not None else "build plan")
                               if best.get("size_checked") else
                               "valid single solid (its shape was not measured against your request)")
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
           "plan": plan, "code": best["code"], "ok": ok, "verified": out["verified"], "route": route,
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
