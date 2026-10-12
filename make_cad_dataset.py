"""Synthetic CadQuery dataset: parametric part generators -> run in CadQuery -> keep only valid solids.

Why this instead of (only) DeepCAD-style data
  * The prompt is generated from the SAME operation list that generates the code, so every number in
    the prompt (hole diameter, pitch, fillet radius, pocket depth...) is exactly what the code does.
    Prompt->code is a deterministic mapping the model can actually learn.
  * Code is short, idiomatic CadQuery (box/hole/fillet/shell/revolve...) instead of hundreds of raw
    lineTo() coordinates, so complex parts fit in 1-2k tokens and are not truncated at training time.
  * Canonical frame: base on z=0, centred on the XY origin, Z up. No arbitrary global offsets.
  * Every row is executed; invalid / multi-solid / timed-out geometry is discarded. Bounding box and
    volume come from the real solid and are stored in `meta` for evaluation.
Output: train.jsonl / val.jsonl, each row {"text": <ChatML>, "meta": {...}}  ("text" is what the trainer reads).

  pip install cadquery            # Kaggle: needs Internet ON
  python make_cad_dataset.py --n 20000 --out /kaggle/working/cad_synth --workers 4

NOTE: written without being able to run CadQuery where it was authored. First run: `--n 200 --workers 1`,
read the printed reject reasons, and fix any generator that rejects most of its samples.
"""
import argparse, hashlib, json, math, os, random, signal, sys, collections
from multiprocessing import Pool

SYSTEM = "You are a Large Mechanical Model. Generate Python CadQuery code for the requested geometry."
TIER_WEIGHTS = {1: 0.15, 2: 0.35, 3: 0.35, 4: 0.15}   # 1: base only | 2: 1-2 ops | 3: 3-4 ops | 4: 5-7 ops
TIER_OPS = {1: (0, 0), 2: (1, 2), 3: (3, 4), 4: (5, 7)}

def n(x): return repr(round(float(x), 3))
def R(rng, lo, hi, step=0.5):
    lo, hi = math.ceil(lo / step), math.floor(hi / step)
    return round(rng.randint(lo, max(lo, hi)) * step, 3)
WP = 'workplane(centerOption="CenterOfBoundBox")'

# ================================================================= features
# Each feature: (stage, group, fn(rng, ctx) -> (code_suffix, sentence) | None). Code suffix is appended to `result`.
# stage 0 pre-ops (edge treatment) | 1 top-face cuts | 2 boss | 3 through/side cuts after boss
def f_vfillet(rng, c):
    r = R(rng, 1, min(c["w"], c["d"]) * 0.2)
    return f'.edges("|Z").fillet({n(r)})', f"Round the vertical edges with a {n(r)} mm fillet."
def f_tchamfer(rng, c):
    top = min(c["h"] * 0.4, c.get("r", 1e9) * 0.2, 5)
    if top < 0.5: return None
    v = R(rng, 0.5, top)
    return f'.faces(">Z").chamfer({n(v)})', f"Chamfer the top face edges by {n(v)} mm."
def f_tfillet(rng, c):
    top = min(c["h"] * 0.4, c.get("r", 1e9) * 0.2, 5)
    if top < 1: return None
    v = R(rng, 1, top)
    return f'.faces(">Z").edges().fillet({n(v)})', f"Fillet the top face edges with a {n(v)} mm radius."

def _grid(rng, c, kind):
    dia = R(rng, 3, min(12, c["h"] * 0.9, c["w"] * 0.2, c["d"] * 0.2))
    m = max(dia, 3.0)
    nx, ny = rng.randint(2, 4), rng.randint(2, 4)
    out = []
    for L, k in ((c["w"], nx), (c["d"], ny)):
        smax = (L - 2 * m - dia) / (k - 1)
        if smax < dia * 2: return None
        out.append(round(rng.uniform(dia * 2, smax), 1))
    sx, sy = out
    pat = f'.faces(">Z").{WP}.rarray({n(sx)}, {n(sy)}, {nx}, {ny})'
    ctx_desc = f"{nx} x {ny} grid (spacing {n(sx)} x {n(sy)} mm) centred on the top face"
    if kind == "hole":
        return pat + f'.hole({n(dia)})', f"Drill {nx*ny} through-holes of diameter {n(dia)} mm in a {ctx_desc}."
    if kind == "cbore":
        cb, cd = round(dia * 1.8, 2), R(rng, 1, c["h"] * 0.5)
        if c["h"] < 4: return None
        return (pat + f'.cboreHole({n(dia)}, {n(cb)}, {n(cd)})',
                f"Make {nx*ny} counterbored holes (hole diameter {n(dia)} mm, counterbore diameter {n(cb)} mm, depth {n(cd)} mm) in a {ctx_desc}.")
    if kind == "csk":
        cd = round(dia * 2, 2)
        if (cd - dia) / 2 > c["h"] * 0.9: return None
        return (pat + f'.cskHole({n(dia)}, {n(cd)}, 90)',
                f"Make {nx*ny} 90-degree countersunk holes (hole diameter {n(dia)} mm, countersink diameter {n(cd)} mm) in a {ctx_desc}.")
f_grid_hole  = lambda rng, c: _grid(rng, c, "hole")
f_grid_cbore = lambda rng, c: _grid(rng, c, "cbore")
f_grid_csk   = lambda rng, c: _grid(rng, c, "csk")

def f_pocket(rng, c):
    if c["h"] < 3: return None
    pw, pd, dp = R(rng, c["w"] * .2, c["w"] * .55), R(rng, c["d"] * .2, c["d"] * .55), R(rng, 1, c["h"] * .6)
    return (f'.faces(">Z").{WP}.rect({n(pw)}, {n(pd)}).cutBlind({n(-dp)})',
            f"Cut a centred rectangular pocket {n(pw)} x {n(pd)} mm, {n(dp)} mm deep, into the top face.")
def f_slot(rng, c):
    L, W = R(rng, c["w"] * .3, c["w"] * .7), R(rng, 3, min(c["d"] * .3, 12))
    return (f'.faces(">Z").{WP}.slot2D({n(L)}, {n(W)}, 0).cutThruAll()',
            f"Cut a centred through-slot {n(L)} mm long and {n(W)} mm wide along the X axis.")
def f_center_hole(rng, c):
    dia = R(rng, 3, min(c["w"], c["d"]) * 0.4)
    c["inner_clear"] = min(c.get("inner_clear", 1e9), dia / 2)
    return f'.faces(">Z").{WP}.hole({n(dia)})', f"Drill a {n(dia)} mm through-hole at the centre of the top face."
def f_boss(rng, c):
    r = R(rng, 3, min(c["w"], c["d"]) * 0.3)
    hb = R(rng, 2, 30)
    c["boss_r"] = r
    return f'.faces(">Z").{WP}.circle({n(r)}).extrude({n(hb)})', f"Add a centred cylindrical boss of diameter {n(2*r)} mm and height {n(hb)} mm on the top face."
def f_bore(rng, c):
    lim = c.get("boss_r", c["rmin"] * 0.8)
    lim = min(lim - 1, c.get("hole_clear", 1e9))
    if lim < 2: return None
    dia = R(rng, 3, 2 * lim)
    return f'.faces(">Z").{WP}.hole({n(dia)})', f"Drill a {n(dia)} mm centred bore through the whole part."
def f_side_hole(rng, c):
    if c["h"] < 8: return None
    dia = R(rng, 2.5, min(c["h"] * 0.5, 10))
    return f'.faces(">X").{WP}.hole({n(dia)})', f"Drill a {n(dia)} mm hole along the X axis through the part at mid-height."

def f_polar(rng, c):
    r = c["r"]
    nh = rng.randint(3, 12)
    pr = R(rng, r * .45, r * .8)
    dia = R(rng, 2.5, min(10, r * .15, c["h"] * .9))
    if 2 * pr * math.sin(math.pi / nh) < dia * 2.5: return None
    c["hole_clear"] = min(c.get("hole_clear", 1e9), pr - dia / 2 - 1)
    return (f'.faces(">Z").{WP}.polarArray({n(pr)}, 0, 360, {nh}).hole({n(dia)})',
            f"Drill {nh} through-holes of diameter {n(dia)} mm evenly spaced on a {n(2*pr)} mm bolt circle.")
def f_groove(rng, c):
    r = c["r"]
    if c["h"] < 3: return None
    ro = R(rng, r * .5, r * .85); ri = ro - R(rng, 2, 6)
    dp = R(rng, 1, c["h"] * .5)
    c["hole_clear"] = min(c.get("hole_clear", 1e9), ri - 1)
    return (f'.faces(">Z").{WP}.circle({n(ro)}).circle({n(ri)}).cutBlind({n(-dp)})',
            f"Cut a ring groove (outer diameter {n(2*ro)} mm, inner diameter {n(2*ri)} mm, {n(dp)} mm deep) in the top face.")

BOX_FEATS = [(0, "edge", f_vfillet), (0, "edge2", f_tchamfer), (1, "grid", f_grid_hole), (1, "grid", f_grid_cbore),
             (1, "grid", f_grid_csk), (1, "boss", f_pocket), (1, "slot", f_slot), (1, "ch", f_center_hole),
             (2, "boss", f_boss), (3, "bore", f_bore), (3, "side", f_side_hole)]
CYL_FEATS = [(0, "edge", f_tchamfer), (0, "edge", f_tfillet), (1, "grid", f_polar), (1, "groove", f_groove),
             (2, "boss", f_boss), (3, "bore", f_bore)]   # no side hole: ">X" is ambiguous on a cylinder

def apply_features(rng, base_code, ctx, feats, k):
    chosen, groups = [], set()
    for st, g, fn in rng.sample(feats, len(feats)):
        if len(chosen) >= k: break
        if g in groups: continue
        groups.add(g); chosen.append((st, fn))
    # bore needs a boss/hole context decided earlier -> stage order guarantees that
    chosen.sort(key=lambda t: t[0])
    code, sents = base_code, []
    for _, fn in chosen:
        r = fn(rng, ctx)
        if r is None: continue
        code += f"\nresult = result{r[0]}"
        sents.append(r[1])
    return code, sents

# ================================================================= families
def fam_plate(rng, tier, kind="plate"):
    w, d = R(rng, 30, 200, 1), R(rng, 20, 150, 1)
    h = R(rng, 3, 25) if kind == "plate" else R(rng, 20, 120, 1)
    base = f'result = cq.Workplane("XY").box({n(w)}, {n(d)}, {n(h)}, centered=(True, True, False))'
    name = "rectangular plate" if kind == "plate" else "rectangular block"
    ctx = dict(w=w, d=d, h=h, rmin=min(w, d) / 2)
    return base, f"a {name} {n(w)} x {n(d)} x {n(h)} mm (X x Y x Z), base on z=0, centred on the XY origin", ctx, BOX_FEATS
def fam_block(rng, tier): return fam_plate(rng, tier, "block")
def fam_disc(rng, tier):
    r, h = R(rng, 15, 120), R(rng, 3, 30)
    base = f'result = cq.Workplane("XY").circle({n(r)}).extrude({n(h)})'
    return base, f"a round disc of diameter {n(2*r)} mm and thickness {n(h)} mm, base on z=0, axis along Z", dict(r=r, h=h, w=2 * r, d=2 * r, rmin=r), CYL_FEATS
def fam_prism(rng, tier):
    sides, rc, h = rng.choice([5, 6, 8]), R(rng, 10, 60), R(rng, 5, 50)
    base = f'result = cq.Workplane("XY").polygon({sides}, {n(2*rc)}).extrude({n(h)})'
    ri = rc * math.cos(math.pi / sides)
    return (base, f"a regular {sides}-sided prism with a circumscribed diameter of {n(2*rc)} mm and height {n(h)} mm, base on z=0",
            dict(r=ri, h=h, w=2 * ri, d=2 * ri, rmin=ri), [x for x in CYL_FEATS if x[2] in (f_tchamfer, f_boss, f_bore)])

def fam_lbracket(rng, tier):
    L, W, H, t = R(rng, 40, 120, 1), R(rng, 20, 80, 1), R(rng, 30, 100, 1), R(rng, 3, 8)
    code = (f'result = cq.Workplane("XY").box({n(L)}, {n(W)}, {n(t)}, centered=(True, True, False))\n'
            f'result = result.union(cq.Workplane("XY").box({n(t)}, {n(W)}, {n(H)}, centered=(False, True, False)).translate(({n(-L/2)}, 0, 0)))')
    desc = (f"an L-bracket: a base plate {n(L)} x {n(W)} x {n(t)} mm (X x Y x Z) on z=0 centred on the origin, "
            f"joined to a vertical leg {n(t)} mm thick, {n(W)} mm wide, {n(H)} mm tall at the -X end")
    sents, dia = [], R(rng, 3, min(8, W * .2, t * 2.5))
    sx = None
    if rng.random() < .8:
        nx, ny = 2, rng.randint(1, 2)
        span_x = L / 2 - t - 2 - dia / 2
        sx_max = 2 * span_x / (nx - 1)
        if sx_max > dia * 2:
            sx = round(rng.uniform(dia * 2, sx_max), 1)
            sy = round(max(dia * 2, W * .5), 1) if ny == 2 else 1.0
            if ny == 2 and sy + dia > W - 4: sy = None
            if sy:
                code += f'\nresult = result.faces("<Z").{WP}.rarray({n(sx)}, {n(sy)}, {nx}, {ny}).hole({n(dia)})'
                sents.append(f"Drill {nx*ny} through-holes of diameter {n(dia)} mm in the base plate on a {nx} x {ny} grid (spacing {n(sx)} x {n(sy)} mm), clear of the leg.")
    if tier >= 3 and H > 2 * t + 10 and rng.random() < .8:
        d2 = R(rng, 3, min(8, W * .2, (H - 2 * t) * .4))
        code += f'\nresult = result.faces("<X").{WP}.hole({n(d2)})'
        sents.append(f"Drill a {n(d2)} mm hole through the centre of the vertical leg's outer face.")
    return code, desc, None, sents

def fam_shaft(rng, tier):
    k = rng.randint(2, 5)
    rad = [R(rng, 4, 40)]
    while len(rad) < k:                      # adjacent radii must differ (no zero-length profile segments)
        r_ = R(rng, 4, 40)
        if r_ != rad[-1]: rad.append(r_)
    ln = [R(rng, 8, 60, 1) for _ in range(k)]
    pts, z = [(0, 0)], 0
    for r_, l_ in zip(rad, ln):
        pts += [(r_, z), (r_, z + l_)]; z += l_
    pts += [(0, z)]
    pt = ", ".join(f"({n(a)}, {n(b)})" for a, b in pts)
    code = f'result = cq.Workplane("XZ").polyline([{pt}]).close().revolve(360, (0, 0, 0), (0, 1, 0))'
    sec = "; ".join(f"section {i+1}: diameter {n(2*r_)} mm, length {n(l_)} mm" for i, (r_, l_) in enumerate(zip(rad, ln)))
    desc = f"a stepped shaft along the Z axis starting at z=0, made by revolving a half-section ({sec})"
    sents = []
    if tier >= 2 and rng.random() < .7:
        c = R(rng, 0.5, min(rad[0], rad[-1]) * .3)
        code += f'\nresult = result.faces(">Z").chamfer({n(c)})\nresult = result.faces("<Z").chamfer({n(c)})'
        sents.append(f"Chamfer both end faces by {n(c)} mm.")
    if tier >= 3 and rng.random() < .7:
        dd, dp = R(rng, 3, rad[-1] * 1.2), R(rng, 5, max(6, ln[-1] * .8))
        if dd / 2 < rad[-1] - 1:
            code += f'\nresult = result.faces(">Z").{WP}.hole({n(dd)}, {n(dp)})'
            sents.append(f"Drill a {n(dd)} mm blind axial hole {n(dp)} mm deep in the top end.")
    if tier >= 3 and rng.random() < .6:
        i = max(range(k), key=lambda j: rad[j]); zc = sum(ln[:i]) + ln[i] / 2
        dc = R(rng, 2, min(rad[i] * .8, ln[i] * .6, 12))
        if dc >= 2 and dc < 2 * rad[i]:
            code += (f'\nresult = result.cut(cq.Workplane("YZ").workplane(offset={n(-rad[i]-1)}).center(0, {n(zc)})'
                     f'.circle({n(dc/2)}).extrude({n(2*rad[i]+2)}))')
            sents.append(f"Drill a {n(dc)} mm cross hole along the X axis through the largest section at z={n(zc)} mm.")
    return code, desc, None, sents

def fam_enclosure(rng, tier):
    w, d, h = R(rng, 40, 160, 1), R(rng, 30, 120, 1), R(rng, 15, 60, 1)
    r = R(rng, 3, min(w, d) * .15); wall = R(rng, 1.5, min(2.5, r - 0.5), 0.5)
    code = (f'result = cq.Workplane("XY").box({n(w)}, {n(d)}, {n(h)}, centered=(True, True, False))\n'
            f'result = result.edges("|Z").fillet({n(r)})\nresult = result.faces(">Z").shell({n(-wall)})')
    desc = (f"an open-top enclosure {n(w)} x {n(d)} x {n(h)} mm (X x Y x Z), base on z=0, with {n(r)} mm rounded vertical edges "
            f"and {n(wall)} mm walls (shelled from the top)")
    sents = []
    if tier >= 3 and h >= 20:
        dd = R(rng, 3, min(10, h * .3))
        code += f'\nresult = result.faces(">X").{WP}.hole({n(dd)})'
        sents.append(f"Drill a {n(dd)} mm hole along the X axis through both end walls at mid-height.")
    return code, desc, None, sents


# ---------------------------------------------------------------- minor components (small hardware)
CHECK = {}          # per-sample extra checks set by a family: {"vol_lt": x} or {"vol_gt": x}
ALLOWED_CHARS = "ABCDEFGHKLMNPRSTUVWXYZ0123456789"      # single connected glyphs only

def fam_washer(rng, tier):
    ro = R(rng, 4, 30); ri = R(rng, 1.5, ro * .6); t = R(rng, 0.5, 6)
    if ri > ro - 1: raise ValueError("washer too thin")
    code = f'result = cq.Workplane("XY").circle({n(ro)}).circle({n(ri)}).extrude({n(t)})'
    return code, f"a flat washer: outer diameter {n(2*ro)} mm, inner diameter {n(2*ri)} mm, thickness {n(t)} mm, base on z=0", None, []

def _hexprism(rng, tier, long):
    d = R(rng, 6, 50, 1); m = R(rng, d * .5, d * .9, 0.5) if not long else R(rng, 10, 60, 1)
    bore = R(rng, d * .3, d * .6)
    code = (f'result = cq.Workplane("XY").polygon(6, {n(d)}).extrude({n(m)})\n'
            f'result = result.faces(">Z").{WP}.hole({n(bore)})')
    name = "hex standoff" if long else "hex nut"
    desc = (f"a {name}: regular hexagon with {n(d)} mm across corners (corners on the X axis), {n(m)} mm tall, base on z=0; "
            f"a {n(bore)} mm bore through the centre (no thread modelled)")
    sents = []
    if tier >= 2 and not long and m * .15 >= 0.3:
        c = R(rng, 0.3, m * .15, 0.1)
        code += f'\nresult = result.faces(">Z").chamfer({n(c)})\nresult = result.faces("<Z").chamfer({n(c)})'
        sents.append(f"Chamfer both hexagonal faces by {n(c)} mm.")
    return code, desc, None, sents
def fam_hexnut(rng, tier): return _hexprism(rng, tier, False)
def fam_standoff(rng, tier): return _hexprism(rng, tier, True)

def fam_hexbolt(rng, tier):
    ds, L = R(rng, 3, 24, 1), R(rng, 10, 100, 1)
    dh, k = round(ds * 1.8, 2), round(ds * .65, 1)
    code = (f'result = cq.Workplane("XY").circle({n(ds/2)}).extrude({n(L)})\n'
            f'result = result.faces(">Z").{WP}.polygon(6, {n(dh)}).extrude({n(k)})')
    desc = (f"a hex-head bolt along Z: shank diameter {n(ds)} mm, shank length {n(L)} mm from z=0, hex head {n(dh)} mm across corners "
            f"and {n(k)} mm tall on top of the shank (no thread modelled)")
    sents = []
    if tier >= 2:
        c = R(rng, 0.3, ds * .15, 0.1)
        code += f'\nresult = result.faces("<Z").chamfer({n(c)})'
        sents.append(f"Chamfer the shank tip by {n(c)} mm.")
    return code, desc, None, sents

def fam_pin(rng, tier):
    d, L = R(rng, 2, 20), R(rng, 8, 80, 1); c = R(rng, 0.3, d * .15, 0.1)
    code = (f'result = cq.Workplane("XY").circle({n(d/2)}).extrude({n(L)})\n'
            f'result = result.faces(">Z").chamfer({n(c)})\nresult = result.faces("<Z").chamfer({n(c)})')
    return code, f"a dowel pin: diameter {n(d)} mm, length {n(L)} mm, base on z=0, both ends chamfered {n(c)} mm", None, []

# ---------------------------------------------------------------- major components
def fam_pulley(rng, tier):
    ro, W = R(rng, 20, 70), R(rng, 10, 30)
    g, w1 = R(rng, 2, 6), R(rng, 1.5, W * .25)
    rb = R(rng, 4, ro * .35)
    if rb > ro - g - 2: raise ValueError("pulley bore too big")
    pts = [(rb, 0), (ro, 0), (ro, w1), (ro - g, W / 2), (ro, W - w1), (ro, W), (rb, W)]
    pt = ", ".join(f"({n(a)}, {n(b)})" for a, b in pts)
    code = f'result = cq.Workplane("XZ").polyline([{pt}]).close().revolve(360, (0, 0, 0), (0, 1, 0))'
    desc = (f"a V-groove pulley along Z: outer diameter {n(2*ro)} mm, width {n(W)} mm, bore diameter {n(2*rb)} mm, "
            f"V-groove {n(g)} mm deep centred on the width with {n(w1)} mm flat lands on each side")
    sents = []
    if tier >= 3:
        sr = R(rng, 1.5, min(4, (ro - rb) * .25), 0.5)
        code += f'\nresult = result.cut(cq.Workplane("YZ").center(0, {n(W/2)}).circle({n(sr)}).extrude({n(ro+1)}))'
        sents.append(f"Drill a {n(2*sr)} mm radial set-screw hole from the +X side into the bore at mid-width.")
    return code, desc, None, sents

def fam_flange(rng, tier):
    ro, t = R(rng, 40, 120, 1), R(rng, 6, 20)
    nh, pr = rng.choice([4, 6, 8, 12]), round(ro * rng.uniform(.7, .82), 1)
    dh = R(rng, 5, min(14, ro * .1))
    rh_hi = pr - dh / 2 - 3
    if rh_hi < ro * .25: raise ValueError("hub does not fit")
    rh, hh = R(rng, ro * .25, rh_hi), R(rng, 8, 40)
    if 2 * rh - 6 < 10: raise ValueError("bore too small")
    bore = R(rng, 10, 2 * rh - 6)
    code = (f'result = cq.Workplane("XY").circle({n(ro)}).extrude({n(t)})\n'
            f'result = result.faces(">Z").{WP}.polarArray({n(pr)}, 0, 360, {nh}).hole({n(dh)})\n'
            f'result = result.faces(">Z").{WP}.circle({n(rh)}).extrude({n(hh)})\n'
            f'result = result.faces(">Z").{WP}.hole({n(bore)})')
    desc = f"a pipe flange: disc diameter {n(2*ro)} mm, thickness {n(t)} mm, base on z=0"
    sents = [f"Drill {nh} through-holes of diameter {n(dh)} mm on a {n(2*pr)} mm bolt circle.",
             f"Add a centred hub of diameter {n(2*rh)} mm and height {n(hh)} mm on the top face.",
             f"Drill a {n(bore)} mm bore through the whole flange and hub."]
    return code, desc, None, sents

def fam_bearing_block(rng, tier):
    w, d, h = R(rng, 60, 140, 1), R(rng, 30, 70, 1), R(rng, 30, 70, 1)
    D, dia, m = R(rng, 10, min(h * .6, w * .4)), R(rng, 4, 8), 8
    cb, cd = round(dia * 1.8, 2), R(rng, 2, h * .35)
    if w / 2 - m - dia / 2 <= D / 2 + 2: raise ValueError("mount holes hit the bore")
    code = (f'result = cq.Workplane("XY").box({n(w)}, {n(d)}, {n(h)}, centered=(True, True, False))\n'
            f'result = result.faces(">Z").{WP}.rarray({n(w-2*m)}, {n(d-2*m)}, 2, 2).cboreHole({n(dia)}, {n(cb)}, {n(cd)})\n'
            f'result = result.faces(">Y").{WP}.hole({n(D)})')
    desc = f"a bearing block {n(w)} x {n(d)} x {n(h)} mm (X x Y x Z), base on z=0, centred on the XY origin"
    sents = [f"Make 4 counterbored mounting holes (diameter {n(dia)} mm, counterbore {n(cb)} mm x {n(cd)} mm deep) at the corners, {m} mm in from each edge.",
             f"Drill a {n(D)} mm bearing bore along the Y axis through the block at mid-height."]
    return code, desc, None, sents

# ---------------------------------------------------------------- characters / text
def _text(rng, k): return "".join(rng.choice(ALLOWED_CHARS) for _ in range(k))

def fam_nameplate(rng, tier):
    w, d, t = R(rng, 40, 150, 1), R(rng, 15, 50, 1), R(rng, 2, 8)
    txt = _text(rng, rng.randint(2, 7)); fs = R(rng, 4, d * .5)
    if 0.8 * fs * len(txt) > w - 8: raise ValueError("text too long for plate")
    base = f'result = cq.Workplane("XY").box({n(w)}, {n(d)}, {n(t)}, centered=(True, True, False))'
    desc = f"a rectangular nameplate {n(w)} x {n(d)} x {n(t)} mm, base on z=0, centred on the XY origin"
    if rng.random() < .5:
        dp = R(rng, 0.5, t * .5)
        CHECK["vol_lt"] = w * d * t
        code = base + f'\nresult = result.faces(">Z").{WP}.text("{txt}", {n(fs)}, {n(-dp)}, combine="cut")'
        s = f'Engrave the text "{txt}" centred on the top face, font size {n(fs)} mm, {n(dp)} mm deep.'
    else:
        dp = R(rng, 0.5, 3)
        CHECK["vol_gt"] = w * d * t
        code = base + f'\nresult = result.faces(">Z").{WP}.text("{txt}", {n(fs)}, {n(dp)}, combine=True)'
        s = f'Add raised text "{txt}" centred on the top face, font size {n(fs)} mm, {n(dp)} mm high.'
    return code, desc, None, [s]

def fam_letter(rng, tier):
    ch, fs, h = rng.choice(ALLOWED_CHARS), R(rng, 10, 60, 1), R(rng, 2, 15)
    code = f'result = cq.Workplane("XY").text("{ch}", {n(fs)}, {n(h)}, combine=True)'
    return code, f'the single character "{ch}" in the default font, font size {n(fs)} mm, extruded {n(h)} mm upward from z=0, centred on the origin', None, []

FAMILIES = [(fam_plate, 3), (fam_block, 2), (fam_disc, 3), (fam_prism, 1), (fam_lbracket, 1.5), (fam_shaft, 2), (fam_enclosure, 1),
            (fam_washer, 1), (fam_hexnut, 1.5), (fam_hexbolt, 1.5), (fam_standoff, 1), (fam_pin, 1),
            (fam_pulley, 1.5), (fam_flange, 2), (fam_bearing_block, 1.5), (fam_nameplate, 1.5), (fam_letter, 1)]
CLASS = {"washer": "minor", "hexnut": "minor", "hexbolt": "minor", "standoff": "minor", "pin": "minor",
         "nameplate": "char", "letter": "char"}          # everything else: major / simple stock shapes

# ================================================================= sample assembly
def build_spec(rng, tier):
    fam = rng.choices([f for f, _ in FAMILIES], [w for _, w in FAMILIES])[0]
    base, desc, ctx, feats = fam(rng, tier)
    if ctx is None:                               # family handled its own features (bracket/shaft/enclosure)
        code, sents = base, list(feats)
        return fam.__name__[4:], code, desc, sents
    lo, hi = TIER_OPS[tier]
    code, sents = apply_features(rng, base, ctx, feats, rng.randint(lo, hi))
    return fam.__name__[4:], code, desc, sents

def make_prompt(rng, desc, sents, bbox):
    size = f"Overall size: {bbox[0]:g} x {bbox[1]:g} x {bbox[2]:g} mm."
    if rng.random() < .5:
        body = f"Create a CadQuery model of {desc}."
        if sents: body += " " + " ".join(sents)
        return f"{body} {size} Units are millimeters, Z up."
    steps = "\n".join(f"{i+2}) {s}" for i, s in enumerate(sents))
    return f"Create a CadQuery model. Units are millimeters, Z up. {size}\nSteps:\n1) Base: {desc}." + (f"\n{steps}" if steps else "")

class Timeout(Exception): pass
def _alarm(*a): raise Timeout()

def run_code(code):
    import cadquery as cq
    ns = {"cq": cq, "math": math}
    alarm = hasattr(signal, "SIGALRM")            # Unix only; on Windows there is no per-sample timeout
    if alarm: signal.signal(signal.SIGALRM, _alarm); signal.alarm(30)
    try:
        exec(code, ns)
    finally:
        if alarm: signal.alarm(0)
    res = ns["result"]
    solids = res.solids().vals()
    if len(solids) != 1: raise ValueError(f"{len(solids)} solids")
    s = res.val()
    if not s.isValid(): raise ValueError("invalid shape")
    vol = s.Volume()
    if vol < 1e-3: raise ValueError("zero volume")
    bb = s.BoundingBox()
    return [round(bb.xlen, 3), round(bb.ylen, 3), round(bb.zlen, 3)], round(vol, 3), len(res.faces().vals())

def gen_one(args):
    seed, i, tw = args
    rng = random.Random(f"{seed}-{i}")
    tier = rng.choices(list(tw), list(tw.values()))[0]
    try:
        CHECK.clear()
        fam, ops, desc, sents = build_spec(rng, tier)
        plan = "\n".join(["# Plan:", f"# 1) base: {desc}"] + [f"# {i+2}) {x}" for i, x in enumerate(sents)])
        code = "import cadquery as cq\n\n" + plan + "\n" + ops
        bbox, vol, nf = run_code(code)
        if "vol_lt" in CHECK and not vol < CHECK["vol_lt"] - 1e-3: raise ValueError("engraving did not remove material")
        if "vol_gt" in CHECK and not vol > CHECK["vol_gt"] + 1e-3: raise ValueError("raised text did not add material")
    except Exception as e:
        return {"reject": f"{type(e).__name__}: {str(e)[:60]}"}
    prompt = make_prompt(rng, desc, sents, bbox)
    text = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n```python\n{code}\n```<|im_end|>\n")
    return {"text": text, "meta": {"family": fam, "tier": tier, "n_ops": len(sents), "bbox": bbox, "volume": vol,
                                   "n_faces": nf, "class": CLASS.get(fam, "major"), "prompt": prompt, "code_sha": hashlib.sha1(code.encode()).hexdigest()}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--out", default="cad_synth")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--holdout_families", default="", help="comma list, e.g. pulley,hexbolt: kept OUT of train, written to heldout.jsonl")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.seed, i, TIER_WEIGHTS) for i in range(int(a.n * 1.4))]   # over-generate: some are rejected
    seen, rows, rej = set(), [], collections.Counter()
    with Pool(a.workers) as p:
        for r in p.imap_unordered(gen_one, jobs, chunksize=16):
            if "reject" in r: rej[r["reject"]] += 1; continue
            if r["meta"]["code_sha"] in seen: rej["duplicate"] += 1; continue
            seen.add(r["meta"]["code_sha"]); rows.append(r)
            if len(rows) >= a.n: break
    hold = set(x for x in a.holdout_families.split(",") if x)
    held = [r for r in rows if r["meta"]["family"] in hold]
    rows = [r for r in rows if r["meta"]["family"] not in hold]
    if held:
        with open(os.path.join(a.out, "heldout.jsonl"), "w") as f:
            for r in held: f.write(json.dumps(r) + "\n")
        print(f"held-out families {sorted(hold)}: {len(held)} rows -> heldout.jsonl")
    rows.sort(key=lambda r: r["meta"]["code_sha"])
    val = [r for r in rows if int(r["meta"]["code_sha"][:8], 16) / 16**8 < a.val_frac]
    vs = {r["meta"]["code_sha"] for r in val}
    tr = [r for r in rows if r["meta"]["code_sha"] not in vs]
    for name, part in (("train", tr), ("val", val)):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for r in part: f.write(json.dumps(r) + "\n")
    import cadquery as _cq
    json.dump({"cadquery": _cq.__version__}, open(os.path.join(a.out, "cadquery_version.json"), "w"))
    print(f"kept {len(rows)} (train {len(tr)}, val {len(val)}) from {len(jobs)} attempts")
    print("tier mix :", dict(collections.Counter(r["meta"]["tier"] for r in rows)))
    print("class    :", dict(collections.Counter(r["meta"]["class"] for r in rows)))
    print("family   :", dict(collections.Counter(r["meta"]["family"] for r in rows)))
    print("top reject reasons:"); [print(f"   {c:6d}  {k}") for k, c in rej.most_common(12)]

if __name__ == "__main__":
    main()
