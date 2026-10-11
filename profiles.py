"""Deterministic shape builders: a 2D profile (from the user's polygon editor or from the
planner) -> readable CadQuery code + the numbers the result must match.

Two spec types (all lengths in mm):

  polygon  extrude an outline
    {"type": "polygon", "points": [[x, y], ...], "thickness": t,
     "holes":   [{"x": x, "y": y, "d": d}, ...],          # round through-holes
     "cutouts": [[[x, y], ...], ...]}                      # polygonal through-cut-outs

  revolve  spin a half-section around the Z axis
    {"type": "revolve", "profile": [[r, z], ...],          # closed outline, r >= 0
     "radial": [{"count": n, "polygon": [[r, z], ...], "thickness": t}]}   # fins, ribs, spokes

build(spec) validates everything and returns (code, plan). It raises SpecError with a
message that is safe to show to the user. Nothing here needs CadQuery, so it can be unit
tested anywhere; the returned code is run in the normal sandbox.
"""
import math

MAX_POINTS = 200
MAX_COORD = 5000.0
MIN_AREA = 1e-3


class SpecError(ValueError):
    pass


# ------------------------------------------------------------------ geometry helpers

def _num(v, what):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise SpecError(f"{what} must be a number")
    if not math.isfinite(f) or abs(f) > MAX_COORD:
        raise SpecError(f"{what} is out of range (max {MAX_COORD:g} mm)")
    return f


def _pts(raw, what, min_n=3):
    if not isinstance(raw, (list, tuple)):
        raise SpecError(f"{what} must be a list of [x, y] points")
    out = []
    for p in raw:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise SpecError(f"{what}: every point must be [x, y]")
        q = (round(_num(p[0], what), 3), round(_num(p[1], what), 3))
        if not out or math.dist(q, out[-1]) > 1e-6:
            out.append(q)
    if len(out) > 1 and math.dist(out[0], out[-1]) < 1e-6:
        out.pop()
    if len(out) < min_n:
        raise SpecError(f"{what} needs at least {min_n} different points")
    if len(out) > MAX_POINTS:
        raise SpecError(f"{what} has too many points (max {MAX_POINTS})")
    return out


def area(p):
    """Signed shoelace area."""
    return 0.5 * sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
                     for i in range(len(p)))


def centroid_x(p):
    """x of the centroid of a simple polygon (for Pappus' volume)."""
    a = area(p)
    if abs(a) < 1e-12:
        return 0.0
    s = 0.0
    for i in range(len(p)):
        x0, y0 = p[i]
        x1, y1 = p[(i + 1) % len(p)]
        s += (x0 + x1) * (x0 * y1 - x1 * y0)
    return s / (6.0 * a)


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_cross(a, b, c, d):
    """True when open segments ab and cd properly cross or overlap."""
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    eps = 1e-9
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    return False


def _touch_interior(a, b, c):
    """True when point c lies on segment ab but is not an endpoint (a vertex touching an edge)."""
    if abs(_orient(a, b, c)) > 1e-7 * max(1.0, math.dist(a, b)):
        return False
    if min(math.dist(c, a), math.dist(c, b)) < 1e-6:
        return False
    return (min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and
            min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9)


def self_intersects(p):
    n = len(p)
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            c, d = p[j], p[(j + 1) % n]
            if _seg_cross(a, b, c, d) or _touch_interior(a, b, c) or _touch_interior(a, b, d) \
               or _touch_interior(c, d, a) or _touch_interior(c, d, b):
                return True
    return False


def inside(pt, poly):
    x, y = pt
    c = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            c = not c
    return c


def seg_dist(pt, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / L2))
    return math.hypot(pt[0] - (ax + t * dx), pt[1] - (ay + t * dy))


def _edge_dist(pt, poly):
    return min(seg_dist(pt, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def _simple(p, what):
    if self_intersects(p):
        raise SpecError(f"{what} crosses itself - move the points so the outline does not cross")
    if abs(area(p)) < MIN_AREA:
        raise SpecError(f"{what} has no area (all points on one line?)")


def _fmt(p):
    return "[" + ", ".join(f"({x:g}, {y:g})" for x, y in p) + "]"


def _num_pos(v, what, hi=2000.0):
    f = _num(v, what)
    if f <= 0 or f > hi:
        raise SpecError(f"{what} must be between 0 and {hi:g} mm")
    return f


# ------------------------------------------------------------------ polygon (extrude)

def _clean_polygon(spec):
    pts = _pts(spec.get("points"), "outline")
    _simple(pts, "outline")
    t = _num_pos(spec.get("thickness", 5), "thickness", 1000.0)
    holes = []
    for h in spec.get("holes") or []:
        if not isinstance(h, dict):
            raise SpecError("each hole must be {x, y, d}")
        x, y, d = _num(h.get("x"), "hole x"), _num(h.get("y"), "hole y"), _num_pos(h.get("d"), "hole diameter", 1000.0)
        if not inside((x, y), pts):
            raise SpecError(f"hole at ({x:g}, {y:g}) is outside the outline")
        if _edge_dist((x, y), pts) < d / 2 + 0.2:
            raise SpecError(f"hole at ({x:g}, {y:g}), d={d:g} touches or crosses the outline edge")
        holes.append({"x": round(x, 3), "y": round(y, 3), "d": round(d, 3)})
    if len(holes) > 100:
        raise SpecError("too many holes (max 100)")
    cutouts = []
    for k, c in enumerate(spec.get("cutouts") or [], 1):
        cp = _pts(c, f"cut-out {k}")
        _simple(cp, f"cut-out {k}")
        for q in cp:
            if not inside(q, pts) or _edge_dist(q, pts) < 0.2:
                raise SpecError(f"cut-out {k} must lie completely inside the outline")
        for i in range(len(cp)):
            for j in range(len(pts)):
                if _seg_cross(cp[i], cp[(i + 1) % len(cp)], pts[j], pts[(j + 1) % len(pts)]):
                    raise SpecError(f"cut-out {k} crosses the outline")
        cutouts.append(cp)
    for i, a in enumerate(cutouts):          # cut-outs must not overlap each other or the holes
        for j, b in enumerate(cutouts):
            if i < j and (any(inside(q, b) for q in a) or any(inside(q, a) for q in b)):
                raise SpecError(f"cut-outs {i + 1} and {j + 1} overlap")
    for h in holes:
        c = (h["x"], h["y"])
        for k, cp in enumerate(cutouts, 1):
            if inside(c, cp) or _edge_dist(c, cp) < h["d"] / 2 + 0.2:
                raise SpecError(f"a hole overlaps cut-out {k}")
    for i, a in enumerate(holes):
        for b in holes[i + 1:]:
            if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < (a["d"] + b["d"]) / 2 + 0.2:
                raise SpecError("two holes overlap")
    return {"type": "polygon", "points": [list(p) for p in pts], "thickness": round(t, 3),
            "holes": holes, "cutouts": [[list(p) for p in c] for c in cutouts]}


def _polygon_code(s):
    pts = [tuple(p) for p in s["points"]]
    t = s["thickness"]
    L = ["import cadquery as cq", "",
         f"thickness = {t:g}",
         f"outline = {_fmt(pts)}",
         'result = cq.Workplane("XY").polyline(outline).close().extrude(thickness)']
    for h in s["holes"]:
        L.append(f'result = result.cut(cq.Workplane("XY").workplane(offset=-1).center({h["x"]:g}, {h["y"]:g})'
                 f'.circle({h["d"] / 2:g}).extrude(thickness + 2))   # hole d={h["d"]:g}')
    for c in s["cutouts"]:
        L.append(f'result = result.cut(cq.Workplane("XY").workplane(offset=-1)'
                 f'.polyline({_fmt([tuple(p) for p in c])}).close().extrude(thickness + 2))')
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    vol = (abs(area(pts)) - sum(math.pi * (h["d"] / 2) ** 2 for h in s["holes"])
           - sum(abs(area([tuple(p) for p in c])) for c in s["cutouts"])) * t
    plan = {"overall_size_mm": [round(max(xs) - min(xs), 3), round(max(ys) - min(ys), 3), t],
            "expect_volume_mm3": round(vol, 3)}
    return "\n".join(L) + "\n", plan


# ------------------------------------------------------------------ revolve

def _outer_r_at(profile, z):
    """Largest radius of the half-section at height z (None when z is outside the part)."""
    best = None
    n = len(profile)
    for i in range(n):
        (r0, z0), (r1, z1) = profile[i], profile[(i + 1) % n]
        if abs(z1 - z0) < 1e-9:
            if abs(z - z0) < 1e-6:
                r = max(r0, r1)
            else:
                continue
        elif min(z0, z1) - 1e-9 <= z <= max(z0, z1) + 1e-9:
            r = r0 + (r1 - r0) * (z - z0) / (z1 - z0)
        else:
            continue
        best = r if best is None else max(best, r)
    return best


def _clean_revolve(spec):
    prof = _pts(spec.get("profile"), "profile")
    if any(r < -1e-9 for r, _ in prof):
        raise SpecError("profile radius cannot be negative (the profile is the right half of the section)")
    prof = [(max(r, 0.0), z) for r, z in prof]
    _simple(prof, "profile")
    if max(r for r, _ in prof) <= 0:
        raise SpecError("profile has no radius")
    radial = []
    for k, f in enumerate(spec.get("radial") or [], 1):
        if not isinstance(f, dict):
            raise SpecError("each radial feature must be {count, polygon, thickness}")
        n = int(_num(f.get("count", 3), "fin count"))
        if not 1 <= n <= 24:
            raise SpecError("fin count must be 1..24")
        poly = _pts(f.get("polygon"), f"fin {k} polygon")
        _simple(poly, f"fin {k} polygon")
        if any(r < -1e-9 for r, _ in poly):
            raise SpecError(f"fin {k} polygon radius cannot be negative")
        t = _num_pos(f.get("thickness", 2), f"fin {k} thickness", 500.0)
        rmin = min(r for r, _ in poly)
        poly2 = []
        for r, z in poly:            # sink the root into the body so the fin fuses (one solid)
            if r <= rmin + 1e-6:
                ro = _outer_r_at(prof, z)
                if ro is None:
                    raise SpecError(f"fin {k} root at z={z:g} is outside the body height")
                r = max(0.0, min(r, ro - max(0.5, 0.1 * ro)))
            poly2.append((round(r, 3), z))
        radial.append({"count": n, "polygon": [list(p) for p in poly2], "thickness": round(t, 3)})
    return {"type": "revolve", "profile": [list(p) for p in prof], "radial": radial}


def _revolve_code(s):
    prof = [tuple(p) for p in s["profile"]]
    L = ["import cadquery as cq", "",
         "# half-section: (radius, height) points, revolved 360 deg around the Z axis",
         f"profile = {_fmt(prof)}",
         'body = cq.Workplane("XZ").polyline(profile).close().revolve(360, (0, 0, 0), (0, 1, 0))',
         "result = body"]
    for i, f in enumerate(s["radial"], 1):
        L += [f"fin{i} = {_fmt([tuple(p) for p in f['polygon']])}",
              f'fin{i}_solid = cq.Workplane("XZ").polyline(fin{i}).close().extrude({f["thickness"] / 2:g}, both=True)',
              f"result = result.union(radial(fin{i}_solid, {f['count']}))"]
    zs = [p[1] for p in prof] + [p[1] for f in s["radial"] for p in f["polygon"]]
    rmax = max(p[0] for p in prof)
    plan = {"expect_height_mm": round(max(zs) - min(zs), 3)}
    if not s["radial"]:
        plan["overall_size_mm"] = [round(2 * rmax, 3), round(2 * rmax, 3), plan["expect_height_mm"]]
        plan["expect_volume_mm3"] = round(2 * math.pi * abs(centroid_x(prof)) * abs(area(prof)), 3)
    return "\n".join(L) + "\n", plan


# ------------------------------------------------------------------ public

def clean(spec):
    if not isinstance(spec, dict):
        raise SpecError("shape spec must be an object")
    kind = spec.get("type")
    if kind == "polygon":
        return _clean_polygon(spec)
    if kind == "revolve":
        return _clean_revolve(spec)
    raise SpecError("type must be 'polygon' or 'revolve'")


def build(spec):
    """-> (code, plan, cleaned_spec). plan has overall_size_mm / expect_* for scoring."""
    s = clean(spec)
    code, plan = _polygon_code(s) if s["type"] == "polygon" else _revolve_code(s)
    plan.update(assembly=False, type=s["type"], profile_spec=s)
    return code, plan, s
