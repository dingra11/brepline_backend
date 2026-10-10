"""Geometry helpers made available to every generated CadQuery script.

They are injected into the script namespace by runner.py, so the LLM can call
them without importing. Keeping them here (instead of asking the LLM to
re-derive them each time) removes the most common failure modes:
disconnected radial parts (fins, spokes) and bolt-hole patterns.
"""
import math
import cadquery as cq


def radial(shape, n, r=0):
    """Copy `shape` n times around the Z axis and union the copies.

    `shape` must be modelled pointing along +X from the origin. `r` shifts each
    copy outward before rotating. Use r=0 and let the shape overlap the central
    body so the result stays one connected solid (rocket fins, spokes, ribs).
    """
    out = None
    for i in range(int(n)):
        s = shape.translate((r, 0, 0)).rotate((0, 0, 0), (0, 0, 1), 360.0 * i / n)
        out = s if out is None else out.union(s)
    return out


def bolt_circle(body, n, r, d, face=">Z"):
    """Drill `n` through-holes of diameter `d` on a circle of radius `r`
    centred on the Z axis, starting from `face` of `body`."""
    return body.faces(face).workplane(centerOption="CenterOfBoundBox") \
        .polarArray(r, 0, 360, int(n)).hole(d)


def spur_gear(m, z, t, bore=0):
    """Simple trapezoid-tooth spur gear: module m, z teeth, thickness t (mm).
    Not a true involute, but a valid solid that looks and meshes roughly right.
    """
    rp = m * z / 2.0
    ra = rp + m
    rf = max(rp - 1.25 * m, 0.5)
    pitch = 2 * math.pi / z
    pts = []
    for i in range(int(z)):
        a0 = i * pitch
        for frac, rad in ((0.0, rf), (0.2, ra), (0.45, ra), (0.65, rf)):
            a = a0 + frac * pitch
            pts.append((rad * math.cos(a), rad * math.sin(a)))
    g = cq.Workplane("XY").polyline(pts).close().extrude(t)
    if bore:
        g = g.faces(">Z").workplane(centerOption="CenterOfBoundBox").hole(bore)
    return g
