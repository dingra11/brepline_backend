"""Executed in a sandboxed subprocess by cadpipe.run_sandbox.

usage: python runner.py <code.py> <out_dir>
Writes out_dir/meta.json, out_dir/mesh.json, out_dir/model.step, out_dir/model.stl
meta.json always exists; meta["ok"] tells whether the script produced a solid.
"""
import sys, os, json, math, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    code_path, out = sys.argv[1], sys.argv[2]
    meta = {"ok": False}

    def done():
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f)

    try:
        import cadquery as cq
        import numpy
        from helpers import radial, bolt_circle, spur_gear

        src = open(code_path).read()
        ns = {"__name__": "__cad__", "cq": cq, "math": math, "np": numpy,
              "numpy": numpy, "radial": radial, "bolt_circle": bolt_circle,
              "spur_gear": spur_gear}
        exec(compile(src, "model.py", "exec"), ns)

        res, parts = ns.get("result"), ns.get("parts")
        named = {}
        if isinstance(parts, dict) and parts:
            named = {str(k): to_shape(cq, v) for k, v in parts.items()}
            shape = cq.Compound.makeCompound(list(named.values()))
        elif res is not None:
            shape = to_shape(cq, res)
        else:
            raise ValueError("script did not define `result`")

        solids = shape.Solids()
        meta["n_solids"] = len(solids)
        meta["valid"] = bool(shape.isValid())
        meta["volume"] = float(sum(s.Volume() for s in solids))
        bb = shape.BoundingBox()
        meta["bbox"] = [bb.xlen, bb.ylen, bb.zlen]
        meta["bbox_min"] = [bb.xmin, bb.ymin, bb.zmin]
        if not solids:
            raise ValueError("result contains no solid")
        try:  # extra analysis for the 3D info panel (never fatal)
            meta["area"] = float(shape.Area())
            c = shape.Center()
            meta["com"] = [c.x, c.y, c.z]
            meta["counts"] = {"solids": len(solids), "faces": len(shape.Faces()),
                              "edges": len(shape.Edges()), "vertices": len(shape.Vertices())}
        except Exception:
            pass

        diag = math.sqrt(bb.xlen ** 2 + bb.ylen ** 2 + bb.zlen ** 2) or 1.0
        tol = max(0.05, diag / 500.0)

        verts, tris, tri_part, names = [], [], [], list(named.keys())
        groups = list(named.items()) if named else [("", shape)]
        for pi, (_, sh) in enumerate(groups):
            v, t = sh.tessellate(tol, 0.3)
            base = len(verts) // 3
            for p in v:
                verts += [round(p.x, 3), round(p.y, 3), round(p.z, 3)]
            for a, b, c in t:
                tris += [base + a, base + b, base + c]
                tri_part.append(pi)
        with open(os.path.join(out, "mesh.json"), "w") as f:
            json.dump({"verts": verts, "tris": tris, "tri_part": tri_part,
                       "parts": names}, f)

        wp = cq.Workplane("XY").newObject([shape])
        cq.exporters.export(wp, os.path.join(out, "model.step"))
        cq.exporters.export(wp, os.path.join(out, "model.stl"), tolerance=tol)
        meta["ok"] = True
    except BaseException:
        meta["error"] = "".join(traceback.format_exc().splitlines(True)[-6:])[-1500:]
    done()


def to_shape(cq, obj):
    if isinstance(obj, cq.Assembly):
        return obj.toCompound()
    if isinstance(obj, cq.Workplane):
        vals = [v for v in obj.vals() if isinstance(v, cq.Shape)]
        if not vals:
            raise ValueError("Workplane holds no solids (empty stack)")
        return vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)
    if isinstance(obj, cq.Shape):
        return obj
    raise TypeError("result must be a cq.Workplane, cq.Assembly or cq.Shape")


if __name__ == "__main__":
    main()
