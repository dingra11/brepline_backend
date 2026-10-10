"""Few-shot retrieval library: (tags, request, code). Every example is run
through the sandbox by smoke_test.py, so keep them valid.
Conventions: millimetres, Z up, final object in `result`."""

EXAMPLES = [
    (["bolt", "screw", "hex", "fastener", "thread"],
     "hex bolt M8 x 30 mm",
     '''import cadquery as cq
head = cq.Workplane("XY").polygon(6, 14.4).extrude(5.3)
shank = cq.Workplane("XY").workplane(offset=-30).circle(4).extrude(30)
result = head.union(shank).faces("<Z").chamfer(0.8)
'''),
    (["nut", "hex", "fastener"],
     "M8 hex nut",
     '''import cadquery as cq
result = cq.Workplane("XY").polygon(6, 14.4).extrude(6.5)
result = result.faces(">Z").workplane(centerOption="CenterOfBoundBox").hole(6.8)
'''),
    (["washer", "ring", "spacer"],
     "washer, 16 mm outer, 8.4 mm inner, 1.6 mm thick",
     '''import cadquery as cq
result = cq.Workplane("XY").circle(8).circle(4.2).extrude(1.6)
'''),
    (["flange", "housing", "pipe", "bearing", "cylinder", "bolt holes", "boss"],
     "flanged housing: 60 mm diameter, 50 mm tall tube, 100 mm flange 10 mm thick, 40 mm bore, 6 bolt holes of 6 mm on a 80 mm circle",
     '''import cadquery as cq
body = cq.Workplane("XY").circle(30).extrude(50)
flange = cq.Workplane("XY").circle(50).extrude(10)
result = body.union(flange)
result = result.faces(">Z").workplane(centerOption="CenterOfBoundBox").hole(40)
result = bolt_circle(result, 6, 40, 6, face="<Z")
'''),
    (["bracket", "angle", "mount", "plate", "holes"],
     "L bracket 60 mm wide, 40 mm base, 40 mm tall wall, 5 mm thick, two 6 mm mounting holes in the base",
     '''import cadquery as cq
base = cq.Workplane("XY").box(60, 40, 5, centered=(True, True, False))
wall = cq.Workplane("XY").box(60, 5, 40, centered=(True, True, False)).translate((0, -17.5, 0))
result = base.union(wall)
holes = cq.Workplane("XY").pushPoints([(-20, 5), (20, 5)]).circle(3).extrude(5)
result = result.cut(holes)
'''),
    (["rocket", "missile", "fin", "fins", "nose", "radial", "spoke"],
     "rocket 30 mm diameter, 120 mm body, cone nose 40 mm, 4 fins",
     '''import cadquery as cq
body = cq.Workplane("XY").circle(15).extrude(120)
nose = cq.Workplane("XY").workplane(offset=120).circle(15).workplane(offset=40).circle(1).loft()
fin = cq.Workplane("XY").box(25, 2, 35, centered=(False, True, False)).translate((13, 0, 0))
result = body.union(nose).union(radial(fin, 4))
'''),
    (["gear", "cog", "teeth", "sprocket"],
     "spur gear module 2, 20 teeth, 8 mm thick, 8 mm bore",
     '''import cadquery as cq
result = spur_gear(2, 20, 8, bore=8)
'''),
    (["enclosure", "box", "case", "shell", "hollow", "container", "cutout"],
     "electronics enclosure 80 x 50 x 30 mm, 2 mm walls, open top, USB cutout on one side",
     '''import cadquery as cq
result = cq.Workplane("XY").box(80, 50, 30).faces(">Z").shell(-2)
cut = cq.Workplane("XY").box(10, 12, 8).translate((40, 0, 0))
result = result.cut(cut)
'''),
    (["pulley", "wheel", "revolve", "groove", "lathe", "sheave"],
     "V-groove pulley 60 mm diameter, 20 mm wide, 16 mm bore",
     '''import cadquery as cq
profile = [(8, 0), (30, 0), (30, 6), (26, 10), (30, 14), (30, 20), (8, 20)]
result = cq.Workplane("XZ").polyline(profile).close().revolve(360, (0, 0, 0), (0, 1, 0))
'''),
    (["hydraulic", "piston", "cylinder", "rod", "actuator", "barrel"],
     "hydraulic cylinder: 50 mm barrel 40 mm bore, 150 mm long, piston and 16 mm rod, end cap",
     '''import cadquery as cq
barrel = cq.Workplane("XY").circle(25).circle(20).extrude(150)
piston = cq.Workplane("XY").circle(20).extrude(15).translate((0, 0, 60))
rod = cq.Workplane("XY").circle(8).extrude(120).translate((0, 0, 70))
cap = cq.Workplane("XY").circle(30).extrude(10).translate((0, 0, -10))
result = barrel.union(piston).union(rod).union(cap)
'''),
]
