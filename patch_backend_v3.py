"""Adapt the Brepline backend to the v3 model (idempotent). Run in the folder with cadpipe.py / serve_model.py:

    python patch_backend_v3.py            # edits cadpipe.py and serve_model.py in place (writes .bak copies)

cadpipe.py     : planner now writes the v3 prompt grammar ("Create a CadQuery model. ... Overall size: ...
                 Steps: 1) Base: ..."), spec extraction + size check match that grammar.
serve_model.py : repetition_penalty 1.0 (the penalty also hits tokens in the prompt, so it fights the
                 model's number copying), same 4-bit config as training (double quant).
"""
import re, shutil, sys

FENCE = "`" * 3

NEW_SPEC = r'''SPEC_SYS = """You convert a CAD request into a precise build spec for a fine-tuned CadQuery model.
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
'''

NEW_EXTRACT = '''def extract_spec(text):
    """Pull the v3 'Create a CadQuery model ... Steps: 1) Base: ...' spec out of a planner reply."""
    i = (text or "").find("Create a CadQuery model")
    if i < 0:
        return ""
    s = text[i:].split("<|im_end|>")[0].replace(FENCE, "").strip()
    return s if "Steps:" in s and "Base:" in s else ""
'''

NEW_ROUTE = """# v3 routing: the fine-tuned model now handles plates, text, L-brackets, nuts, bolts, pulleys, shafts, flanges.
ROUTE_SYS = \"\"\"Classify a CAD request by how the part is best built. Answer with exactly one word.
REVOLVE = an organic round shape whose profile curves freely along its axis: rocket, vase, bottle, goblet, glass, bell, cone, funnel, chess piece, knob, bullet, dome, lamp base, candle holder.
POLYGON = a flat part whose outline is a free shape that is not a rectangle, circle or L-bracket: star, arrow, heart, triangle, gusset, pentagon plate, or any shape given as a list of points.
PRISM = everything else, including plates (with holes, text or engraving), nameplates, anything with letters or text, blocks, washers, rings, tubes, discs, flanges, hex nuts, bolts, standoffs, pins, pulleys, stepped shafts, L-brackets, enclosures, bearing blocks and boxes with pockets.\"\"\"
ROUTE_SHOTS = [("make a rocket", "REVOLVE"), ("washer, 16 mm outer, 8 mm inner, 2 mm thick", "PRISM"),
               ("L bracket 60 x 40 mm", "PRISM"), ("plate 100 x 60 x 8 with four corner holes", "PRISM"),
               ("a vase 120 mm tall", "REVOLVE"), ("6 point star, 50 mm", "POLYGON"),
               ("a nameplate that says HELLO", "PRISM"), ("hex nut for a 10 mm bolt", "PRISM"),
               ("V-groove pulley 60 mm", "PRISM"), ("heart shaped pendant, 40 mm", "POLYGON")]
"""

def patch_cadpipe(path="cadpipe.py"):
    s = open(path).read()
    orig = s
    if "v3 prompt grammar" not in s:
        a = s.index('SPEC_SYS = """')
        b = s.index("# ------------------------------------------------------------------ LLM")
        s = s[:a] + "# v3 prompt grammar (see make_cad_dataset.py)\n" + NEW_SPEC + "\n\n" + s[b:]
        a = s.index("def extract_spec(text):")
        b = s.index("def extract_json(text):")
        s = s[:a] + NEW_EXTRACT + "\n\n" + s[b:]
        old = 're.search(r"bounding box of ([\\d.]+) x ([\\d.]+) x ([\\d.]+) mm", spec)'
        assert old in s, "bbox regex not found"
        s = s.replace(old, 're.search(r"Overall size:\\s*([\\d.]+) x ([\\d.]+) x ([\\d.]+) mm", spec)')
        print("cadpipe.py: v3 grammar patched")
    # an EXTERNAL planner (PLAN_BASE_URL) may be a reasoning model (gpt-oss, qwen3.x): give it room to think
    old = ('    if LLM_MODE == "lmm":\n'
           '        # the fine-tuned model has a fixed output size; reasoning-model padding does not apply\n'
           '        if planner and not own_planner:\n')
    new = ('    if LLM_MODE == "lmm" and not own_planner:\n'
           '        # the fine-tuned model has a fixed output size; reasoning-model padding does not apply\n'
           '        if planner:\n')
    if old in s:
        s = s.replace(old, new); print("cadpipe.py: reasoning-model planner patched")
    if "# v3 routing" not in s:
        a = s.index('ROUTE_SYS = """')
        b = s.index("def route_kind(req):")
        s = s[:a] + NEW_ROUTE.replace('\\"\\"\\"', '"' * 3) + "\n\n" + s[b:]
        print("cadpipe.py: routing patched (model keeps plates, text, brackets, nuts, pulleys)")
    if s != orig:
        shutil.copy(path, path + ".bak")
        open(path, "w").write(s)
    else:
        print("cadpipe.py already fully patched")

def patch_serve(path="serve_model.py"):
    s = open(path).read()
    if "repetition_penalty=1.0," in s:
        print("serve_model.py already patched"); return
    shutil.copy(path, path + ".bak")
    assert "repetition_penalty=1.05" in s, "repetition_penalty not found"
    s = s.replace("repetition_penalty=1.05,", "repetition_penalty=1.0,")
    s = re.sub(r'(bnb_4bit_quant_type="nf4",)(\s*)(bnb_4bit_compute_dtype)',
               r'\1\2bnb_4bit_use_double_quant=True,\2\3', s, count=1)
    open(path, "w").write(s); print("patched serve_model.py")

if __name__ == "__main__":
    which = sys.argv[1:] or ["cadpipe", "serve"]
    if "cadpipe" in which: patch_cadpipe()
    if "serve" in which: patch_serve()
