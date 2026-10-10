"""Data factory: have the LLM invent diverse requests, run each through the pipeline,
and let storage keep the verified ones as training data.

  python factory.py --per-category 20 --workers 2
Needs LLM_* env vars. Costs LLM calls (about N_CAND+2 per prompt).
"""
import argparse, json, re
from concurrent.futures import ThreadPoolExecutor
import cadpipe as cp

CATEGORIES = {
    "fasteners": "bolts, screws, nuts, washers, studs, standoffs with explicit sizes",
    "brackets": "L-brackets, mounting plates, angle brackets, gussets with holes and slots",
    "housings": "flanged housings, bearing blocks, pipe fittings, caps, bushings",
    "enclosures": "electronics boxes, shells, trays with cutouts, bosses and ribs",
    "rotational": "pulleys, shafts, spools, knobs, flywheels, stepped shafts",
    "gears": "spur gears, gear blanks, sprockets, ratchets",
    "aero": "rockets, fins, nose cones, drone arms, propeller hubs",
    "hydraulic": "cylinders, pistons, manifolds, valve bodies",
    "furniture": "table legs, shelf brackets, hooks, handles, simple frames",
    "misc": "hinges, clips, springs-like shapes, funnels, lamp parts, toy parts",
}


def ideas(cat, desc, n):
    txt = cp.llm([{"role": "user", "content":
        f"Write {n} different, natural-language requests a user might type to a text-to-CAD tool for: {desc}. "
        "Vary the style: some terse ('make a screw'), some detailed with dimensions in mm. "
        "One object per request, buildable as a single part. Output a JSON list of strings only."}],
        temperature=0.9, max_tokens=1800)
    m = re.search(r"\[.*\]", txt, re.S)
    try:
        return [s for s in json.loads(m.group(0)) if isinstance(s, str)][:n]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    prompts = []
    for cat, desc in CATEGORIES.items():
        got = ideas(cat, desc, a.per_category)
        print(f"{cat}: {len(got)} prompts")
        prompts += got

    def one(p):
        try:
            out, _ = cp.generate(p, fresh=True, source="factory")
            return out.get("verified", False)
        except Exception as e:
            print("error:", e)
            return False

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, prompts))
    print(f"verified {sum(res)}/{len(res)}")


if __name__ == "__main__":
    main()
