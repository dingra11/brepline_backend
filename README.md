# Brepline backend

Text prompt -> CadQuery code -> validated solid -> STEP / STL / mesh / mass, behind a small
FastAPI job API. Every job is saved as training data (local JSONL + Firebase Realtime DB).

## Current state (honest version)

**Architecture.** An instruct LLM (any OpenAI-compatible API) does the work: planner -> retrieved
few-shot examples -> N parallel candidates -> sandboxed run -> score -> repair loop. **Your
fine-tuned DeepCAD model is not in this path.** The plan is to *distil* this pipeline: its verified
outputs become training data (below), and later a fine-tuned model can replace or pre-filter the API LLM.

| Area | Status |
|---|---|
| Code written, all files compile (`py_compile`) | done |
| AST safety validator (blocks os/sys/open/dunder etc.) | tested, works |
| Storage + export round trip (local JSONL -> chatml/messages/prompt_completion/alpaca) | tested, works |
| CadQuery examples, helpers (`radial`, `bolt_circle`, `spur_gear`), `runner.py` | **untested**: CadQuery could not be installed where this was assembled (no internet). Run `python smoke_test.py` first; fix any failing example before deploying. |
| FastAPI server (jobs, polling, files, feedback, export, rate limit) | **untested** (not run) |
| LLM planner / candidates / repair loop | **untested** against a real LLM |
| Firebase writes | **untested**: depends on your rules / `FIREBASE_AUTH`; check with `python storage.py` |
| Frontend integration (`frontend_integration/`) | **untested** and not merged into your `App.jsx` (see INTEGRATION.md) |
| FEA / safety factor | **not implemented** (remove from UI) |
| Mass | geometry volume x material density only |

Earlier sessions showed that this style of pipeline (instruct model + examples + repair) produced a
hex bolt, pulley, flanged housing, hydraulic cylinder and a rocket. Expect good results on simple
and medium single parts; expect failures on organic shapes, threads, true involute gears, and
multi-part assemblies with motion.

Known limitations: no true threads; gears are trapezoid-tooth approximations; the sandbox is
process isolation with resource limits, **not** a hard security boundary (run it in a container, as
the Dockerfile does); output files live on ephemeral disk (24 h TTL by default).

## Layout

```
cadpipe.py          pipeline (planner, retrieval, candidates, repair, scoring, persistence)
runner.py           sandboxed subprocess: executes the script, tessellates, exports STEP/STL
helpers.py          radial / bolt_circle / spur_gear available to generated code
profiles.py         exact builders: drawn polygon (extrude) / half-section (revolve) -> CadQuery code + checks
examples_lib.py     few-shot library (add more examples here = better results)
storage.py          local JSONL + Firebase RTDB (REST, background queue, retry spool)
server.py           FastAPI API
export_training.py  build fine-tuning files from saved data
factory.py          LLM-generated prompts -> pipeline -> verified training data
smoke_test.py       no-LLM test of examples, validator, storage, export
firebase.rules.json suggested database rules
frontend_integration/  api.js, MeshViewer.jsx, INTEGRATION.md for the React app
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill LLM_*; then:  set -a; source .env; set +a
python smoke_test.py            # CadQuery examples must all print OK
uvicorn server:app --port 7860
curl localhost:7860/api/health
```

## Deploy free (Hugging Face Space)

New Space -> SDK **Docker** -> upload this folder. Add secrets: `LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_MODEL`, `ACCESS_CODE`, `ALLOWED_ORIGINS` (your website origin), `ADMIN_TOKEN`,
`FIREBASE_AUTH`. The API is at `https://<user>-<space>.hf.space`. Free CPU Spaces sleep when idle
and storage is ephemeral, which is why training data goes to Firebase.

## API

| Method | Path | Notes |
|---|---|---|
| POST | `/api/jobs` | `{prompt, material?, parent_job?, fresh?}` -> `{job}`; header `X-Access-Code` |
| GET | `/api/jobs/{job}` | `{status: queued/running/done/error, log[], result?}`; result has `mesh`, `bbox`, `mass_g`, `code`, `verified`, `warning` |
| GET | `/api/file/{job}/{step\|stl\|py}` | downloads |
| POST | `/api/feedback` | `{job, vote: up/down/none, comment?}` |
| GET | `/api/export?fmt=messages&source=local` | header `X-Admin-Token` |
| POST | `/api/shape` | `{spec, name?, material?, parent_job?}` -> `{job}` (poll `/api/jobs/{job}`). Builds a drawn polygon / revolve profile directly, no LLM |
| POST | `/api/shape/validate` | `{spec}` -> `{ok, error?, bbox, volume_mm3}` instant check while drawing |
| GET | `/api/health` | |

`parent_job` makes the request an edit of that job's code ("make the hole bigger").

## Saving data for training

Collections (local `DATA_DIR/<name>.jsonl` and Firebase `/brepline/<name>`):

* **generations** (`brepline.gen/1`), one per job: `job, ts, source (user|factory), prompt, is_edit,
  parent_job, parent_code, material, plan, code, ok, verified, size_checked, score, issues, bbox,
  volume_mm3, mass_g, n_solids, attempts, repaired_from, llm_model, pipeline_version, retrieved_examples`
* **failures** (`brepline.fail/1`): every low-score attempt with `code, stage, score, issues` (negatives, and
  broken -> fixed repair pairs)
* **feedback** (`brepline.vote/1`): `job, vote, comment, hashed ip`

The schema is versioned and records are plain JSON, so new fields can be added without migration.
Meshes are not stored (large); STEP can be regenerated from `code`.

Firebase: writes use REST on a background thread (never blocks requests); failed writes are spooled to
`firebase_pending.jsonl` (`python storage.py --flush` retries). The web `apiKey` in your config is a
public identifier, not a secret, so **what protects the data is the rules**: paste
`firebase.rules.json` into the console (reads off, create-only writes) and set `FIREBASE_AUTH` (database
secret or access token) on the server so writes are authenticated. Note that with unauthenticated
create-only rules anyone who knows your database URL can write junk, so prefer `FIREBASE_AUTH` with
`".write": false` rules. Check connectivity with `python storage.py`.

Privacy: prompts are stored verbatim. Tell users data is kept for model training.

### Export

```bash
python export_training.py --source firebase --format all --out export/   # or --source local
```
Keeps only verified (score 10), non-edit, not down-voted, de-duplicated by code hash (`--include-edits`,
`--min-score`, `--with-plan` available). Formats: `chatml` (`text`, Qwen), `messages`,
`prompt_completion`, `alpaca`, `repair`.

### Grow the dataset

`python factory.py --per-category 20` has the LLM invent diverse requests across 10 categories and runs them
through the pipeline. Then export, fine-tune on `messages.jsonl`, and benchmark against the API pipeline on a
held-out prompt set before trusting it.

## Using your own fine-tuned model (LLM_MODE=lmm)

The adapter was trained on one fixed format, so the pipeline has a mode that matches it:

1. **Planner** turns the plain-English request into a build spec
   (`Create a CadQuery model with a bounding box of ... Steps: 1) new body: sketch on XY plane ...`).
   Six worked examples live in `SPEC_SHOTS` in `cadpipe.py`. The adapter is switched off for this step.
2. **Coder** gets the training system prompt + that spec, with the ```` ```python ```` fence pre-filled.
3. Retries re-sample the same spec (the adapter never saw repair turns) and results are checked
   against the spec's bounding box (5 % tolerance).

`serve_model.py` must accept `prefill` and `adapter` fields (see the patch cells in the chat history / `serve_model_patch.py`).
The stored `plan.spec` of every job is a ready-made (request -> spec) and (spec -> code) training pair.

## Round parts, polygons and the "Verified" badge

* **Routing (lmm mode, `ROUTE=1`, set `ROUTE=0` to disable).** A one-word classifier call (base model, adapter off) sends
  *round parts whose radius changes* (rocket, vase, bottle, pulley...) to the **revolve** builder and *outline parts*
  (L / T / U brackets, stars, hex nuts, any list of points) to the **polygon** builder. The planner only produces a small JSON
  profile; `profiles.py` validates it (no self-crossing, holes inside, fins fused to the body) and writes the CadQuery code.
  Everything else (washers, flanges, plates with holes...) still goes to your fine-tuned adapter. If the profile cannot be
  made valid after 3 tries the request falls back to the adapter.
* **Edits** of a polygon / revolve design re-plan the JSON ("make it twice as tall") instead of patching code.
* **Custom polygons from the UI.** `frontend_integration/ShapeEditor.jsx` posts the user's own outline to `/api/shape`.
  Spec format is documented at the top of `profiles.py`. The result's `plan.profile_spec` reopens in the same editor.
* **What "Verified" means now.** Valid, single solid, AND measured against something: polygon / revolve builds are checked
  against exact expected volume and size; adapter builds against the plan's bounding box. A plain box returned for a request
  that does not mention a box is rejected, and `check_detail` says what was actually checked.
  A "Verified" model is still only as right as the plan, so thumbs-down bad ones: they are excluded from training.

## Next steps (suggested order)

1. `python smoke_test.py` and fix failing examples. 2. Run one real prompt end to end. 3. Merge the frontend
changes. 4. Set Firebase rules/auth and verify with `python storage.py`. 5. Run the factory for a few hundred
verified rows. 6. Fine-tune and compare head-to-head.
