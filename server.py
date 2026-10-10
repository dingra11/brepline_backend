"""FastAPI server: async job queue + polling.  Run: uvicorn server:app --port 7860"""
import os, re, time, uuid, shutil, hashlib, asyncio, collections
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import cadpipe as cp
import storage
import export_training as et

ACCESS_CODE = os.getenv("ACCESS_CODE", "")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
RATE = int(os.getenv("RATE_PER_HOUR", "30"))
MAX_JOBS = int(os.getenv("MAX_JOBS", "2"))
JOB_TTL_H = float(os.getenv("JOB_TTL_H", "24"))
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="Brepline backend")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"],
                   allow_headers=["*"])
SEM = asyncio.Semaphore(MAX_JOBS)
JOBS = {}
HITS = collections.defaultdict(collections.deque)


class JobReq(BaseModel):
    prompt: str = Field(min_length=2, max_length=1500)
    material: str = "Al 6061"
    parent_job: str | None = None
    fresh: bool = False


class VoteReq(BaseModel):
    job: str
    vote: str
    comment: str = ""


def client_ip(req: Request):
    xf = req.headers.get("x-forwarded-for")
    return (xf.split(",")[0].strip() if xf else req.client.host) or "?"


def guard(req: Request, code: str | None):
    if ACCESS_CODE and code != ACCESS_CODE:
        raise HTTPException(401, "invalid access code")
    ip = client_ip(req)
    q = HITS[ip]
    now = time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE:
        raise HTTPException(429, "rate limit reached, try again later")
    q.append(now)
    return ip


def gc():
    now = time.time()
    for j in [j for j, v in JOBS.items() if now - v["t"] > 3600]:
        JOBS.pop(j, None)
    ttl = JOB_TTL_H * 3600
    for d in os.listdir(cp.OUT_DIR):
        p = os.path.join(cp.OUT_DIR, d)
        try:
            if now - os.path.getmtime(p) > ttl:
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        except Exception:
            pass


async def run_job(job, req: JobReq):
    async with SEM:
        J = JOBS[job]
        J["status"] = "running"
        try:
            out, _ = await asyncio.to_thread(
                cp.generate, req.prompt, req.parent_job, req.material,
                lambda m: J["log"].append(m), job,
                req.fresh, "user")
            J["result"] = out
            J["status"] = "done" if out.get("ok") else "error"
            if not out.get("ok"):
                J["error"] = out.get("error", "generation failed")
        except Exception as e:
            J["status"], J["error"] = "error", str(e)[:300]
            J["log"].append("error")


@app.post("/api/jobs")
async def create_job(body: JobReq, request: Request, x_access_code: str | None = Header(None)):
    guard(request, x_access_code)
    if body.parent_job and not cp.valid_job(body.parent_job):
        raise HTTPException(400, "bad parent_job")
    gc()
    job = uuid.uuid4().hex[:12]
    JOBS[job] = {"status": "queued", "log": [], "t": time.time(), "result": None, "error": None}
    asyncio.create_task(run_job(job, body))
    return {"job": job}


@app.get("/api/jobs/{job}")
async def get_job(job: str):
    J = JOBS.get(job)
    if not J:
        out = cp.load_out(job)           # survives restarts / cache hits
        if out:
            return {"status": "done", "log": [], "result": out}
        raise HTTPException(404, "unknown job")
    r = {"status": J["status"], "log": J["log"], "error": J["error"]}
    if J["status"] in ("done", "error"):
        r["result"] = J["result"]
    return r


@app.get("/api/file/{job}/{ext}")
async def get_file(job: str, ext: str):
    if not cp.valid_job(job) or ext not in ("step", "stl", "py"):
        raise HTTPException(404)
    p = os.path.join(cp.job_dir(job), "model." + ext)
    if not os.path.exists(p):
        raise HTTPException(404, "file not found (expired?)")
    return FileResponse(p, filename=f"brepline_{job}.{ext}")


@app.post("/api/feedback")
async def feedback(body: VoteReq, request: Request, x_access_code: str | None = Header(None)):
    ip = guard(request, x_access_code)
    if body.vote not in ("up", "down", "none"):
        raise HTTPException(400, "vote must be up/down/none")
    if not cp.valid_job(body.job):
        raise HTTPException(400, "bad job")
    storage.save_feedback(body.job, body.vote, body.comment,
                          hashlib.sha1(ip.encode()).hexdigest()[:10])
    return {"ok": True}


@app.get("/api/export")
async def export(fmt: str = "messages", source: str = "local", min_score: int = 10,
                 x_admin_token: str | None = Header(None)):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(401, "admin token required")
    rows = et.load_rows(source)
    data, stats = et.build(rows, fmt, min_score=min_score)
    return Response(data, media_type="application/x-ndjson",
                    headers={"X-Export-Stats": str(stats).replace("\n", " ")})


@app.get("/api/health")
async def health():
    try:
        import cadquery  # noqa
        cq_ok = True
    except Exception:
        cq_ok = False
    return {"ok": True, "cadquery": cq_ok, "llm_configured": bool(os.getenv("LLM_BASE_URL")),
            "llm_model": os.getenv("LLM_MODEL", ""), "firebase": storage.FB_ON,
            "pipeline": cp.PIPELINE_VERSION, "queued": sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))}
