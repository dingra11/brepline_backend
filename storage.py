"""Training-data storage: local JSONL (always) + Firebase Realtime Database (optional).

Collections (same names locally as `<DATA_DIR>/<name>.jsonl` and remotely as
`/<FIREBASE_ROOT>/<name>/...`):

  generations  one record per finished job            (schema brepline.gen/1)
  failures     every failed / low-score attempt        (schema brepline.fail/1)
  feedback     user votes (up/down) and comments       (schema brepline.vote/1)

Writes to Firebase happen on a background thread so they never slow a request.
If Firebase is unreachable or rejects the write, the record is appended to
`<DATA_DIR>/firebase_pending.jsonl`; run `python storage.py --flush` to retry.

Firebase is spoken to over plain REST (no SDK). The web `apiKey` is NOT used for
database writes; it is only a public project identifier. Use security rules
(see firebase.rules.json) and, preferably, FIREBASE_AUTH (database secret or
OAuth access token) on the server.
"""
import os, json, time, queue, threading, urllib.request, urllib.error, urllib.parse

# Public identifiers from the project's web config. Safe to keep in code.
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyCut-Qi7scoHnjHyE8UBuN53PHWnMMKqSE",
    "authDomain": "sitee-f6a0c.firebaseapp.com",
    "databaseURL": "https://sitee-f6a0c-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "sitee-f6a0c",
    "storageBucket": "sitee-f6a0c.firebasestorage.app",
    "messagingSenderId": "284183052545",
    "appId": "1:284183052545:web:ef85ecc7be844cede8db00",
    "measurementId": "G-HZHS31QH07",
}

DATA_DIR = os.getenv("DATA_DIR", "/tmp/cad_data")
FB_URL = os.getenv("FIREBASE_DB_URL", FIREBASE_CONFIG["databaseURL"]).rstrip("/")
FB_ROOT = os.getenv("FIREBASE_ROOT", "brepline").strip("/")
FB_AUTH = os.getenv("FIREBASE_AUTH", "")
FB_ON = os.getenv("FIREBASE_ENABLED", "1") == "1"

_lock = threading.Lock()
_q = queue.Queue(maxsize=5000)
_worker = None


def local_path(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name + ".jsonl")


def _append(name, rec):
    with _lock, open(local_path(name), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _fb_request(method, path, rec=None, timeout=15):
    url = f"{FB_URL}/{FB_ROOT}/{path}.json"
    if FB_AUTH:
        url += "?" + urllib.parse.urlencode({"auth": FB_AUTH})
    data = None if rec is None else json.dumps(rec, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else None


def _loop():
    while True:
        method, path, rec = _q.get()
        ok = False
        for i in range(3):
            try:
                _fb_request(method, path, rec)
                ok = True
                break
            except Exception as e:  # network, 401/403 rules, etc.
                err = str(e)
                time.sleep(2 ** i)
        if not ok:
            with _lock, open(os.path.join(DATA_DIR, "firebase_pending.jsonl"), "a") as f:
                f.write(json.dumps({"method": method, "path": path, "rec": rec,
                                    "err": err[:200]}) + "\n")
        _q.task_done()


def _remote(method, path, rec):
    global _worker
    if not FB_ON:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    if _worker is None:
        _worker = threading.Thread(target=_loop, daemon=True)
        _worker.start()
    try:
        _q.put_nowait((method, path, rec))
    except queue.Full:
        pass  # local copy already written


def _clean_key(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))[:64] or "x"


# ----------------------------------------------------------------- public API

def save_generation(rec):
    rec = {"schema": "brepline.gen/1", "ts": time.time(), **rec}
    _append("generations", rec)
    _remote("PUT", f"generations/{_clean_key(rec['job'])}", rec)


def save_failure(rec):
    rec = {"schema": "brepline.fail/1", "ts": time.time(), **rec}
    _append("failures", rec)
    _remote("POST", f"failures/{_clean_key(rec.get('job', 'x'))}", rec)


def save_feedback(job, vote, comment="", ip_hash=""):
    rec = {"schema": "brepline.vote/1", "ts": time.time(), "job": job,
           "vote": vote, "comment": (comment or "")[:500], "ip": ip_hash}
    _append("feedback", rec)
    _remote("POST", f"feedback/{_clean_key(job)}", rec)


def read_local(name):
    p = local_path(name)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def read_remote(name):
    """Return a list of records from Firebase collection `name`."""
    data = _fb_request("GET", name, timeout=120) or {}
    rows = []
    for k, v in data.items():
        if isinstance(v, dict) and "schema" in v:
            rows.append(v)
        elif isinstance(v, dict):          # POST collections are {pushId: rec}
            rows += [x for x in v.values() if isinstance(x, dict)]
    return rows


def flush_pending():
    p = os.path.join(DATA_DIR, "firebase_pending.jsonl")
    if not os.path.exists(p):
        print("nothing pending")
        return
    items = [json.loads(l) for l in open(p) if l.strip()]
    left = []
    for it in items:
        try:
            _fb_request(it["method"], it["path"], it["rec"])
        except Exception as e:
            it["err"] = str(e)[:200]
            left.append(it)
    with open(p, "w") as f:
        for it in left:
            f.write(json.dumps(it) + "\n")
    print(f"flushed {len(items) - len(left)}, still pending {len(left)}")


def wait_flush(timeout=10):
    t = time.time()
    while _q.unfinished_tasks and time.time() - t < timeout:
        time.sleep(0.1)


def selftest():
    """Write + read + delete a probe node to check Firebase rules/connectivity."""
    try:
        _fb_request("PUT", "_probe", {"ts": time.time()})
        got = _fb_request("GET", "_probe")
        _fb_request("DELETE", "_probe")
        return True, f"write/read ok ({got})"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read()[:150]!r}  (check rules / FIREBASE_AUTH)"
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    import sys
    if "--flush" in sys.argv:
        flush_pending()
    else:
        print(selftest())
