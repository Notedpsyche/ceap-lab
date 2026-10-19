import asyncio
from contextlib import asynccontextmanager
import csv
from datetime import datetime, timezone
import io
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import threading
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from .config import Config, POLICIES, presets
from .simulation import Episode
from .controller import choose, evaluate
from .experiments import execute_job, summarize, differences, versions

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CEAP_DATA_DIR", str(ROOT/"data")))
DATA.mkdir(parents=True, exist_ok=True)
EPISODES = {}
PROCESSES = {}
GUARD = threading.RLock()


def db():
    con = sqlite3.connect(DATA/"lab.sqlite3", timeout=30)
    con.row_factory = sqlite3.Row
    return con


def initialize():
    with db() as con:
        con.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, kind TEXT, status TEXT, created TEXT, config TEXT)")
        con.execute("UPDATE runs SET status='interrupted' WHERE status IN ('running','queued','ready')")


def register(rid, kind, status, config):
    folder = DATA/rid; folder.mkdir(exist_ok=True)
    with db() as con:
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (rid, kind, status, datetime.now(timezone.utc).isoformat(), json.dumps(config)))
    return folder


def status_update(rid, status):
    with db() as con: con.execute("UPDATE runs SET status=? WHERE id=?", (status, rid))


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value)); temp.replace(path)


def save_episode(rid, item):
    ep = item["ep"]
    atomic_json(DATA/rid/"episode.json", {**ep.public(diagnostics=True), "frames": ep.frames,
                                         "predictions": ep.predictions, "versions": versions()})
    status_update(rid, "running" if item["running"] else ep.status)


def episode_item(rid):
    if rid not in EPISODES:
        if (DATA/rid/"episode.json").exists():
            raise HTTPException(409, "This saved episode is available for replay. Reproduce it to create an active copy.")
        raise HTTPException(404, "Episode not found")
    return EPISODES[rid]


def public_episode(rid, item, diagnostics=False):
    return {"id": rid, **item["ep"].public(diagnostics), "running": item["running"], "error": item.get("error")}


def auto_episode(rid):
    item = EPISODES[rid]
    try:
        while item["running"]:
            with item["lock"]:
                ep = item["ep"]
                if ep.status in ("completed", "failed", "cancelled"):
                    item["running"] = False; save_episode(rid, item); break
                # Hold the episode lock during a complete operation. Pause is a flag
                # checked again after prediction and takes effect before execution.
                action, scores = choose(ep)
                if not item["running"]:
                    save_episode(rid, item); break
                if action is None: ep.fail("No feasible action remains.")
                else: ep.step(action["id"])
                item["scores"] = None
                save_episode(rid, item)
            time.sleep(.04)
    except Exception as exc:
        item["error"] = f"{type(exc).__name__}: {exc}"
        item["running"] = False
        with item["lock"]:
            item["ep"].fail(item["error"]); save_episode(rid, item)


@asynccontextmanager
async def lifespan(app):
    initialize()
    yield
    for item in EPISODES.values(): item["running"] = False
    for rid, process in PROCESSES.items():
        if process.is_alive():
            (DATA/rid/"cancel").touch()


app = FastAPI(title="CEAP Lab", version="0.1.0", lifespan=lifespan)


@app.get("/api/health")
def health(): return {"status": "ok", "versions": versions(), "policies": POLICIES}


@app.get("/api/presets")
def get_presets(): return {"presets": presets(), "policies": POLICIES}


@app.post("/api/episodes")
def create_episode(config: Config):
    ep = Episode(config); ep.maybe_finish(); ep.snapshot()
    rid = uuid4().hex
    register(rid, "episode", ep.status, config.model_dump())
    item = {"ep": ep, "lock": threading.RLock(), "running": False, "scores": None}
    with GUARD: EPISODES[rid] = item
    save_episode(rid, item)
    return public_episode(rid, item)


@app.get("/api/episodes/{rid}")
def get_episode(rid: str, diagnostics: bool = False):
    item = episode_item(rid)
    with item["lock"]: return public_episode(rid, item, diagnostics)


@app.get("/api/episodes/{rid}/actions")
def get_actions(rid: str, predictions: bool = True):
    item = episode_item(rid)
    with item["lock"]:
        ep = item["ep"]
        if predictions and not item["running"] and ep.legal_actions():
            if item["scores"] is None: item["scores"] = evaluate(ep)
            return {"actions": item["scores"], "model_based": True}
        return {"actions": ep.legal_actions(), "model_based": False}


class StepRequest(BaseModel):
    action_id: str | None = None


@app.post("/api/episodes/{rid}/step")
def step_episode(rid: str, request: StepRequest):
    item = episode_item(rid)
    with item["lock"]:
        if item["running"]: raise HTTPException(409, "Pause automatic execution before stepping manually.")
        ep = item["ep"]
        if ep.status in ("failed", "completed", "cancelled"): raise HTTPException(409, "This episode has finished.")
        action_id = request.action_id
        if action_id is None:
            action, _ = choose(ep)
            if action is None: ep.fail("No feasible action remains.")
            else: action_id = action["id"]
        try:
            if action_id: ep.step(action_id)
        except ValueError as exc: raise HTTPException(409, str(exc))
        item["scores"] = None; save_episode(rid, item)
        return public_episode(rid, item)


@app.post("/api/episodes/{rid}/run")
def start_episode(rid: str):
    item = episode_item(rid)
    with item["lock"]:
        if item["ep"].status in ("completed", "failed", "cancelled"): raise HTTPException(409, "Create a new episode to run again.")
        if not item["running"]:
            item["running"] = True
            threading.Thread(target=auto_episode, args=(rid,), daemon=True).start()
    return {"running": True}


@app.post("/api/episodes/{rid}/pause")
def pause_episode(rid: str):
    item = episode_item(rid); item["running"] = False
    return {"running": False, "message": "Pause requested; any in-progress operation finishes atomically."}


@app.get("/api/episodes/{rid}/events")
async def episode_events(rid: str):
    item = episode_item(rid)
    async def generate():
        last = None
        while True:
            # Snapshot on a worker thread so simulation does not block the event loop.
            state = await asyncio.to_thread(get_episode, rid, False)
            token = (len(state["events"]), state["running"], state["observation"]["status"])
            if token != last:
                yield "data: " + json.dumps(state) + "\n\n"; last = token
            else: yield ": heartbeat\n\n"
            if state["observation"]["status"] in ("completed", "failed", "cancelled"): break
            await asyncio.sleep(.4)
    return StreamingResponse(generate(), media_type="text/event-stream")


class JobRequest(BaseModel):
    config: Config
    policies: list[str] = Field(default_factory=lambda: ["ceap", "immediate", "fixed", "threshold", "weakest"])
    episodes: int = Field(default=200, ge=1, le=10000)
    experiment: str = "E3"
    sweep: dict | None = None

    @model_validator(mode="after")
    def check(self):
        if not self.policies or any(p not in POLICIES for p in self.policies): raise ValueError("Select valid policies")
        if self.experiment not in [f"E{i}" for i in range(8)]: raise ValueError("Unknown experiment")
        if self.sweep and len(self.sweep.get("values", [])) > 20: raise ValueError("At most 20 sweep values")
        return self


@app.post("/api/jobs", status_code=202)
def create_job(request: JobRequest):
    # Validate sweep values before creating a worker.
    from .experiments import conditions
    try: conditions(request.config, request.experiment, request.sweep)
    except (ValueError, TypeError) as exc: raise HTTPException(422, str(exc))
    if sum(p.is_alive() for p in PROCESSES.values()) >= 2:
        raise HTTPException(409, "Two experiment workers are already active. Cancel or finish one first.")
    rid = uuid4().hex; payload = request.model_dump()
    folder = register(rid, "experiment", "queued", payload)
    atomic_json(folder/"progress.json", {"status": "queued", "completed": 0, "total": 0, "summaries": []})
    process = multiprocessing.get_context("spawn").Process(target=execute_job, args=(str(folder), payload), daemon=True)
    process.start(); PROCESSES[rid] = process
    return {"id": rid, "status": "queued"}


def valid_folder(rid):
    if not rid.isalnum() or len(rid) != 32 or not (DATA/rid).is_dir(): raise HTTPException(404, "Run not found")
    return DATA/rid


@app.get("/api/jobs/{rid}")
def get_job(rid: str):
    folder = valid_folder(rid)
    if not (folder/"progress.json").exists(): raise HTTPException(404, "Experiment not found")
    progress = json.loads((folder/"progress.json").read_text())
    with db() as con:
        row = con.execute("SELECT status, config FROM runs WHERE id=?", (rid,)).fetchone()
    process = PROCESSES.get(rid)
    if row["status"] == "interrupted" and progress["status"] in ("running", "queued"):
        progress["status"] = "interrupted"
    elif process and not process.is_alive() and progress["status"] in ("running", "queued"):
        progress["status"] = "failed"; progress["error"] = "Worker stopped before completing its checkpoint."
    if progress["summaries"] is None:
        records = read_trials(folder); progress["summaries"] = summarize(records)
    progress["differences"] = differences(read_trials(folder))
    progress.update({"id": rid, "request": json.loads(row["config"])})
    status_update(rid, progress["status"])
    return progress


@app.post("/api/jobs/{rid}/cancel")
def cancel_job(rid: str):
    (valid_folder(rid)/"cancel").touch()
    return {"status": "cancellation_requested", "message": "Completed trials are retained; cancellation is checked between trials."}


@app.get("/api/jobs/{rid}/analysis")
def job_analysis(rid: str):
    folder=valid_folder(rid)
    trials=read_trials(folder)
    outcomes={(r["condition"],r["policy"],r["episode_index"]):r for r in trials}
    points=[]
    path=folder/"events.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try: row=json.loads(line)
            except json.JSONDecodeError: continue
            outcome=outcomes.get((row["condition"],row["policy"],row["episode_index"]))
            if outcome is None: continue
            for prediction in row.get("predictions",[]):
                chosen=next((a for a in prediction["candidates"] if a["id"]==prediction["chosen"]),None)
                if chosen:
                    points.append({"predicted":chosen["predicted_qualified_probability"],"observed":int(outcome["qualified"]),"policy":row["policy"],"episode_index":row["episode_index"]})
    bins=[]
    for i in range(10):
        sample=[p for p in points if min(9,int(p["predicted"]*10))==i]
        if sample:
            bins.append({"predicted":sum(p["predicted"] for p in sample)/len(sample),"observed":sum(p["observed"] for p in sample)/len(sample),"count":len(sample)})
    return {"calibration":bins,"choices":len(points),"note":"Descriptive chosen-action predictions vs eventual outcomes. Multiple choices within an episode are correlated; forecasts use a threshold continuation, while executed episodes may make later CEAP decisions. No independence-based intervals or causal interpretation are asserted."}


@app.get("/api/jobs/{rid}/events")
async def job_events(rid: str):
    valid_folder(rid)
    async def generate():
        while True:
            state = await asyncio.to_thread(get_job, rid)
            yield "data: " + json.dumps(state) + "\n\n"
            if state["status"] in ("completed", "failed", "cancelled", "interrupted"): break
            await asyncio.sleep(1)
    return StreamingResponse(generate(), media_type="text/event-stream")


def read_trials(folder):
    path = folder/"trials.jsonl"
    if not path.exists(): return []
    records = []
    for line in path.read_text().splitlines():
        try: records.append(json.loads(line))
        except json.JSONDecodeError: pass  # A worker may be writing its next line.
    return records


@app.get("/api/runs")
def list_runs():
    with db() as con: rows = con.execute("SELECT * FROM runs ORDER BY created DESC LIMIT 200").fetchall()
    return {"runs": [{**dict(r), "config": json.loads(r["config"])} for r in rows]}


@app.get("/api/runs/{rid}")
def run_detail(rid: str):
    folder = valid_folder(rid)
    if (folder/"episode.json").exists(): return {"id": rid, **json.loads((folder/"episode.json").read_text())}
    return get_job(rid)


@app.get("/api/runs/{rid}/export")
def export_run(rid: str, format: str = "json"):
    folder = valid_folder(rid)
    episode = (folder/"episode.json").exists()
    if format in ("events", "predictions"):
        if episode:
            data = json.loads((folder/"episode.json").read_text())[format]
            content = "\n".join(json.dumps(e) for e in data)
            return Response(content, media_type="application/x-ndjson", headers={"Content-Disposition": f'attachment; filename="ceap-{rid[:8]}-{format}.jsonl"'})
        if format == "events": return FileResponse(folder/"events.jsonl", filename=f"ceap-{rid[:8]}-events.jsonl")
        raise HTTPException(400, "Experiment predictions are included in the events export.")
    if format == "config":
        with db() as con: row = con.execute("SELECT config FROM runs WHERE id=?", (rid,)).fetchone()
        return Response(row["config"], media_type="application/json", headers={"Content-Disposition": f'attachment; filename="ceap-{rid[:8]}-config.json"'})
    if format == "json":
        data = run_detail(rid)
        return Response(json.dumps(data, indent=2), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="ceap-{rid[:8]}.json"'})
    if format == "csv":
        rows = [json.loads((folder/"episode.json").read_text())["result"]] if episode else read_trials(folder)
        if not rows and (folder/"progress.json").exists():
            rows=json.loads((folder/"progress.json").read_text()).get("validation",[])
        rows = [r for r in rows if r]
        out = io.StringIO()
        if rows:
            fields = sorted(set().union(*(r.keys() for r in rows)))
            writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
            for row in rows: writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
        return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="ceap-{rid[:8]}-trials.csv"'})
    raise HTTPException(400, "Unknown export format")


@app.get("/api/jobs/{rid}/chart")
def export_chart(rid: str, metric: str = "qualified_probability", format: str = "svg"):
    if metric not in ("qualified_probability", "success_probability", "fidelity", "latency_ms", "qualified_throughput", "attempts_per_request") or format not in ("svg", "png"):
        raise HTTPException(400, "Unsupported chart metric or format")
    import matplotlib
    os.environ.setdefault("MPLCONFIGDIR", str(DATA/"matplotlib"))
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summaries = get_job(rid)["summaries"]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    labels = [f"{r['label']}\n{r['condition']}" for r in summaries]
    vals = [r.get(metric) or 0 for r in summaries]
    bars = ax.bar(range(len(vals)), vals, color=["#087f8c" if r["policy"] == "ceap" else "#5383a5" for r in summaries])
    ax.set_xticks(range(len(vals)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " ")); ax.set_title("CEAP Lab · measured simulation results")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    ci_key = {"qualified_probability": "qualified_ci", "success_probability": "success_ci", "fidelity": "fidelity_ci", "latency_ms": "latency_ci", "qualified_throughput": "throughput_ci"}.get(metric)
    if ci_key:
        for i, row in enumerate(summaries):
            ci = row.get(ci_key)
            if ci and ci[0] is not None and row.get(metric) is not None:
                ax.errorbar(i, vals[i], yerr=[[max(0, vals[i]-ci[0])], [max(0, ci[1]-vals[i])]], color="#263746", capsize=3, fmt="none")
    data = io.BytesIO(); fig.savefig(data, format=format, dpi=160); plt.close(fig)
    return Response(data.getvalue(), media_type="image/svg+xml" if format == "svg" else "image/png", headers={"Content-Disposition": f'attachment; filename="ceap-{metric}.{format}"'})


app.mount("/", StaticFiles(directory=ROOT/"web", html=True), name="web")
