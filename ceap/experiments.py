import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np
from .config import Config, POLICIES
from .simulation import Episode
from .controller import run_to_end


def versions():
    out = {"python": platform.python_version(), "ceap_lab": "0.1.0"}
    digest=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode());digest.update(path.read_bytes())
    out["simulator_source_sha256"]=digest.hexdigest()
    for name in ("numpy", "fastapi", "qiskit", "qiskit-aer"):
        try: out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return out


def wilson(success, total):
    if not total:
        return [None, None]
    z = 1.95996398454; p = success/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    delta = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [max(0, center-delta), min(1, center+delta)]


def bootstrap(values, seed=123, count=400, statistic="mean"):
    values = np.asarray(values, float)
    if not len(values): return [None, None]
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(count, len(values)))]
    estimates = np.mean(samples, axis=1) if statistic == "mean" else np.median(samples, axis=1)
    return [float(v) for v in np.quantile(estimates, [.025, .975])]


def summarize(records):
    groups = {}
    for row in records:
        key = (row.get("condition", "baseline"), row["policy"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (condition, policy), rows in groups.items():
        n = len(rows); completed = [r for r in rows if r["success"]]
        qualified = sum(r["qualified"] for r in rows)
        f = [r["fidelity"] for r in completed]; latency = [r["latency_ms"] for r in completed]
        elapsed = sum(r["latency_ms"] for r in rows)
        rng = np.random.default_rng(2026)
        throughput = []
        for _ in range(400):
            sample = [rows[i] for i in rng.integers(0, n, n)]
            duration = sum(r["latency_ms"] for r in sample)
            throughput.append(1000*sum(r["qualified"] for r in sample)/duration if duration else 0)
        action_counts = {}
        for row in rows:
            for a, c in row["action_counts"].items(): action_counts[a] = action_counts.get(a, 0)+c
        summaries.append({"condition": condition, "policy": policy, "label": POLICIES[policy], "trials": n,
                          "success_probability": len(completed)/n, "success_ci": wilson(len(completed), n),
                          "qualified_probability": qualified/n, "qualified_ci": wilson(qualified, n),
                          "fidelity": float(np.mean(f)) if f else None, "fidelity_ci": bootstrap(f),
                          "worst_input_fidelity": min((r["worst_input_fidelity"] for r in completed), default=None),
                          "latency_ms": float(np.mean(latency)) if latency else None, "latency_ci": bootstrap(latency),
                          "median_latency_ms": float(np.median(latency)) if latency else None,
                          "p95_latency_ms": float(np.quantile(latency, .95)) if latency else None,
                          "qualified_throughput": qualified*1000/elapsed if elapsed else 0,
                          "throughput_ci": [float(x) for x in np.quantile(throughput, [.025, .975])],
                          "attempts_per_request": sum(r["attempts"] for r in rows)/n,
                          "pairs_per_request": sum(r["generated_pairs"] for r in rows)/n,
                          "pairs_per_qualified": sum(r["generated_pairs"] for r in rows)/qualified if qualified else None,
                          "discarded_pairs_per_request": sum(r["discarded_pairs"] for r in rows)/n,
                          "attempts_per_qualified": sum(r["attempts"] for r in rows)/qualified if qualified else None,
                          "purification_failures": sum(r["purification_failures"] for r in rows), "action_counts": action_counts,
                          "policy_runtime_s": sum(r["policy_runtime_s"] for r in rows), "sparse": len(completed) < 30})
    return summaries


def differences(records):
    out = []
    for condition in sorted(set(r.get("condition", "baseline") for r in records)):
        by_policy = {}
        for r in records:
            if r.get("condition", "baseline") == condition:
                by_policy.setdefault(r["policy"], {})[r["episode_index"]] = r
        if "ceap" not in by_policy: continue
        for policy, rows in by_policy.items():
            if policy == "ceap": continue
            shared = sorted(set(rows) & set(by_policy["ceap"]))
            delta = [int(by_policy["ceap"][i]["qualified"])-int(rows[i]["qualified"]) for i in shared]
            if delta:
                out.append({"condition": condition, "against": policy, "metric": "qualified_probability",
                            "difference": float(np.mean(delta)), "ci": bootstrap(delta), "matched_episodes": len(delta)})
    return out


def conditions(config, experiment, sweep=None):
    out = []
    def add(label, c): out.append((label, Config.model_validate(c.model_dump())))
    if experiment == "E4":
        for weak in (.75, .85, .95):
            c = config.model_copy(deep=True); c.links[len(c.links)//2].fidelity = weak; add(f"middle F={weak}", c)
    elif experiment == "E5":
        for budget in (12, 30, 60):
            for deadline in (55, 110, 220):
                c = config.model_copy(deep=True); c.attempt_budget = budget; c.deadline_ms = deadline
                add(f"{budget} attempts / {deadline} ms", c)
    elif experiment == "E7":
        for sigma in (0, .02, .05):
            c = config.model_copy(deep=True); c.calibration_sigma = sigma; add(f"calibration σ={sigma}", c)
    elif sweep and sweep.get("values"):
        field = sweep["field"]
        for value in sweep["values"]:
            c = config.model_copy(deep=True)
            if field == "link_fidelity":
                for link in c.links: link.fidelity = value
            elif field in ("attempt_budget", "deadline_ms", "memory_ms", "gate_noise", "readout_error"):
                setattr(c, field, value)
            else: raise ValueError("Unknown sweep field")
            add(f"{field}={value}", c)
    else: add("baseline", config)
    return out


def experiment_policies(experiment, selected):
    return {"E1": ["immediate", "fixed"], "E2": ["threshold", "weakest"],
            "E6": ["ceap", "ablation"], "E7": ["ceap", "oracle"]}.get(experiment, selected)


def execute_job(folder, payload):
    """Separate-process entry point. Files are checkpointed after each trial."""
    folder = Path(folder)
    config = Config.model_validate(payload["config"])
    policies = experiment_policies(payload.get("experiment", "E3"), payload["policies"])
    grid = conditions(config, payload.get("experiment", "E3"), payload.get("sweep"))
    total = len(grid)*len(policies)*payload["episodes"]
    records = []; last_error = None
    def progress(status):
        state = {"status": status, "completed": len(records), "total": total,
                 "summaries": summarize(records) if status != "running" or len(records) % 10 == 0 else None,
                 "differences": differences(records) if status != "running" else [], "error": last_error}
        temp = folder/"progress.tmp"; temp.write_text(json.dumps(state)); temp.replace(folder/"progress.json")
    try:
        (folder/"manifest.json").write_text(json.dumps({**payload, "versions": versions(), "random_streams": "BLAKE2b(seed, physical edge, attempt / operation ancestry); counterfactual streams separate"}, indent=2))
        if payload.get("experiment") == "E0":
            from .validation import physics_checks
            checks = physics_checks()
            (folder/"progress.json").write_text(json.dumps({"status":"completed" if all(c["passed"] for c in checks) else "failed", "completed":len(checks), "total":len(checks), "summaries":[], "differences":[], "validation":checks, "error":None}))
            return
        progress("running")
        for condition, base in grid:
            for index in range(payload["episodes"]):
                for policy in policies:
                    if (folder/"cancel").exists(): progress("cancelled"); return
                    c = base.model_copy(deep=True); c.seed = (base.seed+index*7919) % (2**31-1)
                    c.input_state = list(q_six())[index % 6]; c.policy = policy
                    ep = run_to_end(Episode(c))
                    row = {**ep.result, "condition": condition, "episode_index": index, "configuration": c.model_dump()}
                    records.append(row)
                    with (folder/"trials.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
                    with (folder/"events.jsonl").open("a") as f:
                        f.write(json.dumps({"condition": condition, "policy": policy, "episode_index": index, "events": ep.events, "predictions": ep.predictions})+"\n")
                    progress("running")
        progress("completed")
    except Exception as exc:
        import traceback
        last_error = f"{type(exc).__name__}: {exc}"
        (folder/"error.log").write_text(traceback.format_exc())
        progress("failed")


def q_six():
    return ("0", "1", "+", "-", "+i", "-i")
