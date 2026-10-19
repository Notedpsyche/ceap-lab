"""Information-limited policies and seeded, failure-aware counterfactual rollouts."""
import hashlib
import time
import numpy as np
from . import quantum as q


def heuristic(ep, policy="threshold"):
    actions = ep.legal_actions()
    if not actions:
        return None
    swaps = [a for a in actions if a["kind"] == "swap"]
    purifies = [a for a in actions if a["kind"] == "purify"]
    generates = [a for a in actions if a["kind"] == "regenerate"]
    wait = next((a for a in actions if a["kind"] == "wait"), None)
    pairs = list(ep.pairs.values())
    if policy == "immediate":
        if swaps:
            return swaps[0]
        missing = [a for a in generates if not any(p.left <= a["edge"] < p.right for p in pairs)
                   and not any(e["edge"] == a["edge"] for e in ep.pending)]
        return (missing or ([wait] if wait else []) or generates or actions)[0]

    def needs(p):
        if p.right - p.left != 1:
            return False
        if policy == "fixed":
            return p.rounds == 0
        return q.fidelity(p.belief) < ep.config.local_threshold

    eligible = [a for a in purifies if any(needs(ep.pairs[pid]) for pid in a["pairs"])]
    if eligible:
        return min(eligible, key=lambda a: min(q.fidelity(ep.pairs[x].belief) for x in a["pairs"]))
    # Create missing elementary resources before preparing auxiliary copies.
    missing = [a for a in generates if not any(p.left <= a["edge"] < p.right for p in pairs)
               and not any(e["edge"] == a["edge"] for e in ep.pending)]
    if missing:
        return min(missing, key=lambda a: ep.estimates[a["edge"]]) if policy == "weakest" else missing[0]
    want = sorted([p for p in pairs if needs(p)], key=lambda p: q.fidelity(p.belief))
    for p in want:
        target = next((a for a in generates if a["edge"] == p.left and not any(e["edge"] == p.left for e in ep.pending)), None)
        if target:
            return target
    if want and wait:
        return wait
    if swaps:
        return swaps[0]
    return wait or (generates[0] if generates else actions[0])


def run_to_end(ep, policy=None):
    ep.maybe_finish()
    while ep.status not in ("completed", "failed", "cancelled"):
        selected, _ = choose(ep, policy or ep.config.policy)
        if selected is None:
            ep.fail("No feasible continuation remains."); break
        ep.step(selected["id"])
    if ep.result:
        ep.result["policy_runtime_s"] = ep.policy_seconds
    return ep


def evaluate(ep, oracle=False):
    actions = ep.legal_actions()
    if not actions:
        return []
    base = heuristic(ep, "immediate")
    output = []
    start = time.perf_counter()
    # Same model-future seeds across candidates; completely separate from live RNG.
    for action in actions:
        successes = []; latencies = []; attempts = []; fidelities = []
        for sample in range(ep.config.rollouts):
            seed = int.from_bytes(hashlib.blake2b(repr((ep.config.seed, "counterfactual", ep.decisions, sample)).encode(), digest_size=4).digest(), "little")
            clone = ep.clone_belief(seed, oracle=oracle)
            # The unknown input is not an observation. Integrate over the public
            # six-state evaluation ensemble instead of inspecting the live input.
            clone.config.input_state = tuple(q.SIX)[sample % 6]
            clone.step(action["id"])
            while clone.status not in ("completed", "failed", "cancelled"):
                nxt = heuristic(clone, "threshold")
                if nxt is None:
                    clone.fail("No feasible continuation remains."); break
                clone.step(nxt["id"])
            result = clone.result or clone.summary()
            successes.append(float(result["qualified"]))
            latencies.append(clone.t-ep.t)
            attempts.append(clone.attempts-ep.attempts)
            if result["fidelity"] is not None:
                fidelities.append(result["fidelity"])
        prob = float(np.mean(successes)); dt = float(np.mean(latencies)); cost = float(np.mean(attempts))
        utility = prob - ep.config.time_weight*dt/ep.config.deadline_ms - ep.config.resource_weight*cost/ep.config.attempt_budget
        output.append({**action, "predicted_qualified_probability": prob, "predicted_extra_ms": dt,
                       "predicted_attempts": cost, "predicted_fidelity": float(np.mean(fidelities)) if fidelities else None,
                       "utility": utility, "rollouts": ep.config.rollouts, "reference": action["id"] == base["id"]})
    reference = next(a for a in output if a["reference"])
    for a in output:
        a["delta_probability"] = a["predicted_qualified_probability"]-reference["predicted_qualified_probability"]
        a["delta_utility"] = a["utility"]-reference["utility"]
    ep.policy_seconds += time.perf_counter()-start
    return sorted(output, key=lambda a: (-round(a["utility"], 12), a["predicted_extra_ms"], a["predicted_attempts"], a["id"]))


def choose(ep, policy=None):
    policy = policy or ep.config.policy
    if policy in ("ceap", "oracle"):
        scores = evaluate(ep, oracle=policy == "oracle")
        if not scores:
            return None, []
        if ep.record:
            ep.predictions.append({"decision": ep.decisions, "time_ms": ep.t, "chosen": scores[0]["id"], "candidates": scores})
        return scores[0], scores
    if policy == "ablation":
        actions = ep.legal_actions()
        purifies = [a for a in actions if a["kind"] == "purify"]
        # One-step quality only. No downstream rollouts, budget estimates, or truth.
        if purifies:
            scored = []
            for a in purifies:
                p1, p2 = [ep.pairs[p] for p in a["pairs"]]
                total = value = 0.0
                for reported in (0, 3):
                    prob, rho = q.conditional("purify", p1.belief, p2.belief, reported, ep.config.gate_noise, ep.config.readout_error)
                    value += prob*q.fidelity(rho); total += prob
                scored.append((value/max(total, 1e-15)-q.fidelity(p1.belief), a))
            if max(scored, key=lambda s: s[0])[0] > 0:
                return max(scored, key=lambda s: s[0])[1], []
        return heuristic(ep, "weakest"), []
    return heuristic(ep, policy), []
