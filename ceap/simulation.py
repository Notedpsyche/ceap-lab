"""Event-driven fixed-path repeater environment, independent of the web layer."""
import copy
import hashlib
import time
from dataclasses import dataclass, field
import numpy as np
from .config import Config
from . import quantum as q


def stream(seed, *key):
    token = repr((seed, key)).encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little"))


@dataclass
class Pair:
    id: str
    left: int
    right: int
    rho: np.ndarray
    belief: np.ndarray
    created: float
    updated: float
    ancestry: tuple
    rounds: int = 0
    operations: list = field(default_factory=list)


class Episode:
    def __init__(self, config: Config, record=True):
        self.config = config.model_copy(deep=True)
        self.t = 0.0
        self.pairs = {}
        self.pending = []
        self.attempts = 0
        self.edge_attempts = [0] * (config.nodes - 1)
        self.generated = self.discarded = self.purifications = self.purification_failures = self.swaps = 0
        self.peak_memory = 0
        self.status = "ready"
        self.reason = None
        self.result = None
        self.events = []
        self.frames = []
        self.record = record
        self.serial = 0
        self.decisions = 0
        self.action_counts = {}
        self.predictions = []
        self.policy_seconds = 0.0
        self.estimates = [float(np.clip(link.fidelity + stream(config.seed, "calibration", i).normal(0, config.calibration_sigma), .25, 1)) for i, link in enumerate(config.links)]
        self.log("initialized", "Request initialized; elementary pairs must be generated.")

    def next_id(self):
        self.serial += 1
        return f"p{self.serial}"

    def log(self, kind, message, **data):
        if self.record:
            self.events.append({"index": len(self.events), "time_ms": round(self.t, 6), "kind": kind, "message": message, **data})

    def usage(self):
        use = [0] * self.config.nodes
        for p in self.pairs.values():
            use[p.left] += 1; use[p.right] += 1
        for pending in self.pending:
            use[pending["edge"]] += 1; use[pending["edge"] + 1] += 1
        return use

    def add_pair(self, left, right, rho, belief, ancestry, rounds=0, created=None, operations=None):
        p = Pair(self.next_id(), left, right, rho, belief, self.t if created is None else created, self.t, tuple(ancestry), rounds, operations or [])
        self.pairs[p.id] = p
        self.peak_memory = max(self.peak_memory, max(self.usage()))
        return p

    def legal_actions(self):
        if self.status in ("completed", "failed", "cancelled"):
            return []
        actions = []
        ps = sorted(self.pairs.values(), key=lambda p: (p.left, p.right, p.id))
        for a in ps:
            for b in ps:
                if a.right == b.left and a.left < a.right < b.right:
                    actions.append({"id": f"swap:{a.id}:{b.id}", "kind": "swap", "pairs": [a.id, b.id], "junction": a.right,
                                    "label": f"Swap at {self.node_name(a.right)}", "detail": f"Join {a.id} and {b.id}; consume both pairs."})
            for b in ps:
                if a.id < b.id and (a.left, a.right) == (b.left, b.right):
                    actions.append({"id": f"purify:{a.id}:{b.id}", "kind": "purify", "pairs": [a.id, b.id], "junction": a.left,
                                    "label": f"Purify {self.node_name(a.left)}–{self.node_name(a.right)}", "detail": "Two copies available; rejection consumes both."})
        use = self.usage()
        if self.attempts < self.config.attempt_budget:
            for i, link in enumerate(self.config.links):
                due = self.t + link.attempt_ms + link.classical_ms
                if due <= self.config.deadline_ms and sum(e["edge"] == i for e in self.pending) < 2:
                    if use[i] < self.config.memories and use[i+1] < self.config.memories:
                        actions.append({"id": f"regenerate:{i}", "kind": "regenerate", "edge": i, "label": f"Generate on link {i+1}", "detail": "Reserve two endpoint memories; spend one attempt."})
                    else:
                        for p in ps:
                            if (p.left,p.right) == (i,i+1) and use[i]-1 < self.config.memories and use[i+1]-1 < self.config.memories:
                                actions.append({"id":f"regenerate:{i}:replace:{p.id}","kind":"regenerate","edge":i,"discard_pair":p.id,
                                                "label":f"Replace {p.id} on link {i+1}","detail":"Explicitly discard this stored pair, then spend one new attempt. Generation can fail."})
        if self.pending:
            actions.append({"id": "wait", "kind": "wait", "label": "Wait for heralding", "detail": "Advance to the next pending generation result."})
        return actions

    def node_name(self, i):
        return "Alice" if i == 0 else "Bob" if i == self.config.nodes - 1 else f"R{i}"

    def advance(self, target):
        target = min(target, self.config.deadline_ms)
        while self.pending and min(e["due"] for e in self.pending) <= target + 1e-10:
            next_t = min(e["due"] for e in self.pending)
            self.age_to(next_t)
            due = [e for e in self.pending if e["due"] <= self.t + 1e-9]
            self.pending = [e for e in self.pending if e not in due]
            for e in due:
                i = e["edge"]; link = self.config.links[i]
                success = stream(self.config.seed, "generation", i, e["attempt"]).random() < link.success_probability
                if success:
                    self.generated += 1
                    # Pair exists at transmission completion and ages through heralding.
                    rho = q.age(q.link_state(link.fidelity, link.model), link.classical_ms, self.config.memory_ms)
                    belief = q.age(q.link_state(self.estimates[i], link.model), link.classical_ms, self.config.memory_ms)
                    p = self.add_pair(i, i+1, rho, belief, ((i, e["attempt"]),), created=self.t-link.classical_ms)
                    self.log("generated", f"Link {i+1} heralded a shared pair ({p.id}).", pair_id=p.id, edge=i, attempt=e["attempt"])
                else:
                    self.log("generation_failed", f"Link {i+1} attempt {e['attempt']} failed; reserved memories released.", edge=i, attempt=e["attempt"])
        self.age_to(target)

    def age_to(self, target):
        for p in self.pairs.values():
            dt = max(0, target - p.updated)
            p.rho = q.age(p.rho, dt, self.config.memory_ms)
            p.belief = q.age(p.belief, dt, self.config.memory_ms)
            p.updated = target
        self.t = target

    def fail(self, reason):
        self.status = "failed"; self.reason = reason
        self.log("failed", reason)
        self.result = self.summary()

    def maybe_finish(self):
        if self.status in ("failed", "completed", "cancelled"):
            return
        endpoint = [p for p in self.pairs.values() if p.left == 0 and p.right == self.config.nodes-1]
        if endpoint:
            p = max(endpoint, key=lambda x: q.fidelity(x.belief))
            duration = 2*self.config.gate_ms + self.config.measurement_ms + sum(l.classical_ms for l in self.config.links)
            if self.t + duration > self.config.deadline_ms + 1e-9:
                self.advance(self.config.deadline_ms); self.fail("Deadline expired before final teleportation corrections."); return
            self.advance(self.t + duration)
            psi = q.input_vector(self.config.input_state, self.config.theta, self.config.phi)
            f, out = q.teleport_fidelity(p.rho, psi, self.config.readout_error)
            six = {label: q.teleport_fidelity(p.rho, state, self.config.readout_error)[0] for label, state in q.SIX.items()}
            self.status = "completed"
            self.result = self.summary()
            self.result.update({"fidelity": f, "qualified": f + 1e-12 >= self.config.target_fidelity,
                                "average_fidelity": float(np.mean(list(six.values()))), "worst_input_fidelity": min(six.values()),
                                "six_fidelities": six, "output_bloch": q.bloch(out), "input_bloch": q.bloch(np.outer(psi, psi.conj())),
                                "output_density_matrix": q.serialize_matrix(out), "resource_bell_fidelity": q.fidelity(p.rho)})
            del self.pairs[p.id]
            self.log("teleported", f"Teleportation complete: F = {f:.5f}; {'qualified' if self.result['qualified'] else 'below target'}.", fidelity=f, qualified=self.result["qualified"])
        elif self.t >= self.config.deadline_ms - 1e-9:
            self.fail("Request deadline reached.")
        elif not self.legal_actions():
            self.fail("No feasible action remains: resource or memory exhaustion.")

    def step(self, action_id):
        actions = {a["id"]: a for a in self.legal_actions()}
        if action_id not in actions:
            raise ValueError("Action is unavailable; refresh the episode before trying again.")
        a = actions[action_id]
        self.status = "ready"; self.decisions += 1
        self.action_counts[a["kind"]] = self.action_counts.get(a["kind"], 0) + 1
        if a["kind"] == "regenerate":
            i = a["edge"]; link = self.config.links[i]
            if a.get("discard_pair"):
                del self.pairs[a["discard_pair"]]; self.discarded += 1
                self.log("discarded",f"Explicitly discarded {a['discard_pair']} before regeneration.",pair_id=a["discard_pair"])
            self.attempts += 1; self.edge_attempts[i] += 1
            self.pending.append({"edge": i, "attempt": self.edge_attempts[i], "due": self.t+link.attempt_ms+link.classical_ms})
            self.peak_memory = max(self.peak_memory, max(self.usage()))
            self.log("attempt", f"Link {i+1}: attempt {self.edge_attempts[i]} started.", edge=i)
        elif a["kind"] == "wait":
            self.advance(min(e["due"] for e in self.pending))
        else:
            left, right = [self.pairs[pid] for pid in a["pairs"]]
            # Swap bits travel from the junction to the new right endpoint;
            # purification parity compares the two common endpoints.
            first_hop = left.right if a["kind"] == "swap" else left.left
            last_hop = right.right if a["kind"] == "swap" else left.right
            classical = sum(l.classical_ms for l in self.config.links[first_hop:last_hop])
            dt = 2*self.config.gate_ms + self.config.measurement_ms + classical
            if self.t + dt > self.config.deadline_ms + 1e-9:
                for p in (left, right): self.pairs.pop(p.id)
                self.discarded += 2
                self.advance(self.config.deadline_ms); self.fail("Deadline expired during a quantum operation."); self.snapshot(); return
            self.advance(self.t + dt)
            rng = stream(self.config.seed, a["kind"], left.ancestry, right.ancestry, left.rounds, right.rounds)
            truth, belief, actual, reported = q.operate(a["kind"], left.rho, right.rho, left.belief, right.belief, self.config.gate_noise, self.config.readout_error, rng)
            for p in (left, right): self.pairs.pop(p.id)
            ancestry = left.ancestry + right.ancestry
            operations = left.operations + right.operations + [a["kind"]]
            if a["kind"] == "purify":
                self.purifications += 1
                if truth is None:
                    self.purification_failures += 1; self.discarded += 2
                    self.log("purification_failed", "Purification rejected: both input pairs consumed.", recorded_bits=f"{reported:02b}", diagnostic_true_bits=f"{actual:02b}")
                else:
                    self.discarded += 1
                    p = self.add_pair(left.left, left.right, truth, belief, ancestry, max(left.rounds, right.rounds)+1, operations=operations)
                    self.log("purified", f"Purification accepted; retained {p.id} after an ideal Werner twirl.", pair_id=p.id, recorded_bits=f"{reported:02b}", diagnostic_true_bits=f"{actual:02b}")
            else:
                self.swaps += 1
                p = self.add_pair(left.left, right.right, truth, belief, ancestry, operations=operations)
                self.log("swapped", f"Swapped at {self.node_name(left.right)}; corrected segment {p.id} is ready.", pair_id=p.id, junction=left.right, recorded_bits=f"{reported:02b}", diagnostic_true_bits=f"{actual:02b}")
        if self.decisions >= 512:
            self.fail("Decision safety limit reached.")
        self.maybe_finish()
        self.snapshot()

    def summary(self):
        return {"success": self.status == "completed", "qualified": False, "fidelity": None,
                "latency_ms": self.t, "attempts": self.attempts, "generated_pairs": self.generated,
                "discarded_pairs": self.discarded, "purifications": self.purifications,
                "purification_failures": self.purification_failures, "swaps": self.swaps,
                "peak_memory": self.peak_memory, "reason": self.reason, "seed": self.config.seed,
                "input_state": self.config.input_state, "policy": self.config.policy,
                "action_counts": self.action_counts.copy(), "policy_runtime_s": self.policy_seconds}

    def observation(self):
        return {"time_ms": self.t, "status": self.status, "attempts": self.attempts,
                "remaining_attempts": self.config.attempt_budget-self.attempts,
                "remaining_ms": max(0, self.config.deadline_ms-self.t), "memory_use": self.usage(),
                "pairs": [{"id": p.id, "left": p.left, "right": p.right, "age_ms": self.t-p.created,
                           "estimated_fidelity": q.fidelity(p.belief), "rounds": p.rounds} for p in self.pairs.values()],
                "pending": copy.deepcopy(self.pending), "calibrated_link_fidelities": self.estimates.copy()}

    def public(self, diagnostics=False):
        out = {"config": self.config.model_dump(), "observation": self.observation(),
               "events": [{k: v for k, v in e.items() if not k.startswith("diagnostic_")} for e in self.events],
               "result": self.result, "actions": self.legal_actions(), "frame_count": len(self.frames)}
        if diagnostics:
            out["diagnostics"] = {"label": "Simulator truth — unavailable to CEAP", "pairs": [
                {"id": p.id, "true_fidelity": q.fidelity(p.rho), "density_matrix": q.serialize_matrix(p.rho)} for p in self.pairs.values()]}
        return out

    def snapshot(self):
        if self.record:
            self.frames.append({"observation": self.observation(), "event_count": len(self.events), "result": copy.deepcopy(self.result)})

    def clone_belief(self, seed, oracle=False):
        # Clone only public sufficient state, never historical frames, raw true
        # measurements, or the live random state. This is also much cheaper.
        other = copy.copy(self)
        other.config = self.config.model_copy(deep=True)
        other.pairs = {pid: copy.copy(p) for pid, p in self.pairs.items()}
        other.pending = copy.deepcopy(self.pending)
        other.edge_attempts = self.edge_attempts.copy()
        other.action_counts = self.action_counts.copy()
        other.record = False; other.events = []; other.frames = []; other.predictions = []
        other.config.seed = seed
        for i, link in enumerate(other.config.links):
            link.fidelity = self.config.links[i].fidelity if oracle else self.estimates[i]
        for p in other.pairs.values():
            p.rho = (p.rho if oracle else p.belief).copy()
            p.belief = p.rho.copy()
            p.operations = p.operations.copy()
        other.estimates = [l.fidelity for l in other.config.links]
        return other
