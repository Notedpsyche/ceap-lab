"""Runtime physics checks, also exposed as experiment E0."""
import numpy as np
from . import quantum as q


def physics_checks():
    checks = []
    def check(name, error, tolerance=1e-8):
        checks.append({"name": name, "passed": bool(error < tolerance), "maximum_error": float(error), "tolerance": tolerance})
    err = max(abs(q.teleport_fidelity(q.BELL[0], psi)[0]-1) for psi in q.SIX.values())
    check("Ideal teleportation · all six input states", err)
    err = max(abs(q.fidelity(q.conditional("swap", q.BELL[0], q.BELL[0], bits)[1])-1) for bits in range(4))
    check("Ideal swapping · all four Bell outcomes", err)
    for f in (.6, .8, .95, 1.):
        p = f*f+2*f*(1-f)/3+5*(1-f)**2/9
        target = (f*f+(1-f)**2/9)/p
        measured_p = sum(q.conditional("purify", q.werner(f), q.werner(f), bits)[0] for bits in (0, 3))
        actual_f = q.fidelity(q.conditional("purify", q.werner(f), q.werner(f), 0)[1])
        check(f"BBPSSW analytic success and fidelity · F={f}", max(abs(measured_p-p), abs(actual_f-target)))
    for f in (.25, .5, .8, 1.):
        check(f"Exact amplitude damping Bell-overlap calibration · F={f}", abs(q.fidelity(q.link_state(f, "amplitude_damping"))-f))
    rho = q.age(q.werner(.9), 50, 100)
    check("Memory channel has unit trace and positive spectrum", max(abs(np.trace(rho)-1), max(0, -np.linalg.eigvalsh(rho).min())))
    check("Noisy final readout matches explicit six-state average", abs(q.average_teleport_fidelity(q.werner(.83), .04)-np.mean([q.teleport_fidelity(q.werner(.83), p, .04)[0] for p in q.SIX.values()])))
    # Independent Qiskit/Aer implementation: order q0,q1,q2 is little endian.
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit.quantum_info import DensityMatrix, partial_trace
        backend = AerSimulator(method="density_matrix")
        circuit = QuantumCircuit(4)
        circuit.h(0); circuit.cx(0, 1); circuit.h(2); circuit.cx(2, 3)
        circuit.cx(0, 2); circuit.cx(1, 3); circuit.save_density_matrix()
        rho = np.asarray(backend.run(transpile(circuit, backend)).result().data(0)["density_matrix"])
        prob = 0.; accumulated = np.zeros((4,4), complex)
        for a,b in ((0,0),(1,1)):
            idx = [i for i in range(16) if ((i>>2)&1)==a and ((i>>3)&1)==b]
            sub = rho[np.ix_(idx,idx)]; prob += np.trace(sub).real; accumulated += sub
        check("Independent Qiskit/Aer ideal purification circuit", max(abs(prob-1), abs(q.fidelity(accumulated)-1)))
        qerr = 0.
        for psi in q.SIX.values():
            # Coherent feed-forward replaces terminal measurement for this
            # independent ideal teleportation check (deferred measurement).
            c = QuantumCircuit(3)
            c.initialize(psi,0); c.h(1); c.cx(1,2); c.cx(0,1); c.h(0)
            c.cx(1,2); c.cz(0,2); c.save_density_matrix()
            state=backend.run(transpile(c,backend)).result().data(0)["density_matrix"]
            out=np.asarray(partial_trace(DensityMatrix(state),[0,1]))
            qerr=max(qerr,abs(np.vdot(psi,out@psi).real-1))
        check("Independent Qiskit/Aer teleportation · six inputs", qerr)
        from qiskit_aer.noise import depolarizing_error
        def reverse_order(matrix):
            n=int(np.log2(len(matrix)))
            order=list(reversed(range(n)))+list(reversed(range(n,2*n)))
            return matrix.reshape([2]*(2*n)).transpose(order).reshape(matrix.shape)
        for kind in ("swap","purify"):
            left,right=q.werner(.81),q.werner(.87)
            c=QuantumCircuit(4)
            c.set_density_matrix(reverse_order(np.kron(left,right)))
            if kind=="swap":
                c.cx(1,2);c.append(depolarizing_error(.02,2),[1,2])
                c.h(1);c.append(depolarizing_error(.02,1),[1])
                measured=(1,2)
            else:
                c.cx(0,2);c.append(depolarizing_error(.02,2),[0,2])
                c.cx(1,3);c.append(depolarizing_error(.02,2),[1,3])
                measured=(2,3)
            c.save_density_matrix()
            independent=reverse_order(np.asarray(backend.run(transpile(c,backend)).result().data(0)["density_matrix"]))
            expected=q.branches_raw(kind,left,right,.02)
            error=0.
            for outcome in range(4):
                bits=divmod(outcome,2)
                idx=[i for i in range(16) if all(((i>>(3-target))&1)==bit for target,bit in zip(measured,bits))]
                error=max(error,float(np.max(np.abs(independent[np.ix_(idx,idx)]-expected[outcome]))))
            check(f"Independent Qiskit/Aer noisy {kind} · four physical branches",error)
    except ImportError as exc:
        # Standalone CPython on some Linux distributions cannot dlopen Aer’s
        # OpenMP library late. Validate in an isolated process with it preloaded.
        # A relative preload avoids the dynamic loader splitting this workspace's
        # path at the space in "Quantum computing".
        import os, sys, subprocess, json
        from pathlib import Path
        if "static TLS" in str(exc) and not os.environ.get("CEAP_AER_VALIDATION_CHILD"):
            root = Path(__file__).resolve().parent.parent
            libs = list(Path(sys.prefix).glob("lib/python*/site-packages/qiskit_aer.libs/libgomp*.so*"))
            if libs:
                env = dict(os.environ, LD_PRELOAD=os.path.relpath(libs[0], root), OMP_NUM_THREADS="1", CEAP_AER_VALIDATION_CHILD="1")
                child = subprocess.run([sys.executable,"-c","import json; from ceap.validation import physics_checks; print(json.dumps([c for c in physics_checks() if c['name'].startswith('Independent')]))"], cwd=root, env=env, capture_output=True, text=True, timeout=60)
                if child.returncode == 0:
                    checks.extend(json.loads(child.stdout)); return checks
                exc = RuntimeError(child.stderr[-1000:])
        checks.append({"name":"Independent Qiskit/Aer validation", "passed":False, "error":str(exc)})
    return checks
