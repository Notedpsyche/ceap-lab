"""Small density-matrix circuits. Qubit zero is the leftmost tensor factor.

All fast Bell-diagonal kernels are derived from these same physical circuits;
no arbitrary fidelity bonuses or empirical purification formulas are used.
"""
from functools import lru_cache
from itertools import product
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
PAULIS = (I, X, Y, Z)
PHI = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
BELL = np.array([np.outer(np.kron(I, p) @ PHI, (np.kron(I, p) @ PHI).conj()) for p in PAULIS])
SIX = {
    "0": np.array([1, 0], complex), "1": np.array([0, 1], complex),
    "+": np.array([1, 1], complex) / np.sqrt(2), "-": np.array([1, -1], complex) / np.sqrt(2),
    "+i": np.array([1, 1j], complex) / np.sqrt(2), "-i": np.array([1, -1j], complex) / np.sqrt(2),
}


def clean(rho):
    rho = (rho + rho.conj().T) / 2
    tr = np.trace(rho).real
    return rho / tr if tr > 1e-15 else rho


def fidelity(rho):
    return float(np.clip(np.vdot(PHI, rho @ PHI).real, 0, 1))


def werner(f):
    return f * BELL[0] + (1 - f) / 3 * (BELL[1] + BELL[2] + BELL[3])


def twirl(rho):
    return werner(fidelity(rho))


def coefficients(rho):
    return np.einsum("bij,ji->b", BELL, rho).real


def is_diagonal(rho):
    c = coefficients(rho)
    return np.max(np.abs(rho - np.einsum("b,bij->ij", c, BELL))) < 1e-11


@lru_cache(maxsize=256)
def embed(n, target, which):
    out = np.array([[1]], complex)
    for q in range(n):
        out = np.kron(out, PAULIS[which] if q == target else I)
    return out


@lru_cache(maxsize=128)
def gate(n, name, a, b=-1):
    if name == "h":
        return (embed(n, a, 1) + embed(n, a, 3)) / np.sqrt(2)
    dim = 2**n
    out = np.zeros((dim, dim), complex)
    for col in range(dim):
        row = col ^ (1 << (n - 1 - b)) if col & (1 << (n - 1 - a)) else col
        out[row, col] = 1
    return out


def unitary(rho, u):
    return u @ rho @ u.conj().T


def depolarize(rho, targets, lam):
    """D_lam on target subsystem: (1-lam)rho + lam I/d ⊗ tr_target rho."""
    rho = np.asarray(rho, dtype=complex)
    if lam <= 0:
        return rho
    n = int(np.log2(len(rho)))
    mixed = np.zeros_like(rho)
    for choices in product(range(4), repeat=len(targets)):
        p = np.eye(len(rho), dtype=complex)
        for target, choice in zip(targets, choices):
            p = p @ embed(n, target, choice)
        mixed += unitary(rho, p)
    return (1 - lam) * rho + lam * mixed / (4**len(targets))


def apply_gate(rho, n, name, a, b, lam):
    return depolarize(unitary(rho, gate(n, name, a, b)), (a,) if name == "h" else (a, b), lam)


@lru_cache(maxsize=128)
def projection_indices(n, measured, bits):
    return np.array([i for i in range(2**n) if all(((i >> (n - 1 - q)) & 1) == bit for q, bit in zip(measured, bits))])


def branches_raw(kind, left, right, lam=0):
    rho = np.kron(left, right)
    if kind == "purify":
        rho = apply_gate(rho, 4, "cx", 0, 2, lam)
        rho = apply_gate(rho, 4, "cx", 1, 3, lam)
        measured = (2, 3)
    else:
        rho = apply_gate(rho, 4, "cx", 1, 2, lam)
        rho = apply_gate(rho, 4, "h", 1, -1, lam)
        measured = (1, 2)
    return np.array([rho[np.ix_(idx, idx)] for bits in product((0, 1), repeat=2)
                     for idx in [projection_indices(4, measured, bits)]])


@lru_cache(maxsize=32)
def bell_kernel(kind, lam):
    # 16 elementary basis inputs, four physical measurement outcomes.
    return np.array([[branches_raw(kind, a, b, lam) for b in BELL] for a in BELL])


def branches(kind, left, right, lam=0):
    if is_diagonal(left) and is_diagonal(right):
        return np.einsum("a,b,abtij->tij", coefficients(left), coefficients(right), bell_kernel(kind, float(lam)))
    return branches_raw(kind, left, right, lam)


def correct(rho, reported):
    a, b = divmod(reported, 2)
    op = np.kron(I, np.linalg.matrix_power(Z, a) @ np.linalg.matrix_power(X, b))
    return unitary(rho, op)


def reporting_probability(actual, reported, error):
    distance = (actual ^ reported).bit_count()
    return error**distance * (1 - error)**(2 - distance)


def conditional(kind, left, right, reported, lam=0, readout=0):
    raw = branches(kind, left, right, lam)
    out = sum(reporting_probability(t, reported, readout) * raw[t] for t in range(4))
    prob = float(np.trace(out).real)
    if prob > 1e-15:
        out = clean(out)
        out = twirl(out) if kind == "purify" else correct(out, reported)
    return prob, out


def operate(kind, left, right, belief_left, belief_right, lam, readout, rng):
    raw = branches(kind, left, right, lam)
    probs = np.maximum(0, np.trace(raw, axis1=1, axis2=2).real)
    probs /= probs.sum()
    actual = int(rng.choice(4, p=probs))
    flips = (int(rng.random() < readout) << 1) | int(rng.random() < readout)
    reported = actual ^ flips
    accepted = kind != "purify" or reported in (0, 3)
    if not accepted:
        return None, None, actual, reported
    truth = clean(raw[actual])
    truth = twirl(truth) if kind == "purify" else correct(truth, reported)
    _, belief = conditional(kind, belief_left, belief_right, reported, lam, readout)
    return clean(truth), clean(belief), actual, reported


def link_state(f, model="werner"):
    if model == "werner":
        return werner(f)
    if model == "phase_flip":
        return f * BELL[0] + (1 - f) * BELL[3]
    # Parameterize damping by Bell overlap, not by an undocumented raw gamma.
    # F = (1 + sqrt(1-gamma))² / 4, covers F in [0.25, 1].
    gamma = 1 - (2 * np.sqrt(f) - 1)**2
    k0 = np.kron(I, np.diag([1, np.sqrt(max(0, 1 - gamma))]))
    k1 = np.kron(I, np.array([[0, np.sqrt(max(0, gamma))], [0, 0]]))
    return clean(k0 @ BELL[0] @ k0.conj().T + k1 @ BELL[0] @ k1.conj().T)


def age(rho, dt, lifetime):
    if dt <= 0 or lifetime <= 0:
        return rho
    factor = np.exp(-dt / lifetime)
    if is_diagonal(rho):
        return factor**2 * rho + (1 - factor**2) * np.eye(4) / 4
    return depolarize(depolarize(rho, (0,), 1 - factor), (1,), 1 - factor)


def input_vector(label="+", theta=1.5707963267948966, phi=0):
    return SIX[label] if label in SIX else np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])


def teleport(resource, psi, readout=0):
    """Trace-preserving output, summing all actual and recorded Bell outcomes."""
    rho = np.kron(np.outer(psi, psi.conj()), resource)
    rho = unitary(unitary(rho, gate(3, "cx", 0, 1)), gate(3, "h", 0))
    out = np.zeros((2, 2), complex)
    for actual, bits in enumerate(product((0, 1), repeat=2)):
        idx = projection_indices(3, (0, 1), bits)
        branch = rho[np.ix_(idx, idx)]
        for reported in range(4):
            a, b = divmod(reported, 2)
            u = np.linalg.matrix_power(Z, a) @ np.linalg.matrix_power(X, b)
            out += reporting_probability(actual, reported, readout) * unitary(branch, u)
    return clean(out)


def teleport_fidelity(resource, psi, readout=0):
    out = teleport(resource, psi, readout)
    return float(np.clip(np.vdot(psi, out @ psi).real, 0, 1)), out


def average_teleport_fidelity(resource, readout=0):
    # Standard teleportation Pauli-twirls the resource. Identity correction
    # probability after noisy final readout determines average channel fidelity.
    c = coefficients(resource)
    # BELL order I,X,Y,Z versus recorded bit order I,X,Z,XZ.
    probs = np.array([(1-readout)**2, readout*(1-readout), readout**2, readout*(1-readout)])
    return float((1 + 2*np.dot(c, probs))/3)


def bloch(rho):
    return [float(np.trace(rho @ p).real) for p in (X, Y, Z)]


def serialize_matrix(rho):
    return [[[float(v.real), float(v.imag)] for v in row] for row in rho]
