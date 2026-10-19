import numpy as np
import pytest
from ceap import quantum as q
from ceap.validation import physics_checks


def assert_density(rho):
    assert np.allclose(rho, rho.conj().T)
    assert np.trace(rho).real == pytest.approx(1)
    assert np.linalg.eigvalsh(rho).min() > -1e-10


@pytest.mark.parametrize("state", list(q.SIX))
def test_ideal_teleportation(state):
    f, out = q.teleport_fidelity(q.BELL[0], q.SIX[state])
    assert f == pytest.approx(1)
    assert_density(out)


@pytest.mark.parametrize("reported", range(4))
def test_all_swapping_branches(reported):
    prob, rho = q.conditional("swap", q.BELL[0], q.BELL[0], reported)
    assert prob == pytest.approx(.25)
    assert q.fidelity(rho) == pytest.approx(1)


@pytest.mark.parametrize("f", [.25, .5, .6, .8, .95, 1])
def test_purification_analytic(f):
    p = f*f + 2*f*(1-f)/3 + 5*(1-f)**2/9
    output = (f*f + (1-f)**2/9)/p
    branches = [q.conditional("purify", q.werner(f), q.werner(f), i) for i in (0,3)]
    assert sum(b[0] for b in branches) == pytest.approx(p)
    for _,rho in branches:
        assert q.fidelity(rho) == pytest.approx(output)
        assert_density(rho)


@pytest.mark.parametrize("kind", ["swap", "purify"])
@pytest.mark.parametrize("noise", [0, .01, .2])
def test_fast_kernel_matches_full_circuit(kind, noise):
    a,b=q.werner(.83),q.link_state(.71,"phase_flip")
    assert np.allclose(q.branches(kind,a,b,noise),q.branches_raw(kind,a,b,noise))


def test_readout_changes_correction_not_measurement_probability():
    # True outcome 00, incorrectly recorded as 01, introduces X on Bob.
    raw=q.branches_raw("swap",q.BELL[0],q.BELL[0])
    wrong=q.correct(q.clean(raw[0]),1)
    assert q.fidelity(wrong) == pytest.approx(0)
    assert np.trace(wrong).real == pytest.approx(1)


def test_noise_locality_and_complete_depolarization():
    rho=q.depolarize(q.BELL[0],(0,),1)
    assert np.allclose(rho,np.eye(4)/4)
    zero=np.diag([1,0]);one=np.diag([0,1]);rho=np.kron(zero,one)
    assert np.allclose(q.depolarize(rho,(0,),1),np.kron(np.eye(2)/2,one))


def test_memory_composes_without_double_aging():
    rho=q.link_state(.8,"amplitude_damping")
    assert np.allclose(q.age(q.age(rho,13,100),17,100),q.age(rho,30,100))
    assert np.allclose(q.age(rho,1e6,0),rho)


def test_independent_validation():
    checks=physics_checks()
    assert all(row["passed"] for row in checks), checks
