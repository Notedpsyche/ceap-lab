import copy
import json
import numpy as np
import pytest
from ceap.config import Config, LinkConfig, presets
from ceap.simulation import Episode
from ceap.controller import run_to_end, evaluate
from ceap import quantum as q


def ideal(nodes=4,**kwargs):
    return Config(nodes=nodes,links=[LinkConfig(fidelity=1,success_probability=1) for _ in range(nodes-1)],policy="immediate",gate_noise=0,readout_error=0,memory_ms=0,**kwargs)


@pytest.mark.parametrize("nodes",[2,3,4,5])
@pytest.mark.parametrize("label",list(q.SIX))
def test_complete_ideal_chain(nodes,label):
    ep=run_to_end(Episode(ideal(nodes,input_state=label)))
    assert ep.result["success"]
    assert ep.result["fidelity"] == pytest.approx(1)
    assert ep.result["swaps"] == nodes-2
    assert ep.result["attempts"] == nodes-1


def test_invalid_action_changes_nothing():
    ep=Episode(ideal());before=copy.deepcopy(ep.public())
    with pytest.raises(ValueError): ep.step("purify:p1:p2")
    assert ep.public()==before


def test_generation_overlap_memory_reservation():
    ep=Episode(ideal())
    ep.step("regenerate:0");ep.step("regenerate:1");ep.step("regenerate:2")
    assert ep.t==0
    assert ep.usage()==[1,2,2,1]
    ep.step("wait")
    assert ep.t==11
    assert len(ep.pairs)==3
    assert ep.attempts==3


def test_classical_swap_corrections_follow_the_receiver_segment():
    ep=run_to_end(Episode(ideal(4)))
    # Parallel generation: 11ms; each swap: 0.07+1ms;
    # final teleportation: 0.07+3ms along the complete chain.
    assert ep.result["latency_ms"]==pytest.approx(16.21)


def test_age_after_heralding_exact_once():
    c=Config(nodes=3,links=[LinkConfig(fidelity=1,success_probability=1),LinkConfig()],memory_ms=100)
    ep=Episode(c);ep.step("regenerate:0");ep.step("wait")
    p=next(iter(ep.pairs.values()))
    assert np.allclose(p.rho,q.age(q.BELL[0],1,100))
    ep.advance(16)
    assert np.allclose(p.rho,q.age(q.BELL[0],6,100))


def test_exhausted_generation_budget_still_allows_swap_and_teleport():
    ep=run_to_end(Episode(ideal(attempt_budget=3)))
    assert ep.result["success"]
    assert ep.attempts==3


def test_all_loss_terminates_and_has_no_fidelity():
    c=ideal(attempt_budget=5)
    for l in c.links:l.success_probability=0
    ep=run_to_end(Episode(c))
    assert not ep.result["success"]
    assert ep.result["fidelity"] is None
    assert ep.attempts==5
    assert ep.result["latency_ms"]>0


def test_deadline_includes_final_classical_corrections():
    c=ideal(2,deadline_ms=11.5)
    ep=run_to_end(Episode(c))
    assert not ep.result["success"]
    assert ep.t==11.5


def test_purification_consumption_and_illegal_with_one_copy():
    ep=Episode(ideal(3))
    ep.step("regenerate:0");ep.step("wait")
    assert not any(a["kind"]=="purify" for a in ep.legal_actions())
    ep.step("regenerate:0");ep.step("wait")
    action=next(a for a in ep.legal_actions() if a["kind"]=="purify")
    ep.step(action["id"])
    assert ep.purifications==1 and ep.discarded==1 and len(ep.pairs)==1
    assert ep.attempts==2 and ep.generated==2


def test_counterfactuals_never_mutate_live_episode_or_access_truth():
    c=ideal(3,rollouts=2);c.policy="ceap"
    ep=Episode(c);ep.step("regenerate:0");ep.step("wait")
    before=ep.public(diagnostics=True);before.pop("result")
    scores=evaluate(ep)
    after=ep.public(diagnostics=True);after.pop("result")
    assert before==after
    altered=copy.deepcopy(ep)
    for p in altered.pairs.values():p.rho=q.werner(.3)
    assert evaluate(altered)==scores
    altered.config.input_state="-i"
    assert evaluate(altered)==scores
    assert "density_matrix" not in json.dumps(ep.observation())


def test_reproducibility_ignores_wall_runtime():
    c=Config(policy="fixed")
    a=run_to_end(Episode(c));b=run_to_end(Episode(c))
    assert a.result==b.result
    assert a.events==b.events


def test_memory_limit_holds_for_all_baselines():
    for policy in ["immediate","fixed","threshold","weakest","ablation"]:
        c=Config(policy=policy,memories=2)
        ep=run_to_end(Episode(c))
        assert all(max(f["observation"]["memory_use"])<=2 for f in ep.frames)
        assert ep.t<=c.deadline_ms


def test_replacement_is_explicit_and_counts_discard():
    ep=Episode(ideal(3,memories=2))
    ep.step("regenerate:0");ep.step("wait")
    ep.step("regenerate:0");ep.step("wait")
    a=next(a for a in ep.legal_actions() if a.get("discard_pair"))
    before=ep.attempts
    ep.step(a["id"])
    assert ep.discarded==1 and ep.attempts==before+1
    assert a["discard_pair"] not in ep.pairs
    assert max(ep.usage())<=2
