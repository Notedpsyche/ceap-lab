import json
import numpy as np
import pytest
from ceap.config import Config
from ceap.experiments import wilson, summarize, execute_job, conditions


def test_wilson_and_failure_metrics():
    lo,hi=wilson(0,10)
    assert lo==pytest.approx(0,abs=1e-12) and hi>0
    row={"policy":"immediate","success":False,"qualified":False,"fidelity":None,"latency_ms":10,
         "attempts":2,"generated_pairs":0,"discarded_pairs":0,"purification_failures":0,"action_counts":{"regenerate":2},"policy_runtime_s":0}
    out=summarize([row]*10)[0]
    assert out['fidelity'] is None and out['latency_ms'] is None
    assert out['qualified_probability']==0 and out['qualified_throughput']==0
    assert out['attempts_per_qualified'] is None


def test_conditional_metrics_and_throughput_include_failed_duration():
    base={"policy":"immediate","latency_ms":10,"attempts":2,"generated_pairs":1,"discarded_pairs":0,"purification_failures":0,"action_counts":{},"policy_runtime_s":0}
    a={**base,"success":True,"qualified":True,"fidelity":.95,"worst_input_fidelity":.94}
    b={**base,"success":False,"qualified":False,"fidelity":None}
    row=summarize([a,b])[0]
    assert row['fidelity']==.95 and row['success_probability']==.5
    assert row['qualified_throughput']==50
    assert row['attempts_per_qualified']==4


def test_cli_worker_reproducible_raw_data(tmp_path):
    config=Config(policy='immediate').model_dump()
    payload={'config':config,'policies':['immediate','fixed'],'episodes':6,'experiment':'E1'}
    for name in ['one','two']:
        folder=tmp_path/name;folder.mkdir();execute_job(folder,payload)
        progress=json.loads((folder/'progress.json').read_text())
        assert progress['status']=='completed' and progress['completed']==12
    assert (tmp_path/'one/trials.jsonl').read_text()==(tmp_path/'two/trials.jsonl').read_text()


def test_cancel_preserves_prior_checkpoints(tmp_path):
    (tmp_path/'cancel').touch()
    execute_job(tmp_path,{'config':Config().model_dump(),'policies':['immediate'],'episodes':5,'experiment':'E3'})
    out=json.loads((tmp_path/'progress.json').read_text())
    assert out['status']=='cancelled' and out['completed']==0


def test_dedicated_experiment_grids():
    c=Config()
    assert len(conditions(c,'E4'))==3
    assert len(conditions(c,'E5'))==9
    assert len(conditions(c,'E7'))==3
