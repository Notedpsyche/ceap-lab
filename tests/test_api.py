from fastapi.testclient import TestClient
from ceap.app import app
from ceap.config import presets


def test_episode_workflow_and_exports():
    with TestClient(app) as client:
        assert client.get('/api/health').status_code==200
        config=presets()[1]['config'];config['policy']='immediate'
        response=client.post('/api/episodes',json=config)
        assert response.status_code==200,response.text
        ep=response.json();rid=ep['id']
        bad=client.post(f'/api/episodes/{rid}/step',json={'action_id':'purify:unknown:missing'})
        assert bad.status_code==409
        for _ in range(30):
            if ep['observation']['status'] in ('completed','failed'):break
            response=client.post(f'/api/episodes/{rid}/step',json={})
            assert response.status_code==200,response.text
            ep=response.json()
        assert ep['result']['success'] and ep['result']['fidelity']>.99999
        assert 'diagnostics' not in ep
        assert 'diagnostics' in client.get(f'/api/episodes/{rid}?diagnostics=true').json()
        detail=client.get(f'/api/runs/{rid}').json()
        assert len(detail['frames'])>1
        assert client.get(f'/api/runs/{rid}/export?format=csv').status_code==200
        assert client.get(f'/api/runs/{rid}/export?format=events').status_code==200
        assert any(r['id']==rid for r in client.get('/api/runs').json()['runs'])


def test_validation_errors():
    with TestClient(app) as client:
        assert client.post('/api/episodes',json={'nodes':5}).status_code==422
        assert client.post('/api/episodes',json={'gate_noise':-1}).status_code==422
        assert client.post('/api/jobs',json={'config':presets()[0]['config'],'policies':[]}).status_code==422
        assert client.get('/api/runs/not-a-run').status_code==404
