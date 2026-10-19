"""Full browser acceptance workflow. Run with CEAP Lab serving on port 8000."""
import json
import os
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'test-results';OUT.mkdir(exist_ok=True)


def main():
    errors=[]
    with sync_playwright() as p:
        executable=os.environ.get('CEAP_CHROMIUM')
        if not executable:
            found=list((Path.home()/'.cache/ms-playwright').glob('chromium-*/chrome-linux64/chrome'))
            executable=str(found[-1]) if found else None
        browser=p.chromium.launch(headless=True,executable_path=executable,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1512,'height':1100},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://127.0.0.1:8000',wait_until='networkidle')
        page.wait_for_function("document.querySelector('#connection').textContent.includes('connected')")
        page.screenshot(path=str(OUT/'workbench-desktop.png'),full_page=True)
        assert page.locator('#network svg').count()==1
        assert page.evaluate("document.querySelector('#config-form').checkValidity()")
        page.select_option('#preset','ideal')
        page.select_option('#policy','immediate')
        page.click('#run-button')
        page.wait_for_function("document.querySelector('#episode-status').textContent.includes('Qualified')",timeout=30000)
        assert page.locator('#metric-fidelity').inner_text()=='1.0000'
        page.screenshot(path=str(OUT/'completed-desktop.png'),full_page=True)
        page.click('#evaluate-button',force=True) if not page.locator('#evaluate-button').is_disabled() else None
        # Manual execution and information-limited counterfactual inspection.
        page.select_option('#preset','heterogeneous')
        page.locator('summary',has_text='Counterfactual controller').click()
        page.locator('[name="rollouts"]').fill('2')
        page.click('#step-button')
        page.wait_for_function("document.querySelector('#metric-attempts').textContent.trim().startsWith('1')")
        page.click('#evaluate-button')
        page.wait_for_selector('.action-estimate')
        page.screenshot(path=str(OUT/'counterfactual-desktop.png'),full_page=True)
        page.locator('[data-action]').first.click()
        page.wait_for_function("!document.querySelector('#step-button').disabled")
        page.select_option('#preset','ideal')
        page.select_option('#policy','immediate')
        # Seeded policy comparison with all six input states.
        page.locator('[data-view="comparison"]').click()
        for checkbox in page.locator('#policy-options input').all(): checkbox.uncheck()
        for policy in ['immediate','fixed','ceap']:
            page.locator(f'#policy-options input[value="{policy}"]').check()
        page.fill('#comparison-episodes','6')
        page.click('#start-comparison')
        page.wait_for_function("document.querySelector('#comparison-results .status-pill')?.textContent==='completed'",timeout=120000)
        assert page.locator('#comparison-results tbody tr').count()==3
        assert '1.0000' in page.locator('#comparison-results').inner_text()
        page.locator('[data-calibration-job]').click()
        page.wait_for_selector('.calibration-panel svg')
        job_id=page.locator('[data-job]').first.get_attribute('data-job')
        for ext in ['svg','png']:
            response=page.request.get(f'http://127.0.0.1:8000/api/jobs/{job_id}/chart?format={ext}')
            assert response.status==200,response.text()
            assert len(response.body())>100
        csv=page.request.get(f'http://127.0.0.1:8000/api/runs/{job_id}/export?format=csv')
        assert csv.status==200 and 'fidelity' in csv.text()
        page.screenshot(path=str(OUT/'comparison-desktop.png'),full_page=True)
        # Independent Qiskit circuit checks through the production job API.
        page.locator('[data-view="experiments"]').click()
        page.select_option('#experiment-kind','E0')
        page.click('#start-experiment')
        page.wait_for_function("['completed','failed'].includes(document.querySelector('#experiment-results .status-pill')?.textContent)",timeout=90000)
        assert page.locator('#experiment-results .check-fail').count()==0
        assert page.locator('#experiment-results .check-pass').count()>=12
        # Cooperative cancellation retains completed trial records.
        page.select_option('#experiment-kind','E3')
        page.fill('#experiment-episodes','100')
        page.click('#start-experiment')
        page.locator('#experiment-results [data-cancel-job]').click()
        page.wait_for_function("document.querySelector('#experiment-results .status-pill')?.textContent==='cancelled'",timeout=60000)
        page.locator('[data-view="history"]').click()
        page.wait_for_selector('.history-card')
        assert page.locator('.history-card').count()>=3
        page.locator('.history-card [data-open-run]').last.click()
        page.wait_for_selector('#detail-dialog[open]')
        page.locator('[data-replay-run]').click()
        page.wait_for_selector('#view-workbench.active')
        page.locator('#replay-slider').fill('1')
        assert 'Frame' in page.locator('#replay-label').inner_text()
        page.click('#live-button')
        # Narrow-screen navigation and no horizontal page overflow.
        page.set_viewport_size({'width':390,'height':844})
        page.screenshot(path=str(OUT/'workbench-mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile page overflows horizontally'
        page.locator('[data-view="theory"]').click()
        assert page.locator('#theory-teleport').is_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        browser.close()
    assert not errors,errors
    print(json.dumps({'browser_errors':errors,'screenshots':str(OUT),'status':'passed'}))


if __name__=='__main__':main()
