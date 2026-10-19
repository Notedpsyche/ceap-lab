const $ = (selector, root=document) => root.querySelector(selector);
const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];
const state = {config:null, presets:[], policies:{}, episode:null, dirty:false, busy:false, stream:null, jobStreams:new Map(), jobs:new Map(), replay:null, view:'workbench'};
const escapeHTML = value => String(value??'').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (value, digits=2) => value==null?'—':Number(value).toFixed(digits);
const percentage = value => value==null?'—':`${(100*value).toFixed(1)}%`;
const human = text => text.replaceAll('_',' ');
const endStates = ['completed','failed','cancelled','interrupted'];
let toastTimer;

function toast(message, error=false) {
  const el=$('#toast');el.textContent=message;el.className=`show ${error?'error':''}`;
  clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.className='',6500);
}
async function api(url, options={}) {
  const response=await fetch(url,{headers:{'Content-Type':'application/json'},...options});
  if(!response.ok){let message;try{const data=await response.json();message=typeof data.detail==='string'?data.detail:JSON.stringify(data.detail);}catch{message=response.statusText;}throw new Error(message);}
  return response.json();
}
const post=(url,body={})=>api(url,{method:'POST',body:JSON.stringify(body)});
function busy(on,label='Computing…') {
  state.busy=on;
  $('#run-button').disabled=on||!!state.episode?.running;
  $('#step-button').disabled=on||!!state.episode?.running;
  $('#new-episode').disabled=on;
  $('#evaluate-button').disabled=on||!!state.episode?.running;
  if(on)$('#episode-status').innerHTML=`<span class="spinner"></span>${escapeHTML(label)}`;
}
async function guarded(task,label) {
  if(state.busy)return;
  busy(true,label);
  try{await task();}catch(error){toast(error.message,true);}finally{busy(false);if(state.episode)renderStatus(state.episode);}
}
function view(name) {
  if(!['workbench','comparison','experiments','history','theory'].includes(name))return;
  state.view=name;
  $$('.view').forEach(el=>el.classList.toggle('active',el.id===`view-${name}`));
  $$('.nav-item').forEach(el=>el.classList.toggle('active',el.dataset.view===name));
  $('#breadcrumb-current').textContent={workbench:'Workbench',comparison:'Policy comparison',experiments:'Experiments',history:'Run history',theory:'Theory & methods'}[name];
  if(name==='history')loadHistory();
}
function dirty(){state.dirty=true;$('#config-dirty').classList.remove('hidden');renderNetwork();}
function nodeName(i,n=state.config?.nodes){return i===0?'Alice':i===n-1?'Bob':`R${i}`;}

function populate(config) {
  state.config=structuredClone(config);
  const form=$('#config-form');
  for(const [key,value] of Object.entries(config)){const field=form.elements.namedItem(key);if(field)field.value=value;}
  renderLinks();state.dirty=false;$('#config-dirty').classList.add('hidden');
  if(!state.episode)renderNetwork();
}
function renderLinks(){
  $('#link-fields').innerHTML=state.config.links.map((link,i)=>`<div class="link-config" data-edge="${i}">
    <label class="link-name" for="link-f-${i}"><strong>${nodeName(i)} <span style="color:#b4c0b4">—</span> ${nodeName(i+1)}</strong><output id="link-output-${i}">${number(link.fidelity,2)}</output></label>
    <input id="link-f-${i}" aria-label="Link ${i+1} Bell fidelity" data-link-field="fidelity" type="range" min="0.25" max="1" step="0.01" value="${link.fidelity}">
    <details><summary>Link model & generation</summary><label class="field">Noise model<select data-link-field="model"><option value="werner" ${link.model==='werner'?'selected':''}>Werner / depolarizing</option><option value="phase_flip" ${link.model==='phase_flip'?'selected':''}>Phase flip</option><option value="amplitude_damping" ${link.model==='amplitude_damping'?'selected':''}>Amplitude damping</option></select></label><div class="two-fields"><label class="field">Success probability<input data-link-field="success_probability" type="number" min="0" max="1" step="0.01" value="${link.success_probability}" required></label><label class="field">Attempt · ms<input data-link-field="attempt_ms" type="number" min="0.01" max="10000" step="any" value="${link.attempt_ms}" required></label></div><label class="field">Classical / acknowledgment · ms<input data-link-field="classical_ms" type="number" min="0" max="1000" step="any" value="${link.classical_ms}" required></label></details></div>`).join('');
}
function readConfig(validate=true){
  const form=$('#config-form');
  if(validate&&!form.checkValidity()){
    const invalid=form.querySelector(':invalid');
    for(let parent=invalid?.parentElement;parent&&parent!==form;parent=parent.parentElement)if(parent.tagName==='DETAILS')parent.open=true;
    form.reportValidity();throw new Error('Please correct the highlighted configuration field.');
  }
  const c=structuredClone(state.config);
  for(const [key,value] of new FormData(form))c[key]=['policy','input_state'].includes(key)?value:Number(value);
  $$('.link-config').forEach(el=>{const i=Number(el.dataset.edge);$$('[data-link-field]',el).forEach(f=>c.links[i][f.dataset.linkField]=f.dataset.linkField==='model'?f.value:Number(f.value));});
  return c;
}
async function newEpisode(){
  if(state.episode?.running)await post(`/api/episodes/${state.episode.id}/pause`);
  const config=readConfig();state.config=config;
  state.stream?.close();state.replay=null;
  state.episode=await post('/api/episodes',config);state.dirty=false;
  localStorage.setItem('ceap-active-episode',state.episode.id);
  localStorage.setItem('ceap-last-config',JSON.stringify(config));
  $('#config-dirty').classList.add('hidden');renderEpisode();
}
async function ensureEpisode(){if(!state.episode||state.dirty||endStates.includes(state.episode.observation.status))await newEpisode();}

function renderNetwork(observation=null){
  let c=state.dirty?readConfigSafe():state.episode?.config||state.config;
  if(!c)return;
  const obs=observation||(!state.dirty?state.episode?.observation:null);
  const n=c.nodes,w=650,h=210,pad=55,spacing=(w-2*pad)/(n-1),y=106;
  const x=i=>pad+i*spacing;
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${n}-node quantum chain, ${obs?.pairs?.length||0} stored pairs"><defs><pattern id="network-dots" x="0" y="0" width="14" height="14" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".6" fill="#edf1e9"/></pattern></defs><rect x="0" y="15" width="650" height="170" fill="url(#network-dots)"/>`;
  c.links.forEach((link,i)=>{
    const mid=(x(i)+x(i+1))/2;
    const pending=obs?.pending?.filter(p=>p.edge===i).length||0;
    svg+=`<g role="button" tabindex="0" data-inspect-link="${i}" aria-label="Inspect link ${i+1}"><line x1="${x(i)+24}" y1="${y}" x2="${x(i+1)-24}" y2="${y}" stroke="#b9cbbd" stroke-width="1.3" stroke-dasharray="4 5"/><rect x="${mid-35}" y="${y-34}" width="70" height="26" rx="4" fill="white"/><text class="link-value" x="${mid}" y="${y-18}" text-anchor="middle">${number(link.fidelity,2)}</text><text class="link-label" x="${mid}" y="${y+23}" text-anchor="middle">${pending?`${pending} pending`:`LINK ${i+1}`}</text>${pending?`<circle cx="${mid}" cy="${y}" r="4" fill="#d7ad61"/>`:''}</g>`;
  });
  const pairs=obs?.pairs||[];
  pairs.forEach((p,i)=>{
    const same=pairs.slice(0,i).filter(a=>a.left===p.left&&a.right===p.right).length;
    const arch=40+same*17+(p.right-p.left-1)*7;
    svg+=`<g role="button" tabindex="0" data-inspect-pair="${escapeHTML(p.id)}" aria-label="Inspect stored pair ${escapeHTML(p.id)}"><path d="M${x(p.left)} ${y-21} Q${(x(p.left)+x(p.right))/2} ${y-arch*2} ${x(p.right)} ${y-21}" fill="none" stroke="#239c8a" stroke-width="2" opacity=".85"/><text x="${(x(p.left)+x(p.right))/2}" y="${y-arch-7}" text-anchor="middle" font-size="8" fill="#258c79">${escapeHTML(p.id)} · ${number(p.estimated_fidelity,3)}</text></g>`;
  });
  for(let i=0;i<n;i++){
    const endpoint=i===0||i===n-1;
    svg+=`<g><circle cx="${x(i)}" cy="${y}" r="23" class="node ${endpoint?'endpoint':''}"/><text x="${x(i)}" y="${y+5}" text-anchor="middle" class="node-icon">${endpoint?'◇':'⋈'}</text><text x="${x(i)}" y="${y+50}" text-anchor="middle" class="node-label">${nodeName(i,n)}</text><text x="${x(i)}" y="${y+64}" text-anchor="middle" class="node-sub">${endpoint?(i===0?'SENDER':'RECEIVER'):'REPEATER'}</text>`;
    const use=obs?.memory_use?.[i]||0;
    for(let m=0;m<Math.min(c.memories,8);m++)svg+=`<rect x="${x(i)-(Math.min(c.memories,8)*6-2)/2+m*6}" y="${y+75}" width="4" height="4" rx="1" fill="${m<use?'#5fa58a':'#e2eadd'}"/>`;
    svg+='</g>';
  }
  svg+='</svg>';$('#network').innerHTML=svg;
  $('#topology-info').textContent=`${n-1} ${n===2?'link':'links'} · ${n-2} ${n===3?'repeater':'repeaters'}`;
  $('#memory-info').textContent=`${c.memories} memories per node`;
}
function readConfigSafe(){try{return readConfig(false);}catch{return state.config;}}
function renderStatus(ep){
  const el=$('#episode-status'), status=ep.running?'running':ep.observation.status;
  const labels={ready:'Ready for a decision',running:'Episode running',completed:ep.result?.qualified?'Qualified completion':'Completed · below target',failed:'Request failed',cancelled:'Cancelled'};
  el.className=`status-pill ${status}`;el.textContent=labels[status]||status;
  $('#pause-button').disabled=!ep.running;
  if(!state.busy){$('#run-button').disabled=ep.running;$('#step-button').disabled=ep.running;$('#evaluate-button').disabled=ep.running||endStates.includes(ep.observation.status);}
}
function renderEpisode(){
  const ep=state.episode;if(!ep)return;
  const obs=state.replay?.observation||ep.observation,result=state.replay?state.replay.result:ep.result;
  renderNetwork(obs);renderStatus(ep);
  $('#clock-value').innerHTML=`${number(obs.time_ms)} <small>ms</small>`;
  $('#metric-fidelity').textContent=number(result?.fidelity,4);
  $('#fidelity-note').textContent=result?.success?(result.qualified?'Target achieved':'Below target fidelity'):obs.status==='failed'?'No output state':'Available after completion';
  $('#metric-attempts').innerHTML=`${obs.attempts} <small>/ ${ep.config.attempt_budget}</small>`;
  $('#metric-pairs').textContent=obs.pairs.length;
  $('#pending-note').textContent=`${obs.pending.length} pending ${obs.pending.length===1?'attempt':'attempts'}`;
  const events=state.replay?ep.events.slice(0,state.replay.event_count):ep.events;
  $('#metric-purifications').textContent=events.filter(e=>['purified','purification_failed'].includes(e.kind)).length;
  $('#purification-note').textContent=`${events.filter(e=>e.kind==='purification_failed').length} rejected attempts`;
  $('#event-count').textContent=`${events.length} EVENTS`;
  $('#timeline').innerHTML=events.map(e=>`<div class="timeline-event"><time>${number(e.time_ms,1)} ms</time><div><span class="event-kind">${escapeHTML(human(e.kind))}</span><br>${escapeHTML(e.message)}${e.recorded_bits?` <span class="tag">${escapeHTML(e.recorded_bits)}</span>`:''}</div></div>`).join('');
  if(!state.replay)$('#timeline').scrollTop=$('#timeline').scrollHeight;
  if(result)renderOutput(result);else $('#output').innerHTML='<div class="bloch-placeholder"><svg viewBox="0 0 160 130"><circle cx="80" cy="65" r="48"/><ellipse cx="80" cy="65" rx="48" ry="15"/><path d="M80 8v113M22 65h116M48 100l64-70"/><circle class="origin" cx="80" cy="65" r="3"/></svg><p>Awaiting final teleportation</p></div>';
  $('#replay-slider').max=Math.max(0,ep.frame_count-1);$('#replay-slider').disabled=!ep.frame_count;
  if(!state.replay){$('#replay-slider').value=Math.max(0,ep.frame_count-1);$('#replay-label').textContent='Live';}
  $('#export-episode').disabled=false;
  if(!state.replay)renderActions(ep.actions||[]);
  if(ep.error)toast(ep.error,true);
}
function renderOutput(result){
  if(!result.success){$('#output').innerHTML=`<div class="output-result"><div class="failure">No output state</div><p>${escapeHTML(result.reason||'The request did not complete.')}</p><dl><div><dt>Attempts consumed</dt><dd>${result.attempts}</dd></div><div><dt>Elapsed time</dt><dd>${number(result.latency_ms)} ms</dd></div></dl></div>`;return;}
  const vec=result.output_bloch||[0,0,0],input=result.input_bloch||[0,0,0];
  const project=v=>[80+v[0]*43-v[1]*19,65-v[2]*43+v[1]*16];const b=project(vec),a=project(input);
  $('#output').innerHTML=`<div class="output-result"><svg viewBox="0 0 160 130" aria-label="Input and output Bloch vectors"><circle cx="80" cy="65" r="47" fill="none" stroke="#dfe8dc"/><ellipse cx="80" cy="65" rx="47" ry="15" fill="none" stroke="#e4ecdf"/><path d="M80 9v112M24 65h112" stroke="#e5ebdf"/><line x1="80" y1="65" x2="${a[0]}" y2="${a[1]}" stroke="#91a1bc" stroke-width="2" stroke-dasharray="3 2"/><line x1="80" y1="65" x2="${b[0]}" y2="${b[1]}" stroke="#16887f" stroke-width="2.5"/><circle cx="${b[0]}" cy="${b[1]}" r="3" fill="#16887f"/><text x="86" y="16" font-size="8" fill="#a0ad95">Z</text><text x="139" y="68" font-size="8" fill="#a0ad95">X</text></svg><div class="big-number">${number(result.fidelity,4)}</div><p>${result.qualified?'Qualified teleportation':'Completed, below target'}<br><small style="color:#90a085">Dashed: input · teal: output</small></p><dl><div><dt>Six-state mean fidelity</dt><dd>${number(result.average_fidelity,4)}</dd></div><div><dt>Worst tested input</dt><dd>${number(result.worst_input_fidelity,4)}</dd></div><div><dt>Output Bloch vector</dt><dd>${vec.map(v=>number(v,2)).join(', ')}</dd></div></dl></div>`;
}
function renderActions(actions,evaluated=false){
  if(!actions.length){$('#actions').innerHTML=`<div class="empty-state"><h3>${state.episode&&endStates.includes(state.episode.observation.status)?'This request has finished.':'No action is ready.'}</h3><p>${state.episode&&endStates.includes(state.episode.observation.status)?'Start a new episode or replay the recorded operations.':'Generate resources before attempting swapping or purification.'}</p></div>`;return;}
  $('#actions').innerHTML=actions.map((a,i)=>`<button class="action-card ${evaluated&&i===0?'best':''}" data-action="${escapeHTML(a.id)}" ${state.episode?.running||state.replay?'disabled':''}><span class="action-type">${escapeHTML(a.kind==='regenerate'?'GENERATE':a.kind.toUpperCase())}</span><div class="action-main"><strong>${escapeHTML(a.label)}${evaluated&&i===0?'<span class="best-label">CEAP CHOICE</span>':''}</strong><p>${evaluated?`${number(a.predicted_extra_ms,1)} ms expected · ${number(a.predicted_attempts,1)} attempts · U ${number(a.utility,3)}`:escapeHTML(a.detail)}</p></div>${evaluated?`<div class="action-estimate">${percentage(a.predicted_qualified_probability)}<small>qualified · Δ ${a.delta_probability>=0?'+':''}${number(a.delta_probability*100,1)} pp</small></div>`:''}<span class="action-arrow">→</span></button>`).join('');
}
function connectEpisode(){
  state.stream?.close();const stream=new EventSource(`/api/episodes/${state.episode.id}/events`);state.stream=stream;
  stream.onmessage=e=>{state.episode=JSON.parse(e.data);state.replay=null;renderEpisode();if(endStates.includes(state.episode.observation.status)){stream.close();toast(state.episode.result?.success?'Teleportation completed. Inspect the measured output.':state.episode.result?.reason||'Episode finished.');}};
  stream.onerror=()=>{if(state.episode?.running)$('#connection').textContent='Reconnecting to local engine…';};
}

function inspectLink(i){
  const c=state.dirty?readConfigSafe():state.episode?.config||state.config,link=c.links[i];
  $('#selection-detail').innerHTML=`<strong>Link ${i+1}: ${nodeName(i,c.nodes)}–${nodeName(i+1,c.nodes)}</strong> · ${escapeHTML(human(link.model))} · Bell F ${number(link.fidelity,3)} · generation p ${percentage(link.success_probability)} · ${number(link.attempt_ms+link.classical_ms,1)} ms herald cycle`;
}
function inspectPair(id){
  const obs=state.replay?.observation||state.episode?.observation,p=obs?.pairs.find(a=>a.id===id);if(!p)return;
  $('#selection-detail').innerHTML=`<strong>Pair ${escapeHTML(p.id)}: ${nodeName(p.left)}–${nodeName(p.right)}</strong> · estimated Bell F ${number(p.estimated_fidelity,4)} · age ${number(p.age_ms,2)} ms · ${p.rounds} accepted purification rounds`;
}

const metricLabels={qualified_probability:'Qualified completion probability',success_probability:'Completion probability',fidelity:'Conditional teleportation fidelity',qualified_throughput:'Qualified throughput / simulated second',latency_ms:'Conditional latency · ms',attempts_per_request:'Generation attempts per request'};
function ciText(values,digits=3){return values?.[0]!=null?`95% CI ${number(values[0],digits)} – ${number(values[1],digits)}`:'No completed outputs';}
function barChart(rows,metric){
  if(!rows.length)return '<div class="empty-state"><h3>The evidence starts with a run.</h3><p>Results will appear as completed trials are checkpointed.</p></div>';
  const w=850,rowH=43,h=Math.max(190,rows.length*rowH+45),left=200,right=65,top=20,plotW=w-left-right;
  const probability=['qualified_probability','success_probability','fidelity'].includes(metric);
  const ciKey={qualified_probability:'qualified_ci',success_probability:'success_ci',fidelity:'fidelity_ci',latency_ms:'latency_ci',qualified_throughput:'throughput_ci'}[metric];
  let max=probability?1:Math.max(...rows.map(r=>r[ciKey]?.[1]||r[metric]||0),1)*1.1;
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${escapeHTML(metricLabels[metric])}">`;
  for(let i=0;i<=4;i++){const x=left+i*plotW/4;svg+=`<line x1="${x}" y1="12" x2="${x}" y2="${h-25}" stroke="#edf1e9"/><text x="${x}" y="${h-7}" text-anchor="middle" font-size="9" fill="#9aa88f">${number(max*i/4,probability?2:1)}</text>`;}
  rows.forEach((r,i)=>{const y=top+i*rowH,v=r[metric];let label=r.label+(r.condition==='baseline'?'':` · ${r.condition}`);if(label.length>36)label=label.slice(0,34)+'…';
    svg+=`<text x="${left-12}" y="${y+12}" text-anchor="end" font-size="10" fill="#667e5c">${escapeHTML(label)}</text><rect x="${left}" y="${y}" width="${Math.max(0,(v||0)/max*plotW)}" height="20" rx="2" fill="${r.policy==='ceap'?'#168b80':'#8aadc0'}"/>`;
    if(r[ciKey]?.[0]!=null&&v!=null){const lo=left+r[ciKey][0]/max*plotW,hi=left+r[ciKey][1]/max*plotW;svg+=`<path d="M${lo} ${y+6}v8M${lo} ${y+10}H${hi}M${hi} ${y+6}v8" stroke="#365950" fill="none"/>`;}
    svg+=`<text x="${left+(v||0)/max*plotW+9}" y="${y+14}" font-size="10" fill="#688260">${number(v,probability?3:1)}</text>`;
  });return svg+'</svg>';
}
function paretoChart(rows){
  const points=rows.filter(r=>r.fidelity!=null&&r.latency_ms!=null),w=400,h=220;
  if(!points.length)return '<p class="muted">Awaiting completed outputs.</p>';
  const max=Math.max(...points.map(p=>p.latency_ms),1)*1.15;
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Fidelity latency Pareto comparison"><path d="M45 15V180H380" stroke="#cddac4" fill="none"/><text x="210" y="215" text-anchor="middle" font-size="9" fill="#8e9e82">Mean latency · ms</text><text x="12" y="100" transform="rotate(-90 12 100)" text-anchor="middle" font-size="9" fill="#8e9e82">Conditional fidelity</text>`;
  for(let i=0;i<=4;i++){let x=45+i*335/4;svg+=`<text x="${x}" y="195" text-anchor="middle" font-size="8" fill="#9aab8e">${number(max*i/4,0)}</text>`;}
  for(const p of points){const x=45+p.latency_ms/max*335,y=180-p.fidelity*160;svg+=`<circle cx="${x}" cy="${y}" r="${Math.min(10,3+Math.sqrt(p.attempts_per_request))}" fill="${p.policy==='ceap'?'#0c847b':'#799daf'}" opacity=".75"><title>${escapeHTML(p.label)} · F ${number(p.fidelity,3)} · ${number(p.latency_ms,1)} ms · ${number(p.attempts_per_request,1)} attempts</title></circle>`;}
  return svg+'</svg>';
}
function actionChart(rows){
  const w=400,h=rows.length*27+30,max=Math.max(...rows.map(r=>Object.values(r.action_counts).reduce((a,b)=>a+b,0)),1),colors={regenerate:'#4e8d9b',wait:'#b8c9a7',purify:'#ddb16b',swap:'#6b9b84'};
  let svg=`<svg viewBox="0 0 ${w} ${Math.max(100,h)}" role="img" aria-label="Recorded policy action frequencies">`;
  rows.forEach((r,i)=>{let x=120;const y=10+i*27;svg+=`<text x="110" y="${y+11}" text-anchor="end" font-size="8" fill="#829674">${escapeHTML(r.label.slice(0,22))}</text>`;for(const [a,color] of Object.entries(colors)){const count=r.action_counts[a]||0,bw=count/max*260;svg+=`<rect x="${x}" y="${y}" width="${bw}" height="16" fill="${color}"><title>${escapeHTML(r.label)} · ${a}: ${count}</title></rect>`;x+=bw;}});
  return svg+'</svg>';
}
function jobHTML(job,metric='qualified_probability'){
  const done=endStates.includes(job.status),rows=job.summaries||[],progress=job.total?100*job.completed/job.total:0;
  return `<section class="panel job-panel" data-job="${job.id}"><div class="job-heading"><div><h2>${escapeHTML(job.request?.experiment||'Policy comparison')} <span style="font-weight:400;color:#90a188">/ ${job.id.slice(0,8)}</span></h2><p>${job.status==='running'?'<span class="spinner"></span>':''}${escapeHTML(human(job.status))} · ${job.completed} / ${job.total} ${job.validation?'checks':'trials'}</p></div>${!done?`<button class="button secondary" data-cancel-job="${job.id}">Cancel job</button>`:`<span class="status-pill ${job.status}">${escapeHTML(job.status)}</span>`}</div><div class="progress-track"><div style="width:${progress}%"></div></div><div class="progress-description"><span>${done?'Saved locally. Partial runs remain labeled.':'Results update as trials finish; you can keep exploring.'}</span><span>${number(progress,0)}%</span></div>${job.error?`<div class="job-error">${escapeHTML(job.error)}</div>`:''}
  ${job.validation?`<div style="margin-top:20px">${job.validation.map(v=>`<div class="validation-result"><span>${escapeHTML(v.name)}</span><strong class="${v.passed?'check-pass':'check-fail'}">${v.passed?'PASS':'FAIL'}</strong></div>`).join('')}</div>`:`<div class="chart-toolbar"><select data-chart-metric="${job.id}" aria-label="Chart metric">${Object.entries(metricLabels).map(([k,v])=>`<option value="${k}" ${k===metric?'selected':''}>${v}</option>`).join('')}</select><div><a class="button secondary" href="/api/jobs/${job.id}/chart?metric=${metric}&format=svg">SVG ↓</a> <a class="button secondary" href="/api/jobs/${job.id}/chart?metric=${metric}&format=png">PNG ↓</a></div></div><div class="chart">${barChart(rows,metric)}</div>
  ${rows.length?`<div class="table-wrap"><table class="results-table"><thead><tr><th>POLICY / CONDITION</th><th>QUALIFIED</th><th>COMPLETED</th><th>FIDELITY</th><th>LATENCY</th><th>ATTEMPTS</th></tr></thead><tbody>${rows.map(r=>`<tr class="${r.policy==='ceap'?'ceap-row':''}"><td>${escapeHTML(r.label)}<small>${escapeHTML(r.condition)} · n=${r.trials}${r.sparse?' · sparse completions':''}</small></td><td>${percentage(r.qualified_probability)}<small>${ciText(r.qualified_ci)}</small></td><td>${percentage(r.success_probability)}<small>${ciText(r.success_ci)}</small></td><td>${number(r.fidelity,4)}<small>${ciText(r.fidelity_ci,4)}</small></td><td>${number(r.latency_ms,1)} ms<small>${ciText(r.latency_ci,1)}</small></td><td>${number(r.attempts_per_request,1)} / request<small>${number(r.attempts_per_qualified,1)} / qualified</small></td></tr>`).join('')}</tbody></table></div><div class="chart-grid"><div class="mini-chart"><h3>Fidelity · latency · resources</h3>${paretoChart(rows)}<p class="subtle-note">Circle area reflects generation attempts. Hover for details.</p></div><div class="mini-chart"><h3>Recorded action frequencies</h3>${actionChart(rows)}<p class="subtle-note">Blue: generation · pale green: wait · amber: purification · green: swap</p></div></div>${job.differences?.length?`<div class="mini-chart"><h3>CEAP qualified-completion difference</h3>${job.differences.map(d=>`<p class="subtle-note">vs ${escapeHTML(state.policies[d.against]||d.against)} · ${escapeHTML(d.condition)}: ${d.difference>=0?'+':''}${number(d.difference*100,1)} percentage points (${ciText(d.ci.map(v=>v*100),1)}); ${d.matched_episodes} matched episodes.</p>`).join('')}</div>`:''}`:''}`}
  <div class="result-exports"><a class="button secondary" href="/api/runs/${job.id}/export?format=csv">Trials CSV ↓</a><a class="button secondary" href="/api/runs/${job.id}/export?format=config">Configuration JSON ↓</a><a class="button secondary" href="/api/runs/${job.id}/export?format=json">Summary JSON ↓</a><a class="button secondary" href="/api/runs/${job.id}/export?format=events">Events & predictions ↓</a><button class="button text" data-calibration-job="${job.id}">Prediction calibration ↗</button><button class="button text" data-print-job="${job.id}">Print results ↗</button></div><p class="subtle-note">95% intervals. Fidelity and latency are conditional on completion; throughput includes failed-request duration. Different policies may create different events within a matched scenario.</p></section>`;
}
function renderJob(job,container){
  state.jobs.set(job.id,{job,container});const old=$(`[data-chart-metric="${job.id}"]`,container)?.value||'qualified_probability';container.innerHTML=jobHTML(job,old);
}
function calibrationChart(data){
  const w=400,h=250;
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Predicted vs realized chosen-action outcomes"><path d="M45 15V205H365" fill="none" stroke="#cad9c4"/><path d="M45 205L365 15" stroke="#c4d4bd" stroke-dasharray="4 4"/><text x="200" y="239" text-anchor="middle" font-size="9" fill="#869e7a">Predicted qualified probability</text><text x="12" y="110" text-anchor="middle" transform="rotate(-90 12 110)" font-size="9" fill="#869e7a">Observed eventual qualification</text>`;
  for(const p of data.calibration)svg+=`<circle cx="${45+p.predicted*320}" cy="${205-p.observed*190}" r="${Math.min(12,3+Math.sqrt(p.count))}" fill="#168a80" opacity=".8"><title>${p.count} choices · predicted ${percentage(p.predicted)} · observed ${percentage(p.observed)}</title></circle>`;
  return `<div class="mini-chart"><h3>Predicted vs realized action outcomes</h3>${svg}</svg><p class="subtle-note">${escapeHTML(data.note)}</p></div>`;
}
function watchJob(id,container){
  state.jobStreams.get(id)?.close();const stream=new EventSource(`/api/jobs/${id}/events`);state.jobStreams.set(id,stream);
  stream.onmessage=e=>{const job=JSON.parse(e.data);renderJob(job,container);if(endStates.includes(job.status)){stream.close();toast(`Experiment ${job.status}: ${job.completed} completed ${job.validation?'checks':'trials'}.`,job.status==='failed');}};
  stream.onerror=()=>{const info=state.jobs.get(id);if(info&&!endStates.includes(info.job.status))toast('Experiment connection interrupted; the worker keeps running.');};
}
async function launchJob(experiment,container,episodes,policies,sweep=null){
  if(!Number.isInteger(episodes)||episodes<1||episodes>10000)throw new Error('Episodes must be an integer between 1 and 10,000.');
  const request={config:readConfig(),policies,episodes,experiment,sweep};
  const job=await post('/api/jobs',request);
  renderJob({id:job.id,status:'queued',completed:0,total:0,summaries:[],request},container);watchJob(job.id,container);
  localStorage.setItem(`ceap-${container.id}`,job.id);
}
async function loadHistory(){
  try{
    const data=await api('/api/runs');$('#history-list').innerHTML=data.runs.length?data.runs.map(r=>{
      const config=r.kind==='episode'?r.config:r.config.config;
      return `<div class="panel history-card"><div><h3>${r.kind==='episode'?escapeHTML(state.policies[config.policy]||config.policy)+' episode':escapeHTML(r.config.experiment)+' experiment'} <span style="font-weight:400;color:#9bad91">/ ${r.id.slice(0,8)}</span></h3><p>${config.nodes} nodes · seed ${config.seed} · ${escapeHTML(new Date(r.created).toLocaleString())} · <span>${escapeHTML(human(r.status))}</span></p></div><div class="history-actions"><button class="button secondary" data-open-run="${r.id}">Inspect</button><button class="button secondary" data-reproduce-run="${r.id}">Reproduce</button><a class="button text" href="/api/runs/${r.id}/export?format=json">JSON ↓</a></div></div>`;
    }).join(''):'<div class="empty-state history-empty"><h3>Your research record starts here.</h3><p>Run an episode or comparison to save its configuration and outcomes.</p></div>';
  }catch(error){toast(error.message,true);}
}
async function openRun(id){
  const data=await api(`/api/runs/${id}`);
  if(data.observation){
    $('#dialog-title').textContent=`Episode ${id.slice(0,8)}`;
    $('#dialog-body').innerHTML=`<div class="detail-content"><div class="status-detail"><span>${escapeHTML(data.observation.status)}</span><span>${number(data.observation.time_ms)} ms</span><span>${data.observation.attempts} attempts</span></div><p>Fidelity: <strong>${number(data.result?.fidelity,4)}</strong> · ${data.result?.qualified?'qualified':'not qualified'}</p><button class="button primary" data-replay-run="${id}">Replay recorded episode →</button> <a class="button secondary" href="/api/runs/${id}/export?format=events">Event log ↓</a><h3 style="margin-top:20px">Configuration & result</h3><pre>${escapeHTML(JSON.stringify({config:data.config,result:data.result,versions:data.versions},null,2))}</pre></div>`;$('#detail-dialog').showModal();
  }else{view('experiments');location.hash='experiments';renderJob(data,$('#experiment-results'));if(!endStates.includes(data.status))watchJob(id,$('#experiment-results'));}
}
async function reproduce(id){
  const data=await api(`/api/runs/${id}`);
  if(data.observation){await pauseIfRunning();state.episode=null;populate(data.config);view('workbench');location.hash='workbench';await newEpisode();toast('Recreated the configuration and seed. Run or step to reproduce.');}
  else{populate(data.request.config);view('experiments');location.hash='experiments';const job=await post('/api/jobs',data.request);watchJob(job.id,$('#experiment-results'));toast('A new experiment is running with the saved configuration.');}
}
async function pauseIfRunning(){if(state.episode?.running)await post(`/api/episodes/${state.episode.id}/pause`);state.stream?.close();}
async function replayRun(id){
  const data=await api(`/api/runs/${id}`);await pauseIfRunning();
  state.episode={...data,id,frame_count:data.frames.length,running:false};state.savedFrames=data.frames;populate(data.config);view('workbench');location.hash='workbench';$('#detail-dialog').close();state.replay=data.frames[0];renderEpisode();$('#replay-slider').value=0;$('#replay-label').textContent='Frame 1';
}

document.addEventListener('click',e=>{
  const nav=e.target.closest('[data-view]');if(nav){view(nav.dataset.view);location.hash=nav.dataset.view;}
  const action=e.target.closest('[data-action]');if(action)guarded(async()=>{state.replay=null;state.episode=await post(`/api/episodes/${state.episode.id}/step`,{action_id:action.dataset.action});renderEpisode();},'Executing operation…');
  const link=e.target.closest('[data-inspect-link]');if(link)inspectLink(Number(link.dataset.inspectLink));
  const pair=e.target.closest('[data-inspect-pair]');if(pair)inspectPair(pair.dataset.inspectPair);
  const cancel=e.target.closest('[data-cancel-job]');if(cancel)post(`/api/jobs/${cancel.dataset.cancelJob}/cancel`).then(()=>toast('Cancellation requested. Completed trials are retained.')).catch(x=>toast(x.message,true));
  const open=e.target.closest('[data-open-run]');if(open)openRun(open.dataset.openRun).catch(x=>toast(x.message,true));
  const repro=e.target.closest('[data-reproduce-run]');if(repro)guarded(()=>reproduce(repro.dataset.reproduceRun),'Reproducing configuration…');
  const replay=e.target.closest('[data-replay-run]');if(replay)replayRun(replay.dataset.replayRun).catch(x=>toast(x.message,true));
  const print=e.target.closest('[data-print-job]');if(print)window.print();
  const calibration=e.target.closest('[data-calibration-job]');if(calibration){calibration.disabled=true;api(`/api/jobs/${calibration.dataset.calibrationJob}/analysis`).then(data=>{const container=calibration.closest('.job-panel');let panel=container.querySelector('.calibration-panel');if(!panel){panel=document.createElement('div');panel.className='calibration-panel';container.append(panel);}panel.innerHTML=data.choices?calibrationChart(data):'<p class="subtle-note">No counterfactual predictions have been recorded in this job yet.</p>';}).catch(e=>toast(e.message,true)).finally(()=>calibration.disabled=false);}
});
document.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)&&e.target.matches('[data-inspect-link],[data-inspect-pair]')){e.preventDefault();e.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
window.addEventListener('hashchange',()=>{const hash=location.hash.slice(1);if(!hash.startsWith('theory-'))view(hash);});
$('#help-button').onclick=()=>{view('theory');location.hash='theory';};
$('#close-dialog').onclick=()=>$('#detail-dialog').close();
$('#preset').onchange=async()=>{await pauseIfRunning();state.episode=null;const preset=state.presets.find(p=>p.id===$('#preset').value);populate(preset.config);$('#preset-description').textContent=preset.description;$('#actions').innerHTML='<div class="empty-state"><h3>Scenario loaded.</h3><p>Run an episode or step once to initialize the resources.</p></div>';state.dirty=true;$('#config-dirty').classList.remove('hidden');};
$('#config-form').addEventListener('input',e=>{
  if(e.target.dataset.linkField){const parent=e.target.closest('[data-edge]'),i=Number(parent.dataset.edge);state.config.links[i][e.target.dataset.linkField]=e.target.dataset.linkField==='model'?e.target.value:Number(e.target.value);if(e.target.dataset.linkField==='fidelity')$(`#link-output-${i}`).textContent=number(e.target.value,2);}
  dirty();
});
$('#nodes').addEventListener('change',()=>{const n=Number($('#nodes').value);state.config.nodes=n;state.config.links=Array.from({length:n-1},(_,i)=>state.config.links[i]||{fidelity:.9,success_probability:.6,attempt_ms:10,classical_ms:1,model:'werner'});renderLinks();dirty();});
$('#new-episode').onclick=()=>guarded(newEpisode,'Initializing…');
$('#run-button').onclick=()=>guarded(async()=>{await ensureEpisode();await post(`/api/episodes/${state.episode.id}/run`);state.episode.running=true;renderEpisode();connectEpisode();},'Starting episode…');
$('#step-button').onclick=()=>guarded(async()=>{await ensureEpisode();state.replay=null;state.episode=await post(`/api/episodes/${state.episode.id}/step`);renderEpisode();},'Evaluating next decision…');
$('#pause-button').onclick=async()=>{try{await post(`/api/episodes/${state.episode.id}/pause`);toast('Pausing after the current atomic operation.');}catch(e){toast(e.message,true);}};
$('#evaluate-button').onclick=()=>guarded(async()=>{await ensureEpisode();const data=await api(`/api/episodes/${state.episode.id}/actions`);renderActions(data.actions,data.model_based);},'Evaluating modeled futures…');
$('#diagnostics-button').onclick=()=>guarded(async()=>{if(!state.episode)throw new Error('Initialize an episode first.');const data=await api(`/api/episodes/${state.episode.id}?diagnostics=true`);$('#diagnostics-output').textContent=JSON.stringify(data.diagnostics,null,2);},'Inspecting diagnostics…');
$('#export-episode').onclick=()=>{if(state.episode)location.href=`/api/runs/${state.episode.id}/export?format=json`;};
$('#replay-slider').oninput=async()=>{if(!state.episode||state.episode.running)return;try{if(!state.savedFrames||state.savedFramesId!==state.episode.id){const data=await api(`/api/runs/${state.episode.id}`);state.savedFrames=data.frames;state.savedFramesId=state.episode.id;}const index=Number($('#replay-slider').value);state.replay=state.savedFrames[index];renderEpisode();$('#replay-slider').value=index;$('#replay-label').textContent=`Frame ${index+1}`;$('#actions').innerHTML='<div class="empty-state"><h3>Recorded frame</h3><p>Return to live mode to execute actions.</p></div>';}catch(e){toast(e.message,true);}};
$('#live-button').onclick=()=>{state.replay=null;renderEpisode();};
$('#start-comparison').onclick=async()=>{const b=$('#start-comparison');b.disabled=true;try{const policies=$$('#policy-options input:checked').map(e=>e.value);if(!policies.length)throw new Error('Select at least one policy.');await launchJob('E3',$('#comparison-results'),Number($('#comparison-episodes').value),policies);}catch(e){toast(e.message,true);}finally{b.disabled=false;}};
$('#start-experiment').onclick=async()=>{const b=$('#start-experiment');b.disabled=true;try{let sweep=null;if($('#sweep-field').value){const values=$('#sweep-values').value.split(',').map(s=>Number(s.trim()));if(!$('#sweep-values').value.trim()||values.some(v=>!Number.isFinite(v)))throw new Error('Enter numeric, comma-separated sweep values.');sweep={field:$('#sweep-field').value,values};}await launchJob($('#experiment-kind').value,$('#experiment-results'),Number($('#experiment-episodes').value),['ceap','immediate','fixed','threshold','weakest'],sweep);}catch(e){toast(e.message,true);}finally{b.disabled=false;}};
document.addEventListener('change',e=>{if(e.target.dataset.chartMetric){const info=state.jobs.get(e.target.dataset.chartMetric);info.container.innerHTML=jobHTML(info.job,e.target.value);}});
$('#refresh-history').onclick=loadHistory;

async function init(){
  try{
    const data=await api('/api/presets');state.presets=data.presets;state.policies=data.policies;
    $('#preset').innerHTML=data.presets.map(p=>`<option value="${p.id}">${escapeHTML(p.name)}</option>`).join('');$('#preset-description').textContent=data.presets[0].description;
    $('#policy').innerHTML=Object.entries(data.policies).map(([id,label])=>`<option value="${id}">${escapeHTML(label)}</option>`).join('');
    $('#policy-options').innerHTML=Object.entries(data.policies).filter(([id])=>id!=='oracle').map(([id,label])=>`<label class="policy-option"><input type="checkbox" value="${id}" ${id!=='ablation'?'checked':''}>${escapeHTML(label)}</label>`).join('');
    let config=data.presets[0].config;try{const saved=JSON.parse(localStorage.getItem('ceap-last-config'));if(saved?.links)config=saved;}catch{}
    populate(config);$('#connection').textContent='Local engine connected';
    const hash=location.hash.slice(1);if(hash)view(hash);
    const id=localStorage.getItem('ceap-active-episode');if(id){try{state.episode=await api(`/api/episodes/${id}`);populate(state.episode.config);renderEpisode();if(state.episode.running)connectEpisode();}catch{localStorage.removeItem('ceap-active-episode');}}
    for(const target of ['comparison-results','experiment-results']){const jobId=localStorage.getItem(`ceap-${target}`);if(jobId){try{const job=await api(`/api/jobs/${jobId}`);renderJob(job,$(`#${target}`));if(!endStates.includes(job.status))watchJob(jobId,$(`#${target}`));}catch{}}}
  }catch(error){$('#connection').textContent='Engine unavailable';toast(`Cannot reach the Python engine: ${error.message}`,true);}
}
init();
