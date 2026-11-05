'use strict';
const LOSS_LABELS = Object.freeze({chunking:'Chunking loss',retrieval:'Retrieval loss',ranking:'Ranking loss'});
const LOSS_DESCRIPTIONS = Object.freeze({
  chunking:'Complete gold evidence was not preserved as a usable chunk.',
  retrieval:'Complete gold evidence was absent from the retained retrieval candidates.',
  ranking:'Evidence reached the candidate set but was not selected for final context.'
});
const $ = (id)=>document.getElementById(id);
let selectedId='legal_confidentiality_003';
let baselineOnly=false;
let selectedPipelineIndex=0;
let selectedStageIndex=2;
const orderedCases=[...CASES].sort((a,b)=>Number(a.states[0][0])-Number(b.states[0][0])||CASES.indexOf(a)-CASES.indexOf(b));

function assertEvidenceProjection(){
  if(CASES.length!==18 || PIPELINES.length!==4 || new Set(CASES.map(c=>c.id)).size!==18)throw new Error('Reviewed figure coverage mismatch');
  if(CASES.some(c=>c.states.length!==4||c.states.some(s=>s.length!==4||(s[0]===1&&s[3]!==null)||(s[0]===0&&!LOSS_LABELS[s[3]])||(s[0]===0&&s[3]==='ranking'&&s[1]===null))))throw new Error('Reviewed state contract mismatch');
  const expected=[13,16,18,18];
  const actual=PIPELINES.map((p,i)=>CASES.reduce((n,c)=>n+c.states[i][0],0));
  if(actual.some((n,i)=>n!==expected[i]))throw new Error('Reviewed evidence inclusion totals mismatch');
  const losses=CASES.filter(c=>c.states[0][0]===0).map(c=>c.states[0][3]);
  if(losses.length!==5||losses.filter(x=>x==='chunking').length!==3||losses.filter(x=>x==='retrieval').length!==1||losses.filter(x=>x==='ranking').length!==1)throw new Error('Reviewed loss taxonomy mismatch');
  return true;
}
function lossClass(state){return state[0]?'included':state[3]}
function stateLabel(state){return state[0]?'Evidence included':LOSS_LABELS[state[3]]}
function stateDetails(state,pipelineIndex){
  const bits=[];
  if(state[1]!==null)bits.push(`first-stage rank #${state[1]}`);
  if(state[2]!==null)bits.push(`reranked rank #${state[2]}`);
  if(state[0])bits.push('complete evidence reached final selection');
  else bits.push(LOSS_DESCRIPTIONS[state[3]]);
  return bits.join(' · ');
}
function markHtml(state){return `<span class="mark ${lossClass(state)}" aria-hidden="true"></span>`}
function addText(tag,className,text,parent){const node=document.createElement(tag);node.className=className;node.textContent=text;parent.appendChild(node);return node}
function renderRows(){
  const tbody=$('case-rows');tbody.replaceChildren();
  const shown=orderedCases.filter(c=>!baselineOnly||c.states[0][0]===0);
  for(const [idx,c] of shown.entries()){
    const tr=document.createElement('tr');tr.dataset.case=c.id;
    const previous=shown[idx-1];
    if(previous&&previous.states[0][0]===0&&c.states[0][0]===1)tr.classList.add('group-start');
    if(c.id===selectedId)tr.classList.add('selected');
    const th=document.createElement('th');th.scope='row';
    const button=document.createElement('button');button.type='button';button.className='case-select';button.dataset.select=c.id;
    button.setAttribute('aria-pressed',String(c.id===selectedId));
    button.setAttribute('aria-label',`Inspect case ${c.label}`);
    addText('span','case-label',c.label,button);
    addText('span','row-no',String(CASES.indexOf(c)+1).padStart(2,'0'),button);
    th.appendChild(button);tr.appendChild(th);
    for(const [i,state] of c.states.entries()){
      const td=document.createElement('td');
      td.innerHTML=markHtml(state);
      td.setAttribute('aria-label',`${PIPELINES[i].name}: ${stateLabel(state)}${state[1]!==null?`; first-stage rank ${state[1]}`:''}`);
      td.title=`${PIPELINES[i].name}: ${stateLabel(state)}`;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  $('view-count').textContent=`Showing ${shown.length} of 18 fixed cases`;
  const filter=$('toggle-filter');filter.setAttribute('aria-pressed',String(baselineOnly));
  filter.innerHTML=baselineOnly?'Show all eighteen cases <span aria-hidden="true">↗</span>':'Focus on the five baseline losses <span aria-hidden="true">↗</span>';
}
function interpretationFor(c){
  const stages=c.states.map(s=>lossClass(s));
  if(stages.every(s=>s==='included'))return 'All four fixed configurations retained the complete gold evidence. This case does not establish that any generated answer was correct.';
  if(stages[0]==='chunking'&&stages[1]==='ranking')return 'The failure mechanism changes across configurations: the baseline loses complete evidence at chunking; token + dense finds it but does not select it. Hybrid retrieval retains it, and reranking moves the recorded rank from #3 to #1.';
  if(stages[0]==='retrieval'&&stages[1]==='ranking')return 'The baseline misses the complete evidence at retrieval. Token + dense retrieves it at rank #7 but still excludes it. Hybrid retrieval retains it; the reranker moves the recorded rank from #3 to #1.';
  if(stages[0]==='ranking')return 'The baseline retrieves the relevant evidence, but the selection boundary excludes it. Subsequent reviewed configurations include the complete evidence.';
  if(stages[0]==='chunking')return 'The baseline loses the complete evidence at the chunking boundary. All three subsequent reviewed configurations include it.';
  return 'The configurations show different evidence-selection outcomes for the same fixed diagnostic case.';
}
function renderInspector(){
  const c=CASES.find(x=>x.id===selectedId);if(!c)return;
  $('case-family').textContent=c.group.toUpperCase();
  $('inspector-title').textContent=c.label;
  $('case-question').textContent=c.query;
  $('case-sequence').textContent=`${String(CASES.indexOf(c)+1).padStart(2,'0')} / 18`;
  $('case-interpretation').textContent=interpretationFor(c);
  $('case-identifier').textContent=c.id;
  const path=$('case-path');path.replaceChildren();
  for(const [i,state] of c.states.entries()){
    const row=addText('div','path-row','',path);
    const mark=addText('span','path-mark','',row); mark.innerHTML=markHtml(state);
    const body=addText('div','path-content','',row);
    addText('span','path-label',PIPELINES[i].name+' · '+stateLabel(state),body);
    addText('span','path-explanation',stateDetails(state,i),body);
  }
}
function selectCase(id,options={}){
  if(!CASES.some(c=>c.id===id))return;
  if(baselineOnly&&CASES.find(c=>c.id===id).states[0][0]===1){baselineOnly=false;}
  selectedId=id;
  selectedStageIndex=defaultJourneyStage(CASES.find(c=>c.id===selectedId).states[selectedPipelineIndex]);
  renderRows();renderInspector();renderJourney();
  if(options.updateUrl!==false){
    try{const u=new URL(window.location.href);u.searchParams.set('case',id);history.replaceState(null,'',u.href)}catch(err){/* direct local files remain usable */}
  }
}
function init(){
  assertEvidenceProjection();
  const initial=new URL(window.location.href).searchParams.get('case');
  if(initial&&CASES.some(c=>c.id===initial))selectedId=initial;
  selectedStageIndex=defaultJourneyStage(CASES.find(c=>c.id===selectedId).states[selectedPipelineIndex]);
  assertFigure03Metrics();assertFigure04Proof();assertFigure05Public();renderRows();renderInspector();renderJourney();renderComparison();renderComparisonSteps();renderBudget();renderTransfer();
  $('case-rows').addEventListener('click',event=>{
    const button=event.target.closest('button[data-select]');
    if(button){selectCase(button.dataset.select);return}
    const row=event.target.closest('tr[data-case]');if(row)selectCase(row.dataset.case);
  });
  $('toggle-filter').addEventListener('click',()=>{
    baselineOnly=!baselineOnly;
    if(baselineOnly&&CASES.find(c=>c.id===selectedId).states[0][0]===1){
      selectedId='legal_confidentiality_003';
      selectedStageIndex=defaultJourneyStage(CASES.find(c=>c.id===selectedId).states[selectedPipelineIndex]);
    }
    renderRows();renderInspector();renderJourney();
  });
  $('journey-pipelines').addEventListener('click',event=>{
    const button=event.target.closest('button[data-pipeline]');
    if(!button)return;
    selectedPipelineIndex=Number(button.dataset.pipeline);
    selectedStageIndex=defaultJourneyStage(CASES.find(c=>c.id===selectedId).states[selectedPipelineIndex]);
    renderJourney();
  });
  $('comparison-plots').addEventListener('click',event=>{
    const button=event.target.closest('button[data-figure03]');
    if(!button)return;
    comparisonSelectedIndex=Number(button.dataset.figure03);
    renderComparison();
  });
  $('budget-selectors').addEventListener('click',event=>{
    const button=event.target.closest('button[data-budget]');
    if(!button)return;
    budgetSelectedIndex=Number(button.dataset.budget);
    renderBudget();
  });
  $('transfer-pipeline-rows').addEventListener('click',event=>{
    const button=event.target.closest('button[data-transfer]');
    if(!button)return;
    transferSelectedIndex=Number(button.dataset.transfer);
    renderTransfer();
  });
  $('journey-stages').addEventListener('click',event=>{
    const button=event.target.closest('button[data-stage]');
    if(!button)return;
    selectedStageIndex=Number(button.dataset.stage);
    renderJourney();
  });
}


// Figure 02 is a deliberately bounded explanation of one reviewed outcome.
// It does not synthesize intermediate traces from missing candidate/chunk payloads.
const JOURNEY_LABELS=Object.freeze(['Source','Chunking','Retrieval','Ranking','Final context']);
function defaultJourneyStage(state){
  const loss=state[3];
  if(loss==='chunking')return 1;
  if(loss==='retrieval')return 2;
  if(loss==='ranking')return 3;
  return 4;
}
function stageModel(c,state){
  const [included,firstRank,rerankRank,loss]=state;
  const noRank=firstRank===null;
  return [
    {kind:'fixture',value:'Declared',title:'A fixed reference question',detail:'This synthetic evaluation case declares a known evidence span and asks whether the complete span reaches final context selection.',basis:'Source: fixed evaluation case definition. The publication does not show the source passage.'},
    loss==='chunking'
      ? {kind:'loss',value:'Loss recorded',title:'Complete evidence was lost at chunking',detail:'The reviewed outcome classifies the loss as chunking. Complete gold evidence was not preserved as a usable chunk for this pipeline.',basis:'Source: reviewed loss_stage = chunking. Individual emitted chunks are not displayed.'}
      : {kind:'neutral',value:'No loss recorded',title:'Chunking is not the classified loss',detail:'The reviewed outcome does not classify this case as a chunking loss. This is not a separate measured pass-rate for chunking.',basis:'Source: absence of a chunking loss classification in the reviewed outcome.'},
    noRank
      ? loss==='retrieval'
        ? {kind:'loss',value:'Miss recorded',title:'Complete evidence missed the candidate set',detail:'The reviewed outcome classifies the loss as retrieval. No first-stage rank is recorded for complete gold evidence.',basis:'Source: loss_stage = retrieval and retrieved_gold_rank = null.'}
        : {kind:'unavailable',value:'Rank unavailable',title:'No complete-evidence rank is available',detail:'The reviewed outcome records no first-stage rank for complete evidence. Do not infer the ranking of partial evidence.',basis:'Source: retrieved_gold_rank = null; the recorded loss occurred earlier.'}
      : {kind:'measured',value:'#'+firstRank,title:'First-stage evidence position',detail:'Complete gold evidence entered the retained candidate set at first-stage rank #'+firstRank+'. Candidate contents and similarity scores are not part of this public artifact.',basis:'Source: retrieved_gold_rank = '+firstRank+'. Rank refers to complete gold evidence, not a generated answer.'},
    loss==='ranking'
      ? {kind:'loss',value:'Loss recorded',title:'Evidence did not survive final ranking selection',detail:'The reviewed outcome classifies this as a ranking loss after first-stage retrieval at rank #'+firstRank+'. The record does not provide a separate ranked-candidate narrative.',basis:'Source: loss_stage = ranking; retrieved_gold_rank = '+firstRank+'.'}
      : rerankRank!==null
        ? {kind:'measured',value:'#'+rerankRank,title:'Rank after the reranker',detail:'The first-stage rank was #'+firstRank+'; the recorded reranked position was #'+rerankRank+'. This comparison shows positions, not candidate scores or model reasoning.',basis:'Source: retrieved_gold_rank = '+firstRank+'; reranked_gold_rank = '+rerankRank+'.'}
        : noRank
          ? {kind:'unavailable',value:'Not available',title:'No full-evidence rank can be reported',detail:'There is no recorded first-stage rank for complete evidence. The publication does not invent a ranking stage.',basis:'Source: retrieved_gold_rank = null; reranked_gold_rank = null.'}
          : {kind:'neutral',value:'No reranker',title:'First-stage order supplies the recorded position',detail:'This configuration has no reranked position. The reviewed first-stage rank is #'+firstRank+'.',basis:'Source: retrieved_gold_rank = '+firstRank+'; reranked_gold_rank = null.'},
    included
      ? {kind:'included',value:'Included',title:'Complete evidence reached final context selection',detail:'The complete gold evidence was included in the selected context. This does not establish that the model produced a correct or grounded answer.',basis:'Source: gold_evidence_included = true; loss_stage = null.'}
      : {kind:'loss',value:'Not included',title:'Complete evidence did not reach final context',detail:'The case is excluded from final evidence selection. The reviewed classification attributes the loss to '+loss+', not to an invented additional context drop.',basis:'Source: gold_evidence_included = false; loss_stage = '+loss+'.'}
  ];
}
function renderJourney(){
  const c=CASES.find(x=>x.id===selectedId);
  if(!c)return;
  const state=c.states[selectedPipelineIndex];
  const stages=stageModel(c,state);
  $('journey-case-title').textContent=c.label;
  $('journey-case-question').textContent=c.query;
  const configurations=$('journey-pipelines');configurations.replaceChildren();
  PIPELINES.forEach((p,i)=>{
    const button=addText('button','journey-pipeline'+(i===selectedPipelineIndex?' active':''),'',configurations);
    button.type='button';button.dataset.pipeline=String(i);
    button.setAttribute('aria-pressed',String(i===selectedPipelineIndex));
    addText('span','journey-pipeline-no',String(i+1).padStart(2,'0'),button);
    addText('span','journey-pipeline-name',p.name,button);
  });
  let headline=state[0] ? 'The complete evidence survived this configuration.' : 'The complete evidence was lost at '+state[3]+'.';
  $('journey-introduction').textContent=headline+' The selected path contains only recorded ranks and a bounded reading of the reviewed outcome.';
  const canvas=$('journey-stages');canvas.replaceChildren();
  stages.forEach((stage,i)=>{
    const node=addText('button','journey-node '+stage.kind+(i===selectedStageIndex?' active':''),'',canvas);
    node.type='button';node.dataset.stage=String(i);node.setAttribute('aria-pressed',String(i===selectedStageIndex));
    node.setAttribute('aria-label',JOURNEY_LABELS[i]+': '+stage.value+'. Inspect stage');
    const number=addText('span','journey-node-number',String(i+1).padStart(2,'0'),node);
    const indicator=addText('span','journey-indicator','',node);indicator.setAttribute('aria-hidden','true');
    addText('span','journey-node-title',JOURNEY_LABELS[i],node);
    addText('span','journey-node-value',stage.value,node);
  });
  const chosen=stages[selectedStageIndex];
  $('journey-stage-number').textContent=String(selectedStageIndex+1).padStart(2,'0')+' / 05';
  $('journey-stage-heading').textContent=chosen.title;
  $('journey-stage-explanation').textContent=chosen.detail;
  $('journey-stage-basis').textContent=chosen.basis;
}


// Figure 03 uses an independently validated, public, aggregate-only export.
const METRIC_SERIES = Object.freeze([
  {key:'inclusion',title:'Evidence inclusion',note:'Did complete gold evidence reach final context?',counted:true},
  {key:'recallAt5',title:'Recall@5',note:'Was complete evidence retrieved in the first five?',counted:true},
  {key:'mrrAt10',title:'MRR@10',note:'How high was complete evidence ranked?',counted:false}
]);
let comparisonSelectedIndex=3;
function assertFigure03Metrics(){
  const m=FIGURE03_METRICS;
  if(m.artifactId!==DATA_PROVENANCE.artifactId||m.runId!==DATA_PROVENANCE.runId||m.retrievalMetricK!==5)throw new Error('Figure 03 benchmark identity mismatch');
  if(!Array.isArray(m.rows)||m.rows.length!==PIPELINES.length||m.rows.some((r,i)=>r.id!==PIPELINES[i].id))throw new Error('Figure 03 pipeline order mismatch');
  for(const [i,r] of m.rows.entries()){
    for(const key of ['inclusion','recallAt5']){
      const v=r[key];
      if(!v||v.denominator!==18||!Number.isInteger(v.numerator)||v.numerator<0||v.numerator>18||!Number.isFinite(v.value)||Math.abs(v.value-v.numerator/18)>1e-12)throw new Error('Figure 03 invalid count metric');
    }
    if(!Number.isFinite(r.mrrAt10)||r.mrrAt10<0||r.mrrAt10>1)throw new Error('Figure 03 invalid rank metric');
    if(r.inclusion.numerator!==CASES.reduce((n,c)=>n+c.states[i][0],0))throw new Error('Figure 03 inclusion differs from Figure 01');
    if(r.recallAt5.numerator!==CASES.filter(c=>c.states[i][1]!==null&&c.states[i][1]<=5).length)throw new Error('Figure 03 recall differs from reviewed case ranks');
  }
  if(m.rows.map(x=>x.inclusion.numerator).join()!=='13,16,18,18'||m.rows.map(x=>x.recallAt5.numerator).join()!=='14,17,18,18')throw new Error('Frozen Figure 03 metric progression changed');
  return true;
}
function metricValue(row,key){return key==='mrrAt10'?row.mrrAt10:row[key].value}
function metricLabel(row,key){return key==='mrrAt10'?row.mrrAt10.toFixed(3):`${(row[key].value*100).toFixed(1)}%`}
function countLabel(row,key){return key==='mrrAt10'?row.mrrAt10.toFixed(3):`${row[key].numerator}/18`}
function deltaLabel(amount,precision=1){const prefix=amount>0?'+':'';return prefix+amount.toFixed(precision)}
function renderComparison(){
  const plots=$('comparison-plots');plots.replaceChildren();
  for(const metric of METRIC_SERIES){
    const panel=addText('section','comparison-panel','',plots);
    const heading=addText('div','comparison-panel-head','',panel);
    addText('h3','comparison-panel-title',metric.title,heading);
    addText('p','comparison-panel-note',metric.note,heading);
    const wrap=addText('div','comparison-chart-wrap','',panel);
    const chart=addText('div','comparison-chart','',wrap);
    for(const [i,row] of FIGURE03_METRICS.rows.entries()){
      const button=addText('button','comparison-point'+(comparisonSelectedIndex===i?' active':''),'',chart);
      button.type='button';button.dataset.figure03=String(i);
      button.setAttribute('aria-pressed',String(comparisonSelectedIndex===i));
      button.setAttribute('aria-label',`${PIPELINES[i].name}: ${metric.title} ${metricLabel(row,metric.key)}${metric.counted?', '+countLabel(row,metric.key):''}. Compare this configuration`);
      addText('span','comparison-point-code',PIPELINES[i].short,button);
      const ruler=addText('span','comparison-point-track','',button);
      const dot=addText('span','comparison-point-dot','',ruler);dot.setAttribute('aria-hidden','true');
      ruler.style.setProperty('--score',String((metricValue(row,metric.key)*100).toFixed(4)));
      addText('span','comparison-point-value',metric.counted?countLabel(row,metric.key):metricLabel(row,metric.key),button);
    }
    const axis=addText('div','comparison-ruler','',wrap);
    for(const tick of ['0','50','100'])addText('span','',tick,axis);
  }
  const row=FIGURE03_METRICS.rows[comparisonSelectedIndex];
  $('comparison-active-title').textContent=PIPELINES[comparisonSelectedIndex].name;
  const previous=FIGURE03_METRICS.rows[0];
  const pp=(row.inclusion.value-previous.inclusion.value)*100;
  $('comparison-active-copy').textContent=`${row.inclusion.numerator} of 18 cases retain complete evidence; Recall@5 is ${row.recallAt5.numerator} of 18 and MRR@10 is ${row.mrrAt10.toFixed(3)}. Inclusion is ${deltaLabel(pp)} percentage points relative to character + dense. This does not measure final-answer grounding.`;
}
function renderComparisonSteps(){
  const steps=$('comparison-step-list');steps.replaceChildren();
  const names=['Adopt token-aware chunking','Introduce hybrid retrieval','Add reranking and budget-aware selection'];
  const notes=[
    'The fixed configuration comparison has three additional included cases; two ranking losses remain.',
    'Complete evidence is included in all eighteen fixed cases in this configuration.',
    'Evidence inclusion and Recall@5 stay unchanged while the aggregate rank metric rises. The final configuration also changes context selection.'
  ];
  for(let i=1;i<4;i++){
    const before=FIGURE03_METRICS.rows[i-1],after=FIGURE03_METRICS.rows[i];
    const el=addText('div','comparison-step','',steps);
    addText('span','comparison-step-index',String(i).padStart(2,'0'),el);
    const content=addText('div','','',el);
    addText('h3','',names[i-1],content);
    addText('p','comparison-delta',`Evidence included: ${before.inclusion.numerator} → ${after.inclusion.numerator} (+${after.inclusion.numerator-before.inclusion.numerator} cases) · Recall@5: ${before.recallAt5.numerator} → ${after.recallAt5.numerator} · MRR@10: ${before.mrrAt10.toFixed(3)} → ${after.mrrAt10.toFixed(3)}.`,content);
    addText('p','',notes[i-1],content);
  }
}

// Figure 04 is independent of the 18-case benchmark. Its measured mode is
// generated by Python's validated context-pressure explorer, not by the UI.
let budgetSelectedIndex=0;
const BUDGET_LABELS=Object.freeze({verbose_audit:'Verbose audit wrappers',compact_citation:'Compact citation wrappers'});
const BUDGET_SEGMENTS=Object.freeze([
  {field:'staticPromptTokens',name:'Static prompt',cls:'static'},
  {field:'rawEvidenceTokens',name:'Raw evidence',cls:'evidence'},
  {field:'wrapperTaxTokens',name:'Wrapper overhead',cls:'overhead'},
  {field:'remainingTokens',name:'Unused capacity',cls:'unused'},
  {field:'reservedOutputTokens',name:'Output reserve',cls:'reserve'}
]);
function assertFigure04Proof(){
  const v=FIGURE04_PROOF;
  if(v.schemaVersion!=='figure04_context_budget_v1'||v.caseId!=='token_context_notice_018'||v.tokenizer!=='tiktoken:cl100k_base'||v.windowTokens!==468||v.reservedOutputTokens!==120||v.goldRank!==3||v.scope!=='controlled_context_pressure_not_four_pipeline_benchmark')throw new Error('Figure 04 fixed-experiment boundary mismatch');
  if(!Array.isArray(v.profiles)||v.profiles.length!==2||v.profiles[0].id!=='verbose_audit'||v.profiles[1].id!=='compact_citation'||v.profiles[0].wrapperTaxTokens!==77||v.profiles[1].wrapperTaxTokens!==25||v.profiles[0].goldIncluded!==false||v.profiles[1].goldIncluded!==true||v.profiles[0].goldDropReason!=='budget_exhausted')throw new Error('Figure 04 reported controlled findings changed');
  if(v.status==='preview')return true; // review HTML only; CI exporter --check rejects preview asset.
  if(v.status!=='measured')throw new Error('Figure 04 unknown evidence state');
  for(const p of v.profiles){
    const total=p.staticPromptTokens+p.rawEvidenceTokens+p.wrapperTaxTokens+p.remainingTokens+v.reservedOutputTokens;
    if(![p.staticPromptTokens,p.rawEvidenceTokens,p.wrapperTaxTokens,p.remainingTokens].every(n=>Number.isInteger(n)&&n>=0)||total!==v.windowTokens)throw new Error('Figure 04 context budget does not balance');
    if(!Array.isArray(p.decisions)||p.decisions.length<3||p.decisions.length>20||p.decisions.filter(d=>d.isGold).length!==1)throw new Error('Figure 04 invalid decision coverage');
    if(p.decisions.find(d=>d.isGold).rank!==3||p.decisions.find(d=>d.isGold).included!==p.goldIncluded)throw new Error('Figure 04 gold candidate mismatch');
    if(p.decisions.filter(d=>d.included).reduce((n,d)=>n+d.rawTokens,0)!==p.rawEvidenceTokens||p.decisions.filter(d=>d.included).reduce((n,d)=>n+d.wrapperTokens,0)!==p.wrapperTaxTokens)throw new Error('Figure 04 candidate totals mismatch');
  }
  return true;
}
function renderBudget(){
  assertFigure04Proof();
  const v=FIGURE04_PROOF;
  const measured=v.status==='measured';
  const rows=$('budget-profile-rows');rows.replaceChildren();
  for(const [i,p] of v.profiles.entries()){
    const row=addText('div','budget-profile-row','',rows);
    const headline=addText('div','budget-row-heading','',row);
    addText('strong','budget-row-label',BUDGET_LABELS[p.id],headline);
    addText('span','budget-row-outcome',p.goldIncluded?'Gold evidence included':'Gold evidence dropped',headline);
    const bar=addText('div','budget-bar','',row);
    bar.setAttribute('role','img');
    if(measured){
      bar.setAttribute('aria-label',`${BUDGET_LABELS[p.id]}: ${p.staticPromptTokens} static prompt, ${p.rawEvidenceTokens} raw evidence, ${p.wrapperTaxTokens} wrapper overhead, ${p.remainingTokens} unused, ${v.reservedOutputTokens} output reserve; total ${v.windowTokens} tokens`);
      for(const spec of BUDGET_SEGMENTS){
        const value=spec.field==='reservedOutputTokens'?v.reservedOutputTokens:p[spec.field];
        const segment=addText('span','budget-segment '+spec.cls,'',bar);
        segment.style.width=(value/v.windowTokens*100)+'%';
        segment.title=`${spec.name}: ${value} tokens`;
      }
    }else{
      bar.classList.add('budget-bar-preview');
      bar.setAttribute('aria-label',`Preview: ${v.windowTokens} token window with ${v.reservedOutputTokens} reserved for output; full measured allocation requires local export`);
      const left=addText('span','budget-segment preview','',bar);left.style.width=((v.windowTokens-v.reservedOutputTokens)/v.windowTokens*100)+'%';
      const reserve=addText('span','budget-segment reserve','',bar);reserve.style.width=(v.reservedOutputTokens/v.windowTokens*100)+'%';
    }
    const below=addText('div','budget-row-foot','',row);
    addText('span','',`Wrapper overhead: ${p.wrapperTaxTokens} tokens`,below);
    addText('span','',measured?`${p.remainingTokens} tokens unused`:'Full allocation pending verified export',below);
  }
  $('budget-measurement-note').textContent=measured?
    'All segments are measured and sum to 468 tokens per profile. Wrapper tax is included within rendered evidence cost; it is not an additional capacity outside the window.':
    'Design preview: only the documented 468-token window, 120-token output reserve, 77/25-token wrapper overhead, and gold outcomes are shown. The complete measured allocation is exported locally before merge.';
  const selectors=$('budget-selectors');selectors.replaceChildren();
  v.profiles.forEach((p,i)=>{
    const b=addText('button','budget-selector'+(i===budgetSelectedIndex?' selected':''),'',selectors);
    b.type='button';b.dataset.budget=String(i);b.setAttribute('aria-pressed',String(i===budgetSelectedIndex));
    addText('span','budget-selector-number','0'+(i+1),b);addText('span','',BUDGET_LABELS[p.id],b);
  });
  const p=v.profiles[budgetSelectedIndex];
  const head=$('budget-inspection-head');head.replaceChildren();
  addText('span','budget-eyebrow','SELECTED RENDERING PROFILE',head);
  addText('h4','',BUDGET_LABELS[p.id],head);
  addText('p','',p.goldIncluded?'Gold evidence at reranked position #3 reaches the rendered context.':'Gold evidence at reranked position #3 is excluded because the context budget is exhausted.',head);
  const decisions=$('budget-decisions');decisions.replaceChildren();
  if(measured){
    for(const d of p.decisions){
      const row=addText('div','budget-decision'+(d.isGold?' gold':''),'',decisions);
      addText('span','budget-decision-rank','#'+String(d.rank).padStart(2,'0'),row);
      const info=addText('div','budget-decision-info','',row);
      addText('strong','',d.isGold?'Gold evidence candidate':'Candidate '+d.rank,info);
      addText('span','',`${d.renderedTokens} rendered tokens (${d.rawTokens} raw + ${d.wrapperTokens} wrapper)`,info);
      addText('span','budget-decision-state '+(d.included?'included':'dropped'),d.included?'Included':`Dropped: ${d.dropReason.replaceAll('_',' ')}`,row);
    }
  }else{
    addText('p','budget-preview-message','The verified figure will show the individual candidate decisions after running the local exporter. No candidate counts or token costs have been invented for this preview.',decisions);
  }
  $('budget-inspection-foot').textContent=measured?
    `${p.includedCount} candidate(s) included, ${p.droppedCount} dropped. This is deterministic context accounting, not a fresh RAG retrieval run.`:
    'Preview only · requires python scripts/export_research_publication_figure04.py --write before repository validation.';
}

// Figure 05 is an independently reviewed public-corpus probe, NOT a pooled benchmark.
let transferSelectedIndex=3;
const TRANSFER_METRICS = Object.freeze([
  {key:'inclusion',label:'Complete evidence included',description:'Final context selection',counted:true},
  {key:'recallAt5',label:'Recall@5',description:'First-stage candidate retrieval',counted:true},
  {key:'mrrAt10',label:'MRR@10',description:'Evidence ranking quality',counted:false}
]);
function assertFigure05Public(){
  const p=FIGURE05_PUBLIC;
  if(p.schemaVersion!=='figure05_public_transfer_v1'||p.scope!=='separate_reviewed_public_transfer_not_pooled'||p.artifactId!=='public_transfer_squad_v1_dev_v1_reviewed_v1'||p.runId!==p.artifactId||p.datasetId!=='squad_v1.1_dev'||p.datasetVersion!=='1.1'||p.publicCaseCount!==30||p.sourceDocumentCount!==10||p.syntheticCaseCount!==18||p.recallCutoff!==5)throw new Error('Figure 05 public fixture/scope mismatch');
  if(!/^[a-f0-9]{64}$/.test(p.sourceSha256)||!/^[a-f0-9]{64}$/.test(p.fixtureManifestSha256))throw new Error('Figure 05 missing public provenance');
  if(!Array.isArray(p.rows)||p.rows.length!==4||p.rows.some((r,i)=>r.id!==PIPELINES[i].id))throw new Error('Figure 05 pipeline coverage mismatch');
  for(const row of p.rows){
    for(const metric of ['inclusion','recallAt5']){
      const m=row[metric];
      if(!m||m.denominator!==30||!Number.isInteger(m.numerator)||m.numerator<0||m.numerator>30||Math.abs(m.value-m.numerator/30)>1e-12)throw new Error('Figure 05 invalid public count');
    }
    if(!Number.isFinite(row.mrrAt10)||row.mrrAt10<0||row.mrrAt10>1)throw new Error('Figure 05 invalid MRR');
    if(!row.lossStages||Object.values(row.lossStages).some(n=>!Number.isInteger(n)||n<0)||Object.values(row.lossStages).reduce((a,b)=>a+b,0)!==30-row.inclusion.numerator)throw new Error('Figure 05 loss aggregation mismatch');
  }
  if(p.rows.map(r=>r.inclusion.numerator).join()!=='26,27,28,29'||p.rows.map(r=>r.recallAt5.numerator).join()!=='30,29,29,29'||p.rows[3].lossStages.retrieval!==1)throw new Error('Figure 05 reviewed findings changed');
  return true;
}
function transferValue(row,key){return key==='mrrAt10'?row.mrrAt10:row[key].value}
function transferFormat(row,key){return key==='mrrAt10'?row.mrrAt10.toFixed(3):`${row[key].numerator}/30`}
function transferDelta(base,row,key){
  const d=transferValue(row,key)-transferValue(base,key);
  if(key==='mrrAt10')return `${d>=0?'+':''}${d.toFixed(3)}`;
  return `${d>=0?'+':''}${(d*100).toFixed(1)} pp`;
}
function renderTransfer(){
  const all=FIGURE05_PUBLIC.rows, baseline=all[0], current=all[transferSelectedIndex];
  const comps=$('transfer-comparisons');comps.replaceChildren();
  for(const metric of TRANSFER_METRICS){
    const el=addText('section','transfer-metric','',comps);
    const title=addText('div','transfer-metric-heading','',el);
    addText('h4','',metric.label,title);
    addText('span','transfer-metric-delta'+(transferValue(current,metric.key)<transferValue(baseline,metric.key)?' negative':''),transferDelta(baseline,current,metric.key),title);
    addText('p','transfer-metric-subtitle',metric.description,el);
    const tracks=addText('div','transfer-metric-tracks','',el);
    for(const [i,row] of [baseline,current].entries()){
      const item=addText('div','transfer-metric-track','',tracks);
      addText('span','transfer-track-name',i===0?'Baseline':'Inspected',item);
      const line=addText('span','transfer-ruler','',item);
      const dot=addText('span','transfer-ruler-dot'+(i===0?' baseline':''),'',line);
      dot.style.left=(transferValue(row,metric.key)*100)+'%';
      addText('span','transfer-track-value',transferFormat(row,metric.key),item);
      item.setAttribute('aria-label',`${i===0?'Baseline':'Inspected'} ${metric.label}: ${transferFormat(row,metric.key)}`);
    }
    const axis=addText('div','transfer-ticks','',el);
    addText('span','','0',axis);addText('span','','50',axis);addText('span','',metric.key==='mrrAt10'?'1.0':'100%',axis);
  }
  const body=$('transfer-pipeline-rows');body.replaceChildren();
  all.forEach((row,i)=>{
    const tr=addText('tr',i===transferSelectedIndex?'selected':'','',body);
    const th=addText('th','','',tr);th.scope='row';
    const button=addText('button','transfer-row-button','',th);button.type='button';button.dataset.transfer=String(i);
    button.setAttribute('aria-pressed',String(i===transferSelectedIndex));
    addText('span','transfer-row-code',`0${i+1}`,button);
    addText('span','',PIPELINES[i].name,button);
    addText('td','',`${row.inclusion.numerator}/30`,tr);
    addText('td','',`${row.recallAt5.numerator}/30`,tr);
    addText('td','',row.mrrAt10.toFixed(3),tr);
    addText('td','',String(30-row.inclusion.numerator),tr);
  });
  $('transfer-inspector-title').textContent=PIPELINES[transferSelectedIndex].name;
  const inc=current.inclusion.numerator-baseline.inclusion.numerator;
  const rec=current.recallAt5.numerator-baseline.recallAt5.numerator;
  $('transfer-inspector-copy').textContent=`Compared with the public baseline: ${inc>=0?'+':''}${inc} cases include complete evidence; ${rec>=0?'+':''}${rec} cases retain gold evidence in the first five candidates. MRR@10 changes from ${baseline.mrrAt10.toFixed(3)} to ${current.mrrAt10.toFixed(3)}. These are observations within this fixed probe.`;
  const losses=$('transfer-inspector-loss');losses.replaceChildren();
  addText('span','','REMAINING EVIDENCE LOSSES',losses);
  const entries=Object.entries(current.lossStages).filter(([,n])=>n>0);
  addText('p','',entries.length?entries.map(([stage,n])=>`${n} at ${stage.replaceAll('_',' ')}`).join(' · '):'No observed losses at the evidence-selection boundary in this fixed public configuration.',losses);
}

if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
