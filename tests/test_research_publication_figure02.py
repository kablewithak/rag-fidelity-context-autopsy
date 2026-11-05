"""Static and contract checks for the read-only Figure 02 publication extension."""
from __future__ import annotations

import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / 'design' / 'research_publication_gate01'


class _Ids(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get('id'):
            self.ids.append(str(attributes['id']))
        if tag == 'a' and attributes.get('href'):
            self.links.append(str(attributes['href']))


def test_figure02_navigation_and_stage_ids_are_unique() -> None:
    source = (PUBLICATION / 'index.html').read_text(encoding='utf-8')
    parser = _Ids()
    parser.feed(source)
    assert len(parser.ids) == len(set(parser.ids))
    assert {'figure-one', 'figure-two', 'method', 'journey-stages', 'journey-pipelines', 'journey-stage-heading', 'journey-stage-explanation', 'journey-stage-basis'} <= set(parser.ids)
    assert '#figure-one' in parser.links
    assert '#figure-two' in parser.links
    assert '#method' in parser.links
    assert 'FIGURE <strong>02</strong>' in source


def test_figure02_uses_reviewed_projection_not_new_model_calls() -> None:
    source = (PUBLICATION / 'app.js').read_text(encoding='utf-8')
    assert 'function stageModel(c,state)' in source
    assert 'function renderJourney()' in source
    assert 'const state=c.states[selectedPipelineIndex]' in source
    assert "Source: loss_stage = ranking" in source
    assert "gold_evidence_included = true" in source
    assert 'retrieved_gold_rank' in source
    assert 'reranked_gold_rank' in source
    assert 'window.fetch(' not in source
    assert 'fetch(' not in source
    assert 'innerHTML=c.query' not in source
    assert 'textContent=c.query' in source


def test_figure02_preserves_gate01_and_accessibility_states() -> None:
    html = (PUBLICATION / 'index.html').read_text(encoding='utf-8')
    script = (PUBLICATION / 'app.js').read_text(encoding='utf-8')
    css = (PUBLICATION / 'styles.css').read_text(encoding='utf-8')
    assert 'id="case-rows"' in html
    assert 'id="toggle-filter"' in html
    assert 'id="case-path"' in html
    assert 'aria-label="Choose a reviewed retrieval configuration"' in html
    assert 'aria-label="Inspect evidence lifecycle stages"' in html
    assert "setAttribute('aria-pressed'" in script
    assert 'selectedStageIndex=defaultJourneyStage(' in script
    assert 'prefers-reduced-motion:reduce' in css
    assert 'position:sticky' in css
    assert '--nav-height' in css
    assert '.journey-node.active{border-top-color:var(--ink);background:transparent}' in css
    assert '.journey-node.active .journey-node-title{' in css
    assert 'text-decoration-color:var(--clay)' in css
    assert 'background:#f1f0e9' not in css


def test_figure02_real_case_logic_from_reviewed_projection() -> None:
    """Optional JavaScript runtime check; Python-only environments still run the other tests."""
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is not a project runtime requirement; browser interaction test covered separately')
    program = r'''
const fs=require('fs');
const vm=require('vm');
const ctx=vm.createContext({document:{readyState:'loading',addEventListener:()=>{}},URL});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),ctx);
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),ctx);
const verdict=vm.runInContext(`(() => {
  assertEvidenceProjection();
  const byId=(id)=>CASES.find(c=>c.id===id);
  const conf=byId('legal_confidentiality_003');
  const multilingual=byId('multilingual_security_012');
  const enterprise=byId('support_enterprise_s1_006');
  if(stageModel(conf,conf.states[0])[1].kind!=='loss')return false;
  if(stageModel(conf,conf.states[1])[3].kind!=='loss')return false;
  if(stageModel(conf,conf.states[3])[3].value!=='#1')return false;
  if(stageModel(multilingual,multilingual.states[0])[2].kind!=='loss')return false;
  if(stageModel(enterprise,enterprise.states[0])[2].value!=='#4')return false;
  if(stageModel(enterprise,enterprise.states[0])[4].kind!=='loss')return false;
  return CASES.every(c=>c.states.every(s=>{
    const stages=stageModel(c,s);
    return stages.length===5 && stages[4].kind===(s[0]?'included':'loss');
  }));
})()`,ctx);
if(!verdict)process.exit(1);
'''
    result = subprocess.run(
        [node, '-e', program, str(PUBLICATION / 'data.js'), str(PUBLICATION / 'app.js')],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
