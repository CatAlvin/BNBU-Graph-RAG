import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime import configure
configure()
import torch
torch.set_num_threads(8)
from agents.orchestrator import Orchestrator
from agents.aggregator_judge_agent import create_aggregator_judge_agent
def main():
    orch=Orchestrator(checkpoint_path=str(ROOT/'checkpoints/gpt2_124M_router_2stage.pth'),
        aggregator=create_aggregator_judge_agent(use_deepseek_final=False),enable_route_deepseek_correction=False,
        force_route_deepseek_correction=False)
    q='When does the sample campus library close on weekdays?'
    original_route=orch.router_agent.route(q,use_deepseek_correction=False,force_correction=False)
    start=time.perf_counter()
    result=orch.run(q,route_override='RAG_ONLY',use_route_deepseek_correction=False,force_route_correction=False)
    report={'mode':'Live local RAG demonstration; route override RAG_ONLY; no Web/API call',
            'question':q,'original_route':original_route.model_dump(),'elapsed_seconds':time.perf_counter()-start,'result':result.model_dump()}
    out=ROOT/'validation';out.mkdir(exist_ok=True)
    (out/'live_rag.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if result.error_stage:raise SystemExit('Live RAG returned an error stage: '+result.error_stage)
    if not result.rag_pack or not result.rag_pack.claims:raise SystemExit('No evidence-backed claims returned')
    assert any(e.source == 'sample-campus.pdf' and e.page == 1 for c in result.rag_pack.claims for e in c.evidence), 'Missing source-page provenance'
    assert '22:00' in result.final_output.final_answer, 'Answer did not reproduce the supplied library hours'
if __name__=='__main__':main()
