"""Validate the saved GPT-2 router locally, without cloud correction."""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
os.environ['TIKTOKEN_CACHE_DIR']=str(ROOT/'data/tokenizer_cache')
os.environ['ORCH_ENABLE_ROUTE_DEEPSEEK_CORRECTION']='false'
os.environ['ORCH_FORCE_ROUTE_DEEPSEEK_CORRECTION']='false'
import torch
from agents.router_agent import create_router_agent
torch.set_num_threads(8)

def main():
    router=create_router_agent(checkpoint_path=str(ROOT/'checkpoints/gpt2_124M_router_2stage.pth'),enable_deepseek_correction=False,force_deepseek_correction=False)
    data=json.loads((ROOT/'eval/val_large.json').read_text(encoding='utf8'))
    results=[];start=time.perf_counter()
    for i,item in enumerate(data):
        route=router.route(item['question'],use_deepseek_correction=False,force_correction=False)
        results.append({'id':item.get('id'),'question':item['question'],'category':item['category'],'route':route.model_dump()})
        if i%50==0:print(f'Routed {i}/{len(data)}',flush=True)
    correct=sum(r['category']==r['route']['final_label'] for r in results)
    summary={'samples':len(results),'correct':correct,'accuracy':correct/len(results),'seconds':time.perf_counter()-start,
        'device':str(router.router.device),'cloud_correction':False,'evaluation_scope':'Existing course evaluation corpus; train/test disjointness not established',
        'counts':{label:sum(r['route']['final_label']==label for r in results) for label in ['RAG_ONLY','WEB_ONLY','HYBRID','REFUSE']}}
    out=ROOT/'validation';out.mkdir(exist_ok=True)
    (out/'router_predictions.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'router_run.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
