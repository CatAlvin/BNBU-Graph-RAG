"""Index local PDF pages and extract entities for a bounded subset."""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import fitz
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from agents.rag_specialist_agent import Entities

from runtime import configure
configure()
URI=os.environ.get('NEO4J_URI','bolt://127.0.0.1:17687')
def main():
    driver=GraphDatabase.driver(URI,auth=(os.environ.get('NEO4J_USERNAME','neo4j'),os.environ['NEO4J_PASSWORD']));driver.verify_connectivity()
    embed=OllamaEmbeddings(model='nomic-embed-text:latest')
    chunks=[]
    for pdf_path in sorted((ROOT/'data/documents').glob('*.pdf')):
        filename=pdf_path.name
        doc=fitz.open(pdf_path)
        for page_num,page in enumerate(doc,1):
            text=' '.join(page.get_text().split())
            for start in range(0,len(text),850):
                part=text[start:start+1100]
                if len(part)<40:continue
                identity=hashlib.sha256(f'{filename}:{page_num}:{start}'.encode()).hexdigest()[:20]
                chunks.append({'id':identity,'text':part,'title':f'{filename} page {page_num}',
                               'source':filename,'page':page_num,'abstract':part})
    for start in range(0,len(chunks),16):
        group=chunks[start:start+16];vectors=embed.embed_documents([c['text'] for c in group])
        for c,v in zip(group,vectors):c['embedding']=v
        driver.execute_query('UNWIND $rows AS row MERGE (d:Document {id:row.id}) SET d += row',rows=group)
        print(f'Indexed {min(start+16,len(chunks))}/{len(chunks)} PDF chunks',flush=True)
    if not chunks:
        raise SystemExit('No usable PDF text found in data/documents.')
    dim=len(chunks[0]['embedding'])
    driver.execute_query(f"CREATE VECTOR INDEX vector IF NOT EXISTS FOR (d:Document) ON (d.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}")
    driver.execute_query('CREATE FULLTEXT INDEX documentFullTextIndex IF NOT EXISTS FOR (d:Document) ON EACH [d.text,d.source,d.title]')
    driver.execute_query('CREATE FULLTEXT INDEX entityFullTextIndex IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]')
    driver.execute_query('CALL db.awaitIndexes(120)')
    # Limit extraction cost; index all chunks for vector and full-text retrieval.
    selected=list({c["id"]:c for c in chunks[:8]+chunks[-4:]}.values())
    llm=ChatOllama(model='llama3.1:8b',temperature=0,num_predict=700)
    schema=Entities.model_json_schema()
    schema['required']=list(schema['properties'])
    chain=ChatPromptTemplate.from_messages([('system','Extract concrete named entities from this university document. Fill every field with a list; use empty lists only when absent.'),('human','{text}')])|llm.with_structured_output(schema,method='json_schema')
    graph_counts=0;failures=[]
    for i,c in enumerate(selected):
        try:
            entities=chain.invoke({'text':c['text']})
            rows=[{'name':name.strip(),'kind':kind} for kind,values in entities.items() for name in values[:4] if name.strip()]
            driver.execute_query('MATCH (d:Document {id:$id}) UNWIND $rows AS row MERGE (e:__Entity__ {id:row.name}) SET e.kind=row.kind MERGE (e)-[:MENTIONS]->(d)',id=c['id'],rows=rows)
            graph_counts+=len(rows)
        except Exception as exc:failures.append({'chunk':c['id'],'error_type':type(exc).__name__})
        print(f'Entity graph {i+1}/{len(selected)}',flush=True)
    records,_,_=driver.execute_query('MATCH (n) RETURN labels(n)[0] AS label,count(n) AS count')
    report={'source_pdfs':len(list((ROOT/'data/documents').glob('*.pdf'))),'indexed_chunks':len(chunks),'embedding_model':'nomic-embed-text:latest','embedding_dimension':dim,
            'graph_extraction_chunks':len(selected),'entity_mentions_processed':graph_counts,'node_counts':[dict(r) for r in records],
            'graph_extraction_errors':failures,'scope':'All PDF chunks vector-indexed; entity extraction limited to the selected chunks.'}
    out=ROOT/'validation';out.mkdir(exist_ok=True)
    (out/'knowledge_base.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (ROOT/'data/demo_chunks.json').write_text(json.dumps([{k:v for k,v in c.items() if k!='embedding'} for c in chunks],ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True);driver.close()
if __name__=='__main__':main()
