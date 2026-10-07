"""Interactive routing and evidence-grounded document Q&A."""
import streamlit as st
from runtime import ROOT, configure

configure()
st.set_page_config(page_title='BNBU Graph RAG', page_icon='📚', layout='wide')
st.title('BNBU Graph RAG')
st.caption('Local routing · Document and web evidence · Inspectable answers')
mode = st.radio('Mode', ['Automatic routing', 'Document Q&A', 'Router only'], horizontal=True)

@st.cache_resource
def get_orchestrator():
    import torch
    from agents.orchestrator import Orchestrator
    torch.set_num_threads(8)
    return Orchestrator(checkpoint_path=str(ROOT / 'checkpoints/gpt2_124M_router_2stage.pth'))

question = st.text_input('Question', 'When does the sample campus library close on weekdays?')
if st.button('Run', type='primary') and question.strip():
    checkpoint = ROOT / 'checkpoints/gpt2_124M_router_2stage.pth'
    if not checkpoint.exists():
        st.info('Download the router first: python scripts/download_assets.py')
    else:
        try:
            with st.spinner('Running…'):
                orch = get_orchestrator()
                if mode == 'Router only':
                    route = orch.router_agent.route(question, use_deepseek_correction=False, force_correction=False)
                    st.subheader(route.final_label)
                    st.json(route.model_dump())
                else:
                    result = orch.run(question, route_override='RAG_ONLY' if mode == 'Document Q&A' else None)
                    st.subheader('Answer')
                    st.write(result.final_output.final_answer)
                    if result.error_stage:
                        st.warning('An execution stage failed: ' + result.error_stage)
                    with st.expander('Evidence and execution trace'):
                        st.json(result.model_dump(), expanded=2)
        except Exception as exc:
            # Configuration/provider errors can contain credentials in connection URLs.
            st.error(type(exc).__name__ + ': check local configuration and service availability.')
st.caption('The included campus handbook is fictional. Add your own PDFs and rebuild the index for other questions.')
