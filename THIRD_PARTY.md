# Attribution and provenance

## Graph RAG foundation

The small Graph RAG workflow was adapted from [NevroHelios/rag-agent](https://github.com/NevroHelios/rag-agent). `utils.py` and `chain.py` retain this foundation, with compatibility adjustments; `config.py` and the retrieval structure also follow that project. Its README declares MIT licensing, but no standalone LICENSE file was present when reviewed on 7 October 2026. That declaration is recorded here without inventing a missing copyright notice.

Bohan Wu led the added two-stage GPT-2 routing, specialist coordination, evidence/verifier/aggregator contracts, evaluation tooling and application integration. The original graph/vector workflow is credited to its author.

## GPT-2 implementation

The GPT-2 building blocks and `gpt_download.py` derive from Sebastian Raschka's [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch). The upstream license is preserved in [licenses/LLMs-from-scratch.txt](licenses/LLMs-from-scratch.txt). GPT-2 pretrained weights originate from OpenAI; fine-tuning and the routing heads are project-specific. The release router is a research/course model, with its evaluation scope stated in [docs/EVALUATION.md](docs/EVALUATION.md).

## Data and dependencies

The included `sample-campus.pdf` is a fictional document written for the runnable example. Published routing inputs contain course evaluation questions and category labels; original reports, answer traces and campus source PDFs are not included. Neo4j, LangChain, Ollama, Streamlit, PyTorch, Tavily and model providers retain their own licenses and service terms.

Licensing for each retained upstream component remains as described above; no repository-wide license is imposed on third-party material.
