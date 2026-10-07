# Evaluation protocol

## Two-stage router

The saved GPT-2 router is loaded strictly and evaluated with cloud correction disabled. `eval/val_large.json` contains 340 question/label pairs from the existing course corpus. The reproduced result is **296 correct / 340 = 87.0588%**. A held-out train/test separation has not been established, so the number is a corpus-level routing result.

`python scripts/check_router.py` writes a fresh summary and predictions to the ignored `validation/` directory. The published [summary](../evaluation/router_run.json) and [confusion matrix](../evaluation/router-confusion.json) describe the same reproduced run. Class labels are document-only, web-only, hybrid and refusal; the first head judges answerability and the second selects a source route.

## Local retrieval rebuild

The campus demonstration database was rebuilt from two source PDFs: 87 overlapping chunks embedded with `nomic-embed-text:latest` (768 dimensions), and entity extraction on 12 selected chunks yielding 74 unique entity nodes. All chunks were indexed for vector and full-text search. These counts describe the rebuilt campus demonstration, not the small fictional handbook shipped for a self-contained public example.

A live local run asked where BNBU is located and when it was founded. The document-only path returned Zhuhai and 2005 with source-page evidence. This single demonstration is a functional verification, not an answer-accuracy benchmark. Retrieval in this release is vector search plus entity-driven full-text search; graph links are stored, but the active specialist does not perform multi-hop traversal.

Historical course end-to-end traces used model-judged scores. They are not published as objective answer accuracy. This release highlights the directly recomputed routing result and inspectable retrieval behavior.

## Self-contained public fixture

The included fictional handbook was indexed into a separate authenticated Neo4j instance. The complete local pipeline answered “When does the sample campus library close on weekdays?” with “22:00”, citing `sample-campus.pdf`, page 1. The [answer excerpt](../evaluation/public-demo-answer.json) contains the question, result, and verified quote provenance. Its 1 chunk and 9 extracted entity nodes are separate from the 87-chunk campus rebuild above. Three targeted regression cases check citation recovery, fabricated quotations, and ambiguous quotations.

## Self-contained public fixture

The included fictional handbook was indexed into a separate authenticated Neo4j instance. The complete local pipeline answered “When does the sample campus library close on weekdays?” with “22:00”, citing `sample-campus.pdf`, page 1. The [answer excerpt](../evaluation/public-demo-answer.json) contains the question, result, and verified quote provenance. Its 1 chunk and 9 extracted entity nodes are separate from the 87-chunk campus rebuild above. Three targeted regression cases check citation recovery, fabricated quotations, and ambiguous quotations.
