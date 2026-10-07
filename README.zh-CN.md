# BNBU Graph RAG

**先选择信息来源，再检索证据，最后验证与聚合答案的多 Agent 校园问答系统。**

[English](README.md) · [快速运行](README.md#quick-start) · [评测说明](docs/EVALUATION.md)

![系统概览](docs/assets/cover.svg)

我主导了两阶段 GPT-2 路由、多 Agent 编排、证据结构、验证聚合、评测与应用集成。在 NevroHelios/rag-agent 的 Graph RAG 工作流基础上，扩展出本地文档、联网搜索、混合检索和拒答四条路径。

保存的路由模型在 340 题课程语料上答对 296 题，准确率 **87.06%**，未使用云端路由纠正。重建的本地知识库包含 **87 个 PDF 文本块、768 维向量和 74 个实体节点**；实体从 12 个选定文本块提取。Neo4j 与本地 Ollama 的实时问答已返回带来源页码的答案。

![路由混淆矩阵](docs/assets/router-confusion.png)

GPT-2 第一阶段判断是否回答，第二阶段判断应使用哪类信息来源。RAG/Web 专家产生结构化声明与证据，再由 Verifier 和 Aggregator 检查、选择并组织答案。文档检索结合向量与实体驱动的全文检索，执行过程可在界面查看。

课程评测语料与训练集是否完全独立尚未确认，因此 87.06% 表述为既有课程语料上的路由结果。公开包包含同一组路由问题与标签、权重下载入口、评测代码和虚构示例校园文档，可分别验证路由与文档问答。安装步骤见[英文 README](README.md#quick-start)。

Graph RAG 基础参考 [NevroHelios/rag-agent](https://github.com/NevroHelios/rag-agent)；GPT-2 基础实现参考 [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。详见[第三方归属](THIRD_PARTY.md)。
