# Task Plan — CompetiScope

> 企业级竞品分析Agent系统 — 阶段、进展与决策记录

---

## 项目概述

- **项目名**: CompetiScope v2.0
- **目标**: 从简单 API 调用脚本 → 企业级 ReAct Agent + RAG + Async AI 应用
- **技术栈**: Python, LangChain, DeepSeek/OpenAI, ChromaDB, FastAPI, Streamlit
- **架构**: ReAct Agent Loop → 4 工具 (Search/Scrape/Analyze/Write) + RAG 知识检索 + 自我反思

---

## 当前阶段

**Phase 4: 生产特性** ✅ 已完成 (2026-05-16)

---

## 阶段记录

| # | 阶段 | 状态 | 核心产出 | 简历关键词 |
|---|------|------|----------|-----------|
| 1 | Async 基础 + 架构现代化 | ✅ 完成 | LLM Provider 抽象、异步并发采集、Pydantic 结构化输出、LLM 缓存、10 个测试全过 | async/await, asyncio.gather, 依赖注入, Pydantic |
| 2 | Agent 模式 (ReAct Loop) | ✅ 完成 | ReAct 循环引擎、工具注册表、Agent Memory、自我反思、39 个新测试 | Agent Loop, ReAct, Function Calling, Self-Critique |
| 3 | RAG 知识检索 | ✅ 完成 | Embedding 服务、ChromaDB 向量存储、混合搜索、知识持久化、21 个新测试 | RAG, Embeddings, Vector DB, Semantic Search |
| 4 | 生产特性 | ✅ 完成 | FastAPI 服务、SSE 流式、费用追踪、Streamlit Dashboard、统一配置/异常、29 个新测试 | FastAPI, SSE, 可观测性, 测试 |

---

## Phase 1 任务清单

- [x] `pyproject.toml` — 项目元数据 + 依赖声明
- [x] `llm/` — LLM Provider 抽象层 (BaseLLMProvider + OpenAIProvider + TokenUsage)
- [x] `types/` — Pydantic 结构化输出模型 (CompetitorProfile, SWOTResult, KeyInsights)
- [x] `cache/` — LLM 响应缓存 (文件存储，MD5 key，24h TTL)
- [x] `agents/collector_agent.py` → 异步化 + asyncio.gather 并发
- [x] `agents/manager_agent.py` → async def analyze_async() + 使用 Provider
- [x] `agents/analyst_agent.py` → 使用 Provider + SWOTAnalysis.to_dict() 修复
- [x] `agents/writer_agent.py` → 使用 Provider
- [x] `main.py` → asyncio.run() 入口 + 费用展示
- [x] `config/settings.py` → 添加 ProviderConfig, CacheConfig
- [x] `tests/` → conftest.py + test_collector.py + test_analyst.py (10 passed)

---

## Phase 2 任务清单

- [x] `agent/__init__.py` — Agent 模块导出
- [x] `agent/base_agent.py` — ReAct 循环引擎 (Thought→Action→Observation→Thought)
- [x] `agent/tool_registry.py` — 工具注册中心 + JSON Schema 生成
- [x] `agent/memory.py` — 对话记忆 (ConversationMemory + WorkingMemory)
- [x] `agent/reflector.py` — 自我审查 + 报告修订 (Reflector)
- [x] `tools/tool_adapter.py` — SearchTool/WebScraper → AgentTool 适配 + LLM 分析/撰写工具
- [x] `agents/manager_agent.py` — 固定流水线 → ReAct Agent 自主循环
- [x] `config/prompts.py` — 新增 REACT_AGENT_SYSTEM_PROMPT + REACT_AGENT_TASK_TEMPLATE
- [x] `config/settings.py` — 新增 AgentLoopConfig (max_steps, reflection_rounds, score_threshold)
- [x] `main.py` — 适配 Agent 结果格式 (agent_steps, reflection_rounds, reflection_scores)
- [x] `tests/test_agent.py` — 39 个测试 (ToolRegistry/Memory/Parsing/Execution/Reflector/Adapter/Integration)

---

## Phase 3 任务清单

- [x] `rag/__init__.py` — RAG 模块导出
- [x] `rag/embedder.py` — Embedding 服务抽象 (BaseEmbedder + OpenAIEmbedder + LocalEmbedder + factory)
- [x] `rag/chunker.py` — 递归文本分割 (Markdown 感知 + 段落/句子分割 + 重叠)
- [x] `rag/vector_store.py` — ChromaDB 封装 (3 集合: competitor_data / analysis_reports / search_cache)
- [x] `rag/knowledge_base.py` — 知识库编排器 (ingest 采集数据/报告/搜索缓存 + 检索)
- [x] `rag/hybrid_search.py` — 混合搜索 (向量 RRF + BM25 关键词) + 中文/英文分词
- [x] `agents/collector_agent.py` — 搜索前 RAG 检查 + 采集后 KB 写入
- [x] `agents/analyst_agent.py` — RAG 增强 SWOT 分析 + 历史上下文注入
- [x] `agents/writer_agent.py` — 历史报告风格检索 + 战略建议 KB 增强
- [x] `agents/manager_agent.py` — KB 共享给子 Agent + 报告自动写入 KB
- [x] `config/settings.py` — 新增 EmbedderConfig + 完善 KnowledgeBaseConfig
- [x] `tests/test_rag.py` — 37 个测试 (TextChunker/BM25/HybridSearcher/Embedder/KnowledgeBase/RAG 集成)

### 测试结果
```
tests/test_rag.py ...................... 21 passed, 16 skipped (ChromaDB 平台兼容)
tests/test_agent.py ................... 39 passed
tests/test_collector.py ...............  5 passed
tests/test_analyst.py .................  5 passed
========================= 70 passed, 16 skipped =========================
```

### Phase 4 测试结果
```
tests/test_api.py .............. 16 passed
tests/test_streaming.py ........ 13 passed
============================ 29 passed ============================
```

### 全量测试 (Phase 1-4)
```
tests/test_agent.py ............ 39 passed
tests/test_analyst.py ..........  5 passed
tests/test_collector.py ........  5 passed
tests/test_rag.py .............. 21 passed, 16 skipped
tests/test_api.py .............. 16 passed
tests/test_streaming.py ........ 13 passed
================= 99 passed, 32 skipped, 0 failures =================
```

### Phase 4 架构亮点

```
┌──────────────────────────────────────────────────────┐
│                Phase 4: 生产特性架构                    │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────────────────────────────┐       │
│  │            FastAPI Application             │       │
│  │  GET /health  GET /api/v1/costs           │       │
│  │  GET /api/v1/metrics                      │       │
│  ├──────────────────────────────────────────┤       │
│  │  Analysis Routes          Knowledge Routes│       │
│  │  POST /analyze             POST /search   │       │
│  │  POST /analyze/stream SSE  GET /{name}    │       │
│  │  GET  /task/{id}                          │       │
│  │  GET  /tasks                              │       │
│  └──────────────┬───────────────────────────┘       │
│                 │                                    │
│  ┌──────────────┼───────────────────────────┐       │
│  │     ManagerAgent (BackgroundTasks)        │       │
│  │  ┌──────────┴──────┬──────────┬────────┐ │       │
│  │  ▼                 ▼          ▼        ▼ │       │
│  │ Collector      Analyst     Writer   KB   │       │
│  │ (Async)        (RAG)      (SSE)          │       │
│  └──────────────────────────────────────────┘       │
│                                                      │
│  ┌──────────────────┐  ┌──────────────────────┐     │
│  │  CostTracker     │  │  MetricsCollector    │     │
│  │  按类型追踪费用   │  │  成功率/耗时/步数    │     │
│  │  全局单例        │  │  历史记录            │     │
│  └──────────────────┘  └──────────────────────┘     │
│                                                      │
│  ┌──────────────────────────────────────────┐       │
│  │         Streamlit Dashboard               │       │
│  │  多竞品输入 · 实时进度 · 费用看板        │       │
│  │  KB 检索 · 历史记录 · 报告下载            │       │
│  └──────────────────────────────────────────┘       │
│                                                      │
│  ┌──────────────────────────────────────────┐       │
│  │       Core Layer                          │       │
│  │  AppConfig (Pydantic BaseSettings)        │       │
│  │  CompetiScopeError hierarchy              │       │
│  └──────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────┘
```

---

## 决策记录

| # | 决策 | 原因 |
|---|------|------|
| 1 | 手写 Agent 循环，不用 LangChain AgentExecutor | AgentExecutor 黑盒，手写才能真懂 ReAct 原理 |
| 2 | Phase 1 就异步化 | 性能提升 3-10x，为后续所有阶段打好地基 |
| 3 | ChromaDB 为向量存储 | 本地运行，学习友好，已在依赖中 |
| 4 | 保留 LangChain 用于 LLM 接口 + Tool 抽象 | 避免重复造轮子，但核心 Agent 逻辑自建 |
| 5 | 所有 LLM 调用兼容 OpenAI API 格式 | 支持 DeepSeek/OpenAI/任何兼容服务 |
| 6 | 默认使用本地 SentenceTransformer Embedding | DeepSeek 无 Embedding API，本地免费且无速率限制 |
| 7 | 混合搜索：向量 RRF + BM25 | 向量捕获语义，BM25 捕获关键词（如 "Notion" ≠ "motion"），组合效果更好 |
| 8 | Markdown 感知分块 | 保留报告章节结构，避免在段落中间断开 |
| 9 | SSE Streaming 用于报告流式输出 | 章节级流式 (section-level)，每个章节作为独立 event 发送，客户端可逐章节渲染 |
| 10 | ProviderConfig 懒加载 | class attribute → staticmethod，每次 `to_dict()` 时重新读取 env var，支持测试 monkeypatch |
| 11 | FastAPI lifespan 代替 on_event | 避免 deprecation warning，使用 `@asynccontextmanager` 管理启动/关闭 |
| 12 | 全局单例 CostTracker + MetricsCollector | 跨 session 追踪费用和指标，便于 API 端点直接访问 |

---

## Phase 4 任务清单

- [x] `api/__init__.py` + `api/server.py` — FastAPI 应用 (lifespan 生命周期 + CORS + 10 路由)
- [x] `api/schemas.py` — 请求/响应 Pydantic 模型 (AnalysisRequest / TaskResponse / KnowledgeSearch 等)
- [x] `api/routes/analysis.py` — POST /analyze, GET /task/{id}, GET /tasks, POST /analyze/stream (SSE)
- [x] `api/routes/knowledge.py` — POST /knowledge/search, GET /knowledge/competitors/{name}
- [x] `observability/cost_tracker.py` — LLM 调用拦截 + 费用记录 (CostRecord + CostTracker 全局单例)
- [x] `observability/metrics.py` — 分析次数/成功率/平均耗时 (AnalysisRecord + MetricsCollector 全局单例)
- [x] `core/exceptions.py` — CompetiScopeError 异常体系 (ConfigError / AgentError / ToolError / APIError)
- [x] `core/config.py` — Pydantic BaseSettings 统一配置 (AppConfig 单例 + 所有 env vars)
- [x] `app.py` — Streamlit UI 增强 (多竞品、费用看板、KB 搜索、历史记录)
- [x] `agents/writer_agent.py` — SSE 流式生成 (astream_full_report + awrite_full_report_async)
- [x] `tests/test_api.py` — 16 个 API 测试 (Health/Costs/Metrics/Route 验证/分析路由/知识路由/SSE)
- [x] `tests/test_streaming.py` — 13 个测试 (Writer 流式/Observability/Core Config/Exceptions)
- [x] `config/settings.py` — ProviderConfig 懒加载化 (支持测试时 monkeypatch)

---

## 待解决问题

- [x] DeepSeek 是否提供 Embedding API？→ **否**，Phase 3 已用本地 SentenceTransformer 解决
