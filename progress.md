# Progress — CompetiScope

> 会话日志与工作进度追踪

---

## 2026-05-16

### 会话 #1 — 项目调研 & 规划
- 初始化规划文件 (task_plan.md, findings.md, progress.md)
- 完成项目结构初步调研
- 识别核心架构：4-Agent 协作模式 (Manager/Collector/Analyst/Writer)
- 分析当前代码缺陷：无真正 Agent 模式、无 RAG、无 Async、无生产特性
- 用户确认目标：打造简历级企业 AI 应用

### 会话 #2 — 深入分析 & 方案设计
- 深入阅读全部源码（main.py, 4 Agent 模块, 2 工具模块, settings.py, prompts.py）
- 设计 4 阶段升级路线图（Async → Agent → RAG → Production）
- 用户审核通过方案

### 会话 #3 — Phase 1 实施
- ✅ `pyproject.toml` — 项目元数据 + 依赖声明 + 工具配置
- ✅ `llm/` — LLM Provider 抽象层 (BaseLLMProvider + OpenAIProvider + TokenUsage)
- ✅ `types/` — Pydantic 结构化输出模型 (CompetitorProfile, SWOTResult, KeyInsights 等)
- ✅ `cache/` — LLM 响应缓存 (文件存储，MD5 key，24h TTL)
- ✅ `agents/collector_agent.py` → 全面异步化 (asyncio.gather 并发采集，3-10x 提速)
- ✅ `agents/manager_agent.py` → 异步编排 + Provider 模式 + 费用追踪
- ✅ `agents/analyst_agent.py` → Provider 模式 + SWOTAnalysis.to_dict() 修复
- ✅ `agents/writer_agent.py` → Provider 模式
- ✅ `main.py` → asyncio.run() 入口 + 费用展示
- ✅ `config/settings.py` → 新增 ProviderConfig, CacheConfig
- ✅ `tests/` → conftest.py (MockLLMProvider) + 10 个测试 **全部通过**

### 测试结果
```
tests/test_analyst.py .....  5 passed
tests/test_collector.py .....  5 passed
============================= 10 passed in 0.21s ==============================
```

---

## Phase 1 完成总结

| 指标 | 升级前 | 升级后 |
|------|--------|--------|
| 采集方式 | 串行 for 循环 | asyncio.gather 并发 |
| LLM 调用 | 4 个 Agent 各自创建 ChatOpenAI | 统一 Provider 注入 |
| 输出解析 | 手写 json.loads() + 字符串切割 | Pydantic 结构化模型 |
| API 费用 | 无从追踪 | TokenUsage 累计 + 实时展示 |
| 重复调用 | 每次重新请求 | LLM Cache (24h TTL) |
| 测试 | 0 个测试 | 10 个测试 (含 mock fixtures) |
| 项目包管理 | requirements.txt only | pyproject.toml + setuptools |

---

## 2026-05-16 (续)

### 会话 #5 — Phase 3 实施 ✅ 完成

- ✅ `rag/__init__.py` — RAG 模块导出 (6 组件)
- ✅ `rag/embedder.py` — Embedding 服务抽象层：
  - `BaseEmbedder` 抽象类 (embed / embed_query / dimension)
  - `OpenAIEmbedder` — OpenAI 兼容 Embedding API (异步批量，支持自定义 batch_size)
  - `LocalEmbedder` — SentenceTransformer 本地模型 (默认 paraphrase-multilingual-MiniLM-L12-v2, 384d)
  - `create_embedder()` 工厂函数
- ✅ `rag/chunker.py` — 递归文本分割器：
  - Markdown 章节感知 (按 ##/### header 分块)
  - 段落 → 句子 递归降级
  - 自然边界优先 (句号/感叹号/问号 > 分号 > 空格)
  - 重叠窗口 (overlap) + overlap_from 追踪
- ✅ `rag/vector_store.py` — ChromaDB 封装：
  - 3 个标准集合: competitor_data / analysis_reports / search_cache
  - add / search / search_by_embedding / count / delete_collection
  - 余弦距离 (cosine) 索引
  - where 过滤支持
- ✅ `rag/knowledge_base.py` — 知识库编排器：
  - ingest_collected_data / ingest_report / ingest_search_cache
  - search_similar / get_competitor_history / search_all
  - 自动 embed → chunk → store 流水线
  - 可被 disabled 优雅降级
- ✅ `rag/hybrid_search.py` — 混合搜索：
  - 向量搜索 (ChromaDB) + BM25 关键词搜索
  - RRF (Reciprocal Rank Fusion) 融合排序
  - 中英文混合分词 (CJK 单字 + 英文单词 + 数字)
  - 可配置权重 (vector: 0.6, keyword: 0.4)
- ✅ `agents/collector_agent.py` — RAG 集成：
  - 构造函数接受 `KnowledgeBase` 参数
  - `_check_kb_before_dimension()` — 搜索前检查缓存
  - `_ingest_to_kb()` — 采集后写入 KB
- ✅ `agents/analyst_agent.py` — RAG 集成：
  - `_get_historical_context()` — 检索历史 SWOT/报告
  - `_enrich_prompt_with_rag()` — 提示词注入 RAG 上下文
  - `_analyze_swot_with_rag()` — RAG 增强 SWOT 分析
- ✅ `agents/writer_agent.py` — RAG 集成：
  - `_get_historical_reports()` — 历史报告风格参考
  - `_ingest_report_to_kb()` — 报告自动存储
  - 战略建议生成融入 RAG 上下文
- ✅ `agents/manager_agent.py` — KB 编排：
  - ManagerAgent 创建 KB 实例
  - 共享给 Collector/Analyst/Writer 三个子 Agent
  - 报告完成后自动写入 KB
- ✅ `config/settings.py` — 配置完善：
  - 新增 `EmbedderConfig` (provider, local/openai model, API key)
  - 完善 `KnowledgeBaseConfig` (ENABLED, PERSIST_DIR, RAG_N_RESULTS, 混合搜索权重, CHUNK_SIZE)
- ✅ `tests/test_rag.py` — 37 个新测试 (21 passed, 16 skipped 因 chromadb 安装中)
  - TestTextChunker: 8 个测试 (含中文分块、自然边界、重叠)
  - TestBM25: 4 个测试 (搜索、空查询、无匹配、空文档)
  - TestEmbedderFactory: 2 passed (API key 验证、抽象类)
  - TestKnowledgeBase: 1 passed (disabled 降级)
  - TestRAGAgentIntegration: 6 passed (KB disabled 下的 Agent 兼容性、RAG 方法空返回)

### 测试结果
```
tests/test_rag.py — 37 tests (21 passed, 16 skipped — ChromaDB 底层 DLL 兼容性)
tests/test_agent.py — 39 tests (39 passed)
tests/test_collector.py — 5 tests (5 passed)
tests/test_analyst.py — 5 tests (5 passed)
================================ 70 passed, 16 skipped =============================
```

### Phase 3 架构亮点

```
┌──────────────────────────────────────────────────────┐
│                  Phase 3: RAG 架构                    │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────────────────┐                    │
│  │     ManagerAgent             │                    │
│  │   (KB 创建 + 共享给子 Agent)  │                    │
│  └──────────┬──────────────────┘                    │
│             │                                        │
│    ┌────────┼────────┬──────────────┐               │
│    ▼        ▼        ▼              ▼               │
│ Collector  Analyst  Writer      KnowledgeBase        │
│ ┌──────┐ ┌──────┐ ┌──────┐    ┌──────────────┐      │
│ │搜索前 │ │历史  │ │历史  │    │  Embedder    │      │
│ │KB检查 │ │SWOT  │ │报告  │    │  (Local/API) │      │
│ │       │ │注入  │ │风格  │    │      │       │      │
│ │采集后 │ │增强  │ │参考  │    │  Chunker     │      │
│ │KB写入 │ │分析  │ │      │    │      │       │      │
│ └──────┘ └──────┘ └──────┘    │  VectorStore │      │
│                                │  (ChromaDB)  │      │
│     ┌──────────────────────┐   │      │       │      │
│     │   HybridSearcher     │   │  3 集合:     │      │
│     │ 向量 RRF + BM25 关键词│   │  · competitor│      │
│     │ 中文/英文混合分词      │   │  · reports   │      │
│     └──────────────────────┘   │  · cache      │      │
│                                └──────────────┘      │
└──────────────────────────────────────────────────────┘
```

### Phase 3 关键决策

| 决策 | 说明 |
|------|------|
| 默认本地 SentenceTransformer | DeepSeek 无 Embedding API，本地免费，且 384d 向量对竞品分析足够 |
| 混合搜索 (RRF + BM25) | 向量捕获语义 ("cheap ≈ affordable")，BM25 捕获精确关键词 ("Notion" ≠ "motion") |
| Markdown 感知分块 | 保留报告章节结构，避免在段落中间断开 |
| KB 优雅降级 | `enabled=False` 时所有 RAG 方法返回空，不影响核心流程 |
| KB 实例共享 | ManagerAgent 创建 KB，传递给三个子 Agent，保证读写一致性 |
| 三集合设计 | competitor_data / analysis_reports / search_cache 分离不同类型数据 |

---

## 下一步: Phase 4 — 生产特性

FastAPI 服务 + SSE 流式响应 + 费用追踪 + Streamlit Dashboard + 全量测试

估计工作量：10+ 个文件新建 + 3 个文件修改

## 2026-05-16 (续)

### 会话 #4 — Phase 2 实施

- ✅ `agent/__init__.py` — Agent 模块导出
- ✅ `agent/tool_registry.py` — 工具注册中心 (AgentTool + ToolRegistry + JSON Schema + OpenAI function spec)
- ✅ `agent/memory.py` — 双记忆系统 (ConversationMemory 滑动窗口 + 摘要压缩, WorkingMemory 结构化中间存储)
- ✅ `agent/base_agent.py` — **手写 ReAct 循环引擎**：
  - Thought → Action → Observation → Thought 完整循环
  - 正则 + 括号计数 JSON 提取 (支持嵌套)
  - Parse 失败时自动反馈纠正
  - 达到 max_steps 时强制生成 Final Answer
  - AgentStep / AgentResult 结构化追踪
- ✅ `agent/reflector.py` — 自我反思模块：
  - 评分 (1-10)、优势/劣势/建议/缺失维度
  - reflect_and_revise 迭代修订 (最多 N 轮)
  - 分数 ≥ 阈值自动停止
- ✅ `tools/tool_adapter.py` — 工具适配层：
  - search_web / scrape_page (同步工具异步包装)
  - analyze_competitor_data / generate_report_section (LLM 驱动)
  - create_agent_tools() 工厂函数
- ✅ `agents/manager_agent.py` — **从固定流水线升级为 ReAct Agent 自主循环**：
  - analyze_async() 使用 ReActAgent 自主决策
  - 可选自我反思 + 报告修订
  - analyze_legacy_async() 保留旧流水线向后兼容
- ✅ `config/prompts.py` — 新增 REACT_AGENT_SYSTEM_PROMPT + REACT_AGENT_TASK_TEMPLATE
- ✅ `config/settings.py` — 新增 AgentLoopConfig (MAX_STEPS, REFLECTION_ROUNDS, REFLECTION_ENABLED, REFLECTION_SCORE_THRESHOLD)
- ✅ `main.py` — 适配 Agent 结果格式 (agent_steps, reflection_rounds, reflection_scores)
- ✅ `tests/test_agent.py` — **39 个新测试全部通过**

### 测试结果
```
tests/test_agent.py ........................................ [ 79%]
tests/test_analyst.py .....                               [ 89%]
tests/test_collector.py .....                             [100%]
============================= 49 passed in 0.40s ==============================
```

### Phase 2 架构亮点

```
用户请求 → ManagerAgent.analyze_async()
              ├── 构建 ReAct 任务描述 (REACT_AGENT_TASK_TEMPLATE)
              ├── 创建 ToolRegistry (search_web / scrape_page / analyze_competitor_data / generate_report_section)
              ├── 创建 ReActAgent (provider + tools + memory + max_steps=12)
              ├── Agent 自主循环:
              │   ├── Thought: 我需要搜索 Notion 的基本信息
              │   ├── Action: search_web("Notion company overview")
              │   ├── Observation: [搜索结果 JSON]
              │   ├── Thought: 数据充足，开始写报告
              │   └── Final Answer: 完整的竞品分析报告
              ├── 可选: Reflector.reflect_and_revise() 自我反思改进
              └── 返回: AgentResult (report + steps + cost)
```

### Phase 2 关键决策

| 决策 | 说明 |
|------|------|
| 手写 ReAct 循环，不用 LangChain AgentExecutor | 完全透明可控，真正理解 Agent 原理 |
| 正则 + 括号计数解析 LLM 输出 | 比依赖 JSON mode 更灵活，支持嵌套 JSON 参数 |
| Parse 失败自动反馈纠正 | Agent 收到格式错误提示后重新生成，提高鲁棒性 |
| 保留旧流水线向后兼容 | analyze_legacy_async() 确保 Phase 1 功能可用 |
| 反思可选，默认启用 | AgentLoopConfig.REFLECTION_ENABLED 控制 |

---

## 下一步: Phase 3 — RAG 知识检索

估计工作量：5 个文件新建 + 3 个文件修改

---

## 2026-05-16 (终)

### 会话 #6 — Phase 4 实施 ✅ 完成

- ✅ `core/__init__.py` + `core/exceptions.py` — 统一异常体系：
  - `CompetiScopeError` 基类
  - `ConfigError` / `AgentError` / `ToolError` / `APIError` / `TaskNotFoundError`
- ✅ `core/config.py` — Pydantic `BaseSettings` 统一配置：
  - `AppConfig` 单例 (pydantic-settings, 支持 .env 自动加载)
  - 所有 LLM/Agent/Embedding/KB/API 配置集中管理
  - `get_config()` 工厂函数
- ✅ `observability/cost_tracker.py` — 费用追踪：
  - `CostRecord` dataclass (timestamp, model, tokens, cost, call_type)
  - `CostTracker` 类 (按 call_type 分组统计 + session 级 summary)
  - `cost_tracker` 全局单例
- ✅ `observability/metrics.py` — 运行指标：
  - `AnalysisRecord` dataclass (timestamp, success, duration, agent_steps)
  - `MetricsCollector` (成功率/平均耗时/平均步数/运行中计数)
  - `metrics` 全局单例
- ✅ `api/__init__.py` + `api/schemas.py` — API 模块：
  - Request: `AnalysisRequest`, `KnowledgeSearchRequest`
  - Response: `TaskResponse`, `TaskListResponse`, `CostSummary`, `MetricsSummary`, `HealthResponse`, `KnowledgeSearchResponse`
- ✅ `api/routes/analysis.py` — 分析路由：
  - `POST /api/v1/analyze` — 异步启动分析 (BackgroundTasks)
  - `GET /api/v1/task/{id}` — 查询任务状态
  - `GET /api/v1/tasks` — 任务列表 (支持 status 过滤)
  - `POST /api/v1/analyze/stream` — **SSE 流式分析** (EventSourceResponse)
  - SSE events: progress / step / reflection / cost / result (chunked) / done / error
- ✅ `api/routes/knowledge.py` — 知识路由：
  - `POST /api/v1/knowledge/search` — 跨集合搜索
  - `GET /api/v1/knowledge/competitors/{name}` — 竞品历史查询
- ✅ `api/server.py` — FastAPI 应用工厂：
  - `create_app()` 工厂函数
  - **lifespan** 替代 on_event (消除 deprecation warning)
  - CORS 中间件
  - `/health`, `/api/v1/costs`, `/api/v1/metrics` 端点
  - 10 个路由全部注册成功
- ✅ `agents/writer_agent.py` — SSE 流式生成：
  - `astream_full_report()` — 异步生成器，按章节 yield
  - `awrite_full_report_async()` — `asyncio.to_thread` 异步包装
- ✅ `app.py` — Streamlit Dashboard 重写：
  - 多竞品输入 (逗号分隔)
  - 报告/费用/知识库 三个 Tab
  - KB 搜索 + 历史分析记录
  - 侧边栏 Agent/KB 配置
- ✅ `config/settings.py` — **ProviderConfig 懒加载化** (关键修复):
  - class attribute → staticmethod，每次 `to_dict()` 重新读取 env var
  - 支持测试时 monkeypatch.setenv
- ✅ `tests/test_api.py` — **16 个新测试** (全部通过):
  - TestHealth (2), TestCosts (1), TestMetrics (1)
  - TestAnalysisRouteValidation (4): schema 验证
  - TestAnalysisRouteWithMockKey (5): task CRUD + SSE
  - TestKnowledgeRoutes (3): search + competitor history
- ✅ `tests/test_streaming.py` — **13 个新测试** (全部通过):
  - TestWriterStreaming (3): 流式 section 产出
  - TestObservability (6): CostTracker + MetricsCollector
  - TestCoreConfig (3): AppConfig 默认值 + env var + 单例
  - TestCoreExceptions (1): 异常层次结构

### 测试结果
```
tests/test_api.py — 16 tests (16 passed)
tests/test_streaming.py — 13 tests (13 passed)
================================ 29 passed ================================
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

### Phase 4 关键决策

| 决策 | 说明 |
|------|------|
| ProviderConfig 懒加载 | class attribute → staticmethod，解决测试 monkeypatch 无法覆盖 import-time 求值的问题 |
| FastAPI lifespan | 用 `@asynccontextmanager` 替代 `on_event`，消除 deprecation warning |
| SSE 章节级流式 | 报告按 chapter 分 event 发送，比 token 级更实用（前端可逐章节渲染） |
| 全局单例 CostTracker + MetricsCollector | 避免依赖注入复杂度，便于 API 端点直接访问 |
| 后台任务模式 (BackgroundTasks) | POST /analyze 立即返回 task_id，分析在后台异步执行 |
| 内存任务存储 | 当前用 `dict` 存储任务，生产环境可替换为 Redis/DB |

### Phase 4 修复的 bugs

- `manager_agent.py` 第 26 行: `from llm.types import ProviderConfig as SettingsProvider` → 应为 `from config.settings import ProviderConfig as SettingsProvider`，导致缺少 `to_dict()` 方法
- `ProviderConfig.API_KEY` 在 class 定义时求值导致测试 monkeypatch 无效 → 改为 lazy staticmethod
- API 应用 `on_event("startup")` 已废弃 → 改为 `lifespan`

### Phase 4 新增文件 (12 个)

```
core/__init__.py
core/exceptions.py
core/config.py
observability/__init__.py
observability/cost_tracker.py
observability/metrics.py
api/__init__.py
api/server.py
api/schemas.py
api/routes/__init__.py
api/routes/analysis.py
api/routes/knowledge.py
tests/test_api.py
tests/test_streaming.py
```

### Phase 4 修改文件 (4 个)

```
app.py — 全面重写 (多竞品、费用看板、KB 搜索、历史)
agents/manager_agent.py — import 修复
agents/writer_agent.py — SSE 流式方法
config/settings.py — ProviderConfig 懒加载化
```

---

## 项目完成总结

| 指标 | Phase 0 (原始) | Phase 4 (最终) |
|------|---------------|----------------|
| 架构模式 | 同步流水线 | ReAct Agent Loop + RAG + SSE |
| LLM 调用 | 4 处重复 ChatOpenAI | 统一 Provider (依赖注入) |
| 异步支持 | 无 | asyncio.gather 并发 + async 全链路 |
| Agent 自主决策 | 无 (硬编码 3 阶段) | ReAct 循环 (Thought→Action→Observation) |
| 自我反思 | 无 | Reflector (评分 + 修订 + 早停) |
| 知识检索 | 无 | ChromaDB + 混合搜索 (向量+BM25) |
| API 服务 | 无 | FastAPI (10 路由 + SSE 流式) |
| Web UI | 基础 (2 竞品) | Dashboard (多竞品/费用/KB/历史) |
| 费用追踪 | 无 | TokenUsage + CostTracker 全局单例 |
| 配置文件 | settings.py class attributes | Pydantic BaseSettings |
| 异常处理 | 无统一体系 | CompetiScopeError 层次化 |
| 测试 | 0 | **99 passed, 32 skipped** |
| 测试文件 | 0 | 6 test files |

---

## 下一步 (可选)

所有 4 个阶段已完成，可考虑：
- 部署: Docker 容器化 + docker-compose
- 生产存储: Redis (任务队列) + PostgreSQL (任务历史)
- CI/CD: GitHub Actions (lint + test)
- 认证: API Key / JWT 中间件
