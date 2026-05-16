# Findings — CompetiScope

> 调研发现、技术探索与知识沉淀

---

## 项目架构发现

### Phase 1 升级前现状

| 文件 | 当前问题 | 升级方向 |
|------|----------|----------|
| `agents/manager_agent.py` | 固定流水线，无 Agent 自主决策 | Phase 2: ReAct Agent Loop |
| `agents/collector_agent.py` | 串行采集，每个竞品 5 维度依次执行 | Phase 1: asyncio.gather 并发 |
| `agents/analyst_agent.py` | 手写 json.loads() 解析，脆弱 | Phase 1: Pydantic 结构化输出 |
| `agents/writer_agent.py` | 纯同步生成，无流式输出 | Phase 4: SSE Streaming |
| `tools/search_tool.py` | 功能完善，但未封装为 Agent Tool | Phase 2: Tool Adapter 封装 |
| `tools/web_scraper.py` | 同上 | Phase 2: Tool Adapter 封装 |
| `config/settings.py` | 静态类属性，非 Pydantic | Phase 4: BaseSettings |
| `config/prompts.py` | Pipeline 协调者提示词 | Phase 2: 自主 Agent 提示词 |

### 每个 Agent 都独立创建 ChatOpenAI 实例
```python
# 在 4 个 Agent 中各重复一次
self.llm = ChatOpenAI(model=..., api_key=..., base_url=..., ...)
```
→ Phase 1 用 Provider 统一管理

### 并发改进潜力
```
当前（串行）:
  竞品A [搜索1→搜索2→搜索3→搜索4→搜索5] → 竞品B [...] → 竞品C [...]
  
Phase 1（并发）:
  竞品A [搜索1+搜索2+搜索3+搜索4+搜索5] ┐
  竞品B [搜索1+搜索2+搜索3+搜索4+搜索5] ├ asyncio.gather → 3-10x faster
  竞品C [搜索1+搜索2+搜索3+搜索4+搜索5] ┘
```

---

## 技术调研

### DeepSeek API 兼容性
- DeepSeek API 完全兼容 OpenAI SDK 格式
- base_url: `https://api.deepseek.com`
- 支持 Chat Completions（含 streaming）
- **Embedding API: 不提供** — Phase 3 默认使用 SentenceTransformer 本地模型 (paraphrase-multilingual-MiniLM-L12-v2, 384d)

### ChromaDB
- 已在 requirements.txt 但从未使用
- 轻量级向量数据库，支持本地持久化
- Python API 简单友好，适合学习
- 支持 metadata filtering（按竞品名/日期过滤）
- ⚠ DefaultEmbeddingFunction 依赖 ONNX Runtime，部分 Windows 系统有 DLL 兼容问题
- 使用自定义 embedding function 可绕过（通过 `get_or_create(embedding_function=...)`）

### Pydantic v2 + LangChain
- LangChain 的 `with_structured_output()` 需要 Pydantic v2
- 当前项目已依赖 `pydantic>=2.0.0`
- 可用于替换所有脆弱的 `json.loads()` 字符串解析

### async/await 在 LangChain 中
- `ChatOpenAI` 支持 `ainvoke()` 和 `astream()`
- `asyncio.gather()` 可以并发调用多个 LLM 请求
- DuckDuckGo Search 和 httpx 都支持 async

### SentenceTransformer 本地部署
- `paraphrase-multilingual-MiniLM-L12-v2` 对中文友好，384 维
- 首次运行会自动下载模型 (~420MB)
- ⚠ 依赖 torch，Windows 上 torch DLL 可能与 onnxruntime 冲突
- Embedding 操作可通过 `run_in_executor` 异步化

### 混合搜索 RRF（Reciprocal Rank Fusion）
- 标准公式: `score = Σ weight_i / (k + rank_i)` where k=60
- 向量搜索捕获语义相似性
- BM25 捕获精确关键词匹配
- 中文分词: CJK 字符级 + 英文单词级 + 数字标记

### Markdown 感知分块
- 按 `##` / `###` header 分割保留章节结构
- 段落 → 句子逐级降级分块
- 自然边界优先：句号 → 感叹号 → 问号 → 分号 → 空格
- 重叠窗口保持上下文连续性

---

## 数据来源

_（待 Phase 3 RAG 实施时记录置信度高的数据源）_

---

## Phase 4 发现

### FastAPI + BackgroundTasks
- `BackgroundTasks` 适合轻量级异步任务，无需 Celery/RQ 等外部依赖
- 限制：任务在进程内执行，不能跨进程；生产环境应用消息队列
- TestClient 对 SSE `EventSourceResponse` 兼容良好，返回 `text/event-stream`

### SSE (Server-Sent Events) via sse-starlette
- `sse_starlette.sse.EventSourceResponse` 是最简 SSE 实现
- 每个 event 需包含 `event` 和 `data` 字段
- 章节级流式 (section-level) 比 token 级更适合竞品分析报告
- Chrome DevTools EventSource 不支持 POST → 需用 `Fetch` API 手动发送
- Token 级流式用 `provider.astream()` 原生支持 SSE

### ProviderConfig 懒加载 (关键发现)
- **问题**: `class API_KEY = os.getenv(...)` 在 import 时求值，测试 monkeypatch 无法覆盖
- **解决**: 改为 `@staticmethod def _api_key(): return os.getenv(...)` 每次调用时重新读取
- 这保留了 class-attribute 访问方式，同时支持运行时 env var 变更

### FastAPI lifespan vs on_event
- `on_event("startup")` / `on_event("shutdown")` 在最新版 FastAPI 已废弃
- 替代方案: `@asynccontextmanager` + `lifespan` 参数
- 更精细的资源生命周期管理

### Pydantic BaseSettings (pydantic-settings)
- v2 中已分离为独立包 `pydantic-settings>=2.0`
- 支持 `model_config` 中 `env_file` 和 `env_prefix`
- `Field(alias="ENV_VAR")` 自动映射环境变量
- `extra="ignore"` 忽略额外字段避免崩溃

### API 路由架构
- POST /analyze 立即返回 task_id (BackgroundTasks)，前端轮询 GET /task/{id}
- SSE POST /analyze/stream 适合实时 Dashboard，单连接推流进度
- 内存 dict 存储任务适合开发，生产应迁移到 Redis/DB
- 10 个路由 (8 API + health + docs) 清晰分离关注点

### Streamlit 增强
- 侧边栏折叠 (expander) 避免 UI 混乱
- `st.session_state` 保留分析历史和 KB 搜索结果
- `asyncio.run()` 桥接同步 UI 和异步 Agent
- KB 计数 (`vector_store.count()`) 在侧边栏实时显示集合大小

### ManagerAgent import bug
- `from llm.types import ProviderConfig as SettingsProvider` — 错误
- `from config.settings import ProviderConfig as SettingsProvider` — 正确
- from llm.types 的 ProviderConfig 是 Pydantic model，无 `to_dict()` 方法
- from config.settings 的 ProviderConfig 有 `to_dict()` classmethod
- 此 bug 在 Phase 2-3 被掩盖（因为 .env 文件自动设置了 env var 使 ManagerAgent 初始化成功）
