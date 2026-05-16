"""
CompetiScope Streamlit Dashboard (v2)
- Multi-competitor analysis
- Real-time progress via SSE
- Cost dashboard
- Knowledge base search
- Historical task records
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime

import streamlit as st
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from agents.manager_agent import ManagerAgent
from config.settings import (
    LLMConfig, KnowledgeBaseConfig, ProviderConfig, AgentLoopConfig, EmbedderConfig,
)
from observability.cost_tracker import cost_tracker
from observability.metrics import metrics
from rag.knowledge_base import KnowledgeBase

# ---- Page config ----
st.set_page_config(
    page_title="CompetiScope — 竞品分析Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- CSS ----
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E88E5; text-align: center; }
    .subtitle { font-size: 1rem; color: #888; text-align: center; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1rem; border-radius: 0.75rem; text-align: center;
    }
    .metric-card.green { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: #333; }
    .metric-card.orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; }
    .report-card { background-color: #f0f2f6; border-radius: 0.5rem; padding: 1.5rem; margin: 0.5rem 0; }
    .kb-hit { background-color: #e8f5e9; border-left: 3px solid #4CAF50; padding: 0.5rem; border-radius: 0.25rem; margin: 0.25rem 0; }
</style>
""", unsafe_allow_html=True)


def init_session():
    """Initialize session state."""
    defaults = {
        "report_content": None,
        "analysis_done": False,
        "task_history": [],
        "kb_search_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---- Sidebar ----
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ 配置")

        with st.expander("🔑 API 设置", expanded=True):
            api_key = st.text_input("DeepSeek API Key", type="password",
                                    value=os.getenv("DEEPSEEK_API_KEY", ""),
                                    help="留空使用 .env 中的配置")
            model = st.selectbox("模型", ["deepseek-chat", "deepseek-coder"], index=0)
            base_url = st.text_input("Base URL", value="https://api.deepseek.com")

        with st.expander("🤖 Agent 设置"):
            max_steps = st.slider("最大步数", 3, 20, AgentLoopConfig.MAX_STEPS)
            reflection = st.checkbox("启用自我反思", value=AgentLoopConfig.REFLECTION_ENABLED)
            reflection_rounds = st.slider("反思轮数", 1, 5, AgentLoopConfig.REFLECTION_ROUNDS)

        with st.expander("📚 知识库"):
            kb_enabled = st.checkbox("启用 RAG 知识库", value=KnowledgeBaseConfig.ENABLED)
            kb_clear = st.button("清空知识库", type="secondary")
            if kb_clear:
                try:
                    kb = KnowledgeBase(persist_dir=KnowledgeBaseConfig.PERSIST_DIR, enabled=True)
                    kb.vector_store.delete_collection("competitor_data")
                    kb.vector_store.delete_collection("analysis_reports")
                    kb.vector_store.delete_collection("search_cache")
                    st.success("知识库已清空")
                except Exception as exc:
                    st.warning(f"清空失败: {exc}")

        st.divider()
        st.caption(f"CompetiScope v2.0 | Phase 4")

    return api_key, model, base_url, max_steps, reflection, reflection_rounds, kb_enabled


# ---- Analysis runner ----
async def _run_analysis(competitors, report_type, api_key, model, base_url, max_steps, reflection, reflection_rounds, kb_enabled, status_placeholder):
    """Run analysis and update status placeholder."""
    # Apply overrides
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key
    os.environ["AGENT_MAX_STEPS"] = str(max_steps)
    os.environ["AGENT_REFLECTION_ENABLED"] = str(reflection).lower()
    os.environ["AGENT_REFLECTION_ROUNDS"] = str(reflection_rounds)
    os.environ["ENABLE_KNOWLEDGE_BASE"] = str(kb_enabled).lower()

    cost_tracker.reset()
    manager = ManagerAgent()

    with status_placeholder.container():
        status_text = st.empty()
        progress_bar = st.progress(0)
        step_log = st.empty()

        class UICallback:
            def __init__(self):
                self.steps = 0
            def on_step(self, step_num, thought, action):
                self.steps = step_num
                step_log.info(f"Step {step_num}: {action}")

        callback = UICallback()

        status_text.info(f"正在分析: {', '.join(competitors)}")

        result = await manager.analyze_async(
            competitors=competitors,
            analysis_dimensions=None,
            report_type=report_type,
            our_product=None,
            show_progress=False,
            enable_reflection=reflection,
        )

        progress_bar.progress(100)
        status_text.success("分析完成!" if result.get("success") else "分析失败")

    return result


def run_analysis_sync(competitors, report_type, api_key, model, base_url, max_steps, reflection, reflection_rounds, kb_enabled, status_placeholder):
    """Sync wrapper for Streamlit."""
    return asyncio.run(_run_analysis(competitors, report_type, api_key, model, base_url, max_steps, reflection, reflection_rounds, kb_enabled, status_placeholder))


# ---- Main page ----
def render_main_page(api_key, model, base_url, max_steps, reflection, reflection_rounds, kb_enabled):
    st.markdown('<p class="main-title">🔍 CompetiScope</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">企业级竞品分析 Agent · ReAct + RAG + Async</p>', unsafe_allow_html=True)

    # ---- Input area ----
    cols = st.columns([3, 1, 1])
    with cols[0]:
        competitors_str = st.text_input("竞品名称（逗号分隔）", placeholder="Notion, 飞书文档, Confluence", help="至少输入一个竞品")
    with cols[1]:
        report_type = st.selectbox("报告类型", ["full", "summary", "snapshot"])
    with cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

    # ---- Execute analysis ----
    if analyze_btn:
        competitors = [c.strip() for c in competitors_str.split(",") if c.strip()]
        if not competitors:
            st.error("请输入至少一个竞品名称")
            return

        status_placeholder = st.container()
        try:
            result = run_analysis_sync(competitors, report_type, api_key, model, base_url, max_steps, reflection, reflection_rounds, kb_enabled, status_placeholder)

            if result.get("success"):
                st.session_state.report_content = result.get("report", "")
                st.session_state.analysis_done = True
                # Record in history
                st.session_state.task_history.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "competitors": competitors,
                    "report_type": report_type,
                    "agent_steps": result.get("agent_steps", 0),
                    "cost": result.get("cost", {}),
                })
            else:
                st.error(f"分析失败: {result.get('error', '未知错误')}")
        except Exception as exc:
            st.error(f"发生错误: {exc}")

    # ---- Show results ----
    if st.session_state.analysis_done and st.session_state.report_content:
        st.divider()

        # Tabs for different views
        tab_report, tab_cost, tab_kb = st.tabs(["📋 报告", "💰 费用", "📚 知识库"])

        with tab_report:
            st.download_button("📥 下载 Markdown", data=st.session_state.report_content,
                               file_name=f"competiscope_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                               mime="text/markdown")
            with st.expander("完整报告", expanded=True):
                st.markdown(st.session_state.report_content)

        with tab_cost:
            cost_summary = cost_tracker.summary()
            if cost_summary["call_count"] > 0:
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("LLM 调用次数", cost_summary["call_count"])
                mc2.metric("总 Tokens", f"{cost_summary['total_tokens']:,}")
                mc3.metric("输入 Tokens", f"{cost_summary['total_input_tokens']:,}")
                mc4.metric("费用", f"${cost_summary['total_cost_usd']:.6f}")

                if cost_summary.get("by_type"):
                    st.subheader("按调用类型")
                    st.json(cost_summary["by_type"])
            else:
                st.info("暂无费用数据")

        with tab_kb:
            st.subheader("知识库检索")
            kb_query = st.text_input("搜索知识库", placeholder="例如: Notion 定价", key="kb_search")
            if st.button("搜索", key="kb_search_btn") and kb_query:
                try:
                    kb = KnowledgeBase(persist_dir=KnowledgeBaseConfig.PERSIST_DIR, enabled=True)
                    if kb.enabled:
                        results = asyncio.run(kb.search_all(kb_query, n_results=5))
                        for col_name, hits in results.items():
                            if hits:
                                st.markdown(f"**{col_name}** ({len(hits)} results)")
                                for h in hits[:3]:
                                    st.markdown(f'<div class="kb-hit"><strong>{h.get("id", "")}</strong>: {h.get("content", "")[:300]}...</div>', unsafe_allow_html=True)
                    else:
                        st.info("知识库未启用")
                except Exception as exc:
                    st.warning(f"搜索失败: {exc}")

    # ---- History ----
    if st.session_state.task_history:
        st.divider()
        st.subheader("📜 历史分析")
        for i, h in enumerate(reversed(st.session_state.task_history[-5:])):
            with st.expander(f"{h['time']} — {', '.join(h['competitors'])} ({h['report_type']})"):
                st.write(f"Agent 步数: {h['agent_steps']}")
                if h.get("cost"):
                    st.write(f"费用: ${h['cost'].get('total_cost_usd', 0):.6f}")


# ---- Dashboard page (separate tab concept) ----
def render_dashboard():
    """Show operational metrics if available."""
    ms = metrics.summary()
    if ms["total_analyses"] == 0:
        return

    st.divider()
    st.subheader("📊 运行指标")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("总分析数", ms["total_analyses"])
    mc2.metric("成功率", f"{ms['success_rate']*100:.0f}%")
    mc3.metric("平均耗时", f"{ms['avg_duration_seconds']:.1f}s")
    mc4.metric("平均步数", ms["avg_agent_steps"])


# ---- Init KB from UI ----
def _kb_status():
    try:
        kb = KnowledgeBase(persist_dir=KnowledgeBaseConfig.PERSIST_DIR, enabled=True)
        if kb.enabled:
            return {
                "competitor_data": kb.vector_store.count("competitor_data"),
                "analysis_reports": kb.vector_store.count("analysis_reports"),
                "search_cache": kb.vector_store.count("search_cache"),
            }
    except Exception:
        pass
    return None


# ---- Main ----
def main():
    init_session()
    api_key, model, base_url, max_steps, reflection, reflection_rounds, kb_enabled = render_sidebar()
    render_main_page(api_key, model, base_url, max_steps, reflection, reflection_rounds, kb_enabled)
    render_dashboard()

    # Footer status bar
    kb_counts = _kb_status()
    if kb_counts:
        st.sidebar.divider()
        st.sidebar.caption(f"📚 KB: 竞品数据 {kb_counts['competitor_data']} | 报告 {kb_counts['analysis_reports']} | 缓存 {kb_counts['search_cache']}")


if __name__ == "__main__":
    main()
