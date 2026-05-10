"""
竞品分析Agent系统 - Streamlit Web界面

提供可视化界面来执行竞品分析任务
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from agents.manager_agent import ManagerAgent
from config.settings import LLMConfig


# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="竞品分析Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* 副标题样式 */
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* 分析按钮样式 */
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 0.5rem;
    }

    .stButton > button:hover {
        background-color: #1976D2;
    }

    /* 卡片样式 */
    .report-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* 进度信息样式 */
    .progress-info {
        font-size: 0.9rem;
        color: #666;
        padding: 0.5rem;
        background-color: #e3f2fd;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }

    /* 侧边栏标题 */
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E88E5;
        margin-bottom: 1rem;
    }

    /* 成功提示 */
    .success-box {
        padding: 1rem;
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }

    /* 错误提示 */
    .error-box {
        padding: 1rem;
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 辅助函数
# ============================================================

def init_session_state():
    """初始化会话状态"""
    if "report_content" not in st.session_state:
        st.session_state.report_content = None
    if "report_filename" not in st.session_state:
        st.session_state.report_filename = None
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False


def run_competitor_analysis(
    competitor1: str,
    competitor2: str,
    api_key: str,
    model: str,
    base_url: str
) -> dict:
    """
    执行竞品分析

    Args:
        competitor1: 竞品1名称
        competitor2: 竞品2名称
        api_key: API密钥
        model: 模型名称
        base_url: API基础URL

    Returns:
        分析结果
    """
    # 临时设置环境变量
    original_key = os.environ.get("DEEPSEEK_API_KEY")
    original_model = os.environ.get("DEEPSEEK_MODEL")
    original_url = os.environ.get("DEEPSEEK_BASE_URL")

    os.environ["DEEPSEEK_API_KEY"] = api_key
    os.environ["DEEPSEEK_MODEL"] = model
    os.environ["DEEPSEEK_BASE_URL"] = base_url

    try:
        # 重新初始化配置
        LLMConfig.DEEPSEEK_API_KEY = api_key
        LLMConfig.DEEPSEEK_MODEL = model
        LLMConfig.DEEPSEEK_BASE_URL = base_url

        # 执行分析
        manager = ManagerAgent()

        result = manager.analyze(
            competitors=[competitor1, competitor2],
            report_type="full",
            show_progress=False,
        )

        return result

    finally:
        # 恢复环境变量
        if original_key:
            os.environ["DEEPSEEK_API_KEY"] = original_key
        if original_model:
            os.environ["DEEPSEEK_MODEL"] = original_model
        if original_url:
            os.environ["DEEPSEEK_BASE_URL"] = original_url


def get_download_filename(competitors: list[str]) -> str:
    """生成下载文件名"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    competitor_str = "_vs_".join(competitors[:2])
    return f"竞品分析_{competitor_str}_{timestamp}.md"


# ============================================================
# 侧边栏配置
# ============================================================

def render_sidebar():
    """渲染侧边栏配置"""
    with st.sidebar:
        st.markdown('<p class="sidebar-title">⚙️ 配置设置</p>', unsafe_allow_html=True)

        # API配置
        st.subheader("API 配置")

        # API Key输入
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            help="输入您的DeepSeek API密钥"
        )

        # 模型选择
        model = st.selectbox(
            "选择模型",
            options=["deepseek-chat", "deepseek-coder"],
            index=0,
            help="选择要使用的模型"
        )

        # Base URL（可选，高级用户）
        with st.expander("高级设置"):
            base_url = st.text_input(
                "API Base URL",
                value="https://api.deepseek.com",
                help="API服务地址（通常不需要修改）"
            )

        # 分割线
        st.divider()

        # 搜索配置
        st.subheader("搜索配置")

        search_engine = st.selectbox(
            "搜索引擎",
            options=["duckduckgo", "serper"],
            index=0,
            help="选择搜索数据源"
        )

        # 关于信息
        st.divider()
        st.markdown("""
        ### 💡 使用说明

        1. 在左侧配置API Key
        2. 输入两个产品名称
        3. 点击"开始分析"
        4. 等待分析完成，查看报告

        ### 🔍 分析内容

        - 产品功能对比
        - 市场表现分析
        - SWOT分析
        - 竞争格局洞察
        - 战略建议
        """)

    return api_key, model, base_url, provider


# ============================================================
# 主页面
# ============================================================

def render_main_page(api_key: str, model: str, base_url: str):
    """渲染主页面"""

    # 标题区域
    st.markdown('<p class="main-title">🔍 竞品分析Agent</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">智能竞品分析 · 自动报告生成 · 多维度洞察</p>',
        unsafe_allow_html=True
    )

    # 创建两列布局
    col1, col2 = st.columns(2)

    with col1:
        competitor1 = st.text_input(
            "🔵 竞品 1",
            placeholder="例如：Notion",
            help="输入第一个竞品名称"
        )

    with col2:
        competitor2 = st.text_input(
            "🟢 竞品 2",
            placeholder="例如：飞书",
            help="输入第二个竞品名称"
        )

    # 中心按钮区域
    st.markdown("<br>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        analyze_button = st.button(
            "🚀 开始分析",
            type="primary",
            use_container_width=True
        )

    # 分析按钮点击处理
    if analyze_button:
        # 输入验证
        if not api_key:
            st.error("⚠️ 请先在侧边栏配置API Key")
            return

        if not competitor1 or not competitor2:
            st.error("⚠️ 请输入两个竞品名称")
            return

        if competitor1 == competitor2:
            st.error("⚠️ 两个竞品名称不能相同")
            return

        # 执行分析
        with st.spinner("🔄 正在分析竞品，请稍候..."):
            try:
                # 显示进度信息
                progress_container = st.container()

                with progress_container:
                    st.info("📡 正在连接API服务...")
                    st.info("🔍 正在采集竞品数据...")
                    st.info("📊 正在分析竞品信息...")
                    st.info("✍️ 正在生成分析报告...")

                # 执行分析
                result = run_competitor_analysis(
                    competitor1=competitor1,
                    competitor2=competitor2,
                    api_key=api_key,
                    model=model,
                    base_url=base_url
                )

                # 清空进度信息
                progress_container.empty()

                if result.get("success"):
                    # 保存结果到会话状态
                    st.session_state.report_content = result.get("report", "")
                    st.session_state.report_filename = get_download_filename(
                        [competitor1, competitor2]
                    )
                    st.session_state.analysis_done = True

                    st.success("✅ 分析完成！")

                    # 显示结果
                    render_results()

                else:
                    error_msg = result.get("error", "未知错误")
                    st.error(f"❌ 分析失败: {error_msg}")

            except Exception as e:
                st.error(f"❌ 发生错误: {str(e)}")

    # 如果已有分析结果，显示结果
    elif st.session_state.analysis_done and st.session_state.report_content:
        render_results()


def render_results():
    """渲染分析结果"""
    if not st.session_state.report_content:
        return

    report = st.session_state.report_content
    filename = st.session_state.report_filename

    # 结果容器
    st.divider()
    st.subheader("📋 分析报告")

    # 工具栏
    col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 2])

    with col_dl1:
        # 下载按钮
        st.download_button(
            label="📥 下载报告",
            data=report,
            file_name=filename,
            mime="text/markdown",
            type="primary"
        )

    with col_dl2:
        # 复制按钮（使用JavaScript）
        st.button(
            label="📋 复制报告",
            on_click=lambda: None
        )

    # 报告展示
    st.markdown("---")

    # 使用expander包装报告，便于折叠
    with st.expander("📄 展开/收起完整报告", expanded=True):
        st.markdown(report, unsafe_allow_html=False)

    # 显示数据来源
    st.divider()
    st.caption("💡 提示：报告由AI自动生成，数据来源于公开网络信息")


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    # 初始化会话状态
    init_session_state()

    # 渲染侧边栏
    api_key, model, base_url = render_sidebar()

    # 渲染主页面
    render_main_page(api_key, model, base_url)


if __name__ == "__main__":
    main()
