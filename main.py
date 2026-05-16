"""
竞品分析Agent系统 - 主入口 (v2)

提供命令行界面和API接口来启动竞品分析任务
Phase 1 升级: 异步执行 + 费用追踪
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from agents.manager_agent import ManagerAgent
from config.settings import (
    LLMConfig,
    OutputConfig,
    LogConfig,
    validate_config,
)


def setup_logging(level: str = "INFO"):
    """
    配置日志

    Args:
        level: 日志级别
    """
    # 移除默认的logger配置
    logger.remove()

    # 添加控制台输出
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
    )

    # 确保日志目录存在
    LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 添加文件输出
    log_file = LogConfig.LOG_DIR / f"competiscope_{datetime.now().strftime('%Y%m%d')}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=level,
        rotation="00:00",  # 每天轮转
        retention="30 days",  # 保留30天
        encoding="utf-8",
    )


def setup_environment():
    """设置环境变量和验证配置"""
    # 加载.env文件
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"已加载环境变量文件: {env_file}")
    else:
        logger.warning("未找到.env文件，请复制.env.example创建.env文件")

    # 验证配置
    is_valid, errors = validate_config()
    if not is_valid:
        logger.warning(f"配置验证发现问题: {errors}")


def parse_arguments():
    """
    解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="竞品分析Agent系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析指定竞品
  python main.py --competitors "竞品A,竞品B,竞品C"

  # 快速概览模式
  python main.py --competitors "竞品A,竞品B" --report-type snapshot

  # 智能规划模式（自动识别竞品）
  python main.py --target "目标公司"

  # 输出到指定目录
  python main.py --competitors "竞品A" --output ./my_reports
        """
    )

    # 竞品相关参数
    parser.add_argument(
        "-c", "--competitors",
        type=str,
        help="竞品列表，用逗号分隔"
    )
    parser.add_argument(
        "-t", "--target",
        type=str,
        help="目标公司/产品（智能规划模式）"
    )
    parser.add_argument(
        "-d", "--dimensions",
        type=str,
        help="分析维度，用逗号分隔（可选）"
    )
    parser.add_argument(
        "--our-product",
        type=str,
        help="我们产品名称（用于对比）"
    )

    # 报告相关参数
    parser.add_argument(
        "-r", "--report-type",
        type=str,
        choices=["full", "summary", "snapshot"],
        default="full",
        help="报告类型: full(完整报告), summary(执行摘要), snapshot(快速概览)"
    )

    # 输出相关参数
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="输出目录"
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="保存原始采集数据"
    )

    # 系统参数
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示进度信息"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅验证配置，不执行分析"
    )

    return parser.parse_args()


async def run_analysis_async(
    competitors: list[str],
    report_type: str = "full",
    dimensions: Optional[list[str]] = None,
    our_product: Optional[str] = None,
    show_progress: bool = True,
    output_dir: Optional[str] = None
) -> dict:
    """
    异步执行竞品分析

    Args:
        competitors: 竞品列表
        report_type: 报告类型
        dimensions: 分析维度
        our_product: 我们产品
        show_progress: 显示进度
        output_dir: 输出目录

    Returns:
        分析结果
    """
    manager = ManagerAgent()

    result = await manager.analyze_async(
        competitors=competitors,
        analysis_dimensions=dimensions,
        report_type=report_type,
        our_product=our_product,
        show_progress=show_progress,
    )

    if result.get("success") and result.get("report"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        competitor_str = "_".join(competitors[:2])
        filename = f"竞品分析_{competitor_str}_{timestamp}.md"

        output_path = output_dir or str(OutputConfig.OUTPUT_DIR)
        filepath = Path(output_path) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(result["report"])

        result["output_file"] = str(filepath)
        logger.info(f"报告已保存: {filepath}")

    return result


async def run_smart_analysis_async(
    target: str,
    report_type: str = "full",
    show_progress: bool = True,
    output_dir: Optional[str] = None
) -> dict:
    """
    智能规划模式异步分析

    Args:
        target: 目标公司
        report_type: 报告类型
        show_progress: 显示进度
        output_dir: 输出目录

    Returns:
        分析结果
    """
    manager = ManagerAgent()

    result = await manager.analyze_with_planning_async(
        target_company=target,
        report_type=report_type,
        show_progress=show_progress,
    )

    if result.get("success") and result.get("report"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"竞品分析_{target}_{timestamp}.md"

        output_path = output_dir or str(OutputConfig.OUTPUT_DIR)
        filepath = Path(output_path) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(result["report"])

        result["output_file"] = str(filepath)
        logger.info(f"报告已保存: {filepath}")

    return result


# 同步兼容接口
def run_analysis(**kwargs) -> dict:
    return asyncio.run(run_analysis_async(**kwargs))


def run_smart_analysis(**kwargs) -> dict:
    return asyncio.run(run_smart_analysis_async(**kwargs))


def print_banner():
    """打印Banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           🔍  竞品分析Agent系统  CompetiScope               ║
║                                                              ║
║           智能竞品分析 · 自动报告生成 · 多维度洞察           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


async def main_async():
    """异步主函数"""
    args = parse_arguments()
    setup_logging(args.log_level)
    print_banner()
    setup_environment()

    logger.info("竞品分析系统启动 (Async v2)")

    if args.dry_run:
        logger.info("配置验证模式")
        is_valid, errors = validate_config()
        if is_valid:
            print("\n✅ 配置验证通过")
        else:
            print("\n❌ 配置验证失败:")
            for error in errors:
                print(f"   - {error}")
        return

    if not args.competitors and not args.target:
        print("❌ 错误: 请指定竞品列表(--competitors)或目标公司(--target)")
        print("       使用 --help 查看帮助")
        sys.exit(1)

    try:
        dimensions = None
        if args.dimensions:
            dimensions = [d.strip() for d in args.dimensions.split(",")]

        if args.target:
            result = await run_smart_analysis_async(
                target=args.target,
                report_type=args.report_type,
                show_progress=not args.no_progress,
                output_dir=args.output,
            )
        else:
            competitors = [c.strip() for c in args.competitors.split(",")]
            result = await run_analysis_async(
                competitors=competitors,
                report_type=args.report_type,
                dimensions=dimensions,
                our_product=args.our_product,
                show_progress=not args.no_progress,
                output_dir=args.output,
            )

        print("\n" + "=" * 60)
        if result.get("success"):
            print("✅ 竞品分析完成！")
            if result.get("output_file"):
                print(f"📄 报告已保存: {result['output_file']}")
            if result.get("agent_steps"):
                print(f"🤖 Agent 步数: {result['agent_steps']}")
            if result.get("reflection_rounds"):
                print(f"🔍 反思轮数: {result['reflection_rounds']}, 评分: {result.get('reflection_scores', [])}")
            if result.get("collected_data"):
                print(f"📊 分析了 {len(result['collected_data'])} 个竞品")
            if result.get("cost"):
                cost = result["cost"]
                print(f"💰 API费用: ${cost['total_cost_usd']:.6f} ({cost['total_tokens']} tokens)")
        else:
            print("❌ 竞品分析失败")
            if result.get("error"):
                print(f"错误: {result['error']}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断分析")
        sys.exit(130)
    except Exception as e:
        logger.exception("分析过程发生异常")
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)


def main():
    """主函数入口（同步包装）"""
    asyncio.run(main_async())


# API接口函数（供其他模块调用）
async def analyze_competitors_async(
    competitors: list[str],
    report_type: str = "full",
    dimensions: Optional[list[str]] = None,
    our_product: Optional[str] = None,
) -> dict:
    """
    异步分析竞品的API函数

    Args:
        competitors: 竞品列表
        report_type: 报告类型
        dimensions: 分析维度
        our_product: 我们产品

    Returns:
        分析结果字典
    """
    setup_environment()
    return await run_analysis_async(
        competitors=competitors,
        report_type=report_type,
        dimensions=dimensions,
        our_product=our_product,
        show_progress=True,
    )


def analyze_competitors(**kwargs) -> dict:
    """同步兼容接口"""
    return asyncio.run(analyze_competitors_async(**kwargs))


async def analyze_target_async(target: str, report_type: str = "full") -> dict:
    """异步智能分析目标公司的API函数"""
    setup_environment()
    return await run_smart_analysis_async(target=target, report_type=report_type, show_progress=True)


def analyze_target(**kwargs) -> dict:
    """同步兼容接口"""
    return asyncio.run(analyze_target_async(**kwargs))


def main_cli():
    """pyproject.toml [project.scripts] 入口"""
    main()


if __name__ == "__main__":
    main()
