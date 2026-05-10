"""
网页内容抓取工具模块

提供基于BeautifulSoup的网页内容抓取功能，包含：
- 智能内容提取
- 多种内容类型支持
- 错误处理和重试
- 内容清洗和格式化
"""

import time
import re
from typing import Optional, Literal
from dataclasses import dataclass, field
from urllib.parse import urlparse
from loguru import logger

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


@dataclass
class ScrapedContent:
    """抓取内容数据类"""
    url: str                           # 原始URL
    title: str = ""                    # 页面标题
    content: str = ""                  # 主要内容（清洗后）
    raw_content: str = ""              # 原始内容
    summary: str = ""                  # 内容摘要
    author: str = ""                   # 作者
    publish_date: str = ""             # 发布日期
    images: list[str] = field(default_factory=list)  # 图片列表
    links: list[str] = field(default_factory=list)    # 链接列表
    content_type: Literal["article", "product", "list", "unknown"] = "unknown"  # 内容类型
    language: str = "unknown"         # 页面语言

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "author": self.author,
            "publish_date": self.publish_date,
            "images": self.images,
            "links": self.links,
            "content_type": self.content_type,
            "language": self.language
        }


class WebScraperError(Exception):
    """网页抓取异常基类"""
    pass


class ScrapingTimeoutError(WebScraperError):
    """抓取超时异常"""
    pass


class InvalidURLError(WebScraperError):
    """无效URL异常"""
    pass


class ContentExtractionError(WebScraperError):
    """内容提取异常"""
    pass


class WebScraper:
    """
    网页内容抓取工具类

    提供可靠的网页内容抓取功能，支持：
    - 智能内容提取
    - 内容类型识别
    - 自动重试
    - 多语言支持
    """

    # 常见的内容选择器模式
    CONTENT_SELECTORS = [
        # 通用文章内容选择器
        "article",
        "[role='main']",
        "main",
        ".content",
        ".article-content",
        ".post-content",
        ".entry-content",
        ".story-body",
        "#content",
        ".main-content",
        # 新闻类
        ".news-content",
        ".article-body",
        # 产品类
        ".product-description",
        ".product-detail",
        # 论坛类
        ".post-body",
        ".comment-content",
    ]

    # 需要移除的噪声元素选择器
    NOISE_SELECTORS = [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        ".sidebar",
        ".navigation",
        ".menu",
        ".advertisement",
        ".ad",
        ".ads",
        ".social-share",
        ".comments",
        ".related-posts",
        ".popup",
        ".modal",
        "[role='navigation']",
        "[role='banner']",
        "[role='contentinfo']",
    ]

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        request_interval: float = 1.0,
        user_agent: Optional[str] = None
    ):
        """
        初始化网页抓取工具

        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            request_interval: 请求间隔（秒）
            user_agent: 自定义User-Agent（可选）
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_interval = request_interval

        # 默认User-Agent
        self.default_headers = {
            "User-Agent": user_agent or (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        logger.info(
            f"WebScraper初始化完成: timeout={timeout}s, "
            f"max_retries={max_retries}"
        )

    def _validate_url(self, url: str) -> bool:
        """
        验证URL格式

        Args:
            url: 待验证的URL

        Returns:
            是否有效
        """
        try:
            result = urlparse(url)
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False

    def _clean_text(self, text: str) -> str:
        """
        清洗文本内容

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        # 移除多余空白
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", "\n", text)

        # 移除特殊字符（保留中文、英文、数字、常用标点）
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:'\"()。、，！？；：""''（）【】《》]", "", text)

        return text.strip()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        提取页面标题

        Args:
            soup: BeautifulSoup对象

        Returns:
            页面标题
        """
        # 尝试多个选择器
        selectors = [
            "h1",
            "article h1",
            ".article-title",
            ".post-title",
            ".entry-title",
            "[class*='title']",
            "title"
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element and element.get_text(strip=True):
                return element.get_text(strip=True)

        return ""

    def _extract_author(self, soup: BeautifulSoup) -> str:
        """
        提取作者信息

        Args:
            soup: BeautifulSoup对象

        Returns:
            作者名称
        """
        author_selectors = [
            "[rel='author']",
            ".author",
            ".byline",
            "[class*='author']",
            "[itemprop='author']",
            "meta[name='author']"
        ]

        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == "meta":
                    return element.get("content", "")
                return element.get_text(strip=True)

        return ""

    def _extract_publish_date(self, soup: BeautifulSoup) -> str:
        """
        提取发布日期

        Args:
            soup: BeautifulSoup对象

        Returns:
            发布日期
        """
        date_selectors = [
            "time[datetime]",
            "[class*='date']",
            "[class*='time']",
            "[itemprop='datePublished']",
            "meta[property='article:published_time']"
        ]

        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == "meta":
                    return element.get("content", "")
                if element.name == "time":
                    return element.get("datetime", element.get_text(strip=True))
                return element.get_text(strip=True)

        return ""

    def _extract_content(self, soup: BeautifulSoup) -> tuple[str, str]:
        """
        提取主要内容

        Args:
            soup: BeautifulSoup对象

        Returns:
            (主要内容, 内容摘要)
        """
        # 移除噪声元素
        for selector in self.NOISE_SELECTORS:
            for element in soup.select(selector):
                element.decompose()

        # 尝试使用预定义选择器
        content = ""
        for selector in self.CONTENT_SELECTORS:
            element = soup.select_one(selector)
            if element:
                content = element.get_text(separator="\n", strip=True)
                break

        # 如果没有找到，尝试使用body
        if not content:
            body = soup.find("body")
            if body:
                content = body.get_text(separator="\n", strip=True)

        # 生成摘要（取前200字）
        summary = content[:200] + "..." if len(content) > 200 else content

        return content, summary

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """
        提取页面图片

        Args:
            soup: BeautifulSoup对象
            base_url: 基础URL（用于处理相对路径）

        Returns:
            图片URL列表
        """
        images = []
        for img in soup.select("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-original")

            if src:
                # 处理相对URL
                if src.startswith("//"):
                    src = "https:" + src
                elif src.startswith("/"):
                    parsed = urlparse(base_url)
                    src = f"{parsed.scheme}://{parsed.netloc}{src}"

                if src.startswith("http"):
                    images.append(src)

        return images[:20]  # 限制最多20张图片

    def _extract_links(self, soup: BeautifulSoup) -> list[str]:
        """
        提取页面链接

        Args:
            soup: BeautifulSoup对象

        Returns:
            链接URL列表
        """
        links = []
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if href.startswith("http") and "google.com" not in href:
                links.append(href)

        return list(set(links))[:50]  # 去重并限制数量

    def _detect_content_type(self, soup: BeautifulSoup, url: str) -> str:
        """
        识别内容类型

        Args:
            soup: BeautifulSoup对象
            url: 页面URL

        Returns:
            内容类型
        """
        # 根据URL判断
        url_lower = url.lower()
        if "news" in url_lower or "article" in url_lower:
            return "article"
        if "product" in url_lower or "item" in url_lower:
            return "product"

        # 根据页面结构判断
        if soup.select_one("article"):
            return "article"
        if soup.select_one("[class*='product']") or soup.select_one("[class*='price']"):
            return "product"
        if soup.select(".result") or soup.select(".list-item"):
            return "list"

        return "unknown"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True
    )
    def scrape(self, url: str) -> ScrapedContent:
        """
        抓取网页内容

        Args:
            url: 网页URL

        Returns:
            抓取的内容对象

        Raises:
            InvalidURLError: 无效的URL
            ScrapingTimeoutError: 抓取超时
            WebScraperError: 其他抓取错误
        """
        # 验证URL
        if not self._validate_url(url):
            raise InvalidURLError(f"无效的URL: {url}")

        logger.info(f"开始抓取网页: {url}")

        try:
            # 发送请求
            response = httpx.get(
                url,
                headers=self.default_headers,
                timeout=self.timeout,
                follow_redirects=True
            )
            response.raise_for_status()

            # 判断编码
            response.encoding = response.charset_encoding or "utf-8"

            # 解析HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # 移除script和style标签
            for tag in soup(["script", "style"]):
                tag.decompose()

            # 提取各部分内容
            title = self._extract_title(soup)
            author = self._extract_author(soup)
            publish_date = self._extract_publish_date(soup)
            content, summary = self._extract_content(soup)
            images = self._extract_images(soup, url)
            links = self._extract_links(soup)
            content_type = self._detect_content_type(soup, url)

            # 清洗内容
            content = self._clean_text(content)

            result = ScrapedContent(
                url=url,
                title=title,
                content=content,
                raw_content=soup.get_text(separator="\n", strip=True),
                summary=summary,
                author=author,
                publish_date=publish_date,
                images=images,
                links=links,
                content_type=content_type,
                language="zh" if self._is_chinese(title + content) else "en"
            )

            logger.info(
                f"网页抓取完成: {url}, "
                f"标题='{title[:30]}...' "
                if len(title) > 30 else f"标题='{title}', "
                f"内容长度={len(content)}字符"
            )

            # 请求间隔
            time.sleep(self.request_interval)

            return result

        except httpx.TimeoutException:
            logger.error(f"网页抓取超时: {url}")
            raise ScrapingTimeoutError(f"抓取超时: {url}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP状态错误: {e.response.status_code} - {url}")
            raise WebScraperError(f"HTTP错误 {e.response.status_code}: {url}")
        except httpx.RequestError as e:
            logger.error(f"请求错误: {e}")
            raise WebScraperError(f"请求错误: {e}")
        except Exception as e:
            logger.error(f"网页抓取出错: {e}")
            raise WebScraperError(f"抓取失败: {e}")

    def scrape_multiple(self, urls: list[str]) -> list[ScrapedContent]:
        """
        批量抓取多个网页

        Args:
            urls: URL列表

        Returns:
            抓取结果列表（包含成功和失败的结果）
        """
        results = []

        logger.info(f"开始批量抓取 {len(urls)} 个网页")

        for i, url in enumerate(urls):
            try:
                result = self.scrape(url)
                results.append(result)
                logger.info(f"批量抓取进度: {i+1}/{len(urls)}")
            except WebScraperError as e:
                logger.warning(f"跳过无效URL {url}: {e}")
                continue

        logger.info(f"批量抓取完成: 成功 {len(results)}/{len(urls)}")
        return results

    def _is_chinese(self, text: str) -> bool:
        """
        判断文本是否包含中文

        Args:
            text: 待检测文本

        Returns:
            是否包含中文
        """
        if not text:
            return False
        chinese_char_ratio = sum(1 for c in text if '\u4e00' <= c <= '\u9fff') / len(text)
        return chinese_char_ratio > 0.3


# 导出类
__all__ = [
    "ScrapedContent",
    "WebScraper",
    "WebScraperError",
    "ScrapingTimeoutError",
    "InvalidURLError",
    "ContentExtractionError",
]
