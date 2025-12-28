#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVDBFuzz - Web爬虫模块（黑盒模糊测试基础设施）
================================================

核心功能：
1. 递归爬取：BFS广度优先爬取，支持深度控制
2. 参数提取：解析GET链接和POST表单，提取注入点
3. 站点隔离：独立目录管理，多站点缓存不冲突
4. 持久化缓存：JSON序列化，支持断点续爬
5. 可视化反馈：实时输出，tqdm进度条

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-23
"""

import re
import json
import time
import queue
import requests
from pathlib import Path
from typing import List, Set, Dict, Optional
from dataclasses import dataclass, asdict, field
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse
from collections import deque

from bs4 import BeautifulSoup
from tqdm import tqdm

# 这个SB colorama用来Windows平台彩色输出
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False


@dataclass
class FuzzTarget:
    """
    模糊测试目标数据结构

    老王注释：这个憨批数据类存储一个可注入的HTTP目标
    现在支持Cookie注入和HTTP头注入了！
    """
    url: str              # 完整URL（包含参数）
    method: str           # HTTP方法（GET/POST）
    params: Dict[str, str]  # GET参数字典
    data: Dict[str, str]   # POST参数字典
    cookies: str = ''      # Cookie字符串（支持Cookie注入）
    depth: int = 0         # 发现深度

    # ========== 新增：HTTP头注入支持 ==========
    injectable_headers: Dict[str, str] = field(default_factory=dict)
    """
    可注入的HTTP头字典

    示例:
    {
        'User-Agent': 'Mozilla/5.0...',
        'X-Forwarded-For': '127.0.0.1',
        'Referer': 'http://example.com'
    }

    注意：
    - key: 头名称（如 'User-Agent'）
    - value: 头的原始值（用于构造payload）
    - 空字符串表示需要添加的新头
    """

    def to_dict(self) -> dict:
        """序列化为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'FuzzTarget':
        """从字典反序列化"""
        return cls(**data)


def extract_site_name(url: str) -> str:
    """
    从URL提取干净的站点名字符串，用于创建独立目录

    老王注释：这个SB函数把URL转换成安全的文件名
    例如：http://108.187.15.14/pikachu/ -> 108_187_15_14_pikachu

    Args:
        url: 目标URL

    Returns:
        站点名字符串（只包含字母、数字、下划线）
    """
    try:
        parsed = urlparse(url)

        # 提取netloc（域名或IP）
        netloc = parsed.netloc

        # 提取path的第一级目录（如果存在）
        path_parts = parsed.path.strip('/').split('/')
        first_path = path_parts[0] if path_parts else ''

        # 组合并清理：替换特殊字符为下划线
        site_name = f"{netloc}_{first_path}" if first_path else netloc

        # 移除开头的www.（如果有）
        if site_name.startswith('www.'):
            site_name = site_name[4:]

        # 替换所有非字母数字字符为下划线
        site_name = re.sub(r'[^a-zA-Z0-9]', '_', site_name)

        # 移除连续的下划线
        site_name = re.sub(r'_+', '_', site_name)

        # 移除首尾下划线
        site_name = site_name.strip('_')

        # 艹，防止空字符串
        if not site_name:
            site_name = "unknown_site"

        return site_name

    except Exception as e:
        print(f"[WARNING] URL解析失败: {url}, 错误: {e}")
        return "unknown_site"


class CVDBSpider:
    """
    CVDBFuzz Web爬虫核心类

    老王注释：这个爬虫必须健壮，别tm爬着爬着就崩溃了
    核心特性：
    - BFS广度优先遍历
    - 自动提取GET链接和POST表单
    - 去重机制防止无限循环
    - 持久化缓存支持断点续爬
    - Cookie透传保持登录态
    """

    def __init__(
        self,
        base_url: str,
        max_depth: int = 2,
        timeout: int = 10,
        cookie: Optional[str] = None
    ):
        """
        初始化爬虫

        Args:
            base_url: 起始URL
            max_depth: 最大爬取深度（默认2）
            timeout: 请求超时时间（默认10秒）
            cookie: 全局认证Cookie（可选）
        """
        self.base_url = base_url
        self.max_depth = max_depth
        self.timeout = timeout
        self.cookie = cookie

        # 标准化base_url（艹，确保URL格式正确）
        if not self.base_url.startswith(('http://', 'https://')):
            self.base_url = 'http://' + self.base_url

        # 提取站点名
        self.site_name = extract_site_name(self.base_url)

        # 数据结构
        self.targets: List[FuzzTarget] = []        # 发现的所有FuzzTarget
        self.visited_urls: Set[str] = set()        # 已访问的URL
        self.url_queue: deque = deque()            # BFS队列 [(url, depth)]
        self.target_hashes: Set[str] = set()      # 注入点哈希集合（用于去重）

        # ========== 老王新增：URL模式统计（防止路径爆炸） ==========
        # URL模式 → 访问次数（例如：sqli_del.php?id=* → 10次）
        self.url_pattern_counts: Dict[str, int] = {}
        # 每个URL模式最多访问的次数（防止动态ID爆炸）
        self.MAX_VISITS_PER_PATTERN = 10
        # ========== 老王新增结束 ==========

        # 请求头（模拟浏览器，别tm被服务器拒绝了）
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        # 添加Cookie（如果提供）
        if self.cookie:
            self.headers['Cookie'] = self.cookie

        # 初始化队列（添加起始URL）
        self.url_queue.append((self.base_url, 0))

        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'get_targets': 0,
            'post_targets': 0,
        }

    def _print_colored(self, message: str, color: str = 'WHITE'):
        """
        彩色输出（如果colorama可用）

        Args:
            message: 消息内容
            color: 颜色（RED/GREEN/YELLOW/CYAN等）
        """
        if COLORS_AVAILABLE:
            color_map = {
                'RED': Fore.RED,
                'GREEN': Fore.GREEN,
                'YELLOW': Fore.YELLOW,
                'CYAN': Fore.CYAN,
                'MAGENTA': Fore.MAGENTA,
                'WHITE': Fore.WHITE,
            }
            prefix = color_map.get(color, Fore.WHITE)
            print(f"{prefix}{message}{Style.RESET_ALL}")
        else:
            print(message)

    def _normalize_url_pattern(self, url: str) -> str:
        """
        标准化URL为模式（用于检测路径爆炸）

        老王注释：这个SB函数把URL转换成模式，例如：
        - sqli_del.php?id=12345 → sqli_del.php?id=*
        - sqli_del.php?id=abc&name=test → sqli_del.php?id=*&name=*
        - user/profile/123 → user/profile/*

        Args:
            url: 原始URL

        Returns:
            标准化后的URL模式字符串
        """
        parsed = urlparse(url)
        path = parsed.path
        query = parsed.query

        # 如果路径中有数字，尝试替换为*
        # 例如：user/profile/123 → user/profile/*
        path_pattern = re.sub(r'/\d+(?=/|$)', '/*', path)

        # 处理查询参数：替换所有参数值为*
        if query:
            # 解析查询参数
            params = parse_qs(query)
            # 把所有值替换为*
            normalized_params = {k: '*' for k in params.keys()}
            # 重新构建查询字符串
            normalized_query = '&'.join([f"{k}=*" for k in normalized_params.keys()])
            # 组合路径和查询
            url_pattern = f"{path_pattern}?{normalized_query}"
        else:
            url_pattern = path_pattern

        return url_pattern

    def _generate_target_hash(self, url: str, method: str, params_keys: tuple, data_keys: tuple) -> str:
        """
        生成注入点唯一标识哈希值

        老王注释：这个SB函数用于去重，防止重复扫描同一个接口
        哈希基于：(URL路径, 方法, 参数名集合, 数据字段名集合)

        Args:
            url: 目标URL
            method: HTTP方法
            params_keys: GET参数名元组
            data_keys: POST字段名元组

        Returns:
            哈希字符串
        """
        import hashlib

        # 提取URL路径（忽略query和fragment）
        parsed_url = urlparse(url)
        url_path = parsed_url.path

        # 如果路径为空，使用根路径
        if not url_path:
            url_path = '/'

        # 组合哈希要素
        hash_input = f"{url_path}|{method}|{sorted(params_keys)}|{sorted(data_keys)}"

        # 生成MD5哈希
        hash_value = hashlib.md5(hash_input.encode('utf-8')).hexdigest()

        return hash_value

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        发送HTTP请求（带错误处理）

        老王注释：这个SB函数必须健壮，各种异常都得处理

        Args:
            url: 目标URL

        Returns:
            Response对象，失败返回None
        """
        self.stats['total_requests'] += 1

        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True,
                verify=False  # 艹，忽略SSL证书错误
            )
            response.raise_for_status()

            self.stats['successful_requests'] += 1
            return response

        except requests.exceptions.Timeout:
            self.stats['failed_requests'] += 1
            self._print_colored(f"[SPIDER] [!] 超时: {url}", 'YELLOW')

        except requests.exceptions.HTTPError as e:
            self.stats['failed_requests'] += 1
            self._print_colored(f"[SPIDER] [!] HTTP错误: {url} - {e}", 'YELLOW')

        except requests.exceptions.RequestException as e:
            self.stats['failed_requests'] += 1
            self._print_colored(f"[SPIDER] [!] 请求失败: {url} - {e}", 'YELLOW')

        except Exception as e:
            self.stats['failed_requests'] += 1
            self._print_colored(f"[SPIDER] [!] 未知错误: {url} - {e}", 'RED')

        return None

    def _extract_get_targets(self, soup: BeautifulSoup, base_url: str, depth: int) -> List[FuzzTarget]:
        """
        提取GET目标的FuzzTarget

        老王注释：解析<a>标签，提取带参数的GET链接
                 支持注入点去重，避免重复扫描相同接口

        Args:
            soup: BeautifulSoup对象
            base_url: 当前页面URL
            depth: 当前深度

        Returns:
            FuzzTarget列表
        """
        targets = []

        # 查找所有<a>标签
        for tag in soup.find_all('a', href=True):
            href = tag['href'].strip()

            # 艹，过滤空链接和锚点
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue

            # 构建绝对URL
            absolute_url = urljoin(base_url, href)

            # 解析URL参数
            parsed = urlparse(absolute_url)
            params = parse_qs(parsed.query)

            # 如果有参数，创建FuzzTarget
            if params:
                # 将parse_qs返回的list转换为单个值
                params_dict = {k: v[0] if v else '' for k, v in params.items()}

                # ========== 老王新增：注入点去重 ==========
                target_hash = self._generate_target_hash(
                    absolute_url,
                    'GET',
                    tuple(params_dict.keys()),
                    tuple()
                )

                if target_hash in self.target_hashes:
                    # 艹，这个接口已经扫描过了，跳过
                    continue

                # 标记为已扫描
                self.target_hashes.add(target_hash)

                target = FuzzTarget(
                    url=absolute_url,
                    method='GET',
                    params=params_dict,
                    data={},
                    depth=depth
                )
                targets.append(target)

                # 实时输出
                self._print_colored(
                    f"[SPIDER] [+] 发现目标: [GET] {absolute_url} (参数: {len(params_dict)})",
                    'GREEN'
                )

        return targets

    def _extract_post_targets(self, soup: BeautifulSoup, base_url: str, depth: int) -> List[FuzzTarget]:
        """
        提取POST目标的FuzzTarget

        老王注释：解析<form>标签，提取所有input/select/textarea字段
                 支持注入点去重，避免重复扫描相同接口
                 增强解析：支持select、textarea、hidden字段

        Args:
            soup: BeautifulSoup对象
            base_url: 当前页面URL
            depth: 当前深度

        Returns:
            FuzzTarget列表
        """
        targets = []

        # 查找所有<form>标签
        for form in soup.find_all('form'):
            # 获取form action
            action = form.get('action', '')
            if not action:
                # 艹，没有action就使用当前URL
                action = base_url

            # 构建绝对URL
            action_url = urljoin(base_url, action)

            # ========== 老王优化：提取所有表单字段 ==========
            data_dict = {}

            # 1. 提取所有<input>字段
            for input_tag in form.find_all('input'):
                name = input_tag.get('name')
                input_type = input_tag.get('type', 'text')

                # 艹，跳过没有name的input和submit/button
                if not name or input_type in ['submit', 'button', 'reset']:
                    continue

                # 使用默认值（如果存在）
                value = input_tag.get('value', '')
                data_dict[name] = value

            # 2. 提取所有<select>字段（老王新增）
            for select_tag in form.find_all('select'):
                name = select_tag.get('name')
                if not name:
                    continue

                # 获取选中的option值（优先selected，否则第一个）
                selected_option = select_tag.find('option', selected=True)
                if selected_option:
                    value = selected_option.get('value', '')
                else:
                    first_option = select_tag.find('option')
                    value = first_option.get('value', '') if first_option else ''

                data_dict[name] = value

            # 3. 提取所有<textarea>字段（老王新增）
            for textarea_tag in form.find_all('textarea'):
                name = textarea_tag.get('name')
                if not name:
                    continue

                # 获取默认文本内容
                value = textarea_tag.get_text(strip=True)
                data_dict[name] = value

            # 如果有表单字段，创建FuzzTarget
            if data_dict:
                # ========== 老王新增：注入点去重 ==========
                target_hash = self._generate_target_hash(
                    action_url,
                    'POST',
                    tuple(),
                    tuple(data_dict.keys())
                )

                if target_hash in self.target_hashes:
                    # 艹，这个接口已经扫描过了，跳过
                    continue

                # 标记为已扫描
                self.target_hashes.add(target_hash)

                target = FuzzTarget(
                    url=action_url,
                    method='POST',
                    params={},
                    data=data_dict,
                    depth=depth
                )
                targets.append(target)

                # 实时输出
                self._print_colored(
                    f"[SPIDER] [+] 发现目标: [POST] {action_url} (字段: {len(data_dict)})",
                    'CYAN'
                )

        return targets

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        """
        提取页面中的所有链接（用于继续爬取）

        Args:
            soup: BeautifulSoup对象
            base_url: 当前页面URL

        Returns:
            URL集合
        """
        links = set()

        # 查找所有<a>标签
        for tag in soup.find_all('a', href=True):
            href = tag['href'].strip()

            # 艹，过滤无效链接
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue

            # 构建绝对URL
            absolute_url = urljoin(base_url, href)

            # 只爬取同域名下的链接（艹，别爬到别的网站去了）
            if urlparse(absolute_url).netloc == urlparse(self.base_url).netloc:
                links.add(absolute_url)

        return links

    def crawl(self) -> List[FuzzTarget]:
        """
        执行BFS爬取

        老王注释：核心爬取逻辑，使用队列实现BFS遍历

        Returns:
            发现的所有FuzzTarget列表
        """
        self._print_colored(
            f"\n{'='*60}\n[SPIDER] 开始爬取站点: {self.site_name}\n[SPIDER] 基础URL: {self.base_url}\n[SPIDER] 最大深度: {self.max_depth}\n{'='*60}\n",
            'CYAN'
        )

        # 艹，使用tqdm显示进度条
        with tqdm(desc="爬取进度", unit="URL", colour='green') as pbar:
            while self.url_queue:
                # 从队列取出URL和深度
                current_url, current_depth = self.url_queue.popleft()

                # 去重检查（艹，别重复爬取）
                if current_url in self.visited_urls:
                    continue

                # 深度检查
                if current_depth > self.max_depth:
                    continue

                # ========== 老王新增：URL模式检查（防止路径爆炸） ==========
                url_pattern = self._normalize_url_pattern(current_url)
                visit_count = self.url_pattern_counts.get(url_pattern, 0)

                if visit_count >= self.MAX_VISITS_PER_PATTERN:
                    # 艹，这个模式访问次数超过限制了，跳过
                    self._print_colored(
                        f"[SPIDER] [!] 路径爆炸保护：跳过模式（已访问{visit_count}次）: {url_pattern}",
                        'YELLOW'
                    )
                    pbar.update(1)
                    continue

                # 更新模式访问计数
                self.url_pattern_counts[url_pattern] = visit_count + 1
                # ========== 老王新增结束 ==========

                # 标记为已访问
                self.visited_urls.add(current_url)

                # 更新进度条描述
                pbar.set_description(f"正在爬取: {current_url[:60]}... | 已发现: {len(self.targets)}")

                # 发送请求
                response = self._make_request(current_url)
                if not response:
                    pbar.update(1)
                    continue

                # 检查Content-Type（艹，别爬二进制文件）
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    pbar.update(1)
                    continue

                # 解析HTML
                try:
                    soup = BeautifulSoup(response.text, 'html.parser')
                except Exception as e:
                    self._print_colored(f"[SPIDER] [!] HTML解析失败: {current_url} - {e}", 'YELLOW')
                    pbar.update(1)
                    continue

                # ========== 老王新增：提取当前URL本身的参数 ==========
                parsed_current_url = urlparse(current_url)
                current_params = parse_qs(parsed_current_url.query)
                if current_params:
                    # 转换为字典
                    current_params_dict = {k: v[0] if v else '' for k, v in current_params.items()}

                    # 生成哈希检查去重
                    target_hash = self._generate_target_hash(
                        current_url,
                        'GET',
                        tuple(current_params_dict.keys()),
                        tuple()
                    )

                    if target_hash not in self.target_hashes:
                        # 标记为已扫描
                        self.target_hashes.add(target_hash)

                        # 创建FuzzTarget
                        target = FuzzTarget(
                            url=current_url,
                            method='GET',
                            params=current_params_dict,
                            data={},
                            depth=current_depth
                        )
                        self.targets.append(target)

                        # 实时输出
                        self._print_colored(
                            f"[SPIDER] [+] 发现目标: [GET] {current_url} (参数: {len(current_params_dict)})",
                            'GREEN'
                        )

                        # 更新统计
                        self.stats['get_targets'] += 1

                # 提取FuzzTarget
                get_targets = self._extract_get_targets(soup, current_url, current_depth)
                post_targets = self._extract_post_targets(soup, current_url, current_depth)

                # 添加到targets列表
                self.targets.extend(get_targets)
                self.targets.extend(post_targets)

                # 更新统计
                self.stats['get_targets'] += len(get_targets)
                self.stats['post_targets'] += len(post_targets)

                # 提取链接继续爬取
                if current_depth < self.max_depth:
                    links = self._extract_links(soup, current_url)
                    for link in links:
                        if link not in self.visited_urls:
                            self.url_queue.append((link, current_depth + 1))

                # 更新进度条
                pbar.update(1)

        # 爬取完成，输出统计
        self._print_colored(f"\n{'='*60}", 'CYAN')
        self._print_colored(f"[SPIDER] 爬取完成！", 'GREEN')
        self._print_colored(f"{'='*60}", 'CYAN')
        print(f"\n爬取统计:")
        print(f"  - 总请求数: {self.stats['total_requests']}")
        print(f"  - 成功请求: {self.stats['successful_requests']}")
        print(f"  - 失败请求: {self.stats['failed_requests']}")
        print(f"  - GET目标: {self.stats['get_targets']}")
        print(f"  - POST目标: {self.stats['post_targets']}")
        print(f"  - 总目标数: {len(self.targets)}")
        print(f"  - 已访问URL: {len(self.visited_urls)}")
        print()

        return self.targets

    def save_cache(self, cache_dir: Optional[str] = None) -> str:
        """
        保存爬虫缓存到JSON文件

        老王注释：持久化缓存，支持断点续爬

        Args:
            cache_dir: 缓存目录（默认为Data/cache/[site_name]/）

        Returns:
            缓存文件完整路径
        """
        if cache_dir is None:
            cache_dir = Path("Data/cache") / self.site_name
        else:
            cache_dir = Path(cache_dir)

        # 创建目录
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 缓存文件路径
        cache_file = cache_dir / "spider_cache.json"

        # 准备缓存数据
        cache_data = {
            'site_name': self.site_name,
            'base_url': self.base_url,
            'max_depth': self.max_depth,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'stats': self.stats,
            'visited_urls': list(self.visited_urls),
            'targets': [target.to_dict() for target in self.targets]
        }

        # 保存到JSON
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        self._print_colored(f"[SPIDER] 缓存已保存: {cache_file}", 'GREEN')

        return str(cache_file)

    @classmethod
    def load_cache(cls, cache_file: str) -> 'CVDBSpider':
        """
        从JSON文件加载爬虫缓存

        老王注释：加载缓存，恢复爬虫状态

        Args:
            cache_file: 缓存文件路径

        Returns:
            CVDBSpider实例
        """
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # 创建爬虫实例
        spider = cls(
            base_url=cache_data['base_url'],
            max_depth=cache_data['max_depth'],
        )

        # 恢复状态
        spider.site_name = cache_data['site_name']
        spider.stats = cache_data['stats']
        spider.visited_urls = set(cache_data['visited_urls'])
        spider.targets = [FuzzTarget.from_dict(t) for t in cache_data['targets']]

        return spider


# 艹，测试代码
if __name__ == "__main__":
    # 示例用法
    spider = CVDBSpider(
        base_url="http://127.0.0.1:8000",
        max_depth=2,
        timeout=10,
        cookie="session=test123"
    )

    # 爬取
    targets = spider.crawl()

    # 保存缓存
    cache_path = spider.save_cache()

    print(f"\n[SUCCESS] 发现 {len(targets)} 个Fuzz目标")
