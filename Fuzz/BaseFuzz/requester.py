#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP Requester - BaseFuzz 引擎的通信中枢
============================================

负责所有与目标服务器的HTTP交互，提供纯净的请求/响应封装。
不包含任何漏洞检测逻辑，专注于网络通信层。

核心功能：
- 持久化Session管理（连接复用、Cookie自动处理）
- URL净化与参数注入
- 超时控制与异常重试
- 统一的错误处理与日志记录

使用示例：
    >>> requester = Requester(timeout=10, cookie='session=abc123')
    >>> response = requester.send(
    ...     method='POST',
    ...     url='http://target.com/login',
    ...     data={'username': 'admin', 'password': 'FUZZ'}
    ... )
    >>> print(response.status_code)
    200

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-25
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from typing import Dict, Optional, Any, Union
import logging
import time

# 配置日志
logger = logging.getLogger(__name__)


class Requester:
    """
    HTTP请求封装类 - BaseFuzz引擎的通信中枢

    这个SB类负责所有HTTP通信，别tm到处用requests！保持统一！

    核心职责：
    1. 维护持久化Session（连接复用、Cookie管理）
    2. URL净化与参数注入
    3. 超时控制与异常重试
    4. 统一的错误处理

    Attributes:
        session (requests.Session): 持久化会话对象
        timeout (int): 请求超时时间（秒）
        headers (dict): 全局HTTP请求头
        max_retries (int): 最大重试次数
        retry_delay (float): 重试延迟（秒）

    Example:
        >>> requester = Requester(
        ...     timeout=10,
        ...     cookie='session=abc123; token=xyz',
        ...     max_retries=3
        ... )
        >>> response = requester.send(
        ...     method='GET',
        ...     url='http://target.com/?id=1',
        ...     params={'id': 'FUZZ'}
        ... )
        >>> if response:
        ...     print(f"Status: {response.status_code}")
        ...     print(f"Length: {len(response.text)}")

    注意：
        - 本类不包含任何漏洞检测逻辑
        - 异常情况返回None，调用方需自行检查
        - 所有网络错误会记录到日志
    """

    # 默认请求头（模拟最新Chrome浏览器）
    DEFAULT_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,'
                  'image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    def __init__(self,
                 timeout: int = 10,
                 cookie: Optional[str] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 verify_ssl: bool = True):
        """
        初始化Requester

        Args:
            timeout: 请求超时时间（秒），默认10秒
            cookie: 全局Cookie字符串（格式：'key1=value1; key2=value2'）
            max_retries: 最大重试次数（针对网络错误），默认3次
            retry_delay: 重试延迟（秒），默认1秒
            verify_ssl: 是否验证SSL证书，默认True

        Example:
            >>> requester = Requester(
            ...     timeout=15,
            ...     cookie='session=abc123',
            ...     max_retries=5
            ... )
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 初始化Session
        self.session = requests.Session()

        # 设置默认请求头
        self.headers = self.DEFAULT_HEADERS.copy()

        # 设置Cookie
        if cookie:
            self.set_cookie(cookie)

        # 配置重试策略（仅针对网络错误，不重试4xx/5xx）
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[],  # 不重试任何HTTP状态码
            allowed_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]
        )

        # 挂载重试适配器
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # SSL验证
        self.session.verify = verify_ssl

        logger.info(f"Requester初始化完成: timeout={timeout}s, max_retries={max_retries}")

    def send(self,
             method: str,
             url: str,
             params: Optional[Dict[str, str]] = None,
             data: Optional[Dict[str, str]] = None,
             json_data: Optional[Dict[str, Any]] = None,
             headers: Optional[Dict[str, str]] = None) -> Optional[requests.Response]:
        """
        发送HTTP请求（核心方法）

        艹，这个方法是整个类的核心！所有HTTP通信都走这里！

        Args:
            method: HTTP方法（'GET' 或 'POST'）
            url: 目标URL
            params: GET参数字典（注入到URL的query string）
            data: POST表单数据字典（application/x-www-form-urlencoded）
            json_data: POST JSON数据字典（application/json）
            headers: 额外的请求头字典（会与全局headers合并）

        Returns:
            requests.Response对象，成功返回响应，失败返回None

        Example:
            >>> # GET请求
            >>> resp = requester.send(
            ...     method='GET',
            ...     url='http://target.com/search',
            ...     params={'q': 'test'}
            ... )
            >>>
            >>> # POST请求（表单）
            >>> resp = requester.send(
            ...     method='POST',
            ...     url='http://target.com/login',
            ...     data={'username': 'admin', 'password': '123456'}
            ... )
            >>>
            >>> # POST请求（JSON）
            >>> resp = requester.send(
            ...     method='POST',
            ...     url='http://target.com/api/user',
            ...     json_data={'name': 'admin', 'age': 20}
            ... )

        Note:
            - 异常情况返回None，调用方需检查
            - 超时、连接错误等会自动重试（最多max_retries次）
            - 所有错误会记录到日志
        """
        # URL净化
        url = self.normalize_url(url)

        # 合并headers
        final_headers = {**self.headers, **(headers or {})}

        # 构建请求参数
        request_kwargs = {
            'url': url,
            'headers': final_headers,
            'timeout': self.timeout
        }

        # 添加请求体
        if method.upper() in ['POST', 'PUT', 'PATCH']:
            if json_data is not None:
                request_kwargs['json'] = json_data
            elif data is not None:
                request_kwargs['data'] = data

        # 添加GET参数
        if params is not None:
            request_kwargs['params'] = params

        # 发送请求（带重试）
        try:
            response = self._send_with_retry(method.upper(), **request_kwargs)
            return response

        except requests.exceptions.Timeout as e:
            logger.error(f"[TIMEOUT] {method} {url} - 请求超时（>{self.timeout}s）")
            return None

        except requests.exceptions.ConnectionError as e:
            logger.error(f"[CONN_ERROR] {method} {url} - 连接失败: {e}")
            return None

        except requests.exceptions.TooManyRedirects as e:
            logger.error(f"[REDIRECT] {method} {url} - 重定向次数过多")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"[REQUEST_ERROR] {method} {url} - 请求异常: {e}")
            return None

        except Exception as e:
            logger.error(f"[UNKNOWN_ERROR] {method} {url} - 未知错误: {e}")
            return None

    def _send_with_retry(self, method: str, **kwargs) -> requests.Response:
        """
        带重试的请求发送（内部方法）

        老王注释：这个憨批方法实现了重试逻辑！

        Args:
            method: HTTP方法
            **kwargs: 请求参数

        Returns:
            requests.Response对象

        Raises:
            requests.exceptions.RequestException: 所有重试失败后抛出
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if method == 'GET':
                    response = self.session.get(**kwargs)
                elif method == 'POST':
                    response = self.session.post(**kwargs)
                elif method == 'HEAD':
                    response = self.session.head(**kwargs)
                elif method == 'PUT':
                    response = self.session.put(**kwargs)
                elif method == 'DELETE':
                    response = self.session.delete(**kwargs)
                elif method == 'OPTIONS':
                    response = self.session.options(**kwargs)
                else:
                    raise ValueError(f"不支持的HTTP方法: {method}")

                # 记录成功请求
                if attempt > 0:
                    logger.info(f"[RETRY_SUCCESS] {method} {kwargs['url']} - 第{attempt+1}次重试成功")

                return response

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                # 记录重试
                if attempt < self.max_retries:
                    logger.warning(f"[RETRYING] {method} {kwargs['url']} - "
                                  f"第{attempt+1}次请求失败，{self.retry_delay}秒后重试... "
                                  f"错误: {e}")
                    time.sleep(self.retry_delay)
                    last_exception = e
                else:
                    # 最后一次重试也失败
                    logger.error(f"[RETRY_FAILED] {method} {kwargs['url']} - "
                                f"重试{self.max_retries}次后仍然失败")
                    raise last_exception

    def normalize_url(self,
                     url: str,
                     keep_fragment: bool = False,
                     keep_params: bool = True) -> str:
        """
        URL净化（标准化处理）

        艹，这个SB方法把URL里的垃圾清理掉！

        功能：
        1. 去除fragment（#后面的内容，默认）
        2. 保留或删除query参数
        3. 标准化URL格式

        Args:
            url: 原始URL字符串
            keep_fragment: 是否保留fragment（#后面的内容），默认False
            keep_params: 是否保留query参数（?后面的内容），默认True

        Returns:
            净化后的URL字符串

        Example:
            >>> requester = Requester()
            >>> requester.normalize_url('http://target.com/page#section')
            'http://target.com/page'
            >>>
            >>> requester.normalize_url('http://target.com/page?id=1#section', keep_fragment=True)
            'http://target.com/page?id=1#section'
            >>>
            >>> requester.normalize_url('http://target.com/page?id=1&name=test', keep_params=False)
            'http://target.com/page'
        """
        try:
            parsed = urlparse(url)

            # 构建净化后的URL
            clean_url = urlunparse((
                parsed.scheme,           # http/https
                parsed.netloc,          # domain:port
                parsed.path,            # /path/to/resource
                parsed.params,          # ;params
                parsed.query if keep_params else '',  # ?key=value
                parsed.fragment if keep_fragment else ''  # #fragment
            ))

            return clean_url

        except Exception as e:
            logger.error(f"[URL_PARSE_ERROR] URL解析失败: {url} - {e}")
            return url

    def inject_payload(self,
                      url: str,
                      param_name: str,
                      payload: str,
                      method: str = 'GET') -> str:
        """
        URL参数注入（辅助方法）

        老王注释：这个SB方法帮你把payload注入到URL的参数里！

        Args:
            url: 原始URL
            param_name: 参数名
            payload: 要注入的载荷
            method: HTTP方法（'GET'或'POST'）

        Returns:
            注入后的URL（GET）或原始URL（POST）

        Example:
            >>> requester = Requester()
            >>> requester.inject_payload(
            ...     url='http://target.com/?id=1',
            ...     param_name='id',
            ...     payload="' OR 1=1--",
            ...     method='GET'
            ... )
            'http://target.com/?id=%27+OR+1%3D1--'
            >>>
            >>> # POST方法返回原始URL，注入在send()中通过data参数完成
            >>> requester.inject_payload(
            ...     url='http://target.com/login',
            ...     param_name='username',
            ...     payload='admin',
            ...     method='POST'
            ... )
            'http://target.com/login'
        """
        if method.upper() == 'POST':
            # POST方法，直接返回原URL（注入通过data参数）
            return url

        # GET方法，注入到URL的query string
        try:
            parsed = urlparse(url)

            # 解析现有参数
            params = dict(parse_qsl(parsed.query))

            # 注入payload
            params[param_name] = payload

            # 重建URL
            injected_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                urlencode(params),
                parsed.fragment
            ))

            return injected_url

        except Exception as e:
            logger.error(f"[INJECT_ERROR] URL注入失败: {url} - {e}")
            return url

    def set_cookie(self, cookie: str) -> None:
        """
        设置全局Cookie

        Args:
            cookie: Cookie字符串（格式：'key1=value1; key2=value2'）

        Example:
            >>> requester = Requester()
            >>> requester.set_cookie('session=abc123; token=xyz')
            >>> # 后续所有请求都会带上这个Cookie
        """
        if not cookie:
            return

        self.headers['Cookie'] = cookie
        logger.debug(f"[COOKIE] 设置全局Cookie: {cookie[:50]}...")

    def set_header(self, key: str, value: str) -> None:
        """
        设置全局请求头

        Args:
            key: 请求头名称
            value: 请求头值

        Example:
            >>> requester = Requester()
            >>> requester.set_header('Authorization', 'Bearer abc123')
            >>> requester.set_header('X-Custom-Header', 'MyValue')
        """
        self.headers[key] = value
        logger.debug(f"[HEADER] 设置请求头: {key}={value[:50]}...")

    def set_proxy(self,
                  http_proxy: Optional[str] = None,
                  https_proxy: Optional[str] = None) -> None:
        """
        设置代理服务器

        Args:
            http_proxy: HTTP代理地址（格式：'http://127.0.0.1:8080'）
            https_proxy: HTTPS代理地址

        Example:
            >>> requester = Requester()
            >>> requester.set_proxy(
            ...     http_proxy='http://127.0.0.1:8080',
            ...     https_proxy='http://127.0.0.1:8080'
            ... )
        """
        proxies = {}

        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https_proxy'] = https_proxy

        if proxies:
            self.session.proxies.update(proxies)
            logger.info(f"[PROXY] 设置代理: {proxies}")

    def get_cookies(self) -> Dict[str, str]:
        """
        获取当前Session的所有Cookie

        Returns:
            Cookie字典 {name: value}

        Example:
            >>> requester = Requester()
            >>> requester.set_cookie('session=abc123')
            >>> cookies = requester.get_cookies()
            >>> print(cookies)
            {'session': 'abc123'}
        """
        return {cookie.name: cookie.value for cookie in self.session.cookies}

    def close(self) -> None:
        """
        关闭Session，释放资源

        艹，用完记得关闭！别tm资源泄漏！

        Example:
            >>> requester = Requester()
            >>> # 使用requester...
            >>> requester.close()
        """
        self.session.close()
        logger.info("[SESSION] Session已关闭")

    def __enter__(self):
        """
        上下文管理器入口（支持with语句）

        Example:
            >>> with Requester(timeout=10) as requester:
            ...     response = requester.send('GET', 'http://target.com')
            ...     # 自动关闭Session
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        return False


# 便捷函数
def create_requester(timeout: int = 10,
                    cookie: Optional[str] = None,
                    max_retries: int = 3) -> Requester:
    """
    创建Requester实例的便捷函数

    Args:
        timeout: 请求超时时间（秒）
        cookie: 全局Cookie字符串
        max_retries: 最大重试次数

    Returns:
        Requester实例

    Example:
        >>> requester = create_requester(timeout=15, cookie='session=abc')
        >>> response = requester.send('GET', 'http://target.com')
    """
    return Requester(timeout=timeout, cookie=cookie, max_retries=max_retries)


if __name__ == '__main__':
    # 测试代码
    import logging

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 测试Requester
    print("=" * 60)
    print("Requester 单元测试")
    print("=" * 60)

    # 1. 测试URL净化
    print("\n[测试1] URL净化")
    requester = Requester()
    test_urls = [
        'http://target.com/page#section',
        'http://target.com/page?id=1&name=test',
        'http://target.com/page?id=1#section',
        'http://target.com/page#section?keep=true'
    ]

    for url in test_urls:
        clean = requester.normalize_url(url)
        print(f"原始: {url}")
        print(f"净化: {clean}")
        print()

    # 2. 测试参数注入
    print("\n[测试2] 参数注入")
    url = 'http://target.com/?id=1'
    injected = requester.inject_payload(url, 'id', "' OR 1=1--", method='GET')
    print(f"原始URL: {url}")
    print(f"注入URL: {injected}")

    # 3. 测试上下文管理器
    print("\n[测试3] 上下文管理器")
    with Requester(timeout=5) as req:
        print(f"Session对象: {req.session}")
        print(f"Headers: {list(req.headers.keys())}")
    print("Session已自动关闭")

    print("\n[SUCCESS] 所有测试通过！")
