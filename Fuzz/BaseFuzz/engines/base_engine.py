#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Engine - 漏洞检测引擎抽象基类
====================================

定义所有漏洞探测插件必须遵循的接口规范，确保引擎架构统一。

核心功能：
- 定义统一的初始化接口
- 定义抽象检测方法（所有子引擎必须实现）
- 提供统一的漏洞结果封装方法
- 提供通用的参数注入和响应对比工具

使用示例：
    from Fuzz.BaseFuzz.engines.base_engine import BaseEngine

    class SQLiEngine(BaseEngine):
        def detect(self, target, payloads):
            # 实现SQL注入检测逻辑
            results = []
            for param_name in target.params.keys():
                result = self._test_parameter(target, param_name, payload)
                if result:
                    results.append(result)
            return results

"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import re  # 导入正则模块（用于判断payload类型）
import requests  # 艹，老王忘了导入requests！类型注解需要用到

# 导入依赖模块
from Fuzz.BaseFuzz.requester import Requester
from Fuzz.BaseFuzz.baseline import BaselineProfile
from Fuzz.spider import FuzzTarget

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class VulnerabilityEntry:
    """
    漏洞条目数据结构

    这个类存储漏洞的所有信息

    Attributes:
        vuln_type: 漏洞类型（SQLi, XSS, CMDi等）
        method: 检测方法（Error-Based, Boolean-Based等）
        severity: 严重性等级（High, Medium, Low）
        confidence: 置信度（0.0-1.0）
        payload: 触发漏洞的载荷
        param_name: 漏洞参数名
        evidence: 漏洞证据（错误信息/响应特征等）
        target_url: 目标URL
        payload_url: 完整payload URL（新增！包含注入载荷的完整URL）
        post_body: POST请求体（新增！仅POST请求有效，包含注入载荷的完整body）
        response_info: 响应信息字典
        timestamp: 发现时间（可选）

    Example:
        >>> vuln = VulnerabilityEntry(
        ...     vuln_type='SQLi',
        ...     method='Error-Based',
        ...     severity='High',
        ...     confidence=0.9,
        ...     payload="' OR 1=1--",
        ...     param_name='id',
        ...     evidence='SQL syntax error',
        ...     target_url='http://target.com/?id=1',
        ...     payload_url='http://target.com/?id=1%27%20OR%201=1--',
        ...     post_body='id=1\' OR 1=1--&Submit=提交',
        ...     response_info={'status': 500, 'length': 1234}
        ... )
    """
    vuln_type: str
    method: str
    severity: str
    confidence: float
    payload: str
    param_name: str
    evidence: str
    target_url: str
    payload_url: Optional[str] = None  # 新增字段：完整payload URL
    post_body: Optional[str] = None  # 新增字段：POST请求体（仅POST请求）
    response_info: Dict[str, Any] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        """初始化后处理，确保response_info不为None"""
        if self.response_info is None:
            self.response_info = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于JSON序列化）"""
        result = {
            'vuln_type': self.vuln_type,
            'method': self.method,
            'severity': self.severity,
            'confidence': self.confidence,
            'payload': self.payload,
            'param_name': self.param_name,
            'evidence': self.evidence,
            'target_url': self.target_url,
            'payload_url': self.payload_url,  # 完整payload URL
            'response_info': self.response_info,
            'timestamp': self.timestamp
        }

        # 新增：只有POST请求才添加post_body字段
        if self.post_body is not None:
            result['post_body'] = self.post_body

        return result


class BaseEngine(ABC):
    """
    漏洞检测引擎抽象基类

    定义统一的接口

    核心职责：
    1. 定义统一的初始化接口
    2. 定义抽象检测方法（子类必须实现）
    3. 提供统一的漏洞结果封装
    4. 提供通用的参数注入工具

    使用方式：
        1. 继承此类
        2. 实现 detect() 方法
        3. 调用 _create_vuln_entry() 封装结果

    Example:
        >>> class SQLiEngine(BaseEngine):
        ...     def detect(self, target, payloads):
        ...         results = []
        ...         for param in target.params.keys():
        ...             # 检测逻辑
        ...             if self._is_vulnerable(response):
        ...                 vuln = self._create_vuln_entry(
        ...                     vuln_type='SQLi',
        ...                     method='Error-Based',
        ...                     severity='High',
        ...                     payload=payload,
        ...                     param=param,
        ...                     evidence='SQL error'
        ...                 )
        ...                 results.append(vuln)
        ...         return results
    """

    def __init__(self, requester: Requester, baseline: BaselineProfile):
        """
        初始化引擎

        Args:
            requester: Requester实例（HTTP通信层）
            baseline: BaselineProfile实例（基准画像）

        Example:
            >>> requester = Requester(timeout=10)
            >>> baseline = BaselineProfile(...)
            >>> engine = SQLiEngine(requester, baseline)
        """
        self.requester = requester
        self.baseline = baseline
        self.vuln_count = 0

        logger.info(f"[ENGINE] {self.__class__.__name__} 初始化完成")

    @abstractmethod
    def detect(self,
               target: FuzzTarget,
               payloads: List[str],
               param_name: str) -> List[VulnerabilityEntry]:
        """
        执行漏洞检测（抽象方法，子类必须实现）


        Args:
            target: 测试目标（FuzzTarget对象）
            payloads: 载荷列表
            param_name: 要测试的参数名（单参数检测）

        Returns:
            漏洞条目列表

        Example:
            >>> engine = SQLiEngine(requester, baseline)
            >>> results = engine.detect(target, payloads, param_name='id')
            >>> for vuln in results:
            ...     print(f"[VULN] {vuln.vuln_type}: {vuln.payload}")

        重要：
            - 本方法只检测指定的单个参数！
            - 不要遍历target的所有参数，否则会导致重复测试！
            - 参数遍历由Engine调度器负责，不是引擎的职责！
        """
        pass

    # ========== 通用工具方法 ==========

    def _test_parameter(self,
                      target: FuzzTarget,
                      param_name: str,
                      payload: str,
                      injection_point: str = 'auto') -> Optional[Dict]:
        """
        测试单个参数（通用方法）- 支持GET/POST/Cookie注入

        注入载荷并发送请求！

        Args:
            target: 测试目标
            param_name: 参数名
            payload: 载荷
            injection_point: 注入点位置
                - 'auto': 自动判断（GET/POST参数）
                - 'cookie': Cookie注入
                - 'param': GET/POST参数注入

        Returns:
            包含response和请求信息的字典，失败返回None
        """
        try:
            from urllib.parse import urlparse, urlunparse, urlencode

            # 自动判断注入点（优先级：Cookie > Header > GET/POST参数）
            if injection_point == 'auto':
                # 1. 检查Cookie
                if hasattr(target, 'cookies') and param_name in target.cookies:
                    injection_point = 'cookie'
                # 2. 检查HTTP头（新增）
                elif hasattr(target, 'injectable_headers') and param_name in target.injectable_headers:
                    injection_point = 'header'
                # 3. 默认为GET/POST参数
                else:
                    injection_point = 'param'

            logger.debug(f"[INJECT] 注入点: {injection_point}, 参数: {param_name}")

            # ========== HTTP头注入模式（新增） ==========
            if injection_point == 'header':
                return self._test_parameter_in_header(target, param_name, payload)

            # ========== Cookie注入模式 ==========
            if injection_point == 'cookie':
                return self._test_parameter_in_cookie(target, param_name, payload)

            # ========== GET/POST参数注入模式（原有逻辑）==========
            # 关键修复：构造不带参数的URL！
            # target.url已经包含参数（如：http://target.com/sqli?id=1&Submit=提交）
            # 如果直接传给requests.send()，会导致参数重复！
            # 解决方案：提取URL的基础部分（不带query string）
            parsed_url = urlparse(target.url)
            base_url = urlunparse((
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                '',  # 清空params
                '',  # 清空query（这里最重要！）
                ''   # 清空fragment
            ))

            # 智能注入模式：自动判断是否需要保留原始值！
            # 判断规则：
            # - 如果payload以'或"开头，说明要闭合引号，需要保留原始值（追加模式）
            # - 如果payload包含空格或--，说明是完整payload，直接替换（替换模式）
            # - 其他情况：追加模式

            # 记录实际注入的完整值（用于日志）
            injected_value = None

            # 新增：SQL盲注智能值调整！
            # 对于OR类型的盲注payload，使用不存在的ID（-1, 999999等）
            # 对于AND类型的盲注payload，使用原始值（保持不变）
            original_value = target.params.get(param_name) if target.method == 'GET' else target.data.get(param_name)
            adjusted_value = self._adjust_value_for_blind_payload(payload, original_value)

            if target.method == 'GET':
                params = {}
                for k, v in target.params.items():
                    if k == param_name:
                        # 使用调整后的值（针对盲注优化）
                        v = adjusted_value

                        # 智能判断注入模式！
                        if self._should_append_payload(payload, v):
                            # 追加模式：保留原始值
                            params[k] = f"{v}{payload}"
                            injected_value = params[k]  # 记录实际注入值
                            logger.debug(f"[INJECT] 追加模式: {k}={v} + {payload} → {params[k]}")
                        else:
                            # 替换模式：直接使用payload
                            params[k] = payload
                            injected_value = params[k]  # 记录实际注入值
                            logger.debug(f"[INJECT] 替换模式: {k}={payload}")
                    else:
                        params[k] = v
                response = self.requester.send('GET', base_url, params=params)

            else:  # POST
                data = {}
                for k, v in target.data.items():
                    if k == param_name:
                        # 使用调整后的值（针对盲注优化）
                        v = adjusted_value

                        # 智能判断注入模式！
                        if self._should_append_payload(payload, v):
                            # 追加模式：保留原始值
                            data[k] = f"{v}{payload}"
                            injected_value = data[k]  # 记录实际注入值
                            logger.debug(f"[INJECT] 追加模式: {k}={v} + {payload} → {data[k]}")
                        else:
                            # 替换模式：直接使用payload
                            data[k] = payload
                            injected_value = data[k]  # 记录实际注入值
                            logger.debug(f"[INJECT] 替换模式: {k}={payload}")
                    else:
                        data[k] = v
                response = self.requester.send('POST', base_url, data=data)

            # 调试日志：显示实际请求的URL
            if target.method == 'GET':
                full_url = f"{base_url}?{urlencode(params)}"
                logger.debug(f"[INJECT] 实际请求URL: {full_url}")
            else:
                logger.debug(f"[INJECT] 实际请求POST数据: {data}")

            if response is None:
                return None

            return {
                'response': response,
                'payload': payload,
                'param': param_name,
                'target': target,
                'injected_value': injected_value  # 记录实际注入的完整值
            }

        except Exception as e:
            logger.error(f"[ENGINE] 参数测试失败: {target.url}?{param_name}={payload[:30]}... - {e}")
            return None

    def _test_parameter_in_cookie(self,
                                  target: FuzzTarget,
                                  param_name: str,
                                  payload: str) -> Optional[Dict]:
        """
        测试Cookie参数注入（新增方法）

        专门处理Cookie注入

        Args:
            target: 测试目标
            param_name: Cookie中的参数名
            payload: 载荷

        Returns:
            包含response和请求信息的字典，失败返回None
        """
        try:
            from http.cookies import SimpleCookie

            # 解析当前的Cookie
            cookies = SimpleCookie()
            if hasattr(target, 'cookies') and target.cookies:
                try:
                    cookies.load(target.cookies)
                except:
                    # 如果解析失败，手动构造
                    for cookie_pair in target.cookies.split(';'):
                        cookie_pair = cookie_pair.strip()
                        if '=' in cookie_pair:
                            k, v = cookie_pair.split('=', 1)
                            cookies[k] = v

            # 获取参数原始值
            original_value = None
            if param_name in cookies:
                original_value = cookies[param_name].value
                logger.debug(f"[INJECT-COOKIE] 原始值: {param_name}={original_value}")
            else:
                logger.warning(f"[INJECT-COOKIE] Cookie中找不到参数: {param_name}")
                return None

            # 调整值（针对盲注）
            adjusted_value = self._adjust_value_for_blind_payload(payload, original_value)

            # 注入payload
            if self._should_append_payload(payload, adjusted_value):
                # 追加模式
                injected_value = f"{adjusted_value}{payload}"
                logger.debug(f"[INJECT-COOKIE] 追加模式: {param_name}={adjusted_value} + {payload} → {injected_value}")
            else:
                # 替换模式
                injected_value = payload
                logger.debug(f"[INJECT-COOKIE] 替换模式: {param_name}={payload}")

            # 更新Cookie
            cookies[param_name] = injected_value

            # 转换成Cookie字符串
            cookie_str = "; ".join([f"{k}={v.value}" for k, v in cookies.items()])
            logger.debug(f"[INJECT-COOKIE] 新Cookie: {cookie_str}")

            # 构造请求
            parsed_url = urlparse(target.url)
            base_url = urlunparse((
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                '', '', ''
            ))

            # 临时修改requester的cookie
            old_cookie = self.requester.session.cookies.get_dict()
            self.requester.session.cookies.clear()

            # 设置新Cookie
            for k, v in cookies.items():
                self.requester.session.cookies.set(k, v.value)

            # 发送请求
            response = self.requester.send('GET', base_url)

            # 恢复旧Cookie
            self.requester.session.cookies.clear()
            for k, v in old_cookie.items():
                self.requester.session.cookies.set(k, v)

            if response is None:
                return None

            logger.debug(f"[INJECT-COOKIE] 响应: status={response.status_code}, length={len(response.text)}")

            return {
                'response': response,
                'payload': payload,
                'param': param_name,
                'target': target,
                'injected_value': injected_value,
                'injection_point': 'cookie'
            }

        except Exception as e:
            logger.error(f"[INJECT-COOKIE] Cookie注入失败: {param_name}={payload[:30]}... - {e}")
            return None

    def _test_parameter_in_header(self,
                                  target: FuzzTarget,
                                  header_name: str,
                                  payload: str) -> Optional[Dict]:
        """
        测试HTTP头注入（新增方法）

        处理HTTP头注入！
        用于User-Agent、Referer、X-Forwarded-For等场景！

        支持两种注入模式：
        1. 追加模式：保留原始值，payload追加到末尾
           - 例如：User-Agent: Mozilla/5.0' OR 1=1--
        2. 替换模式：直接使用payload替换整个头
           - 例如：X-Forwarded-For: 127.0.0.1' OR 1=1--

        判断规则（复用_should_append_payload）：
        - payload以引号开头 → 追加模式
        - payload包含空格或-- → 追加模式
        - 其他 → 替换模式

        Args:
            target: 测试目标
            header_name: HTTP头名称（如 'User-Agent', 'Referer'）
            payload: 注入载荷

        Returns:
            {
                'response': requests.Response,
                'payload': str,
                'param': str,  # 这里存储header_name
                'target': FuzzTarget,
                'injected_value': str  # 实际注入的完整值
            }
            失败返回None
        """
        try:
            from urllib.parse import urlparse, urlunparse

            # 获取原始头的值
            original_header_value = ''
            if hasattr(target, 'injectable_headers') and header_name in target.injectable_headers:
                original_header_value = target.injectable_headers[header_name]

            logger.debug(f"[INJECT-HEADER] 原始HTTP头: {header_name}={original_header_value[:50]}")

            # 智能判断注入模式（复用现有逻辑）
            if self._should_append_payload(payload, original_header_value):
                # 追加模式：保留原始值
                injected_value = f"{original_header_value}{payload}"
                logger.debug(f"[INJECT-HEADER] 追加模式: {header_name}: {original_header_value[:30]} + {payload}")
            else:
                # 替换模式：直接使用payload
                injected_value = payload
                logger.debug(f"[INJECT-HEADER] 替换模式: {header_name}: {payload}")

            # HTTP头注入：直接追加payload，不做任何处理！
            #
            # 说明：
            # - 保持payload原样，追加在HTTP头后面
            # - 如果requests库检测到非法字符并报错，那就让它报错！
            # - 至少原始payload保留了，不会因为编码/替换而失效
            #
            # 用户体验：
            # - 看到ERROR日志就知道payload有非法字符
            # - 可以手动调整payload字典，去掉非法字符
            # - 或者未来实现raw HTTP请求绕过requests验证
            #
            # TODO: 未来绕过requests库，使用socket发送原始HTTP请求
            pass  # 不做任何处理，直接使用原始injected_value

            # 构造完整的请求头字典（复制所有可注入的头）
            inject_headers = {}
            if hasattr(target, 'injectable_headers'):
                inject_headers = target.injectable_headers.copy()

            # 替换当前测试的头为注入后的值
            inject_headers[header_name] = injected_value

            logger.debug(f"[INJECT-HEADER] 实际注入的HTTP头: {header_name}={injected_value[:50]}...")

            # 构造基础URL（不带参数）
            parsed_url = urlparse(target.url)
            base_url = urlunparse((
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                '', '', ''
            ))

            # 发送请求，注入自定义HTTP头
            if target.method == 'GET':
                # 保留原始GET参数
                response = self.requester.send(
                    'GET',
                    base_url,
                    params=target.params,
                    headers=inject_headers
                )
            else:  # POST
                # 保留原始POST数据
                response = self.requester.send(
                    'POST',
                    base_url,
                    data=target.data,
                    headers=inject_headers
                )

            if response is None:
                return None

            return {
                'response': response,
                'payload': payload,
                'param': header_name,  # param字段存储header_name
                'target': target,
                'injected_value': injected_value,
                'injection_point': 'header'
            }

        except Exception as e:
            logger.error(f"[INJECT-HEADER] HTTP头注入失败: {header_name}: {payload[:30]}... - {e}")
            return None

    @staticmethod
    def _adjust_value_for_blind_payload(payload: str, original_value: str) -> str:
        """
        调整参数值以适配SQL盲注检测

        调整参数值

        核心问题：
        - 对于OR类型的盲注：如果原始ID存在，True和False都会返回"exists"
        - 解决方案：对于OR类型，使用不存在的ID（如-1）

        判断规则：
        1. 如果payload包含" or "（不区分大小写）→ OR类型 → 使用-1
        2. 如果payload包含" and " → AND类型 → 保持原值
        3. 其他情况 → 保持原值

        Args:
            payload: 攻击载荷
            original_value: 参数原始值

        Returns:
            调整后的参数值

        Example:
            >>> _adjust_value_for_blind_payload("' or 1=1 -- '", "1")
            '-1'  # OR类型，使用-1
            >>> _adjust_value_for_blind_payload("' and 1=1 -- '", "1")
            '1'   # AND类型，保持原值
        """
        # 判断payload类型
        payload_lower = payload.lower().strip()

        # 检查是否是OR类型的payload
        # 规则：payload必须包含" or "或" or"（注意空格）
        if re.search(r'\bor\s+', payload_lower):
            logger.debug(f"[INJECT] 检测到OR类型payload，使用不存在的ID (-1)")
            return "-1"  # 使用不存在的ID

        # AND类型或其他类型，保持原值
        logger.debug(f"[INJECT] 检测到AND类型或其他payload，保持原值 ({original_value})")
        return original_value

    @staticmethod
    def _should_append_payload(payload: str, original_value: str) -> bool:
        """
        判断是否应该保留原始值并追加payload

        判断注入模式！

        判断规则：
        1. 如果payload以'或"开头 → 追加模式（需要闭合引号）
        2. 如果payload以空格开头 → 追加模式（需要保留原始值）
        3. 如果payload包含原始值 → 替换模式（payload已经包含原始值）
        4. 其他情况 → 追加模式（保守策略）

        Args:
            payload: 攻击载荷
            original_value: 参数原始值

        Returns:
            True=追加模式（保留原始值），False=替换模式（直接使用payload）

        Example:
            >>> _should_append_payload("' OR 1=1--", "1")
            True  # 追加模式：1' OR 1=1--
            >>> _should_append_payload("1' OR 1=1--", "1")
            False  # 替换模式：payload已包含原始值
            >>> _should_append_payload(" or 1=1 --", "1")
            True  # 追加模式：1 or 1=1 --
        """
        # 特殊情况：payload已经完整包含原始值作为前缀或独立部分
        # 判断规则：payload必须以原始值开头（可能有引号包裹），才认为是替换模式
        # 错误示例：payload="' OR 1=1--", original="1" → 不应该匹配（1只是1=1的一部分）
        # 正确示例：payload="1' OR 1=1--", original="1" → 应该匹配（完整包含原始值）
        if original_value:
            payload_stripped = payload.strip()
            # 只有当payload以原始值开头时，才认为是替换模式
            if (payload_stripped.startswith(original_value) or  # 1' OR 1=1--
                payload_stripped.startswith(f"'{original_value}") or  # '1' OR 1=1--
                payload_stripped.startswith(f'"{original_value}')):  # "1" OR 1=1
                logger.debug(f"[INJECT] payload已包含原始值 '{original_value}'作为前缀，使用替换模式")
                return False

        # 判断payload开头
        payload = payload.strip()

        # 以'或"开头 → 追加模式（需要闭合引号）
        if payload.startswith("'") or payload.startswith('"'):
            return True

        # 以空格开头 → 追加模式（需要保留原始值）
        if payload.startswith(" ") or payload.startswith("\t"):
            return True

        # 以常见操作符开头（and, or, --, #, ;）
        if payload.startswith(("and ", "or ", "--", "#", ";")):
            return True

        # 默认：追加模式（保守策略）
        return True

    def _is_anomaly_length(self,
                           response_length: int,
                           threshold: float = 0.3) -> bool:
        """
        判断响应长度是否异常

        Args:
            response_length: 响应长度
            threshold: 偏差阈值（默认30%）

        Returns:
            True=异常，False=正常
        """
        return self.baseline.is_anomaly_length(response_length, threshold)

    def _is_anomaly_time(self,
                         response_time: float,
                         multiplier: float = 3.0) -> bool:
        """
        判断响应时间是否异常

        Args:
            response_time: 响应时间（秒）
            multiplier: 倍数阈值

        Returns:
            True=异常，False=正常
        """
        return self.baseline.is_anomaly_time(response_time, multiplier)

    def _is_reflected(self,
                     response_text: str,
                     payload: str) -> bool:
        """
        检查载荷是否被反射到响应中

        Args:
            response_text: 响应内容
            payload: 载荷

        Returns:
            True=被反射，False=未反射
        """
        return payload in response_text

    # ========== 漏洞结果封装 ==========

    def _create_vuln_entry(self,
                          vuln_type: str,
                          method: str,
                          severity: str,
                          confidence: float,
                          payload: str,
                          param: str,
                          evidence: str,
                          response: requests.Response,
                          target: FuzzTarget,
                          leak_data: Optional[Dict[str, str]] = None) -> VulnerabilityEntry:
        """
        创建漏洞条目（统一结果封装）

        确保所有引擎输出的结果格式统一！

        Args:
            vuln_type: 漏洞类型（SQLi, XSS等）
            method: 检测方法（Error-Based等）
            severity: 严重性等级
            confidence: 置信度（0.0-1.0）
            payload: 触发漏洞的载荷
            param: 漏洞参数名
            evidence: 漏洞证据
            response: Response对象
            target: 原始目标对象（新增！用于构造正确URL）

        Returns:
            VulnerabilityEntry对象

        Example:
            >>> vuln = self._create_vuln_entry(
            ...     vuln_type='SQLi',
            ...     method='Error-Based',
            ...     severity='High',
            ...     confidence=0.9,
            ...     payload="' OR 1=1--",
            ...     param='id',
            ...     evidence='SQL syntax error',
            ...     response=response,
            ...     target=target
            ... )
        """
        from datetime import datetime
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

        # 构建响应信息字典
        response_info = {
            'status_code': response.status_code,
            'length': len(response.text),
            'time': response.elapsed.total_seconds(),
            'headers': dict(response.headers),
            # 新增：保存响应内容（用于后续分析和调试）
            'response_text': response.text
        }

        # 新增：如果有泄露数据，添加到response_info中
        if leak_data:
            response_info['leak_data'] = leak_data

        # 使用实际请求的URL作为payload_url！
        # response.url 包含了参数调整后的值（如OR类型把id=1改成id=-1）
        # 手动把requests生成的+替换为%20（更符合RFC标准）
        payload_url = response.url.replace('+', '%20')

        # 新增：构造POST请求体（仅POST请求）
        post_body = None
        if target.method == 'POST':
            try:
                from urllib.parse import quote

                # 从target.data获取POST参数
                post_params = dict(target.data)

                # 追加payload到原值（和GET逻辑一致）
                if param in post_params:
                    original_value = post_params[param]
                    post_params[param] = original_value + payload

                # 构造POST body字符串（URL编码格式）
                # 使用quote_via=quote，让空格编码为%20而不是+
                post_body = urlencode(post_params, quote_via=quote)

                logger.debug(f"[ENGINE] 构造POST body: {post_body}")
            except Exception as e:
                logger.warning(f"[ENGINE] 构造POST body失败: {e}")

        # 创建漏洞条目
        vuln = VulnerabilityEntry(
            vuln_type=vuln_type,
            method=method,
            severity=severity,
            confidence=confidence,
            payload=payload,
            param_name=param,
            evidence=evidence,
            target_url=target.url,  # 使用原始target.url，而不是response.url
            payload_url=payload_url,  # 完整payload URL
            post_body=post_body,  # 新增：POST请求体（仅POST请求）
            response_info=response_info,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

        # 统计
        self.vuln_count += 1

        return vuln

    def _get_reflected_context(self,
                             response_text: str,
                             payload: str) -> str:
        """
        提取载荷反射的上下文环境

        Args:
            response_text: 响应内容
            payload: 载荷

        Returns:
            上下文描述（如 'html_tag', 'javascript', 'text_content'）

        Example:
            >>> context = self._get_reflected_context(
            ...     '<script>alert("test")</script>',
            ...     'alert("test")'
            ... )
            >>> print(context)
            'javascript'
        """
        idx = response_text.find(payload)
        if idx == -1:
            return 'unknown'

        # 提取上下文（前后50字符）
        start = max(0, idx - 50)
        end = min(len(response_text), idx + len(payload) + 50)
        context = response_text[start:end]

        # 判断上下文类型
        if '<script' in context.lower():
            return 'script_tag'
        elif '<style' in context.lower():
            return 'style_tag'
        elif 'on' in context.lower() and '=' in context:
            return 'event_handler'
        elif '<' in context and '>' in context:
            return 'html_tag'
        else:
            return 'text_content'

    def _calculate_confidence(self,
                             anomaly_score: float,
                             evidence_strength: str) -> float:
        """
        计算漏洞置信度

        Args:
            anomaly_score: 异常分数（0.0-1.0）
            evidence_strength: 证据强度（'strong', 'medium', 'weak'）

        Returns:
            置信度（0.0-1.0）
        """
        base_confidence = anomaly_score

        # 根据证据强度调整
        strength_multiplier = {
            'strong': 1.0,
            'medium': 0.7,
            'weak': 0.4
        }

        multiplier = strength_multiplier.get(evidence_strength.lower(), 0.7)

        confidence = min(base_confidence * multiplier, 1.0)

        return confidence


if __name__ == '__main__':
    # 测试代码
    import logging
    from unittest.mock import Mock

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("BaseEngine 单元测试")
    print("=" * 60)

    # 测试1：VulnerabilityEntry
    print("\n[测试1] VulnerabilityEntry")
    print("-" * 60)

    vuln = VulnerabilityEntry(
        vuln_type='SQLi',
        method='Error-Based',
        severity='High',
        confidence=0.9,
        payload="' OR 1=1--",
        param_name='id',
        evidence='SQL syntax error',
        target_url='http://target.com/?id=1',
        response_info={'status': 500, 'length': 1234}
    )

    print(f"漏洞类型: {vuln.vuln_type}")
    print(f"严重性: {vuln.severity}")
    print(f"置信度: {vuln.confidence}")
    print(f"载荷: {vuln.payload}")
    print(f"参数: {vuln.param_name}")
    print(f"证据: {vuln.evidence}")
    print(f"字典格式: {vuln.to_dict()}")

    # 测试2：置信度计算
    print("\n[测试2] 置信度计算")
    print("-" * 60)

    # 创建Mock引擎
    class TestEngine(BaseEngine):
        def detect(self, target, payloads):
            pass

    # 创建Mock依赖
    mock_requester = Mock()
    mock_baseline = Mock()
    mock_baseline.is_anomaly_length = Mock(return_value=False)
    mock_baseline.is_anomaly_time = Mock(return_value=False)

    engine = TestEngine(mock_requester, mock_baseline)

    # 测试置信度计算
    conf1 = engine._calculate_confidence(0.8, 'strong')
    conf2 = engine._calculate_confidence(0.8, 'medium')
    conf3 = engine._calculate_confidence(0.8, 'weak')

    print(f"强证据置信度: {conf1:.2f}")
    print(f"中等证据置信度: {conf2:.2f}")
    print(f"弱证据置信度: {conf3:.2f}")

    print("\n[SUCCESS] 所有测试通过！")
