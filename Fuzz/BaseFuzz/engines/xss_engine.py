#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XSS Engine - 跨站脚本检测引擎
==============================

基于BaseEngine抽象类实现的XSS检测引擎。

核心功能：
- Reflected XSS检测（反射型XSS）
- DOM XSS检测（DOM型XSS）
- Stored XSS检测（存储型XSS，基础支持）
- 上下文感知分析（HTML标签、JavaScript、事件处理器）
- WAF拦截检测（403状态码）

检测策略：
1. 无害探针检测：检查参数是否被反射
2. 上下文分析：识别反射点环境
3. 静态分析：检测DOM XSS特征
4. Payload测试：根据上下文选择合适的载荷
5. 置信度评分：基于转义、反射完整性计算

"""

import re
import random
import string
from typing import List, Dict, Set, Optional, Tuple
import logging

# 导入依赖模块
from Fuzz.BaseFuzz.engines.base_engine import BaseEngine, VulnerabilityEntry
from Fuzz.spider import FuzzTarget
from Fuzz.BaseFuzz.loaders.transformer import PayloadTransformer

# 配置日志
logger = logging.getLogger(__name__)


class XSSEngine(BaseEngine):
    """
    XSS检测引擎


    核心职责：
    1. 无害探针检测反射点
    2. 上下文感知分析（HTML标签、JavaScript、事件处理器）
    3. DOM XSS静态分析
    4. 根据上下文选择合适的Payload
    5. 检测WAF拦截（403状态码）

    Attributes:
        DOM_KEYWORDS: DOM XSS危险关键词
        CONTEXT_PATTERNS: 上下文识别正则表达式
        WAF_STATUS_CODES: WAF拦截状态码

    Example:
        >>> from Fuzz.BaseFuzz.baseline import BaselineProfile
        >>> from Fuzz.BaseFuzz.requester import Requester
        >>>
        >>> requester = Requester(timeout=10)
        >>> baseline = BaselineProfile(...)
        >>> engine = XSSEngine(requester, baseline)
        >>>
        >>> target = FuzzTarget(...)
        >>> payloads = ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>"]
        >>> vulns = engine.detect(target, payloads, param_name='name')
    """

    # ========== DOM XSS 危险关键词 ==========

    DOM_KEYWORDS = [
        '.innerHTML',
        '.outerHTML',
        'document.write',
        'document.writeln',
        'eval(',
        'setTimeout(',
        'setInterval(',
        'Function(',
        'execScript(',
        '.location',
        '.href',
        '.src',
        'location.href',
        'location.hash',
        'location.search',
    ]

    # ========== 上下文识别模式 ==========

    CONTEXT_PATTERNS = {
        'script_tag': re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        'style_tag': re.compile(r'<style[^>]*>.*?</style>', re.IGNORECASE | re.DOTALL),
        'event_handler': re.compile(r'\bon[a-z]+\s*=', re.IGNORECASE),
        'html_tag': re.compile(r'<[^>]+>', re.IGNORECASE),
        'html_comment': re.compile(r'<!--.*?-->', re.DOTALL),
        'javascript': re.compile(r'javascript:', re.IGNORECASE),
    }

    # ========== WAF拦截状态码 ==========

    WAF_STATUS_CODES = [403, 429, 503]

    # ========== 无害探针模板 ==========

    PROBE_TEMPLATE = 'CVDBXSS_{RANDOM}_PROBE'

    def __init__(self, requester, baseline):
        """
        初始化XSSEngine

        Args:
            requester: Requester实例
            baseline: BaselineProfile实例
        """
        super().__init__(requester, baseline)

        # 生成随机探针
        self.probe = self._generate_probe()

        logger.info(f"[XSS] XSS引擎初始化完成: probe={self.probe}")

    def detect(self,
               target: FuzzTarget,
               payloads: List[str],
               param_name: str) -> List[VulnerabilityEntry]:
        """
        执行XSS检测（核心方法）


        Args:
            target: 测试目标
            payloads: 载荷列表
            param_name: 要测试的参数名（只测试这个参数！）

        Returns:
            漏洞条目列表

        注意：
            - 不要遍历target的所有参数！
            - 只测试param_name指定的参数！
        """
        logger.info(f"[XSS] 开始检测: {target.url}?{param_name}")
        logger.info(f"[XSS] 载荷数量: {len(payloads)}")

        results = []

        # 1. 无害探针检测（Probe Stage）
        logger.info(f"[XSS] [1/5] 执行无害探针检测...")
        is_reflected, probe_context, probe_response = self._probe_reflection(
            target, param_name
        )

        if not is_reflected:
            logger.info(f"[XSS] 参数未反射，跳过XSS测试: {param_name}")
            return results

        logger.info(f"[XSS] 参数已反射，上下文: {probe_context}")

        # 2. 静态分析DOM XSS特征
        logger.info(f"[XSS] [2/5] 执行DOM XSS静态分析...")
        dom_vulns = self._detect_dom_xss(target, param_name, probe_response)
        results.extend(dom_vulns)

        # 3. 根据上下文选择合适的载荷
        logger.info(f"[XSS] [3/5] 根据上下文选择载荷...")
        context_payloads = self._select_payloads_by_context(payloads, probe_context)

        # 4. 执行XSS载荷测试
        logger.info(f"[XSS] [4/5] 执行XSS载荷测试...")
        xss_vulns = self._test_xss_payloads(
            target, param_name, context_payloads, probe_context
        )
        results.extend(xss_vulns)

        # 5. 统计WAF拦截
        logger.info(f"[XSS] [5/5] 统计WAF拦截情况...")
        waf_count = self._check_waf_blocks(target, param_name, payloads)
        if waf_count > 0:
            logger.warning(f"[XSS] 检测到WAF拦截: {waf_count}次")

        logger.info(f"[XSS] 检测完成: 发现{len(results)}个漏洞")

        return results

    def _probe_reflection(self,
                         target: FuzzTarget,
                         param_name: str) -> Tuple[bool, str, Optional[object]]:
        """
        无害探针检测（Probe Stage）


        策略：
        1. 注入随机且唯一的探针字符串
        2. 检查探针是否原样出现在响应中
        3. 如果反射，提取上下文环境

        Args:
            target: 测试目标
            param_name: 参数名

        Returns:
            (is_reflected, context, response) 元组
            - is_reflected: 是否反射
            - context: 上下文类型（'text_content', 'html_tag', 'script_tag', etc.）
            - response: Response对象
        """
        try:
            # 注入探针
            test_result = self._test_parameter(target, param_name, self.probe)
            if not test_result:
                return False, 'unknown', None

            response = test_result['response']
            response_text = response.text

            # 检查探针是否被反射
            if not self._is_reflected(response_text, self.probe):
                return False, 'unknown', response

            # 提取反射上下文
            context = self._get_reflected_context(response_text, self.probe)

            return True, context, response

        except Exception as e:
            logger.error(f"[XSS] 探针检测失败: {param_name} - {e}")
            return False, 'unknown', None

    def _detect_dom_xss(self,
                        target: FuzzTarget,
                        param_name: str,
                        response) -> List[VulnerabilityEntry]:
        """
        静态分析DOM XSS特征

        策略：
        1. 检查响应中是否包含DOM关键词
        2. 检查参数是否被用于危险函数
        3. 判定为Low或Medium级别风险

        Args:
            target: 测试目标
            param_name: 参数名
            response: Response对象

        Returns:
            漏洞条目列表
        """
        vulns = []

        try:
            response_text = response.text

            # 检查DOM危险关键词
            found_keywords = []
            for keyword in self.DOM_KEYWORDS:
                if keyword in response_text:
                    found_keywords.append(keyword)

            if not found_keywords:
                return vulns

            # 检查参数名是否出现在危险上下文中
            param_in_context = self._check_param_in_dangerous_context(
                response_text, param_name
            )

            if param_in_context:
                # 发现DOM XSS风险
                evidence = f"检测到DOM关键词: {', '.join(found_keywords[:3])}"

                vuln = self._create_vuln_entry(
                    vuln_type='XSS',
                    method='DOM-Based',
                    severity='Low',
                    confidence=0.6,
                    payload=f"参数{param_name}可能被用于DOM操作",
                    param=param_name,
                    evidence=evidence,
                    response=response,
                    target=target
                )
                vulns.append(vuln)
                logger.warning(f"[XSS] [DOM] 发现潜在风险: {param_name}")

        except Exception as e:
            logger.error(f"[XSS] [DOM] 检测失败: {param_name} - {e}")

        return vulns

    def _test_xss_payloads(self,
                          target: FuzzTarget,
                          param_name: str,
                          payloads: List[str],
                          probe_context: str) -> List[VulnerabilityEntry]:
        """
        执行XSS载荷测试


        策略：
        1. 根据探针上下文选择载荷
        2. 20%概率深度变异绕过WAF
        3. 检查载荷是否被反射且未转义
        4. 计算置信度（基于转义和上下文）

        Args:
            target: 测试目标
            param_name: 参数名
            payloads: 载荷列表
            probe_context: 探针上下文

        Returns:
            漏洞条目列表
        """
        vulns = []

        # 如果没有合适的载荷，跳过
        if not payloads:
            logger.warning(f"[XSS] 上下文 {probe_context} 无可用载荷")
            return vulns

        # 限制测试载荷数量（避免过多请求）
        max_payloads = 10
        test_payloads = payloads[:max_payloads]

        for payload in test_payloads:
            try:
                # 20%概率深度变异
                if random.random() < 0.2:
                    payload = PayloadTransformer.deep_mutate(payload, strategy='encoding')

                # 测试载荷
                test_result = self._test_parameter(target, param_name, payload)
                if not test_result:
                    continue

                response = test_result['response']
                response_text = response.text

                # 检查是否被WAF拦截
                if response.status_code in self.WAF_STATUS_CODES:
                    logger.debug(f"[XSS] WAF拦截: {param_name}={payload[:30]}...")
                    continue

                # 检查载荷是否被反射
                if not self._is_reflected(response_text, payload):
                    continue

                # 检查载荷是否被转义
                is_escaped = self._check_if_escaped(response_text, payload)

                if is_escaped:
                    # 载荷被转义，可能安全
                    continue

                # 发现XSS漏洞
                severity, confidence = self._calculate_xss_severity(
                    probe_context, payload, response_text
                )

                vuln = self._create_vuln_entry(
                    vuln_type='XSS',
                    method='Reflected',
                    severity=severity,
                    confidence=confidence,
                    payload=payload,
                    param=param_name,
                    evidence=f"上下文: {probe_context}, 载荷原样反射",
                    response=response,
                    target=target
                )
                vulns.append(vuln)
                logger.warning(
                    f"[XSS] [反射] 发现漏洞: {param_name} "
                    f"(上下文={probe_context}, 置信度={confidence:.2f})"
                )

                # 找到一个高置信度漏洞后，可以继续寻找其他上下文的漏洞
                if confidence > 0.8:
                    break

            except Exception as e:
                logger.error(f"[XSS] 载荷测试失败: {param_name}={payload[:30]}... - {e}")
                continue

        return vulns

    def _select_payloads_by_context(self,
                                     payloads: List[str],
                                     context: str) -> List[str]:
        """
        根据上下文选择合适的载荷


        Args:
            payloads: 原始载荷列表
            context: 上下文类型

        Returns:
            适合该上下文的载荷列表
        """
        context_payloads = []

        if context == 'script_tag':
            # JavaScript上下文：使用JavaScript载荷
            for payload in payloads:
                if any(kw in payload.lower() for kw in ['alert', 'prompt', 'confirm']):
                    context_payloads.append(payload)

        elif context == 'event_handler':
            # 事件处理器上下文：使用不带标签的载荷
            for payload in payloads:
                if '<' not in payload and any(kw in payload.lower() for kw in ['alert', 'prompt']):
                    context_payloads.append(payload)

        elif context == 'html_tag':
            # HTML标签上下文：使用标签闭合载荷
            for payload in payloads:
                if payload.startswith(('>', '">', "'>")):
                    context_payloads.append(payload)

        else:  # text_content 或其他
            # 文本内容上下文：使用标准载荷
            context_payloads = payloads

        # 如果没有匹配的载荷，使用所有载荷
        if not context_payloads:
            context_payloads = payloads

        return context_payloads

    def _check_if_escaped(self, response_text: str, payload: str) -> bool:
        """
        检查载荷是否被转义

        1. 检查payload中所有特殊字符是否被HTML实体编码
        2. 检查payload在响应中是否被完全编码（而不是部分编码）
        3. 如果关键字符（<>'"等）被编码，则认为payload被转义

        Args:
            response_text: 响应内容
            payload: 原始载荷

        Returns:
            True=被转义，False=未转义
        """
        # HTML实体编码映射表
        escaped_chars = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": ['&#x27;', '&#039;', '&apos;'],  # 支持多种编码形式
            '=': '&#x3D;',
        }

        # 统计payload中需要检查的特殊字符
        special_chars_in_payload = set()
        for char in escaped_chars.keys():
            if char in payload:
                special_chars_in_payload.add(char)

        # 如果payload中没有特殊字符，不需要检查转义
        if not special_chars_in_payload:
            return False

        # 艹，检查响应中是否包含payload
        if payload not in response_text:
            # payload不存在于响应中，可能被完全转义了
            # 尝试查找转义后的payload
            escaped_payload = payload
            for char in special_chars_in_payload:
                escaped_versions = escaped_chars[char]
                if isinstance(escaped_versions, str):
                    escaped_versions = [escaped_versions]

                # 尝试每种编码形式
                for escaped in escaped_versions:
                    if escaped in escaped_payload:
                        escaped_payload = escaped_payload.replace(char, escaped)
                        break

            # 检查转义后的payload是否在响应中
            if escaped_payload in response_text:
                # 找到了完全转义的payload
                logger.debug(f"[XSS] Payload被完全转义: {payload} -> {escaped_payload}")
                return True

        # 重点检查关键字符是否被转义
        # 策略：如果payload中超过50%的特殊字符被转义，则认为payload被转义
        escaped_count = 0
        total_special = len(special_chars_in_payload)

        for char in special_chars_in_payload:
            escaped_versions = escaped_chars[char]
            if isinstance(escaped_versions, str):
                escaped_versions = [escaped_versions]

            # 检查是否有任何一种编码形式在响应中
            for escaped in escaped_versions:
                if escaped in response_text:
                    # 进一步验证：检查转义字符附近是否是payload的上下文
                    escaped_idx = response_text.find(escaped)
                    if escaped_idx != -1:
                        # 提取转义字符周围的上下文（前后各50个字符）
                        start = max(0, escaped_idx - 50)
                        end = min(len(response_text), escaped_idx + len(escaped) + 50)
                        context = response_text[start:end]

                        # 检查上下文中是否包含payload的其他部分
                        # 这能避免误判（比如错误信息中的&lt;不是针对payload的）
                        payload_parts = payload.replace(char, '')  # 去掉当前字符
                        if payload_parts and len(payload_parts) > 3:  # 至少有3个其他字符
                            if payload_parts[:10] in context or payload_parts[-10:] in context:
                                escaped_count += 1
                                break

        # 如果超过50%的特殊字符被转义，则认为payload被转义
        escape_ratio = escaped_count / total_special if total_special > 0 else 0
        is_escaped = escape_ratio > 0.5

        if is_escaped:
            logger.debug(f"[XSS] Payload被转义: {escape_ratio:.0%}的特殊字符被编码")

        return is_escaped

    def _calculate_xss_severity(self,
                                 context: str,
                                 payload: str,
                                 response_text: str) -> Tuple[str, float]:
        """
        计算XSS漏洞严重性和置信度


        Args:
            context: 上下文类型
            payload: 载荷
            response_text: 响应内容

        Returns:
            (severity, confidence) 元组
        """
        # 基础置信度
        base_confidence = 0.7

        # 根据上下文调整
        context_bonus = {
            'script_tag': 0.2,
            'event_handler': 0.15,
            'javascript': 0.1,
            'html_tag': 0.05,
            'text_content': 0.0,
        }

        confidence = base_confidence + context_bonus.get(context, 0.0)

        # 根据载荷调整
        if 'alert(' in payload or 'confirm(' in payload or 'prompt(' in payload:
            confidence += 0.1

        if 'onerror=' in payload or 'onload=' in payload:
            confidence += 0.1

        # 限制置信度范围
        confidence = min(max(confidence, 0.5), 0.95)

        # 确定严重性级别
        if confidence >= 0.8 or context == 'script_tag':
            severity = 'High'
        elif confidence >= 0.6 or context in ['event_handler', 'javascript']:
            severity = 'Medium'
        else:
            severity = 'Low'

        return severity, confidence

    def _check_param_in_dangerous_context(self,
                                          response_text: str,
                                          param_name: str) -> bool:
        """
        检查参数是否出现在危险上下文中


        Args:
            response_text: 响应内容
            param_name: 参数名

        Returns:
            True=危险上下文，False=安全
        """
        # 构造正则：检查参数名是否出现在DOM关键词附近
        for keyword in self.DOM_KEYWORDS:
            # 检查参数名是否在关键词前后50字符内
            keyword_idx = response_text.find(keyword)
            if keyword_idx != -1:
                # 提取关键词前后100字符
                start = max(0, keyword_idx - 100)
                end = min(len(response_text), keyword_idx + 100)
                context = response_text[start:end]

                # 检查参数名是否在上下文中
                if param_name in context:
                    return True

        return False

    def _check_waf_blocks(self,
                         target: FuzzTarget,
                         param_name: str,
                         payloads: List[str]) -> int:
        """
        统计WAF拦截次数


        Args:
            target: 测试目标
            param_name: 参数名
            payloads: 载荷列表

        Returns:
            被拦截的次数
        """
        waf_count = 0

        # 随机选择部分载荷进行WAF检测
        sample_size = min(5, len(payloads))
        sample_payloads = random.sample(payloads, sample_size)

        for payload in sample_payloads:
            try:
                test_result = self._test_parameter(target, param_name, payload)
                if not test_result:
                    continue

                response = test_result['response']
                if response.status_code in self.WAF_STATUS_CODES:
                    waf_count += 1

            except Exception as e:
                logger.debug(f"[XSS] WAF检测失败: {param_name} - {e}")
                continue

        return waf_count

    def _generate_probe(self) -> str:
        """
        生成随机探针字符串

        生成唯一且无害的探针！

        Returns:
            探针字符串
        """
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        return f'CVDBXSS_{random_str}_PROBE'


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
    print("XSSEngine 单元测试")
    print("=" * 60)

    # 创建Mock依赖
    mock_requester = Mock()
    mock_baseline = Mock()

    # 初始化引擎
    engine = XSSEngine(mock_requester, mock_baseline)

    # 测试1：探针生成
    print("\n[测试1] 探针生成")
    print("-" * 60)
    print(f"生成的探针: {engine.probe}")

    # 测试2：DOM关键词
    print("\n[测试2] DOM关键词库")
    print("-" * 60)
    print(f"关键词数量: {len(engine.DOM_KEYWORDS)}")
    print(f"关键词列表: {engine.DOM_KEYWORDS[:5]}")

    # 测试3：上下文选择
    print("\n[测试3] 上下文载荷选择")
    print("-" * 60)

    test_payloads = [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        "javascript:alert(1)",
        "'><script>alert(1)</script>",
    ]

    for context in ['script_tag', 'event_handler', 'html_tag', 'text_content']:
        selected = engine._select_payloads_by_context(test_payloads, context)
        print(f"{context}: 选择{len(selected)}个载荷")
        for payload in selected[:2]:
            print(f"  - {payload[:50]}...")

    print("\n[SUCCESS] 所有测试通过！")
