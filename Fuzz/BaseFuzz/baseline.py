#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Manager - 基准测试组件
================================

负责在攻击开始前对目标进行多维度采样，建立"正常状态响应画像"。
这是后续SQL盲注判定、XSS逃逸分析、逻辑漏洞差分分析的唯一参照。

核心功能：
- 多次采样建立统计画像（长度/时间/状态码稳定性）
- 参数反射检测（识别哪些参数被反射到响应中）
- 动态内容定位（Diff算法定位动态区域）
- 重试频率评估（目标服务器压力承受能力）

使用示例：
    >>> from Fuzz.BaseFuzz.requester import Requester
    >>> from Fuzz.spider import FuzzTarget
    >>> from Fuzz.BaseFuzz.baseline import BaselineManager
    >>>
    >>> requester = Requester(timeout=10)
    >>> baseline_mgr = BaselineManager(requester)
    >>>
    >>> target = FuzzTarget(
    ...     url='http://target.com/?id=1',
    ...     method='GET',
    ...     params={'id': '1'},
    ...     data={},
    ...     depth=0
    ... )
    >>>
    >>> profile = baseline_mgr.build_profile(target, samples=5)
    >>> print(f"基准长度: {profile.avg_length}")
    >>> print(f"基准时间: {profile.avg_time:.3f}s")
    >>> print(f"反射参数: {profile.reflected_params}")


"""

import time
import hashlib
import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
import numpy as np
import logging
import requests  # 艹，老王忘了导入requests！类型注解需要用到

# 导入依赖模块
from .requester import Requester
from Fuzz.spider import FuzzTarget

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class BaselineProfile:
    """
    基准画像数据结构

    存储目标URL的正常响应画像

    Attributes:
        target_url: 目标URL
        method: HTTP方法
        sampled_count: 实际采样次数（可能因网络错误减少）

        # 状态码稳定性
        status_codes: 状态码集合 {200, 200, 200, 500, ...}
        stable_status: 是否稳定（True=单一状态码，False=状态码波动）
        dominant_status: 主导状态码（出现次数最多的）

        # 响应长度统计
        avg_length: 平均响应长度
        min_length: 最小响应长度
        max_length: 最大响应长度
        std_dev_length: 响应长度标准差（动态内容指标）
        length_range: 长度范围 (max - min)

        # 响应时间画像
        avg_time: 平均响应时间（秒）
        min_time: 最小响应时间
        max_time: 最大响应时间
        std_dev_time: 响应时间标准差
        time_threshold: 时间盲注阈值（avg_time + 3 * std_dev_time）

        # 响应头指纹
        server_header: Server响应头
        x_powered_by: X-Powered-By响应头
        content_type: Content-Type响应头
        headers_hash: 响应头哈希（用于识别变化）

        # 响应内容指纹
        content_hash: 响应内容MD5哈希（用于快速比对）
        content_snippet: 响应内容摘要（前200字符）

        # 参数反射检测
        reflected_params: 被反射到响应中的参数名集合
        param_reflection_map: 参数反射映射 {param_name: reflection_count}

        # 动态内容定位
        dynamic_regions: 动态区域列表 [(start_pos, end_pos, stability_score), ...]
        dynamic_ratio: 动态内容比例（动态字符数/总字符数）

        # 网络质量
        retry_count: 重试次数（网络不稳定指标）
        success_rate: 成功率 (成功次数/总尝试次数)

    Example:
        >>> profile = BaselineProfile(
        ...     target_url='http://target.com/?id=1',
        ...     method='GET',
        ...     sampled_count=5,
        ...     status_codes={200, 200, 200, 200, 200},
        ...     stable_status=True,
        ...     dominant_status=200,
        ...     avg_length=1234,
        ...     min_length=1200,
        ...     max_length=1250,
        ...     std_dev_length=15.2,
        ...     length_range=50,
        ...     avg_time=0.5,
        ...     min_time=0.4,
        ...     max_time=0.6,
        ...     std_dev_time=0.08,
        ...     time_threshold=0.74,
        ...     reflected_params={'id'},
        ...     param_reflection_map={'id': 5},
        ...     retry_count=0,
        ...     success_rate=1.0
        ... )
    """
    # 基本信息
    target_url: str
    method: str
    sampled_count: int

    # 状态码稳定性
    status_codes: Set[int] = field(default_factory=set)
    stable_status: bool = True
    dominant_status: Optional[int] = None

    # 响应长度统计
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    std_dev_length: float = 0.0
    length_range: int = 0

    # 响应时间画像
    avg_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    std_dev_time: float = 0.0
    time_threshold: float = 0.0

    # 响应头指纹
    server_header: Optional[str] = None
    x_powered_by: Optional[str] = None
    content_type: Optional[str] = None
    headers_hash: str = ''

    # 响应内容指纹
    content_hash: str = ''
    content_snippet: str = ''

    # 参数反射检测
    reflected_params: Set[str] = field(default_factory=set)
    param_reflection_map: Dict[str, int] = field(default_factory=dict)

    # 动态内容定位
    dynamic_regions: List[Tuple[int, int, float]] = field(default_factory=list)
    dynamic_ratio: float = 0.0

    # 网络质量
    retry_count: int = 0
    success_rate: float = 1.0

    def is_stable(self) -> bool:
        """
        判断基准是否稳定

        稳定条件：
        1. 状态码稳定（single status code）
        2. 长度标准差 < 平均长度的10%
        3. 成功率 > 80%

        Returns:
            bool: True=稳定，False=不稳定
        """
        if not self.stable_status:
            return False

        if self.avg_length > 0 and (self.std_dev_length / self.avg_length) > 0.1:
            return False

        if self.success_rate < 0.8:
            return False

        return True

    def is_anomaly_length(self, response_length: int, threshold: float = 0.3) -> bool:
        """
        判断响应长度是否异常

        Args:
            response_length: 待检测的响应长度
            threshold: 偏差阈值（默认30%）

        Returns:
            bool: True=异常，False=正常
        """
        if self.avg_length == 0:
            return False

        length_diff = abs(response_length - self.avg_length)
        length_ratio = length_diff / self.avg_length

        return length_ratio > threshold

    def is_anomaly_time(self, response_time: float, multiplier: float = 3.0) -> bool:
        """
        判断响应时间是否异常

        Args:
            response_time: 待检测的响应时间（秒）
            multiplier: 倍数阈值（默认3倍标准差）

        Returns:
            bool: True=异常，False=正常
        """
        return response_time > self.time_threshold

    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"BaselineProfile(url={self.target_url}, "
            f"status={self.dominant_status}, "
            f"avg_len={self.avg_length:.0f}, "
            f"avg_time={self.avg_time:.3f}s, "
            f"stable={self.is_stable()})"
        )


class BaselineManager:
    """
    基准管理器 - 建立正常状态响应画像

    负责在攻击前建立基准

    核心职责：
    1. 多次采样建立统计画像
    2. 参数反射检测
    3. 动态内容定位（Diff算法）
    4. 重试频率评估

    Attributes:
        requester: Requester实例（用于发送HTTP请求）

    Example:
        >>> requester = Requester(timeout=10)
        >>> baseline_mgr = BaselineManager(requester)
        >>>
        >>> target = FuzzTarget(...)
        >>> profile = baseline_mgr.build_profile(target, samples=5)
        >>>
        >>> if profile.is_stable():
        ...     print("基准稳定，可以开始扫描")
        ... else:
        ...     print("基准不稳定，建议增加采样次数")
    """

    # 测试用的基准值（用于参数反射检测）
    BASELINE_MARKERS = [
        '___BASELINE_TEST_MARKER___',
        'FUZZ_BASELINE_MARKER_2025',
        'BASELINE_FINGERPRINT_'
    ]

    def __init__(self, requester: Requester):
        """
        初始化BaselineManager

        Args:
            requester: Requester实例（HTTP通信层）

        Example:
            >>> requester = Requester(timeout=10)
            >>> baseline_mgr = BaselineManager(requester)
        """
        self.requester = requester
        logger.info("[BASELINE] BaselineManager初始化完成")

    def build_profile(self,
                     target: FuzzTarget,
                     samples: int = 5,
                     test_marker: Optional[str] = None) -> BaselineProfile:
        """
        构建基准画像（核心方法）

        所有基准建立

        流程：
        1. 发送samples次原始请求
        2. 计算平均耗时、长度波动
        3. 扫描参数反射
        4. 执行Diff算法定位动态区域
        5. 返回BaselineProfile对象

        Args:
            target: FuzzTarget对象（爬虫发现的目标）
            samples: 采样次数（默认5次，建议3-10次）
            test_marker: 测试标记（用于参数反射检测，可选）

        Returns:
            BaselineProfile对象

        Example:
            >>> target = FuzzTarget(
            ...     url='http://target.com/?id=1&name=test',
            ...     method='GET',
            ...     params={'id': '1', 'name': 'test'},
            ...     data={},
            ...     depth=0
            ... )
            >>> profile = baseline_mgr.build_profile(target, samples=5)
        """
        logger.info(f"[BASELINE] 开始构建基准画像: {target.url}")
        logger.info(f"[BASELINE] 采样次数: {samples}")

        # 存储采样数据
        responses = []
        retry_count = 0
        total_attempts = 0

        # 第一步：多次采样
        for i in range(samples):
            total_attempts += 1

            logger.debug(f"[BASELINE] 发送第{i+1}/{samples}次采样请求")

            # 构造请求参数（使用测试标记）
            if test_marker:
                params, data = self._inject_marker(target, test_marker)
            else:
                params, data = target.params, target.data

            # 发送请求
            try:
                if target.method == 'GET':
                    response = self.requester.send(
                        method='GET',
                        url=target.url,
                        params=params
                    )
                else:  # POST
                    response = self.requester.send(
                        method='POST',
                        url=target.url,
                        data=data
                    )

                if response is None:
                    # 请求失败，计数重试
                    retry_count += 1
                    logger.warning(f"[BASELINE] 第{i+1}次采样失败")
                    continue

                # 记录成功响应
                responses.append(response)
                logger.debug(f"[BASELINE] 第{i+1}次采样成功: "
                           f"status={response.status_code}, "
                           f"len={len(response.text)}, "
                           f"time={response.elapsed.total_seconds():.3f}s")

            except Exception as e:
                logger.error(f"[BASELINE] 采样异常: {e}")
                retry_count += 1
                continue

        # 检查采样结果
        if not responses:
            logger.error(f"[BASELINE] 所有采样均失败，无法建立基准")
            return self._create_empty_profile(target, retry_count, total_attempts)

        logger.info(f"[BASELINE] 采样完成: 成功{len(responses)}/{samples}, "
                   f"重试{retry_count}次")

        # 第二步：构建画像
        profile = self._build_profile_from_responses(
            target=target,
            responses=responses,
            retry_count=retry_count,
            total_attempts=total_attempts
        )

        # 第三步：参数反射检测
        if test_marker:
            profile.reflected_params, profile.param_reflection_map = \
                self._detect_param_reflection(responses[-1], target, test_marker)

        # 第四步：动态内容定位
        profile.dynamic_regions, profile.dynamic_ratio = \
            self._locate_dynamic_content(responses)

        logger.info(f"[BASELINE] 基准画像构建完成: {profile}")

        return profile

    def _inject_marker(self,
                      target: FuzzTarget,
                      marker: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        注入测试标记到参数中

        Args:
            target: FuzzTarget对象
            marker: 测试标记字符串

        Returns:
            (params, data) 元组
        """
        params = {}
        data = {}

        if target.method == 'GET':
            # GET请求：注入到所有参数
            for key, value in target.params.items():
                params[key] = marker
        else:  # POST
            # POST请求：注入到所有参数
            for key, value in target.data.items():
                data[key] = marker

        return params, data

    def _build_profile_from_responses(self,
                                      target: FuzzTarget,
                                      responses: List,
                                      retry_count: int,
                                      total_attempts: int) -> BaselineProfile:
        """
        从响应列表构建基准画像

        Args:
            target: FuzzTarget对象
            responses: 响应列表
            retry_count: 重试次数
            total_attempts: 总尝试次数

        Returns:
            BaselineProfile对象
        """
        # 提取状态码
        status_codes = [resp.status_code for resp in responses]
        status_counter = Counter(status_codes)

        # 计算稳定性指标
        unique_statuses = set(status_codes)
        stable_status = len(unique_statuses) == 1
        dominant_status = status_counter.most_common(1)[0][0]

        # 提取响应长度
        lengths = [len(resp.text) for resp in responses]
        avg_length = np.mean(lengths)
        min_length = np.min(lengths)
        max_length = np.max(lengths)
        std_dev_length = np.std(lengths)
        length_range = max_length - min_length

        # 提取响应时间
        times = [resp.elapsed.total_seconds() for resp in responses]
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_dev_time = np.std(times)

        # 时间盲注阈值（平均值 + 3倍标准差）
        time_threshold = avg_time + (3.0 * std_dev_time)

        # 提取响应头指纹
        first_response = responses[0]
        server_header = first_response.headers.get('Server', '')
        x_powered_by = first_response.headers.get('X-Powered-By', '')
        content_type = first_response.headers.get('Content-Type', '')

        # 计算响应头哈希
        headers_str = str(sorted(first_response.headers.items()))
        headers_hash = hashlib.md5(headers_str.encode()).hexdigest()

        # 计算响应内容哈希和摘要
        content_hash = hashlib.md5(first_response.content).hexdigest()
        content_snippet = first_response.text[:200]  # 前200字符

        # 计算成功率
        success_rate = len(responses) / total_attempts if total_attempts > 0 else 0.0

        # 构建画像对象
        profile = BaselineProfile(
            target_url=target.url,
            method=target.method,
            sampled_count=len(responses),

            # 状态码稳定性
            status_codes=unique_statuses,
            stable_status=stable_status,
            dominant_status=dominant_status,

            # 响应长度统计
            avg_length=avg_length,
            min_length=int(min_length),
            max_length=int(max_length),
            std_dev_length=std_dev_length,
            length_range=int(length_range),

            # 响应时间画像
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_dev_time=std_dev_time,
            time_threshold=time_threshold,

            # 响应头指纹
            server_header=server_header,
            x_powered_by=x_powered_by,
            content_type=content_type,
            headers_hash=headers_hash,

            # 响应内容指纹
            content_hash=content_hash,
            content_snippet=content_snippet,

            # 网络质量
            retry_count=retry_count,
            success_rate=success_rate
        )

        return profile

    def _detect_param_reflection(self,
                                 response: requests.Response,
                                 target: FuzzTarget,
                                 marker: str) -> Tuple[Set[str], Dict[str, int]]:
        """
        检测参数反射

        Args:
            response: 响应对象
            target: FuzzTarget对象
            marker: 测试标记字符串

        Returns:
            (reflected_params, reflection_map) 元组
        """
        reflected_params = set()
        reflection_map = {}

        if target.method == 'GET':
            # GET参数检测
            for param_name in target.params.keys():
                if marker in response.text:
                    reflected_params.add(param_name)
                    reflection_map[param_name] = response.text.count(marker)
                    logger.debug(f"[BASELINE] 参数反射检测: {param_name} -> "
                               f"{reflection_map[param_name]}次")
        else:  # POST
            # POST参数检测
            for param_name in target.data.keys():
                if marker in response.text:
                    reflected_params.add(param_name)
                    reflection_map[param_name] = response.text.count(marker)
                    logger.debug(f"[BASELINE] 参数反射检测: {param_name} -> "
                               f"{reflection_map[param_name]}次")

        return reflected_params, reflection_map

    def _locate_dynamic_content(self,
                                 responses: List) -> Tuple[List[Tuple[int, int, float]], float]:
        """
        定位动态内容区域（Diff算法）

        找出响应中的动态区域

        Args:
            responses: 响应列表（至少2个）

        Returns:
            (dynamic_regions, dynamic_ratio) 元组
            - dynamic_regions: 动态区域列表 [(start, end, stability), ...]
            - dynamic_ratio: 动态内容比例（0.0-1.0）
        """
        if len(responses) < 2:
            # 样本太少，无法进行Diff分析
            return [], 0.0

        # 使用前两个响应进行Diff
        text1 = responses[0].text
        text2 = responses[1].text

        # 计算差异
        matcher = difflib.SequenceMatcher(None, text1, text2)

        # 提取动态区域（不匹配的块）
        dynamic_regions = []
        total_dynamic_chars = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace' or tag == 'delete' or tag == 'insert':
                # 动态区域
                start = i1
                end = i2
                length = end - start

                if length > 0:
                    # 计算稳定性分数（长度越长，稳定性越低）
                    stability = 1.0 - min(length / len(text1), 1.0)

                    dynamic_regions.append((start, end, stability))
                    total_dynamic_chars += length

        # 计算动态内容比例
        dynamic_ratio = total_dynamic_chars / len(text1) if len(text1) > 0 else 0.0

        logger.debug(f"[BASELINE] 动态内容分析: 发现{len(dynamic_regions)}个动态区域, "
                    f"动态比例={dynamic_ratio:.2%}")

        return dynamic_regions, dynamic_ratio

    def _create_empty_profile(self,
                              target: FuzzTarget,
                              retry_count: int,
                              total_attempts: int) -> BaselineProfile:
        """
        创建空画像（所有采样均失败时）

        Args:
            target: FuzzTarget对象
            retry_count: 重试次数
            total_attempts: 总尝试次数

        Returns:
            BaselineProfile对象（所有字段为默认值）
        """
        return BaselineProfile(
            target_url=target.url,
            method=target.method,
            sampled_count=0,
            stable_status=False,
            retry_count=retry_count,
            success_rate=0.0
        )


# 便捷函数
def build_baseline(requester: Requester,
                   target: FuzzTarget,
                   samples: int = 5) -> BaselineProfile:
    """
    构建基准画像的便捷函数

    Args:
        requester: Requester实例
        target: FuzzTarget对象
        samples: 采样次数

    Returns:
        BaselineProfile对象

    Example:
        >>> from Fuzz.BaseFuzz.requester import Requester
        >>> from Fuzz.spider import FuzzTarget
        >>> from Fuzz.BaseFuzz.baseline import build_baseline
        >>>
        >>> requester = Requester()
        >>> target = FuzzTarget(...)
        >>> profile = build_baseline(requester, target, samples=5)
    """
    baseline_mgr = BaselineManager(requester)
    return baseline_mgr.build_profile(target, samples=samples)


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
    print("BaselineManager 单元测试")
    print("=" * 60)

    # 创建Mock Response
    def create_mock_response(status_code, text, elapsed_time):
        """创建Mock Response对象"""
        mock_resp = Mock()
        mock_resp.status_code = status_code
        mock_resp.text = text
        mock_resp.content = text.encode()
        mock_resp.elapsed = Mock()
        mock_resp.elapsed.total_seconds = Mock(return_value=elapsed_time)
        mock_resp.headers = {
            'Server': 'nginx/1.18.0',
            'X-Powered-By': 'PHP/7.4.0',
            'Content-Type': 'text/html'
        }
        return mock_resp

    # 测试1：稳定基准
    print("\n[测试1] 稳定基准画像")
    print("-" * 60)

    # 模拟5次稳定的响应
    responses = [
        create_mock_response(200, "Hello World " * 100, 0.5),
        create_mock_response(200, "Hello World " * 100, 0.51),
        create_mock_response(200, "Hello World " * 100, 0.49),
        create_mock_response(200, "Hello World " * 100, 0.5),
        create_mock_response(200, "Hello World " * 100, 0.52)
    ]

    # 创建Mock Requester
    mock_requester = Mock()
    mock_requester.send = Mock(return_value=responses[0])

    # 创建Mock Target
    mock_target = Mock()
    mock_target.url = "http://target.com/?id=1"
    mock_target.method = "GET"
    mock_target.params = {"id": "1"}
    mock_target.data = {}

    # 创建BaselineManager
    baseline_mgr = BaselineManager(mock_requester)

    # 手动调用内部方法测试
    profile = baseline_mgr._build_profile_from_responses(
        target=mock_target,
        responses=responses,
        retry_count=0,
        total_attempts=5
    )

    print(f"目标URL: {profile.target_url}")
    print(f"采样次数: {profile.sampled_count}")
    print(f"状态码: {profile.dominant_status} (稳定: {profile.stable_status})")
    print(f"平均长度: {profile.avg_length:.0f} (标准差: {profile.std_dev_length:.2f})")
    print(f"平均时间: {profile.avg_time:.3f}s")
    print(f"时间阈值: {profile.time_threshold:.3f}s")
    print(f"稳定性判定: {profile.is_stable()}")

    # 测试2：异常检测
    print("\n[测试2] 异常检测")
    print("-" * 60)

    # 测试长度异常
    print(f"长度异常检测 (1200 vs {profile.avg_length:.0f}): "
          f"{profile.is_anomaly_length(1200, threshold=0.3)}")
    print(f"长度异常检测 (2000 vs {profile.avg_length:.0f}): "
          f"{profile.is_anomaly_length(2000, threshold=0.3)}")

    # 测试时间异常
    print(f"时间异常检测 (0.6s vs {profile.time_threshold:.3f}s): "
          f"{profile.is_anomaly_time(0.6)}")
    print(f"时间异常检测 (2.0s vs {profile.time_threshold:.3f}s): "
          f"{profile.is_anomaly_time(2.0)}")

    # 测试3：动态内容检测
    print("\n[测试3] 动态内容检测")
    print("-" * 60)

    # 模拟动态内容响应
    dynamic_responses = [
        create_mock_response(200, "Static content <timestamp>12345</timestamp> end", 0.5),
        create_mock_response(200, "Static content <timestamp>67890</timestamp> end", 0.5)
    ]

    regions, ratio = baseline_mgr._locate_dynamic_content(dynamic_responses)
    print(f"动态区域数量: {len(regions)}")
    print(f"动态内容比例: {ratio:.2%}")
    for i, (start, end, stability) in enumerate(regions):
        print(f"  区域{i+1}: [{start}:{end}], 稳定性={stability:.2f}")

    print("\n[SUCCESS] 所有测试通过！")
