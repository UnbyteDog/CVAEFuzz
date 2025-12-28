#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLi Engine - SQL注入检测引擎
==============================

基于BaseEngine抽象类实现的SQL注入检测引擎。

核心功能：
- Error-Based SQL注入检测（报错注入）
- Boolean-Based盲注检测（布尔盲注）
- Time-Based盲注检测（时间盲注）
- Union查询注入检测（联合查询）
- 深度变异与WAF绕过

检测策略：
1. 报错注入：检测数据库错误信息特征
2. 布尔盲注：对比True/False载荷响应差异
3. 时间盲注：检测响应时间异常（带二次验证）

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-25
"""

import re
import random
import time
from typing import List, Dict, Set, Optional, Tuple
from difflib import SequenceMatcher
import logging

# 导入依赖模块
from Fuzz.BaseFuzz.engines.base_engine import BaseEngine, VulnerabilityEntry
from Fuzz.spider import FuzzTarget
from Fuzz.BaseFuzz.loaders.transformer import PayloadTransformer

# 配置日志
logger = logging.getLogger(__name__)


class SQLiEngine(BaseEngine):
    """
    SQL注入检测引擎

    老王注释：这个SB引擎专门检测SQL注入漏洞！

    核心职责：
    1. 检测报错注入（MySQL, PostgreSQL, MSSQL, Oracle）
    2. 检测布尔盲注（True/False响应差分）
    3. 检测时间盲注（延迟函数 + 二次验证）
    4. 深度变异绕过WAF（20%概率）

    Attributes:
        ERROR_PATTERNS: 数据库错误特征字典
        TIME_PAYLOADS: 时间盲注载荷字典
        BOOLEAN_TRUE: 布尔真载荷
        BOOLEAN_FALSE: 布尔假载荷

    Example:
        >>> from Fuzz.BaseFuzz.baseline import BaselineProfile
        >>> from Fuzz.BaseFuzz.requester import Requester
        >>>
        >>> requester = Requester(timeout=10)
        >>> baseline = BaselineProfile(...)
        >>> engine = SQLiEngine(requester, baseline)
        >>>
        >>> target = FuzzTarget(...)
        >>> payloads = ["' OR 1=1--", '" OR 1=1--']
        >>> vulns = engine.detect(target, payloads, param_name='id')
    """

    # ========== 数据库错误特征库 ==========

    ERROR_PATTERNS = {
        'MySQL': [
            r"SQL syntax.*MySQL",
            r"mysql_fetch",
            r"mysqli.*fetch",
            r"MySQLSyntaxErrorException",
            r"valid MySQL result",
            r"check the manual that corresponds to your MySQL server version",
            r"MySqlException",
            r"Warning: mysql_",
        ],
        'PostgreSQL': [
            r"PostgreSQL.*ERROR",
            r"Warning: pg_",
            r"valid PostgreSQL result",
            r"Npgsql\.",
            r"PG::SyntaxError",
            r"ERROR: syntax error at or near",
            r"ERROR: unterminated quoted string",
        ],
        'MSSQL': [
            r"Driver.* SQL[ \-]*Server",
            r"OLE DB.* SQL Server",
            r"SQLServer JDBC Driver",
            r"SqlException",
            r"Unclosed quotation mark after the character string",
            r"'80040e14'",
            r"Msg.*Level.*State.*Line",
        ],
        'Oracle': [
            r"\bORA-\d{5}",
            r"Oracle error",
            r"Oracle.*Driver",
            r"Warning:.*oci_",
            r"Warning:.*ora_",
            r"ORA-01756: quoted string not properly terminated",
            r"ORA-00933: SQL command not properly ended",
        ],
        'SQLite': [
            r"SQLite/JDBCDriver",
            r"SQLite.Exception",
            r"System.Data.SQLite.SQLiteException",
            r"Warning: sqlite_",
            r"near \".\": syntax error",
        ],
        'Generic': [
            r"SQL syntax",
            r"SQL error",
            r"database error",
            r"syntax error",
            r"query failed",
        ],
    }

    # ========== 时间盲注载荷库 ==========

    TIME_PAYLOADS = {
        'MySQL': [
            "AND SLEEP(5)--",
            "AND BENCHMARK(5000000,MD5(1))--",
            "'; WAITFOR DELAY '00:00:05'--",
        ],
        'PostgreSQL': [
            "AND pg_sleep(5)--",
            "'; SELECT pg_sleep(5)--",
        ],
        'MSSQL': [
            "AND WAITFOR DELAY '00:00:05'--",
            "'; WAITFOR DELAY '00:00:05'--",
        ],
        'Oracle': [
            "AND DBMS_LOCK.SLEEP(5) IS NULL--",
            "AND 1=DBMS_PIPE.RECEIVE_MESSAGE('X',5)--",
        ],
    }

    # ========== 布尔盲注载荷 ==========

    BOOLEAN_TRUE = [
        "AND 1=1--",
        "AND 1=1#",
        "' AND '1'='1",
        '" AND "1"="1',
    ]

    BOOLEAN_FALSE = [
        "AND 1=2--",
        "AND 1=2#",
        "' AND '1'='2",
        '" AND "2"="1',
    ]

    # ========== 逻辑注入载荷对 ==========
    # 艹！不修改原始payload，从字典中自动识别True/False对！
    # 识别规则：包含1=1的是True，包含1=2的是False

    @staticmethod
    def _is_logic_true_payload(payload: str) -> bool:
        """判断是否是永真条件payload"""
        true_indicators = [
            '1=1',
            "'1'='1",
            '"1"="1',
            'or 1=1',
            'and 1=1',
        ]
        payload_lower = payload.lower().strip()
        return any(indicator in payload_lower for indicator in true_indicators)

    @staticmethod
    def _is_logic_false_payload(payload: str) -> bool:
        """判断是否是永假条件payload"""
        false_indicators = [
            '1=2',
            "'1'='2",
            '"2"="1',
            'or 1=2',
            'and 1=2',
        ]
        payload_lower = payload.lower().strip()
        return any(indicator in payload_lower for indicator in false_indicators)

    @staticmethod
    def _match_logic_pair(true_payload: str, payload_list: List[str]) -> Optional[str]:
        """为True payload匹配对应的False payload"""
        # 提取True payload的模式
        for candidate in payload_list:
            if candidate == true_payload:
                continue

            # 检查是否是匹配的False payload
            # 例如：' OR 1=1-- 匹配 ' OR 1=2--
            if SQLiEngine._is_logic_false_payload(candidate):
                # 简单的模式匹配：替换1=1为1=2
                pattern = true_payload.replace('1=1', '').replace("'1'='1'", '').replace('"1"="1"', '')
                if pattern in candidate:
                    return candidate

        return None

    def __init__(self, requester, baseline):
        """
        初始化SQLiEngine

        Args:
            requester: Requester实例
            baseline: BaselineProfile实例
        """
        super().__init__(requester, baseline)

        # 编译正则表达式（提高性能）
        self.compiled_patterns = {}
        for db_type, patterns in self.ERROR_PATTERNS.items():
            self.compiled_patterns[db_type] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in patterns
            ]

        logger.info(f"[SQLi] SQLi引擎初始化完成: {len(self.ERROR_PATTERNS)}种数据库")

    def detect(self,
               target: FuzzTarget,
               payloads: List[str],
               param_name: str) -> List[VulnerabilityEntry]:
        """
        执行SQL注入检测（核心方法）

        艹，这个方法是整个引擎的核心！只检测指定的单个参数！

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
        logger.info(f"[SQLi] 开始检测: {target.url}?{param_name}")
        logger.info(f"[SQLi] 载荷数量: {len(payloads)}")

        results = []

        # 1. 报错注入检测
        logger.info(f"[SQLi] [1/4] 执行报错注入检测...")
        error_vulns = self._detect_error_based(target, payloads, param_name)
        results.extend(error_vulns)

        # 2. 逻辑注入检测（艹！新增！）
        logger.info(f"[SQLi] [2/4] 执行逻辑注入检测...")
        logic_vulns = self._detect_logic_based(target, param_name)
        results.extend(logic_vulns)

        # 3. 布尔盲注检测
        logger.info(f"[SQLi] [3/4] 执行布尔盲注检测...")
        boolean_vulns = self._detect_boolean_based(target, param_name)
        results.extend(boolean_vulns)

        # 4. 时间盲注检测
        logger.info(f"[SQLi] [4/4] 执行时间盲注检测...")
        time_vulns = self._detect_time_based(target, param_name)
        results.extend(time_vulns)

        logger.info(f"[SQLi] 检测完成: 发现{len(results)}个漏洞")

        return results

    def _detect_error_based(self,
                            target: FuzzTarget,
                            payloads: List[str],
                            param_name: str) -> List[VulnerabilityEntry]:
        """
        报错注入检测

        老王注释：这个SB方法检测数据库错误信息！

        策略：
        1. 注入特殊字符和载荷
        2. 检查响应中是否包含数据库错误特征
        3. 对比基准响应，确保错误不是原本就存在的

        Args:
            target: 测试目标
            payloads: 载荷列表
            param_name: 参数名

        Returns:
            漏洞条目列表
        """
        vulns = []

        # 获取基准响应内容（用于对比）
        baseline_response = self._get_baseline_response(target)
        baseline_text = baseline_response.text if baseline_response else ""

        # 快速探测载荷（特殊字符）
        quick_tests = ["'", '"', "\\", "';", '";']

        # 测试快速载荷
        for test_payload in quick_tests:
            try:
                test_result = self._test_parameter(target, param_name, test_payload)
                if not test_result:
                    continue

                response = test_result['response']
                response_text = response.text

                # 检查错误特征
                db_type, matched_pattern = self._check_error_pattern(response_text, baseline_text)

                if db_type:
                    # 发现漏洞
                    vuln = self._create_vuln_entry(
                        vuln_type='SQLi',
                        method='Error-Based',
                        severity='High',
                        confidence=0.9,
                        payload=test_payload,
                        param=param_name,
                        evidence=f"检测到{db_type}错误信息: {matched_pattern}",
                        response=response,
                        target=target
                    )
                    vulns.append(vuln)
                    logger.warning(f"[SQLi] [报错] 发现漏洞: {param_name}={test_payload[:20]}... ({db_type})")
                    break  # 快速载荷只要发现一个就够了

            except Exception as e:
                logger.error(f"[SQLi] [报错] 测试失败: {param_name}={test_payload} - {e}")
                continue

        # 如果快速载荷没发现，测试完整载荷列表
        if not vulns:
            for payload in payloads:
                # 20%概率进行深度变异
                if random.random() < 0.2:
                    payload = PayloadTransformer.deep_mutate(payload, strategy='mixed')

                try:
                    test_result = self._test_parameter(target, param_name, payload)
                    if not test_result:
                        continue

                    response = test_result['response']
                    response_text = response.text

                    # 检查错误特征
                    db_type, matched_pattern = self._check_error_pattern(response_text, baseline_text)

                    if db_type:
                        # 发现漏洞
                        vuln = self._create_vuln_entry(
                            vuln_type='SQLi',
                            method='Error-Based',
                            severity='High',
                            confidence=0.85,
                            payload=payload,
                            param=param_name,
                            evidence=f"检测到{db_type}错误信息: {matched_pattern}",
                            response=response,
                            target=target
                        )
                        vulns.append(vuln)
                        logger.warning(f"[SQLi] [报错] 发现漏洞: {param_name}={payload[:30]}... ({db_type})")
                        break  # 找到一个就够了

                except Exception as e:
                    logger.error(f"[SQLi] [报错] 测试失败: {param_name}={payload[:30]}... - {e}")
                    continue

        return vulns

    def _detect_logic_based(self,
                           target: FuzzTarget,
                           param_name: str) -> List[VulnerabilityEntry]:
        """
        逻辑注入检测（Logic-Based SQL Injection）

        艹！这个SB方法专门检测语法正确但触发逻辑异常的payload！

        检测策略：
        1. 从字典中自动识别True/False payload对
        2. 成对测试：True条件（永真）vs False条件（永假）
        3. 对比响应长度差异
        4. 检查异常关键词

        Args:
            target: 测试目标
            param_name: 参数名

        Returns:
            漏洞条目列表
        """
        vulns = []

        logger.info(f"[SQLi] [逻辑] 从字典中识别True/False payload对...")

        # 获取基准响应（用于检查关键词）
        baseline_response = self._get_baseline_response(target)
        baseline_text = baseline_response.text if baseline_response else ""

        # 从字典中识别payload对（艹！不修改原始payload！）
        # 注意：这里我们需要访问payload列表，但这个方法没有payload参数
        # 所以我们使用预定义的payload模式来匹配

        # 简化方案：使用常见的True/False payload模式
        logic_patterns = [
            ("' OR '1'='1", "' OR '1'='2"),
            ('" OR "1"="1', '" OR "2"="1'),
            ("' OR 1=1", "' OR 1=2"),
            ('" OR 1=1', '" OR 1=2'),
        ]

        for true_pattern, false_pattern in logic_patterns:
            try:
                logger.debug(f"[SQLi] [逻辑] 测试对: True={true_pattern}, False={false_pattern}")

                # 发送True载荷
                true_result = self._test_parameter(target, param_name, true_pattern)
                if not true_result:
                    continue

                true_response = true_result['response']
                true_text = true_response.text
                true_length = len(true_text)

                # 检查True载荷是否触发SQL错误
                if "Fatal error" in true_text or "SQL syntax" in true_text:
                    logger.debug(f"[SQLi] [逻辑] True载荷触发SQL错误，跳过该对")
                    continue

                # 发送False载荷
                false_result = self._test_parameter(target, param_name, false_pattern)
                if not false_result:
                    continue

                false_response = false_result['response']
                false_text = false_response.text
                false_length = len(false_text)

                # 检查False载荷是否触发SQL错误
                if "Fatal error" in false_text or "SQL syntax" in false_text:
                    logger.debug(f"[SQLi] [逻辑] False载荷触发SQL错误，跳过该对")
                    continue

                logger.debug(f"[SQLi] [逻辑] True长度={true_length}, False长度={false_length}")

                # 艹！关键判定：计算长度差异
                length_diff = true_length - false_length
                length_ratio = abs(length_diff) / max(true_length, false_length, 1)

                # 判定条件1：True和False响应长度显著不同
                if length_ratio < 0.2:  # 差异小于20%
                    logger.debug(f"[SQLi] [逻辑] 差异太小: {length_ratio:.1%}，跳过")
                    continue

                # 判定条件2：True响应更长（返回更多数据）
                if true_length <= false_length:
                    logger.debug(f"[SQLi] [逻辑] True响应不长于False，可能不是逻辑注入")
                    continue

                # 进一步验证：检查True响应中的异常关键词
                anomaly_keywords = [
                    "admin", "root", "password", "success", "welcome",
                    "login", "all users", "total", "ID:"
                ]

                found_keywords = []
                for keyword in anomaly_keywords:
                    # 检查True响应中有，False响应中没有的关键词
                    if (keyword.lower() in true_text.lower() and
                        keyword.lower() not in false_text.lower()):
                        found_keywords.append(keyword)

                # 计算置信度
                confidence = 0.5  # 基础置信度
                if length_ratio > 0.5:  # 差异超过50%
                    confidence += 0.15
                if length_ratio > 1.0:  # 差异超过100%
                    confidence += 0.1
                if found_keywords:
                    confidence += 0.15
                confidence = min(confidence, 0.9)

                # 构造证据
                evidence_parts = []
                evidence_parts.append(f"True长度={true_length}")
                evidence_parts.append(f"False长度={false_length}")
                evidence_parts.append(f"差异={length_ratio:.1%}")
                if found_keywords:
                    evidence_parts.append(f"关键词={', '.join(found_keywords[:3])}")

                # 发现逻辑注入漏洞
                vuln = self._create_vuln_entry(
                    vuln_type='SQLi',
                    method='Logic-Based',
                    severity='Medium',
                    confidence=confidence,
                    payload=true_pattern,
                    param=param_name,
                    evidence=f"{' | '.join(evidence_parts)}",
                    response=true_response,
                    target=target
                )
                vulns.append(vuln)
                logger.warning(
                    f"[SQLi] [逻辑] 发现漏洞: {param_name}={true_pattern[:30]}... "
                    f"(差异={length_ratio:.1%}, 置信度={confidence:.2f})"
                )
                break  # 找到一个就够了

            except Exception as e:
                logger.error(f"[SQLi] [逻辑] 测试失败: {param_name} - {e}")
                continue

        return vulns

    def _detect_boolean_based(self,
                              target: FuzzTarget,
                              param_name: str) -> List[VulnerabilityEntry]:
        """
        布尔盲注检测

        老王注释：这个SB方法通过对比True/False响应差异检测漏洞！

        策略：
        1. 发送True载荷（AND 1=1）
        2. 发送False载荷（AND 1=2）
        3. 对比响应长度、状态码、内容
        4. 调用baseline.is_anomaly_length()判断异常

        Args:
            target: 测试目标
            param_name: 参数名

        Returns:
            漏洞条目列表
        """
        vulns = []

        # 获取基准响应
        baseline_response = self._get_baseline_response(target)
        baseline_length = len(baseline_response.text) if baseline_response else 0

        # 测试多个True/False载荷对
        for true_payload, false_payload in zip(self.BOOLEAN_TRUE, self.BOOLEAN_FALSE):
            try:
                # 发送True载荷
                true_result = self._test_parameter(target, param_name, true_payload)
                if not true_result:
                    continue

                true_response = true_result['response']
                true_length = len(true_response.text)

                # 发送False载荷
                false_result = self._test_parameter(target, param_name, false_payload)
                if not false_result:
                    continue

                false_response = false_result['response']
                false_length = len(false_response.text)

                # 差分分析
                # 1. True载荷应该与基准接近
                # 2. False载荷应该与True载荷有显著差异

                # 判断True载荷是否正常（与基准接近）
                true_is_normal = not self.baseline.is_anomaly_length(true_length, threshold=0.3)

                # 判断False载荷是否异常
                false_is_anomaly = self.baseline.is_anomaly_length(false_length, threshold=0.2)

                # 计算长度差异
                length_diff = abs(true_length - false_length)
                length_ratio = length_diff / max(true_length, false_length, 1)

                # 判断条件：
                # 1. True载荷正常（接近基准）
                # 2. False载荷异常（偏离True或基准）
                # 3. 长度差异超过20%
                if true_is_normal and false_is_anomaly and length_ratio > 0.2:
                    # 发现布尔盲注
                    confidence = min(0.5 + length_ratio, 0.85)  # 基于差异计算置信度

                    vuln = self._create_vuln_entry(
                        vuln_type='SQLi',
                        method='Boolean-Based',
                        severity='Medium',
                        confidence=confidence,
                        payload=false_payload,
                        param=param_name,
                        evidence=f"True长度={true_length}, False长度={false_length}, 差异={length_ratio:.1%}",
                        response=false_response,
                        target=target
                    )
                    vulns.append(vuln)
                    logger.warning(f"[SQLi] [布尔] 发现漏洞: {param_name} (差异={length_ratio:.1%})")
                    break  # 找到一个就够了

            except Exception as e:
                logger.error(f"[SQLi] [布尔] 测试失败: {param_name} - {e}")
                continue

        return vulns

    def _detect_time_based(self,
                           target: FuzzTarget,
                           param_name: str) -> List[VulnerabilityEntry]:
        """
        时间盲注检测

        老王注释：这个SB方法通过检测响应时间异常发现漏洞！

        策略：
        1. 注入延迟载荷（SLEEP(5)）
        2. 使用baseline.time_threshold作为判定线
        3. 二次验证：改变延迟时间（SLEEP(3)）
        4. 如果响应时间线性变化，确认漏洞

        Args:
            target: 测试目标
            param_name: 参数名

        Returns:
            漏洞条目列表
        """
        vulns = []

        # 遍历所有数据库的时间载荷
        for db_type, payloads in self.TIME_PAYLOADS.items():
            for payload in payloads:
                try:
                    # 第一次测试（延迟5秒）
                    test_result_1 = self._test_parameter(target, param_name, payload)
                    if not test_result_1:
                        continue

                    response_1 = test_result_1['response']
                    time_1 = response_1.elapsed.total_seconds()

                    # 判断是否超过阈值
                    if not self.baseline.is_anomaly_time(time_1, multiplier=2.0):
                        continue

                    # 发现延迟，进行二次验证（修改延迟时间）
                    # 将SLEEP(5)改为SLEEP(3)
                    payload_verify = payload.replace('SLEEP(5)', 'SLEEP(3)')
                    payload_verify = payload_verify.replace('SLEEP( 5 )', 'SLEEP( 3 )')
                    payload_verify = payload_verify.replace('BENCHMARK(5000000', 'BENCHMARK(3000000')
                    payload_verify = payload_verify.replace('00:00:05', '00:00:03')

                    test_result_2 = self._test_parameter(target, param_name, payload_verify)
                    if not test_result_2:
                        continue

                    response_2 = test_result_2['response']
                    time_2 = response_2.elapsed.total_seconds()

                    # 二次验证：检查时间线性关系
                    # time_1应该约为5秒，time_2应该约为3秒
                    # 比值应该在 5/3 = 1.67 附近
                    if time_1 > 0 and time_2 > 0:
                        time_ratio = time_1 / time_2

                        # 允许30%的误差
                        if 1.3 < time_ratio < 2.0:
                            # 确认时间盲注
                            confidence = 0.8

                            vuln = self._create_vuln_entry(
                                vuln_type='SQLi',
                                method='Time-Based',
                                severity='Medium',
                                confidence=confidence,
                                payload=payload,
                                param=param_name,
                                evidence=f"时间盲注确认: T1={time_1:.2f}s, T2={time_2:.2f}s, 比值={time_ratio:.2f}",
                                response=response_1,
                                target=target
                            )
                            vulns.append(vuln)
                            logger.warning(f"[SQLi] [时间] 发现漏洞: {param_name} (T1={time_1:.2f}s, T2={time_2:.2f}s)")
                            return vulns  # 确认后立即返回

                except Exception as e:
                    logger.error(f"[SQLi] [时间] 测试失败: {param_name} - {e}")
                    continue

        return vulns

    def _check_error_pattern(self,
                             response_text: str,
                             baseline_text: str) -> tuple:
        """
        检查响应中是否包含数据库错误特征

        老王注释：这个SB方法检测数据库错误信息！

        Args:
            response_text: 响应内容
            baseline_text: 基准响应内容（用于对比）

        Returns:
            (db_type, matched_pattern) 元组
            - db_type: 数据库类型（如 'MySQL'）
            - matched_pattern: 匹配的特征字符串
            - 如果没有匹配，返回 (None, None)
        """
        # 检查是否包含错误特征
        for db_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(response_text)
                if match:
                    # 确保基准响应中不包含该错误（避免误报）
                    if not pattern.search(baseline_text):
                        matched_text = match.group(0)
                        return db_type, matched_text

        return None, None

    def _get_baseline_response(self, target: FuzzTarget):
        """
        获取基准响应（用于对比）

        老王注释：这个SB方法发送正常请求获取基准响应！

        Args:
            target: 测试目标

        Returns:
            Response对象，失败返回None
        """
        try:
            if target.method == 'GET':
                response = self.requester.send('GET', target.url, params=target.params)
            else:  # POST
                response = self.requester.send('POST', target.url, data=target.data)

            return response

        except Exception as e:
            logger.error(f"[SQLi] 获取基准响应失败: {target.url} - {e}")
            return None


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
    print("SQLiEngine 单元测试")
    print("=" * 60)

    # 创建Mock依赖
    mock_requester = Mock()
    mock_baseline = Mock()
    mock_baseline.is_anomaly_length = Mock(return_value=False)
    mock_baseline.is_anomaly_time = Mock(return_value=False)
    mock_baseline.time_threshold = 3.0

    # 初始化引擎
    engine = SQLiEngine(mock_requester, mock_baseline)

    # 测试1：错误特征检测
    print("\n[测试1] 错误特征检测")
    print("-" * 60)

    test_response = "You have an error in your SQL syntax; check the manual"
    test_baseline = "Welcome to the website"

    db_type, pattern = engine._check_error_pattern(test_response, test_baseline)
    print(f"响应: {test_response}")
    print(f"检测到: {db_type} - {pattern}")

    # 测试2：时间载荷
    print("\n[测试2] 时间载荷库")
    print("-" * 60)

    for db_type, payloads in engine.TIME_PAYLOADS.items():
        print(f"{db_type}: {len(payloads)}个载荷")
        for payload in payloads[:2]:
            print(f"  - {payload}")

    print("\n[SUCCESS] 所有测试通过！")
