#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyzer - 扫描结果分析器
============================

负责对扫描结果进行后处理、去重、重评级和统计分析。

核心功能：
- 结果去重（基于唯一标识）
- 严重性重评级（统一评级标准）
- 置信度汇总（计算平均置信度）
- 风险指数计算（综合评分）
- 统计分析（按类型、严重性分组）

使用示例：
    >>> from Fuzz.BaseFuzz.analysis.analyzer import Analyzer
    >>>
    >>> # 加载扫描结果
    >>> analyzer = Analyzer()
    >>> analyzer.load_results('Results/scan_20251225/vulnerabilities.json')
    >>>
    >>> # 执行分析
    >>> deduplicated = analyzer.deduplicate()
    >>> ranked = analyzer.rank(deduplicated)
    >>>
    >>> # 获取统计信息
    >>> stats = analyzer.get_statistics()
    >>> print(f"发现{stats['total_vulns']}个漏洞")

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-25
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import logging

# 配置日志
logger = logging.getLogger(__name__)


class Analyzer:
    """
    扫描结果分析器

    老王注释：这个SB类负责分析扫描结果，去重和评级！

    核心职责：
    1. 加载扫描结果JSON文件
    2. 去除重复漏洞
    3. 重新评级严重性
    4. 计算统计信息
    5. 生成分析报告

    Attributes:
        results: 原始扫描结果列表
        analyzed_results: 分析后的结果列表
        statistics: 统计信息字典

    Example:
        >>> analyzer = Analyzer()
        >>> analyzer.load_results('Results/scan/vulnerabilities.json')
        >>> deduplicated = analyzer.deduplicate()
        >>> ranked = analyzer.rank(deduplicated)
        >>> stats = analyzer.get_statistics()
    """

    # ========== 严重性重评级规则 ==========

    SEVERITY_MAPPING = {
        # SQL注入
        'SQLi_Error-Based': 'High',
        'SQLi_Boolean-Based': 'Medium',
        'SQLi_Time-Based': 'Medium',
        'SQLi_Union-Based': 'High',

        # XSS
        'XSS_Reflected_script_tag': 'High',
        'XSS_Reflected_event_handler': 'Medium',
        'XSS_Reflected_text_content': 'Low',
        'XSS_DOM-Based': 'Low',

        # 其他类型（未来扩展）
        'CMDi_Error-Based': 'High',
        'CMDi_Time-Based': 'High',
        'LFI_Direct': 'High',
        'RFI_Direct': 'High',
        'Info_Leak': 'Low',
    }

    # ========== 风险指数权重 ==========

    RISK_WEIGHTS = {
        'High': 10.0,
        'Medium': 5.0,
        'Low': 1.0,
    }

    def __init__(self):
        """初始化Analyzer"""
        self.results = []
        self.analyzed_results = []
        self.statistics = {}

        logger.info("[ANALYZER] 分析器初始化完成")

    def load_results(self, results_file: str) -> bool:
        """
        加载扫描结果JSON文件

        艹，这个SB方法加载扫描结果！健壮性第一！

        Args:
            results_file: JSON文件路径

        Returns:
            True=加载成功，False=加载失败

        Example:
            >>> analyzer = Analyzer()
            >>> if analyzer.load_results('Results/scan/vulnerabilities.json'):
            ...     print("加载成功")
        """
        try:
            results_path = Path(results_file)

            # 检查文件是否存在
            if not results_path.exists():
                logger.error(f"[ANALYZER] 文件不存在: {results_file}")
                return False

            # 检查文件大小
            if results_path.stat().st_size == 0:
                logger.warning(f"[ANALYZER] 文件为空: {results_file}")
                self.results = []
                return True

            # 读取JSON文件
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 验证数据格式
            if isinstance(data, list):
                self.results = data
            elif isinstance(data, dict) and 'vulnerabilities' in data:
                self.results = data['vulnerabilities']
            else:
                logger.error(f"[ANALYZER] 无效的JSON格式: {results_file}")
                return False

            logger.info(f"[ANALYZER] 加载成功: {len(self.results)}条结果")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"[ANALYZER] JSON解析失败: {results_file} - {e}")
            # 降级处理：返回空列表
            self.results = []
            return False

        except Exception as e:
            logger.error(f"[ANALYZER] 加载失败: {results_file} - {e}")
            self.results = []
            return False

    def deduplicate(self) -> List[Dict[str, Any]]:
        """
        去重逻辑（核心方法）

        老王注释：这个SB方法去除重复的漏洞结果！

        策略：
        1. 基于唯一标识去重：vuln_type + param + payload[:50]
        2. 保留置信度最高的版本
        3. 保留最早发现的版本

        Returns:
            去重后的结果列表

        Example:
            >>> deduplicated = analyzer.deduplicate()
            >>> print(f"去重前: {len(analyzer.results)}, 去重后: {len(deduplicated)}")
        """
        if not self.results:
            logger.warning("[ANALYZER] 没有结果需要去重")
            return []

        # 用于存储唯一标识到结果的映射
        unique_map = {}

        for vuln in self.results:
            try:
                # 生成唯一标识
                vuln_type = vuln.get('vuln_type', 'Unknown')
                param_name = vuln.get('param_name', 'unknown')
                payload = vuln.get('payload', '')
                target_url = vuln.get('target_url', '')

                # 截取payload前50字符
                payload_key = payload[:50] if payload else ''

                # 生成唯一键
                unique_key = f"{vuln_type}_{param_name}_{payload_key}"

                # 如果该键不存在，直接添加
                if unique_key not in unique_map:
                    unique_map[unique_key] = vuln
                else:
                    # 键已存在，保留置信度更高的版本
                    existing = unique_map[unique_key]
                    existing_conf = existing.get('confidence', 0.0)
                    current_conf = vuln.get('confidence', 0.0)

                    if current_conf > existing_conf:
                        unique_map[unique_key] = vuln

            except Exception as e:
                logger.error(f"[ANALYZER] 去重处理失败: {vuln} - {e}")
                continue

        # 转换为列表
        deduplicated = list(unique_map.values())

        logger.info(f"[ANALYZER] 去重完成: {len(self.results)} → {len(deduplicated)}")

        return deduplicated

    def rank(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        严重性重评级（核心方法）

        老王注释：这个SB方法统一评级标准！

        策略：
        1. 根据vuln_type和method映射严重性
        2. 调整置信度（基于评级规则）
        3. 按严重性和置信度排序

        Args:
            results: 去重后的结果列表

        Returns:
            重评级后的结果列表

        Example:
            >>> ranked = analyzer.rank(deduplicated)
            >>> for vuln in ranked:
            ...     print(f"{vuln['vuln_type']}: {vuln['severity']}")
        """
        if not results:
            logger.warning("[ANALYZER] 没有结果需要评级")
            return []

        ranked_results = []

        for vuln in results:
            try:
                # 复制原始数据（避免修改原数据）
                ranked_vuln = vuln.copy()

                # 获取漏洞类型和方法
                vuln_type = vuln.get('vuln_type', 'Unknown')
                method = vuln.get('method', 'Unknown')

                # 构造映射键
                map_key = f"{vuln_type}_{method}"

                # 根据映射规则重新评级
                new_severity = self.SEVERITY_MAPPING.get(
                    map_key,
                    vuln.get('severity', 'Low')  # 默认Low
                )

                # 更新严重性
                ranked_vuln['severity'] = new_severity
                ranked_vuln['original_severity'] = vuln.get('severity', '')

                # 根据严重性调整置信度
                original_conf = vuln.get('confidence', 0.5)
                severity_bonus = {
                    'High': 0.1,
                    'Medium': 0.0,
                    'Low': -0.1,
                }
                adjusted_conf = min(max(original_conf + severity_bonus.get(new_severity, 0.0), 0.0), 1.0)

                ranked_vuln['confidence'] = round(adjusted_conf, 2)

                ranked_results.append(ranked_vuln)

            except Exception as e:
                logger.error(f"[ANALYZER] 评级处理失败: {vuln} - {e}")
                # 保留原始数据
                ranked_results.append(vuln)
                continue

        # 按严重性和置信度排序
        severity_order = {'High': 0, 'Medium': 1, 'Low': 2}

        ranked_results.sort(
            key=lambda x: (
                severity_order.get(x.get('severity', 'Low'), 2),
                -x.get('confidence', 0.0)
            )
        )

        logger.info(f"[ANALYZER] 评级完成: {len(ranked_results)}个结果")

        return ranked_results

    def get_statistics(self, results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        生成统计信息（核心方法）

        老王注释：这个SB方法计算各种统计数据！

        统计维度：
        1. 总漏洞数
        2. 按严重性分组
        3. 按类型分组
        4. 按参数分组
        5. 平均置信度
        6. 风险指数

        Args:
            results: 结果列表（默认使用analyzed_results）

        Returns:
            统计信息字典

        Example:
            >>> stats = analyzer.get_statistics()
            >>> print(f"总漏洞数: {stats['total_vulns']}")
            >>> print(f"高危漏洞: {stats['by_severity']['High']}")
        """
        if results is None:
            results = self.analyzed_results

        if not results:
            return {
                'total_vulns': 0,
                'by_severity': {},
                'by_type': {},
                'by_param': {},
                'avg_confidence': 0.0,
                'risk_index': 0.0,
            }

        # 按严重性分组
        by_severity = Counter(vuln.get('severity', 'Low') for vuln in results)

        # 按类型分组
        by_type = Counter(vuln.get('vuln_type', 'Unknown') for vuln in results)

        # 按参数分组
        by_param = Counter(vuln.get('param_name', 'unknown') for vuln in results)

        # 计算平均置信度
        total_confidence = sum(vuln.get('confidence', 0.0) for vuln in results)
        avg_confidence = total_confidence / len(results) if results else 0.0

        # 计算风险指数
        risk_index = sum(
            self.RISK_WEIGHTS.get(vuln.get('severity', 'Low'), 1.0) * vuln.get('confidence', 0.5)
            for vuln in results
        )

        # 准备统计信息
        stats = {
            'total_vulns': len(results),
            'by_severity': dict(by_severity),
            'by_type': dict(by_type),
            'by_param': dict(by_param.most_common(10)),  # 前10个参数
            'avg_confidence': round(avg_confidence, 2),
            'risk_index': round(risk_index, 2),
            'high_risk_count': by_severity.get('High', 0),
            'medium_risk_count': by_severity.get('Medium', 0),
            'low_risk_count': by_severity.get('Low', 0),
        }

        logger.info(f"[ANALYZER] 统计完成: {stats['total_vulns']}个漏洞, 风险指数={stats['risk_index']}")

        return stats

    def analyze(self, results_file: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行完整的分析流程（便捷方法）

        艹，这个SB方法一步完成所有分析！

        流程：
        1. 加载结果
        2. 去重
        3. 重评级
        4. 生成统计信息

        Args:
            results_file: 结果文件路径

        Returns:
            (分析后的结果列表, 统计信息) 元组

        Example:
            >>> analyzer = Analyzer()
            >>> results, stats = analyzer.analyze('Results/scan/vulnerabilities.json')
            >>> print(f"发现{stats['total_vulns']}个漏洞")
        """
        # 1. 加载结果
        if not self.load_results(results_file):
            logger.error("[ANALYZER] 分析失败: 无法加载结果")
            return [], {}

        # 2. 去重
        deduplicated = self.deduplicate()

        # 3. 重评级
        ranked = self.rank(deduplicated)

        # 4. 保存分析结果
        self.analyzed_results = ranked

        # 5. 生成统计信息
        stats = self.get_statistics(ranked)

        logger.info(f"[ANALYZER] 分析完成: {len(ranked)}个漏洞")

        return ranked, stats

    def get_top_vulnerabilities(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        获取Top N漏洞（按风险指数排序）

        老王注释：这个SB方法找出最危险的漏洞！

        Args:
            n: 返回数量（默认10）

        Returns:
            Top N漏洞列表

        Example:
            >>> top10 = analyzer.get_top_vulnerabilities(10)
            >>> for vuln in top10:
            ...     print(f"{vuln['severity']}: {vuln['evidence']}")
        """
        if not self.analyzed_results:
            return []

        # 按严重性和置信度排序
        severity_order = {'High': 0, 'Medium': 1, 'Low': 2}

        sorted_results = sorted(
            self.analyzed_results,
            key=lambda x: (
                severity_order.get(x.get('severity', 'Low'), 2),
                -x.get('confidence', 0.0)
            )
        )

        return sorted_results[:n]


if __name__ == '__main__':
    # 测试代码
    import logging

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("Analyzer 单元测试")
    print("=" * 60)

    # 创建测试数据
    test_results = [
        {
            'vuln_type': 'SQLi',
            'method': 'Error-Based',
            'severity': 'High',
            'confidence': 0.9,
            'payload': "' OR 1=1--",
            'param_name': 'id',
            'evidence': 'MySQL error',
            'target_url': 'http://test.com/?id=1',
        },
        {
            'vuln_type': 'SQLi',
            'method': 'Error-Based',
            'severity': 'High',
            'confidence': 0.85,
            'payload': "' OR 1=1--",
            'param_name': 'id',
            'evidence': 'MySQL error',
            'target_url': 'http://test.com/?id=1',
        },
        {
            'vuln_type': 'XSS',
            'method': 'Reflected',
            'severity': 'Medium',
            'confidence': 0.7,
            'payload': '<script>alert(1)</script>',
            'param_name': 'name',
            'evidence': 'Reflected in script tag',
            'target_url': 'http://test.com/?name=test',
        },
    ]

    # 测试1：去重
    print("\n[测试1] 去重")
    print("-" * 60)

    analyzer = Analyzer()
    analyzer.results = test_results

    deduplicated = analyzer.deduplicate()
    print(f"原始结果: {len(test_results)}")
    print(f"去重后: {len(deduplicated)}")

    # 测试2：评级
    print("\n[测试2] 评级")
    print("-" * 60)

    ranked = analyzer.rank(deduplicated)
    for vuln in ranked:
        print(f"{vuln['vuln_type']}-{vuln['method']}: {vuln['severity']} (置信度={vuln['confidence']})")

    # 测试3：统计
    print("\n[测试3] 统计")
    print("-" * 60)

    analyzer.analyzed_results = ranked
    stats = analyzer.get_statistics()
    print(f"总漏洞数: {stats['total_vulns']}")
    print(f"按严重性: {stats['by_severity']}")
    print(f"平均置信度: {stats['avg_confidence']}")
    print(f"风险指数: {stats['risk_index']}")

    print("\n[SUCCESS] 所有测试通过！")
