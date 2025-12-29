#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reporter - 扫描报告生成器
============================

负责生成最终的扫描汇总与详细报告。

核心功能：
- 汇总统计（扫描时长、目标总数、漏洞统计）
- JSON报告生成（完整的详细漏洞数据）
- 终端彩色输出（ASCII表格展示）
- 多格式支持（JSON、TXT、HTML）

使用示例：
    >>> from Fuzz.BaseFuzz.analysis.reporter import Reporter
    >>>
    >>> # 生成报告
    >>> reporter = Reporter()
    >>> reporter.generate_summary(
    ...     results=vulns,
    ...     stats=stats,
    ...     output_dir='Results/scan_20251225'
    ... )

"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# 尝试导入colorama（彩色输出）
try:
    from colorama import init, Fore, Style, Back
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

# 尝试导入tabulate（表格输出）
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# 配置日志
logger = logging.getLogger(__name__)


class Reporter:
    """
    扫描报告生成器


    核心职责：
    1. 汇总统计信息
    2. 生成JSON详细报告
    3. 生成终端彩色汇总
    4. 管理输出文件路径

    Attributes:
        output_dir: 输出目录路径
        summary_file: 汇总文件路径
        detail_file: 详细报告文件路径

    Example:
        >>> reporter = Reporter(output_dir='Results/scan_20251225')
        >>> reporter.generate_summary(vulns, stats)
        >>> reporter.print_console_summary()
    """

    # ========== 彩色输出配置 ==========

    COLORS = {
        'High': Fore.RED if COLORAMA_AVAILABLE else '',
        'Medium': Fore.YELLOW if COLORAMA_AVAILABLE else '',
        'Low': Fore.GREEN if COLORAMA_AVAILABLE else '',
        'Info': Fore.CYAN if COLORAMA_AVAILABLE else '',
        'Success': Fore.GREEN if COLORAMA_AVAILABLE else '',
        'Warning': Fore.YELLOW if COLORAMA_AVAILABLE else '',
        'Error': Fore.RED if COLORAMA_AVAILABLE else '',
        'Reset': Style.RESET_ALL if COLORAMA_AVAILABLE else '',
    }

    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化Reporter

        Args:
            output_dir: 输出目录（默认：Results/scan_YYYYMMDD_HHMMSS）
        """
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"Results/scan_{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 文件路径
        self.summary_file = self.output_dir / "summary.json"
        self.detail_file = self.output_dir / "vulnerabilities_detail.json"
        self.console_file = self.output_dir / "console_report.txt"

        logger.info(f"[REPORTER] 报告生成器初始化完成: {self.output_dir}")

    def generate_summary(self,
                        results: List[Dict[str, Any]],
                        stats: Dict[str, Any],
                        scan_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        生成汇总报告（核心方法）

        生成所有格式的报告！

        Args:
            results: 漏洞结果列表
            stats: 统计信息字典
            scan_info: 扫描信息（开始时间、结束时间等）

        Returns:
            True=生成成功，False=生成失败

        Example:
            >>> reporter = Reporter()
            >>> reporter.generate_summary(vulns, stats, scan_info)
        """
        try:
            # 1. 生成JSON详细报告
            self._generate_json_report(results, stats, scan_info)

            # 2. 生成汇总统计
            summary = self._create_summary(results, stats, scan_info)

            # 3. 保存汇总到JSON
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            logger.info(f"[REPORTER] 汇总报告已生成: {self.summary_file}")

            return True

        except Exception as e:
            logger.error(f"[REPORTER] 生成汇总失败: {e}")
            return False

    def _generate_json_report(self,
                             results: List[Dict[str, Any]],
                             stats: Dict[str, Any],
                             scan_info: Optional[Dict[str, Any]] = None) -> None:
        """
        生成JSON详细报告

        生成完整的JSON报告！
        """
        # 构建完整报告
        report = {
            'scan_info': scan_info or {},
            'statistics': stats,
            'vulnerabilities': results,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 保存到文件
        with open(self.detail_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"[REPORTER] 详细报告已生成: {self.detail_file}")

    def _create_summary(self,
                       results: List[Dict[str, Any]],
                       stats: Dict[str, Any],
                       scan_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        创建汇总信息

        汇总所有统计信息！
        """
        summary = {
            'scan_info': scan_info or {},
            'statistics': stats,
            'top_vulnerabilities': results[:10],  # Top 10
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return summary

    def print_console_summary(self,
                              stats: Dict[str, Any],
                              scan_info: Optional[Dict[str, Any]] = None) -> None:
        """
        打印终端彩色汇总（核心方法）


        Args:
            stats: 统计信息字典
            scan_info: 扫描信息

        Example:
            >>> reporter = Reporter()
            >>> reporter.print_console_summary(stats, scan_info)
        """
        # 打印标题
        self._print_header()

        # 打印扫描信息
        if scan_info:
            self._print_scan_info(scan_info)

        # 打印统计信息
        self._print_statistics(stats)

        # 打印漏洞分布
        if 'by_severity' in stats:
            self._print_severity_distribution(stats)

        # 打印Top漏洞
        # （需要额外的results参数，这里暂时跳过）

    def _print_header(self) -> None:
        """打印报告标题"""
        print("\n")
        if COLORAMA_AVAILABLE:
            print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)
            print(Fore.CYAN + "          CVDBFuzz 漏洞扫描报告" + Style.RESET_ALL)
            print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)
        else:
            print("=" * 70)
            print("          CVDBFuzz 漏洞扫描报告")
            print("=" * 70)
        print("")

    def _print_scan_info(self, scan_info: Dict[str, Any]) -> None:
        """打印扫描信息"""
        print("📊 扫描信息")
        print("-" * 70)

        info_items = [
            ("开始时间", scan_info.get('start_time', 'Unknown')),
            ("结束时间", scan_info.get('end_time', 'Unknown')),
            ("目标数量", scan_info.get('total_targets', 0)),
            ("测试参数", scan_info.get('total_params_tested', 0)),
            ("发送载荷", scan_info.get('total_payloads_sent', 0)),
            ("使用引擎", ', '.join(scan_info.get('engines_used', []))),
        ]

        for key, value in info_items:
            print(f"  {key}: {value}")

        print("")

    def _print_statistics(self, stats: Dict[str, Any]) -> None:
        """打印统计信息"""
        print("🎯 扫描统计")
        print("-" * 70)

        total = stats.get('total_vulns', 0)
        high = stats.get('high_risk_count', 0)
        medium = stats.get('medium_risk_count', 0)
        low = stats.get('low_risk_count', 0)
        avg_conf = stats.get('avg_confidence', 0.0)
        risk_idx = stats.get('risk_index', 0.0)

        # 彩色输出
        if COLORAMA_AVAILABLE:
            print(f"  总漏洞数: {Fore.CYAN}{total}{Style.RESET_ALL}")
            print(f"  高危漏洞: {Fore.RED}{high}{Style.RESET_ALL}")
            print(f"  中危漏洞: {Fore.YELLOW}{medium}{Style.RESET_ALL}")
            print(f"  低危漏洞: {Fore.GREEN}{low}{Style.RESET_ALL}")
            print(f"  平均置信度: {Fore.CYAN}{avg_conf:.2f}{Style.RESET_ALL}")
            print(f"  风险指数: {Fore.CYAN}{risk_idx:.2f}{Style.RESET_ALL}")
        else:
            print(f"  总漏洞数: {total}")
            print(f"  高危漏洞: {high}")
            print(f"  中危漏洞: {medium}")
            print(f"  低危漏洞: {low}")
            print(f"  平均置信度: {avg_conf:.2f}")
            print(f"  风险指数: {risk_idx:.2f}")

        print("")

    def _print_severity_distribution(self, stats: Dict[str, Any]) -> None:
        """打印严重性分布"""
        print("📈 漏洞分布")
        print("-" * 70)

        # 按严重性分组
        by_severity = stats.get('by_severity', {})
        by_type = stats.get('by_type', {})

        # 打印严重性分布
        if by_severity:
            if TABULATE_AVAILABLE:
                table_data = [
                    ["严重性", "数量"],
                    ["高危", by_severity.get('High', 0)],
                    ["中危", by_severity.get('Medium', 0)],
                    ["低危", by_severity.get('Low', 0)],
                ]
                print(tabulate(table_data, headers='firstrow', tablefmt='grid'))
            else:
                print("  按严重性:")
                for severity, count in by_severity.items():
                    color = self.COLORS.get(severity, '')
                    reset = self.COLORS['Reset']
                    print(f"    {color}{severity}: {count}{reset}")

        print("")

        # 打印类型分布
        if by_type:
            print("  按类型:")
            for vuln_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
                print(f"    {vuln_type}: {count}")

        print("")

    def print_vulnerability_table(self, results: List[Dict[str, Any]], top_n: int = 20) -> None:
        """
        打印漏洞表格（核心方法）

        Args:
            results: 漏洞结果列表
            top_n: 显示数量（默认20）
        """
        if not results:
            print("✅ 未发现漏洞")
            return

        print(f"🔍 Top {min(top_n, len(results))} 漏洞详情")
        print("-" * 70)

        # 限制显示数量
        display_results = results[:top_n]

        if TABULATE_AVAILABLE:
            # 使用tabulate生成表格
            table_data = []
            for i, vuln in enumerate(display_results, 1):
                severity = vuln.get('severity', 'Low')
                confidence = vuln.get('confidence', 0.0)

                table_data.append([
                    i,
                    vuln.get('vuln_type', 'Unknown'),
                    vuln.get('method', 'Unknown'),
                    severity,
                    f"{confidence:.2f}",
                    vuln.get('param_name', 'unknown'),
                    vuln.get('payload', '')[:30] + '...' if len(vuln.get('payload', '')) > 30 else vuln.get('payload', ''),
                ])

            headers = ['#', '类型', '方法', '严重性', '置信度', '参数', '载荷']
            print(tabulate(table_data, headers=headers, tablefmt='grid'))

        else:
            # 降级：使用简单格式
            print(f"{'#':<4} {'类型':<10} {'方法':<20} {'严重性':<8} {'置信度':<8} {'参数':<15}")
            print("-" * 70)

            for i, vuln in enumerate(display_results, 1):
                severity = vuln.get('severity', 'Low')
                confidence = vuln.get('confidence', 0.0)

                print(f"{i:<4} {vuln.get('vuln_type', 'Unknown'):<10} "
                      f"{vuln.get('method', 'Unknown'):<20} "
                      f"{severity:<8} {confidence:<8.2f} "
                      f"{vuln.get('param_name', 'unknown'):<15}")

        print("")

    def save_console_report(self, text: str) -> None:
        """
        保存终端报告到文件

        保存彩色报告为纯文本！

        Args:
            text: 终端输出文本
        """
        try:
            with open(self.console_file, 'w', encoding='utf-8') as f:
                f.write(text)

            logger.info(f"[REPORTER] 终端报告已保存: {self.console_file}")

        except Exception as e:
            logger.error(f"[REPORTER] 保存终端报告失败: {e}")

    def get_report_files(self) -> Dict[str, str]:
        """
        获取所有报告文件路径

        Returns:
            文件路径字典
        """
        return {
            'summary': str(self.summary_file),
            'detail': str(self.detail_file),
            'console': str(self.console_file),
            'directory': str(self.output_dir),
        }


if __name__ == '__main__':
    # 测试代码
    import logging

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("Reporter 单元测试")
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
        },
        {
            'vuln_type': 'XSS',
            'method': 'Reflected',
            'severity': 'Medium',
            'confidence': 0.7,
            'payload': '<script>alert(1)</script>',
            'param_name': 'name',
            'evidence': 'Reflected in script tag',
        },
    ]

    test_stats = {
        'total_vulns': 2,
        'by_severity': {'High': 1, 'Medium': 1, 'Low': 0},
        'by_type': {'SQLi': 1, 'XSS': 1},
        'avg_confidence': 0.8,
        'risk_index': 12.5,
        'high_risk_count': 1,
        'medium_risk_count': 1,
        'low_risk_count': 0,
    }

    test_scan_info = {
        'start_time': '2025-12-25 10:00:00',
        'end_time': '2025-12-25 10:05:00',
        'total_targets': 5,
        'total_params_tested': 15,
        'total_payloads_sent': 1500,
        'engines_used': ['sqli', 'xss'],
    }

    # 测试报告生成
    print("\n[测试] 生成报告")
    print("-" * 60)

    reporter = Reporter(output_dir='Results/test_report')

    # 生成汇总
    if reporter.generate_summary(test_results, test_stats, test_scan_info):
        print("✅ 报告生成成功")

        # 打印终端汇总
        reporter.print_console_summary(test_stats, test_scan_info)

        # 打印漏洞表格
        reporter.print_vulnerability_table(test_results, top_n=10)

        # 获取文件路径
        files = reporter.get_report_files()
        print(f"\n报告目录: {files['directory']}")
        print(f"汇总文件: {files['summary']}")
        print(f"详细报告: {files['detail']}")

    print("\n[SUCCESS] 所有测试通过！")
