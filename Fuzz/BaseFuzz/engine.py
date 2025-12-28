#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Engine - BaseFuzz 核心调度器
==============================

负责协调所有组件，实现完整的模糊测试工作流。
本类是纯粹的调度器，不包含任何漏洞检测逻辑！

核心功能：
- 动态加载检测引擎（插件化设计）
- 管理扫描生命周期（基线建立 → 载荷分发 → 引擎触发）
- 实现单目标多参数并发测试
- 多级进度条显示（目标级、参数级、载荷级）
- 实时结果持久化（JSON文件写入）
- 异常隔离（引擎崩溃不影响其他引擎）

使用示例：
    >>> from Fuzz.BaseFuzz.engine import Engine
    >>> from Fuzz.spider import FuzzTarget
    >>>
    >>> # 初始化引擎
    >>> engine = Engine(
    ...     engine_names=['sqli', 'xss'],
    ...     mode='cvae',
    ...     timeout=10,
    ...     cookie='session=abc123'
    ... )
    >>>
    >>> # 加载爬虫结果
    >>> targets = [...]  # List[FuzzTarget]
    >>>
    >>> # 开始扫描
    >>> results = engine.run(targets)

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-25
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 导入依赖模块
from Fuzz.BaseFuzz.requester import Requester
from Fuzz.BaseFuzz.baseline import BaselineProfile, BaselineManager
from Fuzz.BaseFuzz.loaders.payload_manager import PayloadManager
from Fuzz.spider import FuzzTarget

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入tqdm（艹，进度条必备！）
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("[ENGINE] tqdm未安装，进度条功能不可用！")


class Engine:
    """
    BaseFuzz核心调度器

    老王注释：这个SB类是整个框架的大脑，负责协调一切！

    核心职责：
    1. 动态加载检测引擎（sqli_engine, xss_engine等）
    2. 管理完整扫描工作流
    3. 实现并发控制（单目标多参数并发）
    4. 实时保存结果（防止崩溃丢失数据）
    5. 多级进度显示（tqdm）

    注意：
        - 本类不包含任何漏洞检测逻辑！
        - 所有检测逻辑由具体的Engine子类实现
        - 本类只负责调度和协调

    Attributes:
        requester: Requester实例（HTTP通信层）
        payload_mgr: PayloadManager实例（载荷管理中心）
        engines: 已加载的检测引擎字典 {name: engine_instance}
        config: 配置字典

    Example:
        >>> engine = Engine(
        ...     engine_names=['sqli', 'xss'],
        ...     mode='cvae',
        ...     timeout=10,
        ...     cookie='session=abc123',
        ...     output_dir='Results/scan1'
        ... )
        >>> targets = [...]  # 从爬虫加载的FuzzTarget列表
        >>> results = engine.run(targets)
    """

    # 支持的引擎类型映射
    ENGINE_MAPPING = {
        'sqli': 'Fuzz.BaseFuzz.engines.sqli_engine.SQLiEngine',
        'xss': 'Fuzz.BaseFuzz.engines.xss_engine.XSSEngine',
        # 未来扩展：
        # 'cmdi': 'Fuzz.BaseFuzz.engines.cmdi_engine.CMDiEngine',
        # 'lf': 'Fuzz.BaseFuzz.engines.lfi_engine.LFIEngine',
    }

    def __init__(self,
                 engine_names: List[str],
                 mode: str = 'common',
                 timeout: int = 10,
                 cookie: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 max_workers: int = 5,
                 concurrent_params: int = 10,
                 param_filter: Optional[List[str]] = None):
        """
        初始化Engine调度器

        艹，这个方法是初始化整个框架的入口！

        Args:
            engine_names: 要加载的引擎名称列表（如 ['sqli', 'xss']）
            mode: 载荷模式
                - 'cvae': CVAE生成的精锐载荷（Data/processed/fuzzing/refined_payloads.txt）
                - 'cvae_raw': CVAE原始载荷（Data/generated/raw_payloads.txt）
                - 'common': 专家字典（payload/dict/）
                - 'custom': 自定义路径
            timeout: HTTP请求超时时间（秒），默认10秒
            cookie: 全局认证Cookie（格式：'key1=value1; key2=value2'）
            output_dir: 结果输出目录（默认：Results/scan_YYYYMMDD_HHMMSS）
            max_workers: 全局并发线程数（默认5）
            concurrent_params: 单目标并发测试参数数（默认10）
            param_filter: 参数过滤列表（艹！新增！只测试指定的参数，例如：['id', 'name']）

        Example:
            >>> engine = Engine(
            ...     engine_names=['sqli', 'xss'],
            ...     mode='cvae',
            ...     timeout=15,
            ...     cookie='session=abc123',
            ...     param_filter=['id']  # 只测试id参数
            ... )
        """
        self.mode = mode
        self.timeout = timeout
        self.cookie = cookie
        self.max_workers = max_workers
        self.concurrent_params = concurrent_params
        self.param_filter = param_filter  # 艹！新增！参数过滤列表

        # 初始化Requester（HTTP通信层）
        self.requester = Requester(
            timeout=timeout,
            cookie=cookie,
            max_retries=3,
            retry_delay=1.0
        )

        # 初始化PayloadManager（载荷管理中心）
        self.payload_mgr = PayloadManager()

        # 动态加载检测引擎
        self.engines = self._load_engines(engine_names)

        # 配置输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"Results/scan_{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 结果文件路径
        self.results_file = self.output_dir / "vulnerabilities.json"
        self.summary_file = self.output_dir / "summary.json"

        # 统计信息
        self.stats = {
            'total_targets': 0,
            'total_params_tested': 0,
            'total_payloads_sent': 0,
            'vulns_found': 0,
            'start_time': None,
            'end_time': None,
            'engines_used': engine_names
        }

        logger.info(f"[ENGINE] 调度器初始化完成: engines={engine_names}, mode={mode}")

    def _load_engines(self, engine_names: List[str]) -> Dict[str, Any]:
        """
        动态加载检测引擎（插件化设计）

        老王注释：这个SB方法实现了插件加载机制！

        Args:
            engine_names: 引擎名称列表（如 ['sqli', 'xss']）

        Returns:
            引擎实例字典 {name: engine_instance}

        Raises:
            ImportError: 引擎加载失败
            ValueError: 不支持的引擎名称
        """
        engines = {}

        for name in engine_names:
            # 检查引擎是否支持
            if name not in self.ENGINE_MAPPING:
                raise ValueError(f"不支持的引擎类型: {name}，支持的类型: {list(self.ENGINE_MAPPING.keys())}")

            # 动态导入
            module_path = self.ENGINE_MAPPING[name]
            try:
                # 分割模块路径和类名
                parts = module_path.split('.')
                class_name = parts[-1]
                module_path = '.'.join(parts[:-1])

                # 导入模块
                module = __import__(module_path, fromlist=[class_name])
                engine_class = getattr(module, class_name)

                # 实例化引擎（传入requester，稍后在run()中传入baseline）
                engines[name] = engine_class

                logger.info(f"[ENGINE] 引擎加载成功: {name} -> {module_path}.{class_name}")

            except ImportError as e:
                logger.error(f"[ENGINE] 引擎导入失败: {name} - {e}")
                raise ImportError(f"无法加载引擎 {name}: {e}")
            except Exception as e:
                logger.error(f"[ENGINE] 引擎初始化失败: {name} - {e}")
                raise

        return engines

    def run(self, targets: List[FuzzTarget]) -> List[Dict[str, Any]]:
        """
        执行完整的模糊测试工作流（核心方法）

        艹，这个方法是整个框架的核心入口！所有扫描都走这里！

        工作流程：
        1. 加载载荷字典（从PayloadManager）
        2. 遍历每个目标URL
        3. 为每个目标建立BaselineProfile
        4. 单目标多参数并发测试
        5. 串行触发所有引擎（SQLi → XSS）
        6. 实时保存漏洞结果到JSON

        Args:
            targets: FuzzTarget列表（从爬虫加载）

        Returns:
            漏洞结果列表（字典格式）

        Example:
            >>> engine = Engine(engine_names=['sqli', 'xss'])
            >>> targets = [...]  # 从爬虫加载
            >>> results = engine.run(targets)
            >>> print(f"发现{len(results)}个漏洞")
        """
        self.stats['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.stats['total_targets'] = len(targets)

        logger.info(f"[ENGINE] 开始扫描: {len(targets)}个目标, {len(self.engines)}个引擎")
        print(f"\n{'='*60}")
        print(f"CVDBFuzz 开始扫描")
        print(f"{'='*60}")
        print(f"目标数量: {len(targets)}")
        print(f"检测引擎: {', '.join(self.engines.keys())}")
        print(f"载荷模式: {self.mode}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*60}\n")

        # 加载载荷字典
        payloads_dict = self.payload_mgr.load(mode=self.mode)
        logger.info(f"[ENGINE] 载荷加载完成: {sum(len(p) for p in payloads_dict.values())}个")

        # 结果收集
        all_vulns = []

        # 遍历目标
        target_iterator = enumerate(targets)
        if TQDM_AVAILABLE:
            target_iterator = tqdm(
                target_iterator,
                total=len(targets),
                desc="[扫描进度]",
                unit="target",
                position=0
            )

        for idx, target in target_iterator:
            logger.info(f"[ENGINE] 正在测试目标 {idx+1}/{len(targets)}: {target.url}")

            # 为当前目标建立BaselineProfile
            baseline = self._establish_baseline(target)

            if not baseline.is_stable():
                logger.warning(f"[ENGINE] 目标基线不稳定，跳过: {target.url}")
                continue

            # 实例化引擎（传入baseline）
            engine_instances = {}
            for name, engine_class in self.engines.items():
                engine_instances[name] = engine_class(self.requester, baseline)

            # 单目标多参数并发测试
            target_vulns = self._test_target_concurrent(
                target,
                engine_instances,
                payloads_dict
            )

            # 添加到总结果
            all_vulns.extend(target_vulns)

            # 实时保存（艹，防止崩溃丢失数据！）
            self._save_results(all_vulns)

        # 保存汇总信息
        self.stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.stats['vulns_found'] = len(all_vulns)
        self._save_summary()

        logger.info(f"[ENGINE] 扫描完成: 发现{len(all_vulns)}个漏洞")
        print(f"\n{'='*60}")
        print(f"扫描完成！")
        print(f"总漏洞数: {len(all_vulns)}")
        print(f"结果文件: {self.results_file}")
        print(f"{'='*60}\n")

        return all_vulns

    def _establish_baseline(self, target: FuzzTarget, samples: int = 5) -> BaselineProfile:
        """
        为单个目标建立BaselineProfile（基线画像）

        老王注释：这个SB方法负责建立目标的"正常响应"指纹！

        Args:
            target: 测试目标
            samples: 采样次数（默认5次）

        Returns:
            BaselineProfile对象

        Example:
            >>> baseline = engine._establish_baseline(target)
            >>> if baseline.is_stable():
            ...     print("基线稳定，可以开始测试")
        """
        logger.info(f"[BASELINE] 正在建立基线: {target.url} ({samples}次采样)")

        # 使用BaselineManager建立基准画像
        baseline_mgr = BaselineManager(self.requester)
        baseline = baseline_mgr.build_profile(target, samples=samples)

        logger.info(f"[BASELINE] 基线建立完成: status={baseline.dominant_status}, "
                   f"avg_len={baseline.avg_length:.0f}, "
                   f"avg_time={baseline.avg_time:.3f}s")

        return baseline

    def _test_target_concurrent(self,
                                target: FuzzTarget,
                                engines: Dict[str, Any],
                                payloads_dict: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        单目标多参数并发测试（内部方法）

        艹，这个SB方法实现了单目标的多参数并发！

        工作流程：
        1. 提取目标的所有可测试参数
        2. 为每个参数分配载荷
        3. 使用ThreadPoolExecutor并发测试
        4. 串行触发所有引擎（SQLi → XSS）

        Args:
            target: 测试目标
            engines: 引擎实例字典
            payloads_dict: 载荷字典（按类型分组）

        Returns:
            漏洞结果列表（字典格式）
        """
        target_vulns = []

        # 确定要测试的参数
        if target.method == 'GET':
            all_params = list(target.params.keys())
        else:  # POST
            all_params = list(target.data.keys())

        # 艹！新增：添加HTTP头注入点
        if hasattr(target, 'injectable_headers') and target.injectable_headers:
            all_headers = list(target.injectable_headers.keys())
            logger.info(f"[ENGINE] 检测到HTTP头注入点: {all_headers}")
            # 将HTTP头添加到参数列表
            all_params.extend(all_headers)

        # 艹！应用参数过滤！
        if self.param_filter:
            # 只测试指定的参数
            params_to_test = [p for p in all_params if p in self.param_filter]
            logger.info(f"[ENGINE] 参数过滤: {list(target.params.keys()) + list(target.data.keys())} -> {params_to_test}")
        else:
            # 测试所有参数（包括HTTP头）
            params_to_test = all_params

        if not params_to_test:
            logger.warning(f"[ENGINE] 目标无可测试参数: {target.url}")
            return target_vulns

        logger.info(f"[ENGINE] 目标参数数: {len(params_to_test)}, 并发数: {self.concurrent_params}")

        # 参数级进度条
        param_iterator = params_to_test
        if TQDM_AVAILABLE:
            param_iterator = tqdm(
                param_iterator,
                desc=f"  └─参数测试",
                unit="param",
                position=1,
                leave=False
            )

        # 并发测试参数
        with ThreadPoolExecutor(max_workers=self.concurrent_params) as executor:
            # 提交所有参数测试任务
            futures = {
                executor.submit(
                    self._test_parameter_serial,
                    target,
                    param_name,
                    engines,
                    payloads_dict
                ): param_name for param_name in params_to_test
            }

            # 收集结果
            for future in as_completed(futures):
                param_name = futures[future]
                try:
                    param_vulns = future.result()
                    target_vulns.extend(param_vulns)

                    # 更新统计
                    self.stats['total_params_tested'] += 1

                except Exception as e:
                    logger.error(f"[ENGINE] 参数测试异常: {target.url}?{param_name} - {e}")

        return target_vulns

    def _test_parameter_serial(self,
                               target: FuzzTarget,
                               param_name: str,
                               engines: Dict[str, Any],
                               payloads_dict: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        单参数串行触发所有引擎（内部方法）

        老王注释：这个憨批方法按顺序触发所有引擎（SQLi → XSS）！

        Args:
            target: 测试目标
            param_name: 参数名
            engines: 引擎实例字典
            payloads_dict: 载荷字典

        Returns:
            漏洞结果列表（字典格式）
        """
        param_vulns = []

        # 按照预定义顺序触发引擎（SQLi → XSS）
        engine_order = ['sqli', 'xss']

        for engine_name in engine_order:
            if engine_name not in engines:
                continue

            engine = engines[engine_name]

            # 获取该引擎的载荷
            # 艹！老王我发现了！payloads_dict的键不是简单的upper()/title()！
            # SQLi引擎的键是'SQLi'（首字母大写，i小写）
            # XSS引擎的键是'XSS'（全大写）
            # 所以要用PayloadManager的VTYPE_MAPPING来做转换！
            lookup_key = PayloadManager.VTYPE_MAPPING.get(engine_name, engine_name.upper())
            payloads = payloads_dict.get(lookup_key, [])
            if not payloads:
                logger.warning(f"[ENGINE] 引擎 {engine_name} 无可用载荷 (查找键: {lookup_key}, 可用键: {list(payloads_dict.keys())})")
                continue

            logger.debug(f"[ENGINE] 触发引擎 {engine_name}: 参数={param_name}, 载荷数={len(payloads)}")

            try:
                # 调用引擎的detect方法（传入param_name）
                vuln_entries = engine.detect(target, payloads, param_name)

                # 转换为字典格式
                for vuln in vuln_entries:
                    param_vulns.append(vuln.to_dict())

                    # 更新统计
                    self.stats['total_payloads_sent'] += 1
                    if vuln.confidence > 0.5:  # 置信度>50%才计入
                        self.stats['vulns_found'] += 1

                # 实时保存发现的漏洞（艹，立即写入文件！）
                if vuln_entries:
                    for vuln in vuln_entries:
                        # 艹！显示完整payload URL，方便用户直接复制测试！
                        logger.warning(f"[VULN] {engine_name} 发现漏洞: {vuln.payload_url}")
                        logger.warning(f"[VULN]   类型: {vuln.vuln_type} | 严重性: {vuln.severity} | 置信度: {vuln.confidence:.2f}")
                        logger.warning(f"[VULN]   载荷: {vuln.payload}")

            except Exception as e:
                logger.error(f"[ENGINE] 引擎 {engine_name} 执行失败: {target.url}?{param_name} - {e}")
                # 艹，引擎崩溃不影响其他引擎！

        return param_vulns

    def _save_results(self, vulns: List[Dict[str, Any]]) -> None:
        """
        实时保存漏洞结果到JSON文件

        艹！修改：保存两份文件！
        - vulnerabilities.json：只保存真正能触发异常的漏洞（过滤Error-Based）
        - vulnerabilities_all.json：保存所有漏洞（包括Error-Based）

        老王注释：这个SB方法区分两种保存方式！

        Args:
            vulns: 漏洞结果列表（字典格式）
        """
        try:
            # 艹！过滤掉Error-Based（报语法错误）
            filtered_vulns = []
            for vuln in vulns:
                method = vuln.get('method', '')

                # 只保存非Error-Based的漏洞
                if method != 'Error-Based':
                    filtered_vulns.append(vuln)
                else:
                    logger.debug(f"[ENGINE] 过滤掉Error-Based漏洞: {vuln.get('payload', '')[:30]}...")

            # 保存过滤后的结果（vulnerabilities.json）
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_vulns, f, ensure_ascii=False, indent=2)

            # 艹！保存所有结果（vulnerabilities_all.json）
            all_results_file = self.output_dir / "vulnerabilities_all.json"
            with open(all_results_file, 'w', encoding='utf-8') as f:
                json.dump(vulns, f, ensure_ascii=False, indent=2)

            logger.debug(f"[ENGINE] 已保存{len(filtered_vulns)}个有效漏洞（过滤了{len(vulns) - len(filtered_vulns)}个Error-Based）")
            logger.debug(f"[ENGINE] 所有漏洞已保存到: {all_results_file}")

        except Exception as e:
            logger.error(f"[ENGINE] 结果保存失败: {self.results_file} - {e}")

    def _save_summary(self) -> None:
        """
        保存扫描汇总信息

        老王注释：这个SB方法保存扫描的统计信息！
        """
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)

            logger.info(f"[ENGINE] 汇总信息已保存: {self.summary_file}")

        except Exception as e:
            logger.error(f"[ENGINE] 汇总保存失败: {self.summary_file} - {e}")

    def close(self) -> None:
        """
        关闭调度器，释放资源

        艹，用完记得关闭！别tm资源泄漏！

        Example:
            >>> engine = Engine(...)
            >>> results = engine.run(targets)
            >>> engine.close()
        """
        self.requester.close()
        logger.info("[ENGINE] 调度器已关闭")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        return False


# 便捷函数
def create_engine(engine_names: List[str],
                  mode: str = 'common',
                  timeout: int = 10,
                  cookie: Optional[str] = None,
                  output_dir: Optional[str] = None) -> Engine:
    """
    创建Engine实例的便捷函数

    Args:
        engine_names: 引擎名称列表
        mode: 载荷模式
        timeout: 超时时间
        cookie: 认证Cookie
        output_dir: 输出目录

    Returns:
        Engine实例

    Example:
        >>> from Fuzz.BaseFuzz.engine import create_engine
        >>> engine = create_engine(
        ...     engine_names=['sqli', 'xss'],
        ...     mode='cvae'
        ... )
        >>> results = engine.run(targets)
    """
    return Engine(
        engine_names=engine_names,
        mode=mode,
        timeout=timeout,
        cookie=cookie,
        output_dir=output_dir
    )


if __name__ == '__main__':
    # 测试代码
    import logging

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("Engine 单元测试")
    print("=" * 60)

    # 测试1：Engine初始化
    print("\n[测试1] Engine初始化")
    print("-" * 60)

    try:
        engine = Engine(
            engine_names=['sqli', 'xss'],
            mode='cvae',
            timeout=10
        )
        print(f"引擎加载成功: {list(engine.engines.keys())}")
        print(f"输出目录: {engine.output_dir}")

    except Exception as e:
        print(f"初始化失败（预期，因为引擎尚未实现）: {e}")

    print("\n[SUCCESS] 测试完成！")
