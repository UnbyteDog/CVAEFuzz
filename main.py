#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVDBFuzz 主程序入口
=================

支持命令行调用的统一入口，整合数据预处理、模型训练、生成和聚类功能
遵循 Doc/prompt指导.md 中定义的调用结构

使用示例：
    python main.py --preprocess                    # 数据预处理
    python main.py --train                        # 模型训练
    python main.py --generate --cluster            # 生成并聚类
    python main.py --preprocess --data-dir ./data # 自定义参数

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-18
"""

import sys
import os
import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # 添加Data_processing目录到Python路径
    sys.path.insert(0, os.path.join(project_root, "Data_processing"))
    from preprocessor import main as preprocess_main
except ImportError as e:
    print(f"[ERROR] 导入预处理模块失败：{e}")
    print("请确保 Data_processing/preprocessor.py 文件存在")
    sys.exit(1)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CVDBFuzz - 基于CVAE生成与DBSCAN优化的智能Web模糊测试框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
工作流程示例：
    1. 数据预处理：  python main.py --preprocess
    2. 模型训练：    python main.py --train
    3. 生成与聚类：  python main.py --generate --cluster
    4. 智能探测：    python main.py --fuzz --url "http://target.com/index.php?id=FUZZ"
    5. 快速扫描：    python main.py --fuzz --url "http://target.com/index.php?id=FUZZ" --quick-scan
    6. 深度变异：    python main.py --fuzz --url "http://target.com/index.php?id=FUZZ" --radamsa
    7. 登录态探测：  python main.py --fuzz --url "http://target.com/user.php?id=FUZZ" --cookie "session=abc123; token=def456"
    8. 效果分析：    python main.py --analyze
    9. 全站扫描：    python main.py --crawler --url "http://target.com/"
    10. 登录态扫描：  python main.py --crawler --url "http://target.com/" --cookie "session=abc123; token=def456"
    11. 完整流程：   python main.py --preprocess --train --generate --cluster --crawler --analyze

参数说明：
    --preprocess:   执行数据预处理模块
    --train:        执行CVAE模型训练
    --generate:     执行载荷生成
    --cluster:      执行DBSCAN聚类优化
    --fuzz:         执行智能漏洞探测
    --analyze:      执行AI载荷效果评估分析
    --crawler:      启动CVDB-Spider全站自动化扫描
        """
    )

    # 主要操作参数
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='执行数据预处理（字符级分词、序列标准化、词表构建）'
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='训练CVAE模型'
    )

    parser.add_argument(
        '--generate',
        action='store_true',
        help='使用训练好的CVAE生成载荷'
    )

    parser.add_argument(
        '--cluster',
        action='store_true',
        help='使用DBSCAN对生成的载荷进行聚类优化'
    )

    parser.add_argument(
        '--fuzz',
        action='store_true',
        help='使用精锐载荷进行智能漏洞探测'
    )

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='执行AI载荷效果评估分析'
    )

    parser.add_argument(
        '--crawler',
        action='store_true',
        help='启动CVDB-Spider全站自动化扫描'
    )

    # 通用参数
    parser.add_argument(
        '--data-dir',
        type=str,
        default='Data/payload/train',
        help='训练数据目录路径 (默认: Data/payload/train)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='Data/processed',
        help='输出目录路径 (默认: Data/processed)'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=150,
        help='最大序列长度 (默认: 150)'
    )

    parser.add_argument(
        '--vocab-size',
        type=int,
        default=256,
        help='词表大小 (默认: 256)'
    )

    # CVAE训练参数
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='训练轮数 (默认: 50)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批大小 (默认: 32)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='学习率 (默认: 1e-3)'
    )

    parser.add_argument(
        '--embed-dim',
        type=int,
        default=128,
        help='词嵌入维度 (默认: 128)'
    )

    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='GRU隐藏层维度 (默认: 256)'
    )

    parser.add_argument(
        '--latent-dim',
        type=int,
        default=32,
        help='隐空间维度 (默认: 32)'
    )

    # 🔥 KL退火策略参数 - 老王专门修复参数传递链！
    parser.add_argument(
        '--kl-cycles',
        type=int,
        default=1,
        help='KL退火周期数 (默认: 1，单一稳定增长)'
    )

    parser.add_argument(
        '--beta-max',
        type=float,
        default=0.25,
        help='KL退火最大Beta值 (默认: 0.25，强约束力)'
    )

    parser.add_argument(
        '--delay-epochs',
        type=int,
        default=20,
        help='KL退火延迟epoch数 (默认: 20，先学好重构)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.8,
        help='生成温度参数 (默认: 1.8，增加随机性)'
    )

    # 载荷生成参数
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5000,
        help='每种攻击类型生成样本数量 (默认: 5000)'
    )

    parser.add_argument(
        '--attack-type',
        type=str,
        default='ALL',
        help='攻击载荷类型 (默认: ALL，支持: SQLi, XSS, CMDi, Overflow, XXE, SSI, ALL 或逗号分隔的组合)'
    )

    parser.add_argument(
        '--generation-batch-size',
        type=int,
        default=500,
        help='生成批处理大小 (默认: 500)'
    )

    # 聚类参数
    parser.add_argument(
        '--eps',
        type=float,
        help='DBSCAN的eps参数（自动寻找如果未指定）'
    )

    parser.add_argument(
        '--min-samples',
        type=int,
        help='DBSCAN的min_samples参数'
    )

    parser.add_argument(
        '--samples-per-cluster',
        type=int,
        default=5,
        help='每个簇保留的样本数 (默认: 5)'
    )

    parser.add_argument(
        '--reduction-method',
        type=str,
        default='tsne',
        choices=['tsne', 'pca'],
        help='降维方法 (默认: tsne)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='生成聚类可视化图像'
    )

    parser.add_argument(
        '--keep-noise',
        action='store_true',
        help='保留所有噪声点'
    )

    # 其他参数
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出信息'
    )

    # 检查数据目录是否存在
    if args.preprocess:
        data_path = Path(args.data_dir)
        if not data_path.exists():
            errors.append(f"数据目录不存在：{data_path}")
        elif not data_path.is_dir():
            errors.append(f"数据路径不是目录：{data_path}")

    # 检查操作组合的有效性
    operations = [args.preprocess, args.train, args.generate, args.cluster, args.fuzz, args.analyze, args.crawler]
    active_operations = sum(operations)

    if active_operations == 0:
        errors.append("必须指定至少一个操作：--preprocess, --train, --generate, --cluster, --fuzz, --analyze, --crawler")

    # 如果指定了cluster但没有generate，给出警告
    if args.cluster and not args.generate:
        print("[WARNING] --cluster 通常需要与 --generate 一起使用")

    # 如果指定了fuzz但没有url，报错
    if args.fuzz and not args.url:
        errors.append("--fuzz 参数必须配合 --url 参数使用")

    # 如果指定了fuzz，检查URL是否包含FUZZ标记
    if args.fuzz and args.url and 'FUZZ' not in args.url:
        errors.append("--url 参数必须包含 'FUZZ' 标记作为载荷注入点，例如: http://target.com/index.php?id=FUZZ")

    # 如果指定了crawler但没有url，报错
    if args.crawler and not args.url:
        errors.append("--crawler 参数必须配合 --url 参数使用")

    # crawler的URL不需要FUZZ标记，如果检测到FUZZ但没有crawler，则提示使用crawler
    if args.url and 'FUZZ' not in args.url and not args.crawler and not args.fuzz:
        print("[INFO] 检测到URL中不包含FUZZ标记，但未启用--crawler模式")
        print("[INFO] 如果您希望进行全站自动化扫描，请使用: python main.py --crawler --url \"您的URL\"")
        print("[INFO] 如果您希望进行单一URL扫描，请在URL中添加FUZZ标记，例如: --url \"http://target.com/index.php?id=FUZZ\"")

    return errors


def print_banner():
    """打印程序启动横幅"""
    banner = """
================================================================
                    CVDBFuzz v1.0
              基于CVAE生成与DBSCAN优化的智能Web模糊测试框架

  核心功能：
  - CVAE深度生成模型 - 学习攻击载荷语法结构
  - DBSCAN聚类优化 - 去除冗余，保留高价值种子
  - Wfuzz深度变异 - 编码混淆，绕过WAF防护
  - CVDB-Spider全站扫描 - 智能爬虫 + 参数提取 + 自动化探测

  新增特性：
  - 智能递归爬虫 (域名锁定 + 深度控制)
  - 参数自动提取 (GET参数 + POST表单)
  - 全站自动化漏洞扫描 (一键扫描整站)

  老王出品，必属精品！
================================================================
    """
    print(banner)


def execute_preprocess(args):
    """执行数据预处理"""
    print("\n" + "=" * 60)
    print("启动数据预处理模块")
    print("=" * 60)

    # 构建预处理模块的参数
    sys.argv = [
        'main.py',  # 脚本名称
        '--preprocess',
        '--data-dir', args.data_dir,
        '--output-dir', args.output_dir,
        '--max-length', str(args.max_length),
        '--vocab-size', str(args.vocab_size)
    ]

    try:
        # 调用预处理模块的主函数
        preprocess_main()
        print("\n[SUCCESS] 数据预处理模块执行完成")
        return True

    except Exception as e:
        print(f"\n[ERROR] 数据预处理失败：{e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def execute_train(args):
    """执行CVAE模型训练"""
    print("\n" + "=" * 60)
    print("启动CVAE模型训练模块")
    print("=" * 60)

    try:
        import subprocess

        # 构建CVAE训练命令
        cvae_script = os.path.join(project_root, "CVAE", "main.py")
        cvae_args = [
            sys.executable,  # Python解释器路径
            cvae_script,
            '--train',
            '--data-path', os.path.join(project_root, 'Data', 'processed', 'processed_data.pt'),
            '--vocab-path', os.path.join(project_root, 'Data', 'processed', 'vocab.json'),
            '--output-dir', os.path.join(project_root, 'CVAE/checkpoints'),
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--learning-rate', str(args.learning_rate),
            '--embed-dim', str(args.embed_dim),
            '--hidden-dim', str(args.hidden_dim),
            '--latent-dim', str(args.latent_dim),
            # 🔥 老王修复：传递KL退火参数！
            '--kl-cycles', str(args.kl_cycles),
            '--beta-max', str(args.beta_max),
            '--delay-epochs', str(args.delay_epochs),
            '--temperature', str(args.temperature)
        ]

        # 如果有其他训练相关参数，可以传递
        if hasattr(args, 'verbose') and args.verbose:
            cvae_args.append('--verbose')

        print(f"[INFO] 执行命令: {' '.join(cvae_args)}")

        # 执行CVAE训练脚本（实时输出）
        print(f"[INFO] 正在启动CVAE训练...")
        process = subprocess.Popen(cvae_args, cwd=project_root,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   encoding='utf-8',
                                   universal_newlines=True)

        # 实时输出
        for line in process.stdout:
            print(line.rstrip())

        # 等待完成
        process.wait()

        if process.returncode == 0:
            print("\n[SUCCESS] CVAE模型训练模块执行完成")
            return True
        else:
            print(f"\n[ERROR] CVAE训练失败，退出码: {process.returncode}")
            return False

    except Exception as e:
        print(f"[ERROR] CVAE模型训练失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def execute_generate(args):
    """执行载荷生成"""
    print("\n" + "=" * 60)
    print("启动CVAE载荷生成模块")
    print("=" * 60)

    try:
        import subprocess

        # 构建生成器命令
        generator_script = os.path.join(project_root, "CVAE", "generator.py")
        generator_args = [
            sys.executable,  # Python解释器路径
            generator_script,
            '--model-path', os.path.join(project_root, 'CVAE/checkpoints/cvae_final.pth'),
            '--vocab-path', os.path.join(project_root, 'Data/processed/vocab.json'),
            '--attack-type', args.attack_type,
            '--num-samples', str(args.num_samples),
            '--temperature', str(args.temperature),
            '--batch-size', str(args.generation_batch_size),
            '--output-dir', os.path.join(project_root, 'Data/generated')
        ]

        # 详细输出
        if args.verbose:
            generator_args.append('--verbose')

        print(f"[INFO] 执行生成命令: {' '.join(generator_args)}")
        print(f"[INFO] 正在生成 {args.attack_type} 载荷，每类数量: {args.num_samples}")

        # 执行生成脚本 - 修复Windows中文编码问题
        process = subprocess.Popen(generator_args, cwd=project_root,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   encoding='gbk',
                                   errors='replace',
                                   universal_newlines=True,
                                   bufsize=1)  # 行缓冲

        # 实时输出并刷新
        for line in process.stdout:
            print(line.rstrip(), flush=True)

        # 等待完成
        process.wait()

        if process.returncode == 0:
            print("\n[SUCCESS] 载荷生成模块执行完成")
            return True
        else:
            print(f"\n[ERROR] 载荷生成失败，退出码: {process.returncode}")
            return False

    except Exception as e:
        print(f"[ERROR] 载荷生成失败：{e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def execute_cluster(args):
    """执行聚类优化"""
    print("\n" + "=" * 60)
    print("启动DBSCAN聚类优化模块")
    print("=" * 60)

    try:
        import subprocess

        # 构建聚类器命令
        clusterer_script = os.path.join(project_root, "Clusterer", "clusterer.py")
        clusterer_args = [
            sys.executable,  # Python解释器路径
            clusterer_script,
            '--embeddings', os.path.join(project_root, 'Data/generated/latent_embeddings.npy'),
            '--payloads', os.path.join(project_root, 'Data/generated/raw_payloads.txt'),
            '--metadata', os.path.join(project_root, 'Data/generated/payload_metadata.json'),
            '--valid-mask', os.path.join(project_root, 'Data/generated/valid_mask.npy'),
            '--samples-per-cluster', str(args.samples_per_cluster),
            '--reduction-method', args.reduction_method,
            '--output-dir', os.path.join(project_root, 'Data/clustered')
        ]

        # 添加可选参数
        if args.eps:
            clusterer_args.extend(['--eps', str(args.eps)])
        if args.min_samples:
            clusterer_args.extend(['--min-samples', str(args.min_samples)])
        if args.visualize:
            clusterer_args.append('--visualize')
        if args.keep_noise:
            clusterer_args.append('--keep-noise')

        print(f"[INFO] 执行聚类命令: {' '.join(clusterer_args)}")
        print("[INFO] 正在进行DBSCAN聚类分析...")

        # 执行聚类脚本 - 修复Windows中文编码问题
        process = subprocess.Popen(clusterer_args, cwd=project_root,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   encoding='gbk',
                                   errors='replace',
                                   universal_newlines=True,
                                   bufsize=1)  # 行缓冲

        # 实时输出并刷新
        for line in process.stdout:
            print(line.rstrip(), flush=True)

        # 等待完成
        process.wait()

        if process.returncode == 0:
            print("\n[SUCCESS] DBSCAN聚类优化模块执行完成")

            # 检查精锐载荷文件是否生成
            refined_file = os.path.join(project_root, 'Data/fuzzing/refined_payloads.txt')
            if os.path.exists(refined_file):
                with open(refined_file, 'r', encoding='utf-8') as f:
                    refined_count = len([line for line in f.readlines() if line.strip()])
                print(f"[INFO] 精锐载荷已保存: {refined_file}")
                print(f"[INFO] 精锐载荷数量: {refined_count}")

            return True
        else:
            print(f"\n[ERROR] 聚类分析失败，退出码: {process.returncode}")
            return False

    except Exception as e:
        print(f"[ERROR] 聚类分析失败：{e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def execute_analyze(args):
    """执行AI载荷效果评估分析"""
    print("\n" + "=" * 60)
    print("启动AI载荷效果评估分析模块")
    print("=" * 60)

    try:
        # 添加Fuzz模块到Python路径
        fuzz_module_path = os.path.join(project_root, "Fuzz")
        if fuzz_module_path not in sys.path:
            sys.path.insert(0, fuzz_module_path)

        from analyzer import CVDBFuzzAnalyzer

        # 创建分析器实例
        analyzer = CVDBFuzzAnalyzer()

        # 执行分析
        analysis_result = analyzer.run_analysis()

        if analysis_result:
            print("\n[SUCCESS] AI载荷效果评估分析模块执行完成")
            return True
        else:
            print("\n[ERROR] 分析未产生结果")
            return False

    except ImportError as e:
        print(f"[ERROR] 导入分析器模块失败：{e}")
        print("[INFO] 请确保Fuzz/analyzer.py文件存在且依赖库已安装")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False
    except Exception as e:
        print(f"[ERROR] AI载荷效果评估分析失败：{e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def execute_fuzz(args):
    """执行智能漏洞探测 - 🔥 老王优化：支持多任务扫描和Baseline复用"""
    print("\n" + "=" * 60)
    print("启动智能漏洞探测模块")
    print("=" * 60)

    try:
        # 添加Fuzz模块到Python路径
        fuzz_module_path = os.path.join(project_root, "Fuzz")
        if fuzz_module_path not in sys.path:
            sys.path.insert(0, fuzz_module_path)

        from fuzzer import CVDBFuzzer

        # 检查精锐载荷文件是否存在
        refined_payloads_file = os.path.join(project_root, 'Data/fuzzing/refined_payloads.txt')

        # 🔥 老王新增：模式检查
        if args.common:
            print(f"[COMMON] 使用常见载荷模式，无需AI载荷文件")
            # Common模式下不依赖精锐载荷文件
        elif args.hybrid:
            # Hybrid模式下优先使用AI载荷，如果不存在会自动回退到专家载荷
            if not os.path.exists(refined_payloads_file):
                print(f"[HYBRID] AI载荷文件不存在，将使用专家载荷模式: {refined_payloads_file}")
            else:
                print(f"[HYBRID] 找到AI载荷文件，将使用混合模式: {refined_payloads_file}")
        else:
            # Smart模式下必须存在AI载荷文件
            if not os.path.exists(refined_payloads_file):
                print(f"[ERROR] 精锐载荷文件不存在: {refined_payloads_file}")
                print("[INFO] 请先执行 --generate --cluster 生成精锐载荷，或使用 --common 模式")
                return False

        # 🔥 老王新增：构建任务列表 - 支持单一URL和多任务
        fuzz_tasks = []

        if hasattr(args, 'fuzz_tasks') and args.fuzz_tasks:
            # 如果传入的是任务列表
            fuzz_tasks = args.fuzz_tasks
            print(f"[INFO] 使用传入的任务列表，共 {len(fuzz_tasks)} 个任务")
        elif getattr(args, 'tasks_file', None):
            # 从JSON文件加载任务列表
            try:
                with open(args.tasks_file, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)

                # 支持两种JSON格式：
                # 1. {"tasks": [...]}  - 包含tasks字段
                # 2. [...]           - 直接是任务列表
                if isinstance(tasks_data, dict) and 'tasks' in tasks_data:
                    fuzz_tasks = tasks_data['tasks']
                elif isinstance(tasks_data, list):
                    fuzz_tasks = tasks_data
                else:
                    print(f"[ERROR] 任务文件格式不正确: {args.tasks_file}")
                    return False

                print(f"[INFO] 从文件加载任务列表: {args.tasks_file}")
                print(f"[INFO] 任务数量: {len(fuzz_tasks)}")

            except Exception as e:
                print(f"[ERROR] 加载任务文件失败: {e}")
                return False
        elif args.url:
            # 单一URL扫描（兼容原逻辑）
            fuzz_tasks = [args.url]
            print(f"[INFO] 单一URL扫描模式: {args.url}")
        else:
            print("[ERROR] 未指定扫描目标，请使用 --url 或 --tasks-file")
            return False

        # 🔥 老王核心优化：根URL全局基准建立（针对Pikachu靶场优化）
        global_baseline = None
        processed_count = 0
        total_vulnerabilities = 0
        root_url_for_baseline = None

        # 🔥 老王新增：从任务列表中提取根URL用于基准测试
        if fuzz_tasks and len(fuzz_tasks) > 0:
            # 找到第一个任务作为根URL
            first_task = fuzz_tasks[0]
            if isinstance(first_task, dict):
                root_url_for_baseline = first_task.get('url', '')
            else:
                root_url_for_baseline = str(first_task)

            # 提取根域名（去除FUZZ和参数）
            if 'FUZZ' in root_url_for_baseline:
                # 简单处理：将FUZZ替换为"test"作为基准测试URL
                root_url_for_baseline = root_url_for_baseline.replace('FUZZ', '1')

            print(f"[BASELINE] 根URL基准测试目标: {root_url_for_baseline}")

            # 🔥 老王核心修复：针对根 URL 执行一次 establish_baseline()
            try:
                baseline_fuzzer = CVDBFuzzer(
                    url=root_url_for_baseline,
                    threads=min(args.threads, 5),  # 基准测试用较少线程
                    proxy=args.proxy,
                    cookie=args.cookie,
                    timeout=args.fuzz_timeout,
                    delay=args.fuzz_delay,
                    method='GET',
                    common_mode=args.common,
                    hybrid_mode=args.hybrid
                )
                global_baseline = baseline_fuzzer.establish_baseline(baseline_requests=5)
                print(f"[SUCCESS] 全局基准建立完成！基准信息:")
                print(f"  - 正常长度: {global_baseline.normal_length}")
                print(f"  - 正常响应时间: {global_baseline.normal_time:.3f}s")
                print(f"  - 正常状态码: {global_baseline.normal_status}")
                print(f"  - 后续所有GET任务将复用此基准，大幅提升扫描效率")
            except Exception as e:
                print(f"[WARNING] 全局基准建立失败: {e}")
                print(f"[INFO] 各任务将独立建立基准")
                global_baseline = None

        # 🔥 老王核心修复：在 for task in fuzz_tasks 循环中实现任务解包逻辑
        for task in fuzz_tasks:
            try:
                processed_count += 1
                print(f"\n{'-'*60}")
                print(f"[PROGRESS] 处理任务 {processed_count}/{len(fuzz_tasks)}")
                print(f"{'-'*60}")

                # 🔥 老王核心修复：任务解包逻辑 - 增强类型检查
                url = ''
                method = 'GET'
                post_data = None

                if isinstance(task, dict):
                    # 字典格式：提取 url、method 和 data 并传递给 CVDBFuzzer
                    url = task.get('url', '')
                    method = task.get('method', 'GET')
                    post_data = task.get('data', None) or task.get('form_data', None)

                    print(f"[TASK] 字典格式任务: {url}")
                    print(f"[TASK] HTTP方法: {method}")
                    if post_data:
                        print(f"[TASK] POST数据: {post_data}")

                elif isinstance(task, str):
                    # 字符串格式：按原逻辑执行
                    url = task
                    method = 'GET'
                    post_data = None

                    print(f"[TASK] 字符串格式任务: {url}")

                else:
                    # 其他格式：转换为字符串处理
                    url = str(task)
                    method = 'GET'
                    post_data = None

                    print(f"[TASK] 其他格式任务: {url}")

                # 🔥 老王核心优化：强制传入BaselineInfo对象
                use_baseline = global_baseline if method == 'GET' else None

                # 创建CVDBFuzzer实例（统一创建逻辑）
                fuzzer = CVDBFuzzer(
                    url=url,
                    threads=args.threads,
                    proxy=args.proxy,
                    cookie=args.cookie,
                    timeout=args.fuzz_timeout,
                    delay=args.fuzz_delay,
                    method=method,
                    post_data=post_data,
                    baseline=global_baseline if method == 'GET' else None,  # 🔥 强制传入BaselineInfo对象
                    common_mode=args.common,
                    hybrid_mode=args.hybrid
                )

                # 🔥 老王增强：根据模式选择载荷加载方式
                payload_file_to_use = None

                if args.common:
                    # Common模式：使用常见载荷
                    print(f"[COMMON] 加载常见载荷字典...")
                    # 使用特殊标记告诉 fuzzer 使用内置载荷
                    payload_file_to_use = "BUILTIN:COMMON"
                elif args.hybrid:
                    # Hybrid模式：混合载荷
                    print(f"[HYBRID] 加载混合载荷（专家+AI）...")
                    if os.path.exists(refined_payloads_file):
                        payload_file_to_use = refined_payloads_file
                    else:
                        payload_file_to_use = "BUILTIN:HYBRID"
                else:
                    # Smart模式：使用AI生成的载荷
                    payload_file_to_use = refined_payloads_file

                # 🔥 老王核心优化：根据是否传入baseline决定是否跳过基准测试
                if use_baseline:
                    print(f"[BASELINE] 使用全局基准，跳过重复基准测试")
                    # 执行扫描（baseline_requests=0表示跳过基准测试）
                    report = fuzzer.scan(payload_file_to_use, baseline_requests=0, use_full_combination=args.full_combination, quick_scan=args.quick_scan, use_radamsa=args.radamsa)
                else:
                    if method == 'POST':
                        print(f"[BASELINE] POST任务执行独立基准测试...")
                    else:
                        print(f"[BASELINE] 执行独立基准测试...")
                    # 执行扫描（会自动建立baseline）
                    report = fuzzer.scan(payload_file_to_use, baseline_requests=5, use_full_combination=args.full_combination, quick_scan=args.quick_scan, use_radamsa=args.radamsa)

                # 显示扫描结果摘要
                if report and report.get('report_file'):
                    print(f"[SUCCESS] 任务扫描完成: {report.get('report_file', 'N/A')}")
                    if report['statistics']['vulnerable_requests'] > 0:
                        vuln_count = report['statistics']['vulnerable_requests']
                        total_vulnerabilities += vuln_count
                        print(f"[VULNERABLE] 发现 {vuln_count} 个潜在漏洞!")
                        for vuln_type, count in report['statistics']['vulnerabilities_by_type'].items():
                            print(f"[VULNERABLE] {vuln_type}: {count} 个")
                    else:
                        print("[INFO] 未发现明显漏洞")

            except Exception as e:
                print(f"[ERROR] 任务扫描失败: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue

        # 🔥 老王新增：显示总体扫描结果
        print(f"\n{'='*60}")
        print(f"[SUCCESS] 智能漏洞探测模块执行完成")
        print(f"{'='*60}")
        print(f"[SUMMARY] 处理任务总数: {processed_count}/{len(fuzz_tasks)}")
        print(f"[SUMMARY] 发现漏洞总数: {total_vulnerabilities}")
        if global_baseline:
            print(f"[SUMMARY] 已启用全局基准复用优化")

        return True

    except ImportError as e:
        print(f"[ERROR] 导入Fuzzer模块失败：{e}")
        print("[INFO] 请确保Fuzz/fuzzer.py文件存在且依赖库已安装")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False
    except Exception as e:
        print(f"[ERROR] 智能漏洞探测失败：{e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def execute_crawler(args):
    """执行CVDB-Spider全站自动化扫描"""
    print("\n" + "=" * 60)
    print("启动CVDB-Spider全站自动化扫描模块")
    print("=" * 60)

    try:
        # 添加Fuzz模块到Python路径
        fuzz_module_path = os.path.join(project_root, "Fuzz")
        if fuzz_module_path not in sys.path:
            sys.path.insert(0, fuzz_module_path)

        from spider import CVDBSpider

        print(f"[INFO] 目标站点: {args.url}")
        print(f"[INFO] 爬取深度: {args.crawler_depth}")
        print(f"[INFO] 爬虫线程: {args.crawler_threads}")
        print(f"[INFO] 请求超时: {args.crawler_timeout}s")
        print(f"[INFO] 请求延迟: {args.crawler_delay}s")
        print(f"[INFO] 仅爬取模式: {not args.scan}")
        if args.proxy:
            print(f"[INFO] 代理服务器: {args.proxy}")
        if args.cookie:
            print(f"[INFO] 登录Cookie: {args.cookie[:50]}...")

        # 创建爬虫实例
        spider = CVDBSpider(
            base_url=args.url,
            max_depth=args.crawler_depth,
            threads=args.crawler_threads,
            timeout=args.crawler_timeout,
            delay=args.crawler_delay,
            cookie=args.cookie,
            debug=args.verbose  # 🔥 老王修复: 传递debug参数
        )

        # 🔥 老王新增：智能缓存系统
        use_cache = not args.no_cache
        print(f"[INFO] 缓存模式: {'启用' if use_cache else '禁用'}")

        # 开始爬取（支持智能缓存）
        fuzz_targets = spider.start_crawling(use_cache=use_cache)

        if args.scan and fuzz_targets:
            scan_result = None

            # 🔥 老王新增：优先使用 Wfuzz 引擎
            if args.wfuzz:
                print(f"\n[INFO] 使用 Wfuzz 引擎进行扫描 ({len(fuzz_targets)} 个目标)")
                scan_result = integrate_with_wfuzz_engine(
                    fuzz_targets=fuzz_targets,
                    threads=args.threads,
                    timeout=args.fuzz_timeout,
                    delay=args.fuzz_delay,
                    proxy=args.proxy,
                    cookie=args.cookie,
                    common_mode=args.common,
                    hybrid_mode=args.hybrid,
                    verbose=args.verbose
                )
            else:
                # 集成fuzzer进行扫描（原有逻辑）
                scan_result = spider.integrate_with_fuzzer(
                    fuzz_targets=fuzz_targets,
                    threads=args.threads,  # 使用fuzz的线程数
                    timeout=args.fuzz_timeout,
                    delay=args.fuzz_delay,
                    proxy=args.proxy,
                    cookie=args.cookie,
                    common_mode=args.common,
                    hybrid_mode=args.hybrid
                )

            if scan_result and scan_result.get('vulnerabilities_found', 0) > 0:
                print(f"\n[SUCCESS] 全站扫描完成！发现 {scan_result['vulnerabilities_found']} 个潜在漏洞!")
                return True
            else:
                print(f"\n[SUCCESS] 全站扫描完成，未发现明显漏洞")
                return True
        elif not fuzz_targets:
            print(f"[WARNING] 未发现任何FUZZ目标，扫描结束")
            return False
        else:
            print(f"\n[SUCCESS] 爬虫完成，跳过漏洞扫描 (仅爬取模式)")
            return True

    except ImportError as e:
        print(f"[ERROR] 导入Spider模块失败：{e}")
        print("[INFO] 请确保Fuzz/spider.py文件存在且依赖库已安装")
        print("[INFO] 需要安装: pip install requests beautifulsoup4 colorama progressbar")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False
    except Exception as e:
        print(f"[ERROR] CVDB-Spider全站扫描失败：{e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def integrate_with_wfuzz_engine(fuzz_targets: List, threads: int = 20, timeout: int = 10,
                               delay: float = 0.1, proxy: str = None, cookie: str = None,
                               common_mode: bool = False, hybrid_mode: bool = False,
                               verbose: bool = True) -> Optional[Dict]:
    """
    🔥 老王新增：集成 Wfuzz 引擎进行扫描

    Args:
        fuzz_targets: 爬虫发现的目标列表
        threads: 并发线程数
        timeout: 请求超时时间
        delay: 请求间延迟
        proxy: 代理服务器
        cookie: Cookie 字符串
        common_mode: 是否使用常见载荷模式
        hybrid_mode: 是否使用混合模式
        verbose: 详细输出

    Returns:
        扫描结果字典
    """
    try:
        # 导入 wfuzz 引擎
        from wfuzz_plugins.wfuzz_engine import WfuzzEngine

        print(f"[WFUZZ_ENGINE] 初始化高性能扫描引擎...")

        # 创建引擎实例
        engine = WfuzzEngine(
            threads=threads,
            timeout=timeout,
            delay=delay,
            proxy=proxy,
            cookie=cookie,
            verbose=verbose
        )

        # 选择扫描模式
        if common_mode:
            mode = "common"
        elif hybrid_mode:
            mode = "hybrid"
        else:
            mode = "smart"

        print(f"[WFUZZ_ENGINE] 使用扫描模式: {mode}")
        print(f"[WFUZZ_ENGINE] 目标数量: {len(fuzz_targets)}")

        # 确定载荷文件
        payloads_file = None
        if mode == "smart":
            payloads_file = os.path.join(project_root, 'Data', 'fuzzing', 'refined_payloads.txt')

        # 执行批量扫描
        scan_result = engine.scan_multiple_targets(
            fuzz_targets=fuzz_targets,
            mode=mode,
            payloads_file=payloads_file,
            error_threshold='0.3',
            reflection_threshold='0.4',
            time_threshold='5.0',
            count_per_type='10' if mode == "common" else '5',
            expert_count='15' if mode == "hybrid" else '10'
        )

        # 显示统计信息
        stats = engine.get_stats()
        print(f"\n[WFUZZ_ENGINE] 扫描统计:")
        print(f"  - 总请求数: {stats.get('total_requests', 0)}")
        print(f"  - 发现漏洞: {scan_result.get('vulnerabilities_found', 0)}")
        print(f"  - 平均速度: {stats.get('requests_per_second', 0):.1f} req/s")
        print(f"  - 扫描时间: {stats.get('running_time', 0):.1f}s")

        return scan_result

    except ImportError as e:
        print(f"[ERROR] Wfuzz 引擎导入失败: {e}")
        print("[INFO] 回退到传统 Fuzzer 模式")
        return None
    except Exception as e:
        print(f"[ERROR] Wfuzz 引擎执行失败: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def main():
    """主函数"""
    # 如果没有参数，显示帮助信息
    if len(sys.argv) == 1:
        print_banner()
        parser = argparse.ArgumentParser(
            description="CVDBFuzz - 基于CVAE生成与DBSCAN优化的智能Web模糊测试框架",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
工作流程示例：
    1. 数据预处理：  python main.py --preprocess
    2. 模型训练：    python main.py --train
    3. 生成与聚类：  python main.py --generate --cluster
    4. 智能探测：    python main.py --fuzz --url "http://target.com/index.php?id=FUZZ"
    5. 快速扫描：    python main.py --fuzz --url "http://target.com/index.php?id=FUZZ" --quick-scan
    6. 深度变异：    python main.py --fuzz --url "http://target.com/index.php?id=FUZZ" --radamsa
    7. 登录态探测：  python main.py --fuzz --url "http://target.com/user.php?id=FUZZ" --cookie "session=abc123; token=def456"
    8. 效果分析：    python main.py --analyze
    9. 全站扫描：    python main.py --crawler --url "http://target.com/"
    10. 登录态扫描：  python main.py --crawler --url "http://target.com/" --cookie "session=abc123; token=def456"
    11. 完整流程：   python main.py --preprocess --train --generate --cluster --crawler --analyze

参数说明：
    --preprocess:   执行数据预处理模块
    --train:        执行CVAE模型训练
    --generate:     执行载荷生成
    --cluster:      执行DBSCAN聚类优化
    --fuzz:         执行智能漏洞探测
    --analyze:      执行AI载荷效果评估分析
    --crawler:      启动CVDB-Spider全站自动化扫描
        """
        )
        parser.add_argument(
            '--version',
            action='version',
            version='CVDBFuzz v1.0 - 老王专属版'
        )
        parser.print_help()
        return

    # 解析命令行参数
    args = parse_arguments()

    # 🔥 老王新增：缓存管理功能
    if args.cache_info:
        execute_cache_info(args)
        sys.exit(0)

    if args.clear_cache:
        execute_clear_cache(args)
        sys.exit(0)

    # 验证参数
    errors = validate_arguments(args)
    if errors:
        print("[ERROR] 参数验证失败：")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)

    # 显示启动信息
    print_banner()

    # 记录执行状态
    success_count = 0
    total_operations = sum([args.preprocess, args.train, args.generate, args.cluster, args.fuzz, args.analyze, args.crawler])

    # 执行指定的操作
    if args.preprocess:
        if execute_preprocess(args):
            success_count += 1

    if args.train:
        if execute_train(args):
            success_count += 1

    if args.generate:
        if execute_generate(args):
            success_count += 1

    if args.cluster:
        if execute_cluster(args):
            success_count += 1

    if args.fuzz:
        if execute_fuzz(args):
            success_count += 1

    if args.analyze:
        if execute_analyze(args):
            success_count += 1

    if args.crawler:
        if execute_crawler(args):
            success_count += 1

    # 显示执行结果
    print("\n" + "=" * 60)
    print("任务执行结果")
    print("=" * 60)
    print(f"总任务数：{total_operations}")
    print(f"成功完成：{success_count}")
    print(f"执行失败：{total_operations - success_count}")

    if success_count == total_operations:
        print("\n[SUCCESS] 所有任务执行完成！CVDBFuzz框架运行正常！")
        sys.exit(0)
    else:
        print(f"\n[WARNING] {total_operations - success_count} 个任务执行失败")
        sys.exit(1)


def execute_cache_info(args):
    """执行缓存信息显示"""
    from datetime import datetime

    print("\n" + "=" * 60)
    print("CVDB-Spider 缓存信息")
    print("=" * 60)

    try:
        # 添加Fuzz模块到Python路径
        fuzz_module_path = os.path.join(project_root, "Fuzz")
        if fuzz_module_path not in sys.path:
            sys.path.insert(0, fuzz_module_path)

        from spider_cache import get_spider_cache

        cache = get_spider_cache()
        cache_entries = cache.list_cache_entries()

        if not cache_entries:
            print("[INFO] 没有找到任何缓存记录")
            return True

        print(f"[INFO] 共找到 {len(cache_entries)} 个缓存条目：")
        print("-" * 60)

        for entry in cache_entries:
            created_time = datetime.fromisoformat(entry['created_time'])
            age_hours = (datetime.now() - created_time).total_seconds() / 3600

            print(f"域名: {entry['domain']}")
            print(f"URL: {entry['base_url']}")
            print(f"深度: {entry['max_depth']}")
            print(f"URL数量: {entry['urls_count']}")
            print(f"FUZZ目标: {entry['fuzz_targets_count']}")
            print(f"缓存时间: {entry['timestamp']}")
            print(f"缓存年龄: {age_hours:.1f} 小时")
            print("-" * 60)

        # 计算总统计
        total_urls = sum(entry['urls_count'] for entry in cache_entries)
        total_targets = sum(entry['fuzz_targets_count'] for entry in cache_entries)
        unique_domains = len(set(entry['domain'] for entry in cache_entries))

        print(f"[汇总] 域名数量: {unique_domains}")
        print(f"[汇总] URL总数: {total_urls}")
        print(f"[汇总] FUZZ目标总数: {total_targets}")

        return True

    except Exception as e:
        print(f"[ERROR] 缓存信息显示失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def execute_clear_cache(args):
    """执行缓存清理"""
    print("\n" + "=" * 60)
    print(f"CVDB-Spider 缓存清理")
    print("=" * 60)

    try:
        # 添加Fuzz模块到Python路径
        fuzz_module_path = os.path.join(project_root, "Fuzz")
        if fuzz_module_path not in sys.path:
            sys.path.insert(0, fuzz_module_path)

        from spider_cache import get_spider_cache

        cache = get_spider_cache()

        if args.clear_cache.lower() == 'all':
            # 清理所有缓存
            print("[INFO] 清理所有缓存...")
            cleared_count = cache.clear_cache()
        else:
            # 清理指定域名的缓存
            print(f"[INFO] 清理域名 '{args.clear_cache}' 的缓存...")
            cleared_count = cache.clear_cache(domain=args.clear_cache)

        if cleared_count > 0:
            print(f"[SUCCESS] 成功清理 {cleared_count} 个缓存条目")
        else:
            print("[INFO] 没有找到匹配的缓存条目")

        return True

    except Exception as e:
        print(f"[ERROR] 缓存清理失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    main()