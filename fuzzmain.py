#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVDBFuzz - 基于CVAE生成与DBSCAN优化的智能Web模糊测试框架（全阶段）
====================================================================

核心功能：
1. 阶段一：数据预处理 - 字符级分词、序列标准化、词表构建
2. 阶段二：CVAE训练 - 学习攻击载荷的隐式分布
3. 阶段三：生成与聚类 - CVAE生成载荷 + DBSCAN优化
4. 阶段四：黑盒模糊测试 - 递归爬虫 + 智能注入

使用示例：
    # 前三阶段流程
    python fuzzmain.py --preprocess --train --generate --cluster

    # 第四阶段：纯爬虫模式
    python fuzzmain.py --crawl --url http://example.com --depth 2

    # 第四阶段：完整扫描模式
    python fuzzmain.py --scan --url http://example.com --depth 2 --use-cache

    # 分步执行
    python fuzzmain.py --preprocess
    python fuzzmain.py --train --epochs 50
    python fuzzmain.py --generate --cluster --num-samples 10000
    python fuzzmain.py --analyze
"""

import sys
import os
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入预处理模块
try:
    sys.path.insert(0, os.path.join(project_root, "Data_processing"))
    from preprocessor import main as preprocess_main
except ImportError as e:
    print(f"[ERROR] 导入预处理模块失败：{e}")
    print("[INFO] 请确保 Data_processing/preprocessor.py 文件存在")
    sys.exit(1)

# 导入CVAE生成器和聚类器模块
try:
    sys.path.insert(0, os.path.join(project_root, "CVAE"))
    from generator import CVAEGenerator
    sys.path.insert(0, os.path.join(project_root, "Clusterer"))
    from clusterer import CVAEClusterer
except ImportError as e:
    print(f"[ERROR] 导入生成/聚类模块失败：{e}")
    print("[INFO] 请确保 CVAE/generator.py 和 Clusterer/clusterer.py 文件存在")
    sys.exit(1)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CVDBFuzz - 基于CVAE生成与DBSCAN优化的智能Web模糊测试框架（全阶段）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
工作流程示例：
【前三阶段：AI载荷生成】
    1. 数据预处理：  python fuzzmain.py --preprocess
    2. 模型训练：    python fuzzmain.py --train --epochs 50
    3. 生成与聚类：  python fuzzmain.py --generate --cluster --num-samples 5000

【第四阶段：黑盒模糊测试】
    4. 纯爬虫模式：  python fuzzmain.py --crawl --url http://example.com --depth 2
    5. 完整扫描：    python fuzzmain.py --scan --url http://example.com --use-cache

【BaseFuzz智能模糊测试】
    6. 从URL扫描：  python fuzzmain.py --fuzz --url http://example.com
    7. 从缓存扫描：  python fuzzmain.py --fuzz --file Data/cache/example.com/spider_cache.json
    8. 自定义模式：  python fuzzmain.py --fuzz --url http://example.com --mode common --threads 20

【完整流程】
    # AI载荷生成 + BaseFuzz测试
    python fuzzmain.py --preprocess --train --generate --cluster
    python fuzzmain.py --fuzz --url http://target.com --mode cvae

参数说明：
    [阶段一] --preprocess:   执行数据预处理模块
    [阶段二] --train:        执行CVAE模型训练
    [阶段三] --generate:     执行载荷生成
            --cluster:       执行DBSCAN聚类优化
    [阶段四] --crawl:        纯爬虫模式，发现并保存任务缓存
            --scan:         完整扫描模式（爬虫 + 模糊测试）
            --url:          目标基础URL（阶段四必填）
            --depth:        爬虫递归深度（默认2）
            --use-cache:    复用已有爬虫缓存
            --cookie:       全局认证Cookie

    [BaseFuzz] --fuzz:        使用BaseFuzz智能引擎执行模糊测试
            --file:         加载现有的爬虫JSON缓存文件
            --engine:       选择引擎类型 (仅base可用，其他待实现)
            --mode:         载荷模式 (common=专家字典, cvae=AI生成, 默认: cvae)
            --threads:      并发线程数 (默认: 10)
        """
    )

    # ========== 阶段一：数据预处理参数 ==========
    preprocess_group = parser.add_argument_group('阶段一：数据预处理参数')
    preprocess_group.add_argument(
        '--preprocess',
        action='store_true',
        help='执行数据预处理（字符级分词、序列标准化、词表构建）'
    )
    preprocess_group.add_argument(
        '--data-dir',
        type=str,
        default='Data/payload/train',
        help='训练数据目录路径 (默认: Data/payload/train)'
    )
    preprocess_group.add_argument(
        '--output-dir',
        type=str,
        default='Data/processed',
        help='输出目录路径 (默认: Data/processed)'
    )
    preprocess_group.add_argument(
        '--max-length',
        type=int,
        default=150,
        help='最大序列长度 (默认: 150)'
    )
    preprocess_group.add_argument(
        '--vocab-size',
        type=int,
        default=256,
        help='词表大小 (默认: 256)'
    )

    # ========== 阶段二：CVAE训练参数 ==========
    train_group = parser.add_argument_group('阶段二：CVAE模型训练参数')
    train_group.add_argument(
        '--train',
        action='store_true',
        help='训练CVAE模型'
    )
    train_group.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='训练轮数 (默认: 50)'
    )
    train_group.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批大小 (默认: 32)'
    )
    train_group.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='学习率 (默认: 1e-3)'
    )
    train_group.add_argument(
        '--embed-dim',
        type=int,
        default=128,
        help='词嵌入维度 (默认: 128)'
    )
    train_group.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='GRU隐藏层维度 (默认: 256)'
    )
    train_group.add_argument(
        '--latent-dim',
        type=int,
        default=32,
        help='隐空间维度 (默认: 32)'
    )
    train_group.add_argument(
        '--kl-cycles',
        type=int,
        default=1,
        help='KL退火周期数 (默认: 1，单一稳定增长)'
    )
    train_group.add_argument(
        '--beta-max',
        type=float,
        default=0.25,
        help='KL退火最大Beta值 (默认: 0.25，强约束力)'
    )
    train_group.add_argument(
        '--delay-epochs',
        type=int,
        default=20,
        help='KL退火延迟epoch数 (默认: 20，先学好重构)'
    )
    train_group.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='GRU层数 (默认: 2，双层GRU增强表达能力)'
    )
    train_group.add_argument(
        '--condition-dim',
        type=int,
        default=6,
        help='攻击类型标签维度 (默认: 6，支持SQLi/XSS/CMDi等6类)'
    )
    train_group.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='L2正则化权重衰减系数 (默认: 1e-5)'
    )
    train_group.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='训练集比例 (默认: 0.8，20%%作为验证集)'
    )
    train_group.add_argument(
        '--no-oversample',
        action='store_false',
        dest='oversample',
        help='禁用类别过采样 (默认启用，平衡样本分布)'
    )

    # ========== 阶段三：生成与聚类参数 ==========
    generate_group = parser.add_argument_group('阶段三：生成与聚类参数')
    generate_group.add_argument(
        '--generate',
        action='store_true',
        help='使用训练好的CVAE生成载荷'
    )
    generate_group.add_argument(
        '--cluster',
        action='store_true',
        help='使用DBSCAN对生成的载荷进行聚类优化'
    )
    generate_group.add_argument(
        '--temperature',
        type=float,
        default=1.8,
        help='生成温度参数 (默认: 1.8，增加随机性)'
    )
    generate_group.add_argument(
        '--num-samples',
        type=int,
        default=5000,
        help='每种攻击类型生成样本数量 (默认: 5000)'
    )
    generate_group.add_argument(
        '--attack-type',
        type=str,
        default='ALL',
        help='攻击载荷类型 (默认: ALL，支持: SQLi, XSS, CMDi, Overflow, XXE, SSI, ALL 或逗号分隔的组合)'
    )
    generate_group.add_argument(
        '--generation-batch-size',
        type=int,
        default=500,
        help='生成批处理大小 (默认: 500)'
    )
    generate_group.add_argument(
        '--eps',
        type=float,
        help='DBSCAN的eps参数（自动寻找如果未指定）'
    )
    generate_group.add_argument(
        '--min-samples',
        type=int,
        help='DBSCAN的min_samples参数'
    )
    generate_group.add_argument(
        '--samples-per-cluster',
        type=int,
        default=5,
        help='每个簇保留的样本数 (默认: 5)'
    )
    generate_group.add_argument(
        '--reduction-method',
        type=str,
        default='tsne',
        choices=['tsne', 'pca'],
        help='降维方法 (默认: tsne)'
    )
    generate_group.add_argument(
        '--visualize',
        action='store_true',
        help='生成聚类可视化图像'
    )
    generate_group.add_argument(
        '--keep-noise',
        action='store_true',
        help='保留所有噪声点'
    )

    # ========== 效果分析参数 ==========
    analyze_group = parser.add_argument_group('效果分析参数')
    analyze_group.add_argument(
        '--analyze',
        action='store_true',
        help='执行AI载荷效果评估分析（TODO: 待实现）'
    )

    # ========== 爬虫与目标生成参数 ==========
    scan_group = parser.add_argument_group('爬虫与目标生成参数')
    scan_group.add_argument(
        '--crawl',
        action='store_true',
        help='纯爬虫模式：发现并保存任务缓存后退出'
    )
    scan_group.add_argument(
        '--scan',
        action='store_true',
        help='完整扫描模式：爬虫 + 模糊测试'
    )
    scan_group.add_argument(
        '--url',
        type=str,
        help='目标基础URL（必填）'
    )
    scan_group.add_argument(
        '--params',
        type=str,
        help='手动指定测试参数（逗号分隔），格式: --params name1,name2 或 name1=value1,name2 （value表示固定值，Fuzz表示测试）'
    )
    scan_group.add_argument(
        '--method',
        type=str,
        choices=['GET', 'POST'],
        default='GET',
        help='请求方法: GET(默认) 或 POST'
    )
    scan_group.add_argument(
        '--depth',
        type=int,
        default=2,
        help='爬虫递归深度 (默认: 2)'
    )
    scan_group.add_argument(
        '--use-cache',
        action='store_true',
        help='复用已有的爬虫缓存'
    )
    scan_group.add_argument(
        '--cookie',
        type=str,
        help='全局认证Cookie'
    )
    scan_group.add_argument(
        '--headers',
        type=str,
        default='',
        help='指定要测试的HTTP头（逗号分隔）\n示例: --headers "User-Agent,Referer,X-Forwarded-For"\n默认: 空（不测试HTTP头）'
    )
    scan_group.add_argument(
        '--timeout',
        type=int,
        default=10,
        help='请求超时时间(秒) (默认: 10)'
    )

    # ========== BaseFuzz引擎参数 ==========
    basefuzz_group = parser.add_argument_group('BaseFuzz引擎参数')
    basefuzz_group.add_argument(
        '--fuzz',
        action='store_true',
        help='使用BaseFuzz引擎执行模糊测试'
    )
    basefuzz_group.add_argument(
        '--engine',
        type=str,
        default='base',
        choices=['base'],  # 目前只实现base引擎
        help='选择引擎类型 (默认: base，其他引擎待实现)'
    )
    basefuzz_group.add_argument(
        '--mode',
        type=str,
        default='cvae',
        choices=['common', 'cvae'],
        help='载荷模式: common=专家字典, cvae=AI生成 (默认: cvae)'
    )
    basefuzz_group.add_argument(
        '--threads',
        type=int,
        default=10,
        help='并发线程数 (默认: 10)'
    )
    basefuzz_group.add_argument(
        '--file',
        type=str,
        help='加载现有的爬虫JSON缓存文件（BaseFuzz模式）'
    )

    # ========== 通用参数 ==========
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出信息'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='CVDBFuzz v4.0 - 全阶段完整版'
    )

    return parser.parse_args()


def validate_arguments(args):
    """验证命令行参数的有效性"""
    errors = []

    # 检查数据目录是否存在
    if args.preprocess:
        data_path = Path(args.data_dir)
        if not data_path.exists():
            errors.append(f"数据目录不存在：{data_path}")
        elif not data_path.is_dir():
            errors.append(f"数据路径不是目录：{data_path}")

    # 检查操作组合的有效性
    operations = [args.preprocess, args.train, args.generate, args.cluster, args.analyze, args.crawl, args.scan, args.fuzz]
    active_operations = sum(operations)

    if active_operations == 0:
        errors.append("必须指定至少一个操作：--preprocess, --train, --generate, --cluster, --analyze, --crawl, --scan, --fuzz")

    # ========== 新增：analyze功能待实现 ==========
    if args.analyze:
        print("[WARNING] --analyze 功能尚未实现，将在未来版本中推出")
        print("[INFO] 计划包含：重构准确率、有效样本率、聚类质量评估等")
        # 注意：这里不返回，让主流程继续，main()函数中会跳过analyze的执行

    # 如果指定了cluster但没有generate，给出警告
    if args.cluster and not args.generate:
        print("[WARNING] --cluster 通常需要与 --generate 一起使用")

    # ========== 新增：第四阶段参数验证 ==========
    if args.crawl or args.scan:
        if not args.url:
            errors.append("--crawl 和 --scan 模式必须指定 --url 参数")

        # 深度验证
        if args.depth < 0 or args.depth > 10:
            errors.append("--depth 必须在 0-10 之间")

        # 如果同时指定了--crawl和--scan，给出警告
        if args.crawl and args.scan:
            print("[WARNING] --crawl 和 --scan 同时指定，将只执行 --scan 模式")

    # ========== 新增：BaseFuzz参数验证 ==========
    if args.fuzz:
        # BaseFuzz必须指定 --url 或 --file 之一
        if not args.url and not args.file:
            errors.append("--fuzz 模式必须指定 --url 或 --file 参数")

        # 如果指定了--file，检查文件是否存在
        if args.file:
            file_path = Path(args.file)
            if not file_path.exists():
                errors.append(f"--file 指定的文件不存在: {args.file}")
            elif not file_path.suffix == '.json':
                print(f"[WARNING] --file 指定的文件不是JSON格式: {args.file}")

        # 线程数验证
        if args.threads < 1 or args.threads > 100:
            errors.append("--threads 必须在 1-100 之间")

        # 模式验证
        if args.mode == 'cvae':
            # CVAE模式需要检查生成载荷是否存在
            cvae_payloads = Path("Data/processed/fuzzing/refined_payloads.txt")
            if not cvae_payloads.exists():
                print(f"[WARNING] CVAE模式未找到载荷文件: {cvae_payloads}")
                print(f"[INFO] 请先生成载荷: python fuzzmain.py --generate --cluster")

    return errors


def print_banner():
    """打印程序启动横幅"""
    banner = """
================================================================
启动！！！！！
================================================================
"""
    print(banner)


def execute_preprocess(args):
    """执行数据预处理"""
    print("\n" + "=" * 60)
    print("阶段一：数据预处理")
    print("=" * 60)

    try:
        # ========== 🔥 优化1：检查Data_processing目录 ==========
        data_processing_dir = os.path.join(project_root, "Data_processing")
        if not os.path.exists(data_processing_dir):
            print(f"\n[ERROR] Data_processing目录不存在: {data_processing_dir}")
            print("[INFO] 请确保项目结构完整")
            return False

        # ========== 🔥 优化2：检查数据目录中的jsonl文件 ==========
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"\n[ERROR] 数据目录不存在: {data_dir}")
            return False

        # 查找jsonl文件
        jsonl_files = list(data_dir.glob("*.jsonl"))

        if not jsonl_files:
            print(f"\n[WARNING] 数据目录中没有找到.jsonl文件: {data_dir}")
            print(f"[INFO] 发现的文件类型: {list(data_dir.glob('*'))}")
            print("\n[提示] 数据目录为空，可能需要先运行分类器准备数据：")
            print(f"        python Data_processing/categorize_fuzz_dicts.py")
            print(f"\n        或者手动准备数据到: {args.data_dir}")

            # 询问用户是否继续
            try:
                user_input = input("\n是否仍要继续执行预处理？(y/N): ").strip().lower()
                if user_input != 'y':
                    print("[INFO] 用户取消操作")
                    return False
            except (EOFError, KeyboardInterrupt):
                print("\n[INFO] 用户取消操作")
                return False

        # ========== 优化3：显示找到的数据文件 ==========
        if jsonl_files:
            print(f"[INFO] 找到 {len(jsonl_files)} 个数据文件:")
            for jsonl_file in sorted(jsonl_files):
                file_size = jsonl_file.stat().st_size
                print(f"  - {jsonl_file.name}: {file_size:,} 字节")

        # 构建preprocessor参数
        # 修复：必须显式添加'--preprocess'参数，触发实际处理逻辑
        preprocess_args = [
            '--preprocess',  # 必需参数，告诉preprocessor.py执行预处理
            '--data-dir', args.data_dir,
            '--output-dir', args.output_dir,
            '--max-length', str(args.max_length),
            '--vocab-size', str(args.vocab_size)
        ]

        # 修复：移除--verbose参数传递，preprocessor.py不支持此参数
        # 不再传递 --verbose 给 preprocessor.py

        print(f"\n[INFO] 数据目录: {args.data_dir}")
        print(f"[INFO] 输出目录: {args.output_dir}")
        print(f"[INFO] 最大序列长度: {args.max_length}")
        print(f"[INFO] 词表大小: {args.vocab_size}")

        # 调用preprocessor的main函数
        import sys
        old_argv = sys.argv
        sys.argv = ['preprocessor.py'] + preprocess_args
        try:
            preprocess_main()
        finally:
            sys.argv = old_argv

        print("\n[SUCCESS] 数据预处理完成！")
        return True

    except Exception as e:
        print(f"\n[ERROR] 数据预处理失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def execute_train(args):
    """执行CVAE模型训练"""
    print("\n" + "=" * 60)
    print("阶段二：CVAE模型训练")
    print("=" * 60)

    try:
        # ========== 优化1：检查Stage 1输出文件 ==========
        processed_data_path = os.path.join(args.output_dir, "processed_data.pt")
        vocab_path = os.path.join(args.output_dir, "vocab.json")

        if not os.path.exists(processed_data_path):
            print(f"\n[ERROR] 找不到预处理数据文件: {processed_data_path}")
            print("[INFO] 请先运行数据预处理：python fuzzmain.py --preprocess")
            return False

        if not os.path.exists(vocab_path):
            print(f"\n[ERROR] 找不到词表文件: {vocab_path}")
            print("[INFO] 请先运行数据预处理：python fuzzmain.py --preprocess")
            return False

        print(f"[INFO] 预处理数据文件: {processed_data_path}")
        print(f"[INFO] 词表文件: {vocab_path}")

        # ========== 优化2：检查trainer模块是否存在 ==========
        trainer_module_path = os.path.join(project_root, "Data_processing", "trainer.py")
        if not os.path.exists(trainer_module_path):
            print(f"\n[ERROR] trainer.py模块不存在: {trainer_module_path}")
            print("\n[提示] CVAE训练模块尚未实现，需要根据Doc/prompt指导.md创建：")
            print("  - Seq2Seq CVAE架构（Encoder: Bi-GRU, Decoder: GRU）")
            print("  - Gumbel-Softmax重参数化（解决离散文本生成）")
            print("  - KL退火策略（防止Posterior Collapse）")
            print("  - 损失函数：Reconstruction Loss + β·KL Divergence")
            print("\n参考实现:")
            print(f"  数学定义: Doc/prompt指导.md 第2节")
            print(f"  超参数: epochs={args.epochs}, batch_size={args.batch_size}")
            print(f"          embed_dim={args.embed_dim}, hidden_dim={args.hidden_dim}")
            print(f"          latent_dim={args.latent_dim}, num_layers={args.num_layers}")
            return False

        # ========== 优化3：导入trainer模块 ==========
        try:
            from Data_processing.trainer import CVAETrainer
        except ImportError as e:
            print(f"\n[ERROR] 导入trainer模块失败: {e}")
            print(f"[INFO] trainer.py路径: {trainer_module_path}")
            return False

        # ========== 优化4：构建训练配置字典 ==========
        config = {
            # 数据路径
            'data_path': processed_data_path,
            'vocab_path': vocab_path,
            'output_dir': args.output_dir,

            # 模型架构参数
            'vocab_size': args.vocab_size,
            'embed_dim': args.embed_dim,
            'hidden_dim': args.hidden_dim,
            'latent_dim': args.latent_dim,
            'num_layers': args.num_layers,
            'condition_dim': args.condition_dim,

            # 训练超参数
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'train_split': args.train_split,
            'oversample': args.oversample,

            # KL退火参数
            'kl_cycles': args.kl_cycles,
            'beta_max': args.beta_max,
            'delay_epochs': args.delay_epochs,

            # Gumbel-Softmax温度参数
            'tau_init': 1.0,
            'tau_min': 0.5,
            'tau_decay': 0.99995,
        }

        print("\n[INFO] 训练配置参数:")
        print(f"  - 数据文件: {config['data_path']}")
        print(f"  - 词表文件: {config['vocab_path']}")
        print(f"  - 模型架构: vocab_size={config['vocab_size']}, embed_dim={config['embed_dim']}")
        print(f"              hidden_dim={config['hidden_dim']}, latent_dim={config['latent_dim']}")
        print(f"              num_layers={config['num_layers']}, condition_dim={config['condition_dim']}")
        print(f"  - 训练参数: epochs={config['epochs']}, batch_size={config['batch_size']}")
        print(f"              learning_rate={config['learning_rate']}, weight_decay={config['weight_decay']}")
        print(f"              train_split={config['train_split']}, oversample={config['oversample']}")
        print(f"  - KL退火: cycles={config['kl_cycles']}, beta_max={config['beta_max']}, delay_epochs={config['delay_epochs']}")

        # ========== 🔥 老王优化5：初始化trainer并开始训练 ==========
        print("\n[INFO] 初始化CVAE训练器...")
        trainer = CVAETrainer(config)

        print("[INFO] 开始训练CVAE模型...")
        print("[提示] 训练过程中监控以下指标:")
        print("  - Reconstruction Loss (重构损失，应该下降)")
        print("  - KL Divergence (KL散度，应该逐渐上升)")
        print("  - Total Loss (总损失，应该平稳下降)")
        print("  - Beta值 (KL权重，周期性变化)")

        # 调用训练方法
        history = trainer.train()

        # ========== 优化6：验证输出模型文件 ==========
        model_path = os.path.join(args.output_dir, "cvae.pth")
        if not os.path.exists(model_path):
            print(f"\n[WARNING] 训练完成但未找到模型文件: {model_path}")
            print("[INFO] 请检查trainer.py是否正确保存了模型")
        else:
            print(f"\n[SUCCESS] 模型训练完成！")
            print(f"[INFO] 模型文件: {model_path}")
            file_size = os.path.getsize(model_path)
            print(f"[INFO] 模型大小: {file_size:,} 字节 ({file_size/1024/1024:.2f} MB)")

        # 显示训练历史
        if history and 'final_loss' in history:
            print(f"\n[INFO] 训练总结:")
            print(f"  - 最终重构损失: {history.get('final_recon_loss', 'N/A'):.4f}")
            print(f"  - 最终KL散度: {history.get('final_kl_loss', 'N/A'):.4f}")
            print(f"  - 最终总损失: {history['final_loss']:.4f}")

        return True

    except Exception as e:
        print(f"\n[ERROR] CVAE训练失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def execute_generate(args):
    """执行载荷生成"""
    print("\n" + "=" * 60)
    print("阶段三：载荷生成")
    print("=" * 60)

    try:
        # ========== 优化1：检查必要文件是否存在 ==========
        model_path = os.path.join(args.output_dir, "cvae.pth")
        vocab_path = os.path.join(args.output_dir, "vocab.json")

        if not os.path.exists(model_path):
            print(f"\n[ERROR] 找不到CVAE模型文件: {model_path}")
            print("[INFO] 请先训练模型：python fuzzmain.py --train")
            return False

        if not os.path.exists(vocab_path):
            print(f"\n[ERROR] 找不到词表文件: {vocab_path}")
            print("[INFO] 请先运行数据预处理：python fuzzmain.py --preprocess")
            return False

        print(f"[INFO] CVAE模型文件: {model_path}")
        print(f"[INFO] 词表文件: {vocab_path}")

        # ========== 优化2：初始化CVAE生成器 ==========
        print("\n[INFO] 初始化CVAE生成器...")
        generator = CVAEGenerator(
            model_path=model_path,
            vocab_path=vocab_path,
            device='auto'
        )

        # ========== 优化3：处理攻击类型参数 ==========
        if ',' in args.attack_type:
            attack_types = [t.strip() for t in args.attack_type.split(',')]
        else:
            attack_types = args.attack_type

        print(f"[INFO] 生成参数:")
        print(f"  - 攻击类型: {attack_types}")
        print(f"  - 每类样本数: {args.num_samples}")
        print(f"  - 温度参数: {args.temperature}")
        print(f"  - 批处理大小: {args.generation_batch_size}")

        # ========== 优化4：生成载荷 ==========
        print("\n[INFO] 开始生成攻击载荷...")
        payloads, metadata = generator.generate_payloads(
            attack_types=attack_types,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_length=150,
            batch_size=args.generation_batch_size
        )

        print(f"\n[SUCCESS] 载荷生成完成！总计: {len(payloads)} 个")

        # ==========优化5：清洗载荷 ==========
        print("\n[INFO] 清洗无效载荷...")
        cleaned_payloads, cleaned_metadata = generator.clean_payloads(payloads, metadata)

        valid_ratio = len(cleaned_payloads) / len(payloads) * 100
        print(f"[INFO] 清洗完成！有效载荷: {len(cleaned_payloads)}/{len(payloads)} ({valid_ratio:.1f}%)")

        # ========== 优化6：提取隐空间特征 ==========
        print("\n[INFO] 提取隐空间特征...")
        embeddings, valid_mask = generator.get_embeddings(cleaned_payloads, cleaned_metadata)

        print(f"[INFO] 隐空间特征: {embeddings.shape}")
        print(f"[INFO] 有效特征数: {np.sum(valid_mask)}")

        # ========== 优化7：保存生成数据 ==========
        generated_dir = os.path.join(args.output_dir, "generated")
        print(f"\n[INFO] 保存生成数据到: {generated_dir}")
        generator.save_generated_data(cleaned_payloads, cleaned_metadata, generated_dir)

        # 保存隐空间特征和有效性掩码
        embeddings_file = os.path.join(generated_dir, "latent_embeddings.npy")
        valid_mask_file = os.path.join(generated_dir, "valid_mask.npy")

        np.save(embeddings_file, embeddings)
        np.save(valid_mask_file, valid_mask)

        print(f"[SUCCESS] 隐空间特征已保存: {embeddings_file}")
        print(f"[SUCCESS] 有效性掩码已保存: {valid_mask_file}")

        # ========== 优化8：显示生成统计 ==========
        print("\n[INFO] 生成统计:")
        type_counts = {}
        for meta in cleaned_metadata:
            attack_type = meta['type']
            type_counts[attack_type] = type_counts.get(attack_type, 0) + 1

        for attack_type, count in sorted(type_counts.items()):
            percentage = count / len(cleaned_metadata) * 100
            print(f"  - {attack_type:>8}: {count:>6} 条 ({percentage:>5.1f}%)")

        print("\n[SUCCESS] 载荷生成阶段完成！")
        return True

    except Exception as e:
        print(f"\n[ERROR] 载荷生成失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def execute_cluster(args):
    """执行DBSCAN聚类优化"""
    print("\n" + "=" * 60)
    print("阶段三续：DBSCAN聚类优化")
    print("=" * 60)

    try:
        # ========== 优化1：检查必要文件是否存在 ==========
        generated_dir = os.path.join(args.output_dir, "generated")

        embeddings_file = os.path.join(generated_dir, "latent_embeddings.npy")
        payloads_file = os.path.join(generated_dir, "raw_payloads.txt")
        metadata_file = os.path.join(generated_dir, "payload_metadata.json")
        valid_mask_file = os.path.join(generated_dir, "valid_mask.npy")

        # 检查文件是否存在，如果不存在则提示先运行生成
        missing_files = []
        if not os.path.exists(embeddings_file):
            missing_files.append("latent_embeddings.npy")
        if not os.path.exists(payloads_file):
            missing_files.append("raw_payloads.txt")
        if not os.path.exists(metadata_file):
            missing_files.append("payload_metadata.json")

        if missing_files:
            print(f"\n[ERROR] 找不到必要文件: {', '.join(missing_files)}")
            print(f"[INFO] 请先生成载荷：python fuzzmain.py --generate")
            return False

        print(f"[INFO] 隐空间特征文件: {embeddings_file}")
        print(f"[INFO] 载荷文件: {payloads_file}")
        print(f"[INFO] 元数据文件: {metadata_file}")

        # ========== 优化2：加载数据 ==========
        print("\n[INFO] 加载数据文件...")
        embeddings = np.load(embeddings_file)
        print(f"[INFO] 隐空间特征: {embeddings.shape}")

        with open(payloads_file, 'r', encoding='utf-8') as f:
            payloads = [line.strip() for line in f.readlines()]
        print(f"[INFO] 载荷数量: {len(payloads)}")

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"[INFO] 元数据数量: {len(metadata)}")

        # 加载有效性掩码（如果存在）
        valid_mask = None
        if os.path.exists(valid_mask_file):
            valid_mask = np.load(valid_mask_file)
            print(f"[INFO] 有效性掩码: {valid_mask.shape}")
            print(f"[INFO] 有效样本: {np.sum(valid_mask)} ({np.sum(valid_mask)/len(valid_mask)*100:.1f}%)")

        # ========== 优化3：初始化聚类器 ==========
        print("\n[INFO] 初始化CVAE聚类器...")
        clusterer = CVAEClusterer(
            embeddings=embeddings,
            payloads=payloads,
            metadata=metadata,
            valid_mask=valid_mask,
            label_weight=15.0  # 使用强标签权重进行类型隔离
        )

        # ========== 优化4：执行DBSCAN聚类 ==========
        print("\n[INFO] 开始DBSCAN聚类分析...")
        print(f"[INFO] 聚类参数:")
        print(f"  - eps: {args.eps if args.eps else '自动寻找'}")
        print(f"  - min_samples: {args.min_samples if args.min_samples else '3 (默认)'}")

        clustering_results = clusterer.perform_clustering(
            eps=args.eps,
            min_samples=args.min_samples
        )

        # 显示聚类结果
        print(f"\n[SUCCESS] 聚类完成！")
        print(f"[INFO] 聚类结果:")
        print(f"  - 簇数量: {clustering_results['n_clusters']}")
        print(f"  - 噪声点: {clustering_results['n_noise']} ({clustering_results['n_noise']/len(payloads)*100:.1f}%)")

        if clustering_results.get('silhouette_score'):
            print(f"  - 轮廓系数: {clustering_results['silhouette_score']:.3f}")

        # ========== 优化5：降维处理（用于可视化） ==========
        if args.visualize:
            print(f"\n[INFO] 执行降维处理 ({args.reduction_method})...")
            clusterer.reduce_dimensions(method=args.reduction_method)

        # ========== 优化6：筛选精锐载荷 ==========
        print("\n[INFO] 筛选精锐载荷...")
        print(f"[INFO] 筛选参数:")
        print(f"  - 每簇样本数: {args.samples_per_cluster}")
        print(f"  - 保留噪声点: {args.keep_noise}")

        refined_payloads = clusterer.select_refined_payloads(
            samples_per_cluster=args.samples_per_cluster,
            keep_all_noise=args.keep_noise
        )

        # 计算压缩比例
        reduction_ratio = (len(payloads) - len(refined_payloads)) / len(payloads) * 100
        print(f"\n[SUCCESS] 精锐载荷筛选完成！")
        print(f"[INFO] 筛选结果:")
        print(f"  - 原始样本: {len(payloads)}")
        print(f"  - 精锐样本: {len(refined_payloads)}")
        print(f"  - 压缩比例: {reduction_ratio:.1f}%")

        # ========== 优化7：保存聚类结果 ==========
        clustered_dir = os.path.join(args.output_dir, "clustered")
        print(f"\n[INFO] 保存聚类结果到: {clustered_dir}")
        clusterer.save_clustering_results(clustered_dir)

        # ========== 优化8：保存精锐载荷 ==========
        fuzzing_dir = os.path.join(args.output_dir, "fuzzing")
        os.makedirs(fuzzing_dir, exist_ok=True)

        refined_payloads_file = os.path.join(fuzzing_dir, "refined_payloads.txt")
        clusterer.save_refined_payloads(refined_payloads_file)
        print(f"[SUCCESS] 精锐载荷已保存: {refined_payloads_file}")

        # ========== 优化9：生成可视化图像（可选） ==========
        if args.visualize:
            print("\n[INFO] 生成聚类可视化图像...")
            viz_path = os.path.join(clustered_dir, "clustering_visualization.png")
            clusterer.visualize_clusters(
                method=args.reduction_method,
                save_path=viz_path
            )
            print(f"[SUCCESS] 可视化图像已保存: {viz_path}")

        # ========== 优化10：显示簇统计信息 ==========
        print("\n[INFO] 簇统计信息:")
        for cluster_id, cluster_info in clustering_results['cluster_info'].items():
            print(f"  - 簇 {cluster_id}: {cluster_info['size']} 个样本, "
                  f"平均距离: {cluster_info['avg_distance_to_centroid']:.3f}")

        print("\n[SUCCESS] DBSCAN聚类优化阶段完成！")
        print(f"[INFO] 精锐载荷已准备好用于Fuzz测试")
        print(f"[INFO] 载荷文件: {refined_payloads_file}")

        return True

    except Exception as e:
        print(f"\n[ERROR] DBSCAN聚类失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def execute_scan_init(args):
    """
    第四阶段：黑盒模糊测试调度函数

    老王注释：这个SB函数负责：
    1. 站点环境初始化（目录创建）
    2. 爬虫执行或缓存加载
    3. 任务统计和可视化
    4. 准备进入Fuzz引擎阶段

    Args:
        args: 命令行参数对象

    Returns:
        成功返回True，失败返回False
    """
    print("\n" + "=" * 60)
    print("阶段四：黑盒模糊测试")
    print("=" * 60)

    try:
        # ========== 优化1：导入爬虫模块 ==========
        try:
            sys.path.insert(0, os.path.join(project_root, "Fuzz"))
            from spider import CVDBSpider, extract_site_name
        except ImportError as e:
            print(f"\n[ERROR] 导入爬虫模块失败: {e}")
            print(f"[INFO] 请确保 Fuzz/spider.py 文件存在")
            return False

        # ========== 优化2：站点环境初始化 ==========
        site_name = extract_site_name(args.url)
        print(f"\n[INFO] 目标站点: {site_name}")
        print(f"[INFO] 基础URL: {args.url}")

        # 创建目录结构
        results_dir = Path("Data/scan_results") / site_name
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 结果目录: {results_dir}")

        # ========== 优化3：智能模式判断 ==========
        # 艹！提前定义参数过滤列表（包含HTTP头），用于Engine初始化
        param_filter_for_engine = []

        if not args.crawl:
            # ========== 纯扫描模式（不爬虫） ==========
            print(f"\n[INFO] 纯扫描模式：仅测试指定URL")
            print(f"[INFO] 目标URL: {args.url}")
            print(f"[INFO] 请求方法: {args.method}")

            # 导入FuzzTarget
            from Fuzz.spider import FuzzTarget

            # 手动构造单个FuzzTarget
            from urllib.parse import urlparse, parse_qs

            parsed = urlparse(args.url)

            params = {}
            data = {}


            # ========== 新增：支持--params手动指定参数 ==========
            if args.params:
                # 用户手动指定了测试参数
                # 支持两种格式：
                # 1. name1,name2（测试这些参数，值设为Fuzz）
                # 2. name1=value1,name2（name1固定为value1，name2测试设为Fuzz）
                manual_params = [p.strip() for p in args.params.split(',')]
                print(f"[INFO] 手动指定测试参数: {manual_params}")

                # 根据请求方法设置参数
                for param_def in manual_params:
                    if '=' in param_def:
                        # 有=号，解析名称和值
                        param_name, param_value = param_def.split('=', 1)
                        param_name = param_name.strip()
                        param_value = param_value.strip()
                    else:
                        # 没有=号，默认测试参数（值为Fuzz）
                        param_name = param_def
                        param_value = 'Fuzz'

                    # 根据请求方法放到不同的字典
                    if args.method == 'GET':
                        params[param_name] = param_value
                    else:  # POST
                        data[param_name] = param_value
            else:
                # 没有手动指定，自动解析
                if args.method == 'GET':
                    # GET请求：从URL中解析参数
                    if parsed.query:
                        params = {k: v[0] if v else '' for k, v in parse_qs(parsed.query).items()}
                else:  # POST
                    # POST请求：没有手动指定参数就警告
                    print(f"[WARNING] POST请求需要手动指定参数，如: --params id,name")
                    return False

            # 艹！新增：解析HTTP头注入列表
            injectable_headers = {}
            # 艹！同时收集要测试的参数名（用于param_filter）
            param_names_to_test = []
            if args.params:
                # 解析--params参数，提取参数名
                for param_def in args.params.split(','):
                    param_def = param_def.strip()
                    if '=' in param_def:
                        param_name = param_def.split('=', 1)[0].strip()
                    else:
                        param_name = param_def
                    param_names_to_test.append(param_name)

            # 艹！调试日志
            print(f"[DEBUG] args.headers = '{args.headers}'")
            print(f"[DEBUG] args.headers类型 = {type(args.headers)}")

            if args.headers and args.headers.strip():
                header_list = [h.strip() for h in args.headers.split(',')]
                print(f"[INFO] 指定HTTP头注入: {header_list}")

                # 从Requester获取默认头值
                from Fuzz.BaseFuzz.requester import Requester
                temp_requester = Requester(timeout=args.timeout)

                for header_name in header_list:
                    default_value = temp_requester.headers.get(header_name, '')
                    injectable_headers[header_name] = default_value
                    # 艹！把HTTP头也加入到要测试的参数列表中
                    param_names_to_test.append(header_name)
                    print(f"  - 添加HTTP头注入点: {header_name}={default_value[:30]}")
            else:
                print(f"[INFO] 未指定HTTP头注入")

            # 艹！保存到外部变量（用于Engine初始化）
            param_filter_for_engine = param_names_to_test

            # 艹！显示完整测试参数列表
            if param_names_to_test:
                print(f"[INFO] 完整测试参数列表: {param_names_to_test}")
            else:
                print(f"[INFO] 完整测试参数列表: 自动检测所有参数")

            # 构造target
            target = FuzzTarget(
                url=args.url,
                method=args.method,
                params=params,
                data=data,
                injectable_headers=injectable_headers,  # 新增
                depth=0
            )

            targets = [target]

            print(f"[INFO] 已构造扫描目标: {len(targets)} 个")
            print(f"  - [{target.method}] {target.url}")
            if target.params:
                print(f"    GET参数: {list(target.params.keys())}")
            if target.data:
                print(f"    POST参数: {list(target.data.keys())}")
            if target.injectable_headers:
                print(f"    HTTP头注入: {list(target.injectable_headers.keys())}")

        else:
            # ========== 爬虫+扫描模式 ==========
            print(f"[INFO] 爬虫+扫描模式：递归爬取后扫描")
            print(f"[INFO] 爬取深度: {args.depth}")

            # 创建缓存目录
            cache_dir = Path("Data/cache") / site_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] 缓存目录: {cache_dir}")

            # ========== 优化4：缓存加载逻辑 ==========
            cache_file = cache_dir / "spider_cache.json"

            if args.use_cache and cache_file.exists():
                # 复用缓存
                print(f"\n[INFO] 检测到已有缓存，正在加载...")
                print(f"[INFO] 缓存文件: {cache_file}")

                try:
                    spider = CVDBSpider.load_cache(str(cache_file))
                    print(f"[SUCCESS] 缓存加载成功！")
                    print(f"[INFO] 缓存时间: {spider.stats.get('timestamp', 'N/A')}")

                except Exception as e:
                    print(f"[ERROR] 缓存加载失败: {e}")
                    print(f"[INFO] 将重新执行爬取...")
                    args.use_cache = False

            # ========== 优化5：执行爬虫（如果需要） ==========
            if not args.use_cache or not cache_file.exists():
                # 初始化爬虫
                print(f"\n[INFO] 初始化CVDBSpider爬虫...")
                spider = CVDBSpider(
                    base_url=args.url,
                    max_depth=args.depth,
                    timeout=10,
                    cookie=args.cookie
                )

                # 执行爬取
                print(f"[INFO] 开始递归爬取...")
                targets = spider.crawl()

                # 保存缓存
                print(f"\n[INFO] 保存爬虫缓存...")
                cache_path = spider.save_cache(str(cache_dir))
                print(f"[SUCCESS] 缓存已保存: {cache_path}")
            else:
                targets = spider.targets

        # ========== 优化6：任务统计表格 ==========
        # 注意：targets在两种模式下都已经定义好了

        print(f"\n{'='*60}")
        print("任务统计表")
        print(f"{'='*60}")

        # 统计GET和POST任务
        get_targets = [t for t in targets if t.method == 'GET']
        post_targets = [t for t in targets if t.method == 'POST']

        # 按深度统计
        depth_stats = {}
        for target in targets:
            depth = target.depth
            depth_stats[depth] = depth_stats.get(depth, 0) + 1

        # 打印统计表格
        print(f"\n任务类型统计:")
        print(f"  - GET 任务:  {len(get_targets)} 个")
        print(f"  - POST 任务: {len(post_targets)} 个")
        print(f"  - 总任务数:  {len(targets)} 个")

        print(f"\n深度分布统计:")
        for depth in sorted(depth_stats.keys()):
            count = depth_stats[depth]
            percentage = count / len(targets) * 100
            print(f"  - 深度 {depth}:  {count} 个 ({percentage:.1f}%)")

        # 显示前5个目标示例
        if targets:
            print(f"\n目标示例（前5个）:")
            for i, target in enumerate(targets[:5], 1):
                params_info = f"参数: {len(target.params)}个" if target.params else f"字段: {len(target.data)}个"
                print(f"  {i}. [{target.method}] {target.url[:60]}... ({params_info})")

        # ========== 优化6：保存任务列表到文件 ==========
        targets_file = results_dir / "fuzz_targets.json"
        print(f"\n[INFO] 保存任务列表到: {targets_file}")

        targets_data = {
            'site_name': site_name,
            'base_url': args.url,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_targets': len(targets),
            'get_targets': len(get_targets),
            'post_targets': len(post_targets),
            'depth_stats': depth_stats,
            'targets': [target.to_dict() for target in targets]
        }

        with open(targets_file, 'w', encoding='utf-8') as f:
            json.dump(targets_data, f, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] 任务列表已保存")

        # ========== 优化7：模式判断 ==========
        # 只有--crawl（没有--scan）：纯爬虫模式，直接退出
        if args.crawl and not args.scan:
            print(f"\n{'='*60}")
            print(f"[SUCCESS] 纯爬虫模式完成！")
            print(f"[INFO] 已发现 {len(targets)} 个Fuzz目标")
            print(f"[INFO] 使用以下命令进入扫描模式：")
            print(f"        python fuzzmain.py --scan --url {args.url} --use-cache")
            print(f"{'='*60}\n")
            return True

        # 有--scan参数：进入Fuzz引擎
        if args.scan:
            # 扫描模式（可能带爬虫，也可能不带）
            if not args.crawl:
                print(f"\n{'='*60}")
                print(f"[INFO] 纯扫描模式：直接测试指定URL")
                print(f"[INFO] 待注入目标: {len(targets)} 个")
                print(f"{'='*60}\n")
            else:
                print(f"\n{'='*60}")
                print(f"[SUCCESS] 爬虫阶段完成！")
                print(f"[INFO] 目标已保存到缓存文件")
                print(f"[INFO] 待注入目标: {len(targets)} 个")
                print(f"{'='*60}\n")

            return True

    except Exception as e:
        print(f"\n[ERROR] 第四阶段执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def execute_basefuzz(args):
    """
    BaseFuzz引擎执行函数

    负责完整的BaseFuzz流程：
    1. 加载目标列表（从spider缓存或--url爬取）
    2. 初始化BaseFuzz Engine
    3. 执行模糊测试
    4. 生成分析报告

    Args:
        args: 命令行参数对象

    Returns:
        成功返回True，失败返回False
    """
    print("\n" + "=" * 70)
    print("BaseFuzz引擎 - 智能模糊测试")
    print("=" * 70)

    try:
        # ========== 步骤1：导入BaseFuzz模块 ==========
        print("\n[INFO] 导入BaseFuzz模块...")

        try:
            sys.path.insert(0, os.path.join(project_root, "Fuzz/BaseFuzz"))
            from Fuzz.BaseFuzz.engine import Engine
            from Fuzz.BaseFuzz.analysis import Analyzer, Reporter
        except ImportError as e:
            print(f"\n[ERROR] 导入BaseFuzz模块失败: {e}")
            print(f"[INFO] 请确保 Fuzz/BaseFuzz/engine.py 存在")
            return False

        # ========== 步骤2：获取目标列表 ==========
        print("\n[步骤1] 准备测试目标")
        print("-" * 70)

        targets = []

        if args.file:
            # 从文件加载spider缓存
            print(f"[INFO] 从缓存文件加载目标: {args.file}")

            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                # 导入FuzzTarget
                from Fuzz.spider import FuzzTarget

                # 从缓存中提取targets
                targets_data = cache_data.get('targets', [])
                targets = [FuzzTarget.from_dict(t) for t in targets_data]

                print(f"[SUCCESS] 已加载 {len(targets)} 个目标")

            except Exception as e:
                print(f"[ERROR] 缓存文件加载失败: {e}")
                return False

        elif args.url:
            # 先尝试加载缓存
            print(f"[INFO] 目标URL: {args.url}")

            # 检查是否有现有缓存
            from Fuzz.spider import extract_site_name
            from urllib.parse import urlparse, parse_qs
            from Fuzz.spider import FuzzTarget

            site_name = extract_site_name(args.url)
            cache_dir = Path("Data/cache") / site_name
            cache_file = cache_dir / "spider_cache.json"

            if cache_file.exists():
                print(f"[INFO] 发现现有缓存: {cache_file}")

                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)

                    targets_data = cache_data.get('targets', [])
                    targets = [FuzzTarget.from_dict(t) for t in targets_data]

                    print(f"[SUCCESS] 已从缓存加载 {len(targets)} 个目标")

                except Exception as e:
                    print(f"[WARNING] 缓存加载失败: {e}，将直接从URL创建目标")
                    targets = []
            else:
                print(f"[INFO] 未找到缓存文件，将直接从URL创建目标")
                targets = []

            # 如果缓存为空，直接从URL创建FuzzTarget
            if not targets:
                print(f"[INFO] 直接从URL创建测试目标...")

                try:
                    # 解析URL
                    parsed = urlparse(args.url)
                    query_params = parse_qs(parsed.query)

                    # parse_qs返回的值是列表，需要提取第一个值
                    # 例如：{'id': ['1'], 'Submit': ['提交']}
                    # 需要转换为：{'id': '1', 'Submit': '提交'}
                    params = {k: v[0] if v else '' for k, v in query_params.items()}

                    # 新增：支持--params手动指定参数！
                    # 支持语法：
                    # - name=value → 测试name参数，初始值为value
                    # - name=@value → 固定name参数为value，不测试（@前缀表示固定值）
                    # - name → 测试name参数，初始值为Fuzz
                    if args.params:
                        manual_params = [p.strip() for p in args.params.split(',')]
                        print(f"[INFO] 手动指定测试参数: {manual_params}")

                        # 根据请求方法设置参数
                        manual_data = {}
                        manual_params_dict = {}
                        for param_def in manual_params:
                            if '=' in param_def:
                                # 有=号，解析名称和值
                                param_name, param_value = param_def.split('=', 1)
                                param_name = param_name.strip()
                                param_value = param_value.strip()

                                # 检查@前缀（固定值，不测试）
                                if param_value.startswith('@'):
                                    # 去掉@前缀，保持原值
                                    param_value = param_value[1:]
                                    print(f"[INFO]   - {param_name} = {param_value} (固定值，不测试)")
                                else:
                                    print(f"[INFO]   - {param_name} = {param_value} (测试)")
                            else:
                                # 没有=号，默认测试参数（值为Fuzz）
                                param_name = param_def
                                param_value = 'Fuzz'
                                print(f"[INFO]   - {param_name} = {param_value} (测试)")

                            # 根据请求方法放到不同的字典
                            if args.method == 'GET':
                                manual_params_dict[param_name] = param_value
                            else:  # POST
                                manual_data[param_name] = param_value

                        # 手动参数优先，覆盖URL中的参数
                        if args.method == 'GET':
                            params = manual_params_dict
                        else:
                            params = {}  # POST请求，params应该为空
                            data = manual_data
                    else:
                        # 没有手动指定，使用URL中的参数
                        data = {}

                    # 判断请求方法
                    method = args.method.upper()

                    # 新增：解析HTTP头注入列表
                    injectable_headers = {}
                    if args.headers and args.headers.strip():
                        header_list = [h.strip() for h in args.headers.split(',')]
                        print(f"[INFO] 指定HTTP头注入: {header_list}")

                        # 从Requester获取默认头值
                        from Fuzz.BaseFuzz.requester import Requester
                        temp_requester = Requester(timeout=args.timeout)

                        for header_name in header_list:
                            default_value = temp_requester.headers.get(header_name, '')
                            injectable_headers[header_name] = default_value
                            print(f"[INFO]   - 添加HTTP头注入点: {header_name}={default_value[:30]}")
                    else:
                        print(f"[INFO] 未指定HTTP头注入")

                    # 构建FuzzTarget
                    if method == 'GET':
                        target = FuzzTarget(
                            url=args.url,
                            method=method,
                            params=params,
                            data={},
                            injectable_headers=injectable_headers,  # 新增HTTP头注入
                            depth=0  # 直接URL的深度设为0
                        )
                    else:  # POST
                        target = FuzzTarget(
                            url=args.url,
                            method=method,
                            params={},
                            data=data if args.params else {},
                            injectable_headers=injectable_headers,  # 新增HTTP头注入
                            depth=0  # 直接URL的深度设为0
                        )

                    targets = [target]
                    print(f"[SUCCESS] 已创建测试目标: {args.url}")
                    print(f"[INFO] 请求方法: {method}")

                    #显示参数信息（GET和POST分开处理）
                    if method == 'GET':
                        print(f"[INFO] GET参数数量: {len(params)}")
                        if params:
                            print(f"[INFO] GET参数列表: {', '.join(params.keys())}")
                    else:  # POST
                        print(f"[INFO] POST参数数量: {len(data)}")
                        if data:
                            print(f"[INFO] POST参数列表: {', '.join(data.keys())}")

                    #显示HTTP头注入信息
                    if injectable_headers:
                        print(f"[INFO] HTTP头注入: {', '.join(injectable_headers.keys())}")

                except Exception as e:
                    print(f"[ERROR] URL解析失败: {e}")
                    return False

        else:
            print("[ERROR] 必须指定 --file 参数（加载爬虫缓存）")
            print("[HINT] 正确用法:")
            print("  1. 先运行爬虫: python fuzzmain.py --crawl --url <url> --cookie <cookie>")
            print("  2. 再运行BaseFuzz: python fuzzmain.py --fuzz --engine base --mode common --file <cache_file>")
            return False

        # 验证目标列表
        if not targets:
            print("[ERROR] 没有可用的测试目标")
            return False

        # 显示目标统计
        print(f"\n[INFO] 目标统计:")
        print(f"  - 总目标数: {len(targets)}")
        get_count = sum(1 for t in targets if t.method == 'GET')
        post_count = sum(1 for t in targets if t.method == 'POST')
        print(f"  - GET任务: {get_count}")
        print(f"  - POST任务: {post_count}")

        # ========== 步骤3：初始化BaseFuzz Engine ==========
        print("\n[步骤2] 初始化BaseFuzz引擎")
        print("-" * 70)

        # 确定使用的引擎
        engine_names = ['sqli', 'xss']  # 目前支持SQLi和XSS

        # 解析参数过滤列表
        param_filter = None
        if args.params:
            # 修复：提取参数名（忽略值部分）
            # 支持语法：
            # - id=1 → 测试id
            # - id=@1 → 不测试id（@前缀表示固定值）
            # - id → 测试id
            param_filter = []
            for p in args.params.split(','):
                p = p.strip()
                if '=' in p:
                    # 有=号，解析名称和值
                    param_name, param_value = p.split('=', 1)
                    param_name = param_name.strip()
                    param_value = param_value.strip()

                    # 检查@前缀（固定值，不测试）
                    if param_value.startswith('@'):
                        # 跳过固定值参数
                        continue
                    else:
                        # 添加到测试列表
                        param_filter.append(param_name)
                else:
                    # 没有=号，就是参数名（测试）
                    param_filter.append(p)

            # 艹！新增：把HTTP头也加入到参数过滤列表中
            if args.headers:
                header_list = [h.strip() for h in args.headers.split(',')]
                param_filter.extend(header_list)

            if param_filter:
                print(f"[INFO] 参数过滤: 只测试指定的参数 -> {param_filter}")
            else:
                print(f"[INFO] 参数过滤: 所有参数都是固定值，无可测试参数")
                print(f"[WARNING] 警告：没有任何参数会被测试！")

        print(f"[INFO] 引擎配置:")
        print(f"  - 检测引擎: {', '.join(engine_names)}")
        print(f"  - 载荷模式: {args.mode}")
        print(f"  - 并发线程: {args.threads}")
        print(f"  - 超时时间: {args.timeout}秒")

        try:
            # 创建Engine实例
            engine = Engine(
                engine_names=engine_names,
                mode=args.mode,
                timeout=args.timeout,
                cookie=args.cookie,
                max_workers=args.threads,
                concurrent_params=10,  # 参数级并发数
                param_filter=param_filter  # 已包含HTTP头的参数过滤列表
            )

            print("[SUCCESS] BaseFuzz引擎初始化完成")

        except Exception as e:
            print(f"[ERROR] 引擎初始化失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return False

        # ========== 步骤4：执行模糊测试 ==========
        print("\n[步骤3] 执行模糊测试")
        print("-" * 70)
        print("[INFO] 开始扫描...")
        print("[提示] 按Ctrl+C可随时中断扫描")

        import time
        start_time = time.time()

        try:
            # 执行扫描
            results = engine.run(targets)

            elapsed_time = time.time() - start_time

            print(f"\n[SUCCESS] 扫描完成！")
            print(f"[INFO] 扫描耗时: {elapsed_time:.2f}秒")
            print(f"[INFO] 发现漏洞: {len(results)} 个")

        except KeyboardInterrupt:
            print("\n\n[WARNING] 用户中断扫描")
            print("[INFO] 正在保存已发现的结果...")

            # Engine会自动保存已发现的结果
            elapsed_time = time.time() - start_time

            return True

        except Exception as e:
            print(f"\n[ERROR] 扫描执行失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return False

        # ========== 步骤5：生成分析报告 ==========
        if results:
            print("\n[步骤4] 生成分析报告")
            print("-" * 70)

            try:
                # 分析结果（基于过滤后的vulnerabilities.json）
                analyzer = Analyzer()
                analyzed_results, stats = analyzer.analyze(
                    engine.results_file  # Engine保存的过滤后结果（无Error-Based）
                )

                print(f"[INFO] 分析完成:")
                print(f"  - 有效漏洞: {len(analyzed_results)} 条（已过滤Error-Based）")
                print(f"  - 高危漏洞: {stats.get('high_risk_count', 0)} 个")
                print(f"  - 中危漏洞: {stats.get('medium_risk_count', 0)} 个")
                print(f"  - 低危漏洞: {stats.get('low_risk_count', 0)} 个")
                print(f"  - 风险指数: {stats.get('risk_index', 0):.2f}")

                # 生成报告
                print(f"\n[INFO] 生成报告...")

                reporter = Reporter(output_dir=engine.output_dir)

                # 扫描信息
                scan_info = {
                    'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
                    'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_targets': len(targets),
                    'total_params_tested': sum(len(t.params) + len(t.data) for t in targets),
                    'engines_used': engine_names,
                    'mode': args.mode,
                    'engine_type': args.engine,
                }

                # 艹！读取所有漏洞（包括Error-Based）
                all_results_file = engine.output_dir / "vulnerabilities_all.json"
                if all_results_file.exists():
                    import json
                    with open(all_results_file, 'r', encoding='utf-8') as f:
                        all_vulns = json.load(f)
                    print(f"  - 总检测数: {len(all_vulns)} 条（含Error-Based）")
                else:
                    all_vulns = results  # 降级：使用原始results

                # 生成汇总报告（基于过滤后的结果）
                reporter.generate_summary(analyzed_results, stats, scan_info)

                #修改Reporter：生成详细报告时使用所有漏洞
                reporter.detail_file = engine.output_dir / "vulnerabilities_detail.json"
                reporter._generate_json_report(all_vulns, stats, scan_info)

                # 打印终端汇总
                reporter.print_console_summary(stats, scan_info)

                # 打印漏洞表格（Top 20）
                reporter.print_vulnerability_table(analyzed_results, top_n=20)

                # 获取报告文件路径
                report_files = reporter.get_report_files()

                print(f"\n[SUCCESS] 报告生成完成！")
                print(f"[INFO] 报告目录: {report_files['directory']}")
                print(f"[INFO] 汇总报告: {report_files['summary']}")
                print(f"[INFO] 详细报告: {report_files['detail']}")

            except Exception as e:
                print(f"[ERROR] 报告生成失败: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        else:
            print("\n[INFO] 未发现漏洞")

        print("\n" + "=" * 70)
        print("[SUCCESS] BaseFuzz扫描完成！")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n[ERROR] BaseFuzz执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()

    # 验证参数
    errors = validate_arguments(args)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return 1

    # 打印横幅
    print_banner()

    # 记录开始时间
    start_time = time.time()

    # 执行各个阶段
    success = True

    if args.preprocess:
        success = execute_preprocess(args) and success

    if args.train:
        success = execute_train(args) and success

    if args.generate:
        success = execute_generate(args) and success

    if args.cluster:
        success = execute_cluster(args) and success

    # ========== 新增：第四阶段调度 ==========
    if args.crawl or args.scan:
        success = execute_scan_init(args) and success

    # ========== 新增：BaseFuzz调度 ==========
    if args.fuzz:
        success = execute_basefuzz(args) and success

    # 统计总耗时
    elapsed_time = time.time() - start_time

    # 打印总结
    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] 所有任务执行完成！")
        print(f"[TIME] 总耗时: {elapsed_time:.2f}秒")
    else:
        print("[FAILED] 部分任务执行失败，请检查错误信息")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
