#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVAE 训练主程序
==============

支持通过命令行参数启动CVAE训练
可通过 python main.py --train 触发

"""

import sys
import os
import argparse
from pathlib import Path

# 解决Windows中文显示问题
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from trainer import CVAETrainer, create_default_config
except ImportError as e:
    print(f"[ERROR] 导入CVAE模块失败：{e}")
    print("请确保CVAE模块文件存在")
    sys.exit(1)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CVAE 模型训练工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
    python main.py --train
    python main.py --train --epochs 100 --batch-size 64
    python main.py --train --embed-dim 256 --hidden-dim 512
        """
    )

    # 训练参数
    parser.add_argument(
        '--train',
        action='store_true',
        help='启动CVAE模型训练'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='../Data/processed/processed_data.pt',
        help='训练数据文件路径 (默认: ../Data/processed/processed_data.pt)'
    )

    parser.add_argument(
        '--vocab-path',
        type=str,
        default='../Data/processed/vocab.json',
        help='词表文件路径 (默认: ../Data/processed/vocab.json)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='CVAE/checkpoints',
        help='模型输出目录 (默认: CVAE/checkpoints)'
    )

    # 模型参数
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

    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='GRU层数 (默认: 2)'
    )

    # 训练超参数
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
        '--weight-decay',
        type=float,
        default=1e-5,
        help='权重衰减 (默认: 1e-5)'
    )

    # 数据处理参数
    parser.add_argument(
        '--no-oversample',
        action='store_true',
        help='禁用过采样'
    )

    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='训练集比例 (默认: 0.8)'
    )

    # 其他参数
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )

    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='禁用混合精度训练'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Gumbel-Softmax温度参数 (默认: 1.0)'
    )

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
        '--early-stopping',
        action='store_true',
        help='启用早停'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='早停耐心值 (默认: 15)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        help='从检查点恢复训练'
    )

    parser.add_argument(
        '--generate-samples',
        type=int,
        default=10,
        help='训练后生成样本数量 (默认: 10)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出'
    )

    return parser.parse_args()


def main():
    """主函数"""
    parser = parse_arguments()

    if not parser.train:
        parser.print_help()
        return

    print("=" * 80)
    print("🧠 CVAE 模型训练工具")
    print("基于 GRU 的 Seq2Seq 条件变分自编码器")
    print("=" * 80)

    # 创建配置
    config = create_default_config()

    # 根据命令行参数更新配置
    config.update({
        'data_path': parser.data_path,
        'vocab_path': parser.vocab_path,
        'output_dir': parser.output_dir,
        'embed_dim': parser.embed_dim,
        'hidden_dim': parser.hidden_dim,
        'latent_dim': parser.latent_dim,
        'num_layers': parser.num_layers,
        'epochs': parser.epochs,
        'batch_size': parser.batch_size,
        'learning_rate': parser.learning_rate,
        'weight_decay': parser.weight_decay,
        'oversample': not parser.no_oversample,
        'train_split': parser.train_split,
        'random_state': parser.seed,
        'use_amp': not parser.no_amp,
        'temperature': parser.temperature,
        'kl_cycles': parser.kl_cycles,
        'early_stopping': parser.early_stopping,
        'patience': parser.patience,
        'verbose': parser.verbose
    })

    # 检查数据文件是否存在
    data_path = Path(config['data_path'])
    vocab_path = Path(config['vocab_path'])

    if not data_path.exists():
        print(f"❌ 数据文件不存在：{data_path}")
        print("请先运行数据预处理：python main.py --preprocess")
        return

    if not vocab_path.exists():
        print(f"❌ 词表文件不存在：{vocab_path}")
        print("请先运行数据预处理：python main.py --preprocess")
        return

    # 创建输出目录
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 创建训练器
        trainer = CVAETrainer(config)

        # 🔥 无感自动恢复：如果用户没有指定--resume，自动检查最新检查点
        if not parser.resume:
            output_dir = Path(config['output_dir'])
            latest_checkpoint = output_dir / 'checkpoint_latest.pth'
            if latest_checkpoint.exists():
                trainer.load_checkpoint(str(latest_checkpoint))
                print(f"🔄 自动恢复：从最新检查点恢复训练")
                print(f"📂 检查点路径：{latest_checkpoint}")

        # 从检查点恢复训练（如果指定）
        if parser.resume:
            if Path(parser.resume).exists():
                trainer.load_checkpoint(parser.resume)
                print(f"📂 从检查点恢复：{parser.resume}")
            else:
                print(f"⚠️ 检查点文件不存在：{parser.resume}，从头开始训练")

        # 开始训练
        print(f"\n🚀 开始训练 CVAE 模型")
        print(f"📁 数据路径：{config['data_path']}")
        print(f"💾 输出目录：{config['output_dir']}")
        print(f"⚙️  模型配置：embed_dim={config['embed_dim']}, hidden_dim={config['hidden_dim']}, latent_dim={config['latent_dim']}")
        print(f"🎯 训练配置：epochs={config['epochs']}, batch_size={config['batch_size']}, lr={config['learning_rate']}")

        training_history = trainer.train()

        # 生成样本
        if parser.generate_samples > 0:
            print(f"\n🎲 生成 {parser.generate_samples} 个样本...")
            trainer.generate_samples(num_samples=parser.generate_samples)

        print("\n" + "=" * 80)
        print("🎉 CVAE 训练完成！")
        print(f"💾 模型已保存到：{config['output_dir']}/cvae.pth")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 训练失败：{e}")
        import traceback
        traceback.print_exc()
        return

    # 打印最终训练统计
    if 'training_history' in locals() and training_history:
        print(f"\n📊 训练统计：")
        best_epoch = min(range(len(training_history)),
                        key=lambda i: training_history[i]['val_metrics']['val_loss'])
        best_metrics = training_history[best_epoch]['val_metrics']

        print(f"   最佳 epoch: {best_epoch + 1}")
        print(f"   最佳验证损失: {best_metrics['val_loss']:.4f}")
        print(f"   最佳重构准确率: {best_metrics['val_recon_accuracy']:.4f}")
        print(f"   最佳有效率: {best_metrics['val_validity_rate']:.4f}")


if __name__ == "__main__":
    main()