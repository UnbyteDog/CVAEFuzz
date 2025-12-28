#!/usr/bin/env python3
"""
🔥 运行日志记录工具 - 老王出品
专门记录CVAE模型训练过程中的所有信息，包括：
- 训练配置参数
- 每个epoch的训练指标
- 生成样本示例
- 错误信息和异常情况
"""

import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import traceback

class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir: str = "logs"):
        """
        初始化日志记录器

        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"cvae_training_{timestamp}.log")

        # 配置logging
        self._setup_logging()

        # 训练配置
        self.training_config = {}

        # 历史记录
        self.training_history = []

        # 开始日志
        self.logger.info("=" * 60)
        self.logger.info("🔥 CVAE模型训练日志记录器启动")
        self.logger.info(f"📁 日志文件: {self.log_file}")
        self.logger.info("=" * 60)

    def _setup_logging(self):
        """设置logging配置"""
        # 创建logger
        self.logger = logging.getLogger('CVAE_Training')
        self.logger.setLevel(logging.INFO)

        # 清除已有的handler
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 文件handler
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加handler
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_training_config(self, config: Dict[str, Any]):
        """记录训练配置"""
        self.training_config = config.copy()

        self.logger.info("📋 训练配置参数:")
        self.logger.info(json.dumps(config, indent=2, ensure_ascii=False))

        # 保存配置到单独的JSON文件
        config_file = self.log_file.replace('.log', '_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 配置已保存到: {config_file}")

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.logger.info(f"\n🚀 开始训练 Epoch {epoch+1}/{total_epochs}")
        self.logger.info("-" * 50)

    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """记录epoch指标"""
        self.logger.info(f"📊 Epoch {epoch+1} 训练指标:")

        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"   {key}: {value:.6f}")
            else:
                self.logger.info(f"   {key}: {value}")

        # 添加到历史记录
        epoch_record = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        self.training_history.append(epoch_record)

    def log_generation_samples(self, epoch: int, samples: Dict[str, str]):
        """记录生成样本"""
        self.logger.info(f"🎲 Epoch {epoch+1} 生成样本:")
        for attack_type, sample in samples.items():
            self.logger.info(f"   [{attack_type}]: {sample}")

    def log_validation_metrics(self, epoch: int, val_metrics: Dict[str, Any]):
        """记录验证指标"""
        self.logger.info(f"✅ Epoch {epoch+1} 验证指标:")

        for key, value in val_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"   {key}: {value:.6f}")
            else:
                self.logger.info(f"   {key}: {value}")

    def log_best_model(self, epoch: int, metrics: Dict[str, Any]):
        """记录最佳模型保存"""
        self.logger.info(f"🏆 最佳模型更新! Epoch {epoch+1}")
        self.logger.info(f"   最佳验证损失: {metrics.get('val_loss', 'N/A')}")
        self.logger.info(f"   重构准确率: {metrics.get('val_recon_accuracy', 'N/A')}")

    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """记录错误信息"""
        self.logger.error(f"❌ 错误: {error_msg}")

        if exception:
            self.logger.error(f"异常详情:\n{traceback.format_exc()}")

    def log_warning(self, warning_msg: str):
        """记录警告信息"""
        self.logger.warning(f"⚠️  警告: {warning_msg}")

    def log_info(self, info_msg: str):
        """记录一般信息"""
        self.logger.info(f"ℹ️  信息: {info_msg}")

    def save_training_history(self):
        """保存训练历史"""
        history_file = self.log_file.replace('.log', '_history.json')

        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump({
                'training_config': self.training_config,
                'training_history': self.training_history,
                'total_epochs': len(self.training_history)
            }, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 训练历史已保存到: {history_file}")

    def log_training_complete(self, total_epochs: int):
        """记录训练完成"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🎉 CVAE模型训练完成!")
        self.logger.info(f"📊 总训练轮数: {total_epochs}")
        self.logger.info(f"📁 完整日志: {self.log_file}")
        self.logger.info("=" * 60)

        # 保存完整历史
        self.save_training_history()

class SimpleLogger:
    """简化版日志记录器，用于快速测试"""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message: str):
        """简单记录日志"""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

# 便捷函数
def create_logger(log_dir: str = "logs", simple: bool = False) -> Any:
    """
    创建日志记录器

    Args:
        log_dir: 日志目录
        simple: 是否使用简化版

    Returns:
        日志记录器实例
    """
    if simple:
        return SimpleLogger()
    else:
        return TrainingLogger(log_dir)