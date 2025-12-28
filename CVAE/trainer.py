#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVAE 训练器
===========

实现完整的训练循环、模型保存和指标监控
支持通过 python main.py --train 触发

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-18
"""

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm

from cvae_model import CVAE
from training_utils import CVAELoss, CyclicalAnnealingSchedule, CVAEMetrics
from data_loader import create_data_loaders
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import create_logger


class CVAETrainer:
    """CVAE 训练器

    实现完整的训练流程和模型管理
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 🔥 初始化老王牌训练日志记录器
        log_dir = os.path.join(self.config.get('output_dir', 'CVAE/checkpoints'), 'logs')
        self.training_logger = create_logger(log_dir=log_dir, simple=False)

        # 设置原有的基础日志
        self._setup_logging()

        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.kl_scheduler = None
        self.metrics = None
        self.scaler = None

        # 训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []

        # 初始化
        self._initialize()

        # 🔥 记录训练配置到详细日志
        self.training_logger.log_training_config(self.config)

        print(f"🎯 CVAE 训练器初始化完成")
        print(f"📱 设备：{self.device}")
        print(f"📊 模型参数：{sum(p.numel() for p in self.model.parameters()):,}")
        print(f"📝 详细日志：{self.training_logger.log_file}")

    def _setup_logging(self):
        """设置日志记录"""
        log_dir = Path(self.config['output_dir']) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _initialize(self):
        """初始化所有组件"""
        # 加载数据
        self.train_loader, self.val_loader, self.vocab_info = create_data_loaders(
            data_path=self.config['data_path'],
            vocab_path=self.config['vocab_path'],
            batch_size=self.config['batch_size'],
            train_split=self.config['train_split'],
            oversample=self.config['oversample'],
            num_workers=self.config['num_workers'],
            random_state=self.config['random_state']
        )

        # 🔥 创建模型（传递词表信息）
        self.model = CVAE(
            vocab_size=self.vocab_info['vocab_size'],
            embed_dim=self.config['embed_dim'],
            hidden_dim=self.config['hidden_dim'],
            latent_dim=self.config['latent_dim'],
            condition_dim=self.config['condition_dim'],
            num_layers=self.config['num_layers'],
            vocab_info=self.vocab_info  # 🔥 传递完整词表信息
        ).to(self.device)

        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config['learning_rate'] * 0.1
        )

        # 损失函数
        self.loss_fn = CVAELoss()

        # 🔥 KL 退火调度器（完全配置驱动版本）
        total_steps = len(self.train_loader) * self.config['epochs']
        steps_per_epoch = len(self.train_loader)

        # 🔥 构建KL退火配置字典 - 从配置中读取，不再硬编码！
        kl_config = {
            'total_steps': total_steps,
            'n_cycles': self.config.get('kl_cycles', 1),  # 🔥 从配置读取，默认1
            'ratio': self.config.get('kl_ratio', 0.6),  # 🔥 从配置读取，默认0.6
            'beta_max': self.config.get('beta_max', 0.25),  # 🔥 从配置读取，默认0.25
            'delay_epochs': self.config.get('delay_epochs', 20),  # 🔥 从配置读取，默认20
            'steps_per_epoch': steps_per_epoch
        }

        self.kl_scheduler = CyclicalAnnealingSchedule(kl_config)

        # 🔥 指标计算器（传递完整的特殊token索引）
        special_tokens = self.vocab_info['special_tokens']
        self.metrics = CVAEMetrics(
            vocab=self.vocab_info['char_to_idx'],
            pad_idx=special_tokens.get('<PAD>', 2),
            sos_idx=special_tokens.get('<SOS>', 0),
            eos_idx=special_tokens.get('<EOS>', 1),
            unk_idx=special_tokens.get('<UNK>', 3)  # 🔥 传递UNK索引
        )

        # 🔥 修复混合精度训练的废弃警告
        if self.config.get('use_amp', True) and self.device.type == 'cuda':
            from torch.amp import GradScaler, autocast
            self.scaler = GradScaler('cuda')
            self.logger.info("🔥 启用混合精度训练")
        else:
            self.scaler = None

    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_recon_acc = 0.0
        epoch_validity = 0.0

        # 进度条 - 老王我加上平滑显示和mininterval
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}",
                   mininterval=0.1, smoothing=0.1)

        try:
            for batch_idx, (sequences, labels) in enumerate(pbar):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                # 清零梯度
                self.optimizer.zero_grad()

                # 🔥 修复autocast导入和前向传播
                if self.scaler is not None:
                    from torch.amp import autocast
                    with autocast('cuda', enabled=self.scaler is not None):
                        # 使用序列自身作为目标 (训练时)
                        outputs = self.model(
                            x=sequences,
                            c=labels,
                            target_seq=sequences,
                            temperature=self.config.get('temperature', 1.0)
                        )

                        # 获取当前 beta 值
                        beta = self.kl_scheduler.get_beta(self.current_step)

                        # 计算损失
                        loss_dict = self.loss_fn(
                            decoder_output=outputs['decoder_output'],
                            target=sequences,
                            mu=outputs['mu'],
                            logvar=outputs['logvar'],
                            beta=beta
                        )

                        loss = loss_dict['total_loss']
                else:
                    # 不使用混合精度时
                    # 使用序列自身作为目标 (训练时)
                    outputs = self.model(
                        x=sequences,
                        c=labels,
                        target_seq=sequences,
                        temperature=self.config.get('temperature', 1.0)
                    )

                    # 获取当前 beta 值
                    beta = self.kl_scheduler.get_beta(self.current_step)

                    # 计算损失
                    loss_dict = self.loss_fn(
                        decoder_output=outputs['decoder_output'],
                        target=sequences,
                        mu=outputs['mu'],
                        logvar=outputs['logvar'],
                        beta=beta
                    )

                    loss = loss_dict['total_loss']

                # 反向传播
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # 更新统计
                batch_size = sequences.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_recon_loss += loss_dict['recon_loss'].item() * batch_size
                epoch_kl_loss += loss_dict['kl_loss'].item() * batch_size

                # 计算指标
                if batch_idx % self.config.get('metric_interval', 10) == 0:
                    with torch.no_grad():
                        metrics_dict = self.metrics.calculate_metrics(
                            decoder_output=outputs['decoder_output'],
                            target=sequences
                        )
                        epoch_recon_acc += metrics_dict['reconstruction_accuracy'] * batch_size
                        epoch_validity += metrics_dict['validity_rate'] * batch_size

                # 🔥 更新进度条，显示关键指标
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Beta': f"{beta:.3f}",
                    'Recon': f"{loss_dict['recon_loss'].item():.4f}",
                    'KL': f"{loss_dict['kl_loss'].item():.4f}"
                })

                self.current_step += 1

        except Exception as e:
            # 🔥 记录训练错误到老王日志
            error_msg = f"Epoch {self.current_epoch + 1} 训练出错: {str(e)}"
            self.training_logger.log_error(error_msg, e)
            self.logger.error(error_msg)
            raise e

        # 计算平均值
        num_samples = len(self.train_loader.dataset)
        avg_loss = epoch_loss / num_samples
        avg_recon_loss = epoch_recon_loss / num_samples
        avg_kl_loss = epoch_kl_loss / num_samples
        avg_recon_acc = epoch_recon_acc / (len(self.train_loader) // self.config.get('metric_interval', 10) * self.config['batch_size'])
        avg_validity = epoch_validity / (len(self.train_loader) // self.config.get('metric_interval', 10) * self.config['batch_size'])

        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
            'recon_accuracy': avg_recon_acc,
            'validity_rate': avg_validity,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def validate(self) -> Dict[str, float]:
        """验证模型 - 包含Teacher Forcing和Non-Teacher-Forcing测试"""
        self.model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        val_recon_acc = 0.0
        val_validity = 0.0

        # 🔥 新增：生成测试指标
        val_gen_acc = 0.0
        val_gen_validity = 0.0

        with torch.no_grad():
            for sequences, labels in tqdm(self.val_loader, desc="Validation"):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                # === 1. Teacher Forcing 验证（重构测试） ===
                tf_outputs = self.model(
                    x=sequences,
                    c=labels,
                    target_seq=sequences,  # Teacher Forcing模式
                    temperature=self.config.get('temperature', 1.0)
                )

                # 使用 beta = 1.0 进行验证
                beta = 1.0
                tf_loss_dict = self.loss_fn(
                    decoder_output=tf_outputs['decoder_output'],
                    target=sequences,
                    mu=tf_outputs['mu'],
                    logvar=tf_outputs['logvar'],
                    beta=beta
                )

                # 更新Teacher Forcing统计
                batch_size = sequences.size(0)
                val_loss += tf_loss_dict['total_loss'].item() * batch_size
                val_recon_loss += tf_loss_dict['recon_loss'].item() * batch_size
                val_kl_loss += tf_loss_dict['kl_loss'].item() * batch_size

                # 计算Teacher Forcing指标
                tf_metrics = self.metrics.calculate_metrics(
                    decoder_output=tf_outputs['decoder_output'],
                    target=sequences
                )
                val_recon_acc += tf_metrics['reconstruction_accuracy'] * batch_size
                val_validity += tf_metrics['validity_rate'] * batch_size

                # === 2. Non-Teacher-Forcing 验证（生成测试） ===
                gen_outputs = self.model(
                    x=sequences,
                    c=labels,
                    target_seq=None,  # 🔥 关键：不给目标序列，强制自回归生成
                    max_length=sequences.size(1),
                    temperature=self.config.get('temperature', 1.0)
                )

                # 🔥 计算生成指标（真实评估模式，不使用visual_mode）
                gen_metrics = self.metrics.calculate_metrics(
                    decoder_output=gen_outputs['decoder_output'],
                    target=sequences
                )
                val_gen_acc += gen_metrics['reconstruction_accuracy'] * batch_size
                val_gen_validity += gen_metrics['validity_rate'] * batch_size

        # 计算平均值
        num_samples = len(self.val_loader.dataset)
        avg_loss = val_loss / num_samples
        avg_recon_loss = val_recon_loss / num_samples
        avg_kl_loss = val_kl_loss / num_samples
        avg_recon_acc = val_recon_acc / num_samples
        avg_validity = val_validity / num_samples
        avg_gen_acc = val_gen_acc / num_samples
        avg_gen_validity = val_gen_validity / num_samples

        # 🔥 返回扩展的验证指标
        return {
            'val_loss': avg_loss,
            'val_recon_loss': avg_recon_loss,
            'val_kl_loss': avg_kl_loss,
            'val_recon_accuracy': avg_recon_acc,
            'val_validity_rate': avg_validity,
            # 🔥 新增：生成测试指标
            'val_gen_accuracy': avg_gen_acc,
            'val_gen_validity_rate': avg_gen_validity,
            'teacher_forcing_gap': avg_recon_acc - avg_gen_acc  # Teacher Forcing依赖度指标
        }

    def save_checkpoint(self, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'vocab_info': self.vocab_info,
            'training_history': self.training_history
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # 保存最新检查点
        checkpoint_path = Path(self.config['output_dir']) / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = Path(self.config['output_dir']) / 'cvae.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 保存最佳模型：{best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', [])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.logger.info(f"📂 加载检查点：{checkpoint_path}")
        self.logger.info(f"🎯 恢复到 epoch {self.current_epoch}, step {self.current_step}")

    def train(self):
        """完整训练流程"""
        self.logger.info("🚀 开始训练 CVAE 模型")
        self.logger.info(f"📊 训练配置：{self.config}")

        # 🔥 记录训练开始到老王日志
        self.training_logger.log_info("开始CVAE模型训练")

        # 记录训练开始时间
        start_time = time.time()

        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch

            # 🔥 记录epoch开始
            self.training_logger.log_epoch_start(epoch, self.config['epochs'])

            # 训练一个 epoch
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 学习率调度
            self.scheduler.step()

            # 更新最佳验证损失
            val_loss = val_metrics['val_loss']
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # 🔥 记录epoch指标到老王日志
            combined_metrics = {
                **train_metrics,
                **val_metrics,
                'beta': self.kl_scheduler.get_beta(self.current_step),
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_logger.log_epoch_metrics(epoch, combined_metrics)

            # 记录训练历史
            epoch_record = {
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'beta': self.kl_scheduler.get_beta(self.current_step),
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_record)

            # 🔥 如果是最佳模型，记录到老王日志
            if is_best:
                self.training_logger.log_best_model(epoch, val_metrics)

            # 🔥 生成调试样本并记录到老王日志
            generation_samples = self.get_debug_samples_dict(num_samples=5, max_length=50, temperature=1.5)
            self.training_logger.log_generation_samples(epoch, generation_samples)

            # 🔥 打印训练信息（包含生成测试指标）
            print(f"\n📈 Epoch {epoch + 1}/{self.config['epochs']} 完成")
            print(f"训练损失: {train_metrics['loss']:.4f}, 重构损失: {train_metrics['recon_loss']:.4f}, KL损失: {train_metrics['kl_loss']:.4f}")
            print(f"验证损失: {val_metrics['val_loss']:.4f}")
            print(f"📊 Teacher Forcing: 重构准确率={val_metrics['val_recon_accuracy']:.4f}, 有效率={val_metrics['val_validity_rate']:.4f}")
            print(f"🎲 真实生成: 准确率={val_metrics['val_gen_accuracy']:.4f}, 有效率={val_metrics['val_gen_validity_rate']:.4f}")
            print(f"⚠️  Teacher-Forcing依赖度: {val_metrics['teacher_forcing_gap']:.4f} (越小越好)")
            print(f"学习率: {train_metrics['learning_rate']:.6f}, Beta: {epoch_record['beta']:.3f}")

            # 🔥 生成调试样本（每个epoch结束时强制打印前5个生成样本）
            print(f"\n🎲 生成调试样本（Epoch {epoch + 1}）：")
            self.debug_generate_samples(num_samples=5, max_length=50, temperature=1.5)

            # 保存检查点
            self.save_checkpoint(is_best)

            # 早停检查
            if self.config.get('early_stopping', False):
                patience = self.config.get('patience', 10)
                if len(self.training_history) > patience:
                    recent_losses = [h['val_metrics']['val_loss'] for h in self.training_history[-patience:]]
                    if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                        self.logger.info(f"⏰ 早停触发，在 epoch {epoch + 1}")
                        self.training_logger.log_info(f"早停触发，在 epoch {epoch + 1}")
                        break

        # 训练完成
        total_time = time.time() - start_time
        self.logger.info(f"🎉 训练完成！总时间：{total_time:.2f} 秒")
        self.logger.info(f"🏆 最佳验证损失：{self.best_val_loss:.4f}")

        # 🔥 记录训练完成到老王日志
        self.training_logger.log_training_complete(self.config['epochs'])

        # 保存最终模型
        final_path = Path(self.config['output_dir']) / 'cvae_final.pth'
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"💾 最终模型已保存：{final_path}")

        return self.training_history

    def generate_samples(self, num_samples: int = 10, max_length: int = 150) -> None:
        """生成样本用于测试"""
        self.model.eval()

        # 从验证集中获取条件标签
        all_labels = []
        for _, labels in self.val_loader:
            all_labels.extend(labels.tolist())
            if len(all_labels) >= num_samples:
                break

        all_labels = torch.tensor(all_labels[:num_samples]).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                c=all_labels,
                num_samples=1,
                max_length=max_length,
                temperature=1.0
            )

        # 解码并打印生成的样本
        print("\n🎲 生成的样本：")
        class_names = ['SQLi', 'XSS', 'CMDi', 'Overflow', 'XXE', 'SSI']

        for i in range(num_samples):
            label_idx = int(all_labels[i])
            class_name = class_names[label_idx] if label_idx < len(class_names) else f"Class_{label_idx}"

            # 解码序列
            sequence = generated[i, 0]  # [seq_len]
            decoded = self.metrics.decode_sequence(sequence)

            print(f"({i+1}) [{class_name}]: {decoded}")

    def debug_generate_samples(self, num_samples: int = 5, max_length: int = 50, temperature: float = 1.5):
        """生成调试样本用于观察模型是否正常工作

        强制在每个epoch结束后生成样本，观察是否出现重复乱码

        Args:
            num_samples: 生成样本数量
            max_length: 最大生成长度
            temperature: 采样温度参数，控制随机性
        """
        self.model.eval()

        # 使用多种攻击类型进行生成测试
        test_labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=self.device)  # SQLi, XSS, CMDi, Overflow, XXE

        with torch.no_grad():
            for i in range(min(num_samples, len(test_labels))):
                label = test_labels[i:i+1]  # 保持2D形状 [1, 1]

                # 🔥 生成样本（使用更高的温度参数增加随机性）
                generated = self.model.generate(
                    c=label,
                    num_samples=1,
                    max_length=max_length,
                    temperature=temperature
                )

                # 解码序列（使用visual_mode便于观察）
                sequence = generated[0, 0]  # [seq_len]
                decoded = self.metrics.decode_sequence(sequence, visual_mode=True)  # 🔥 视觉模式

                # 获取攻击类型名称
                class_names = ['SQLi', 'XSS', 'CMDi', 'Overflow', 'XXE', 'SSI']
                class_name = class_names[int(label.item())] if int(label.item()) < len(class_names) else f"Class_{int(label.item())}"

                # 检查是否为重复字符或乱码
                is_repetitive = len(set(decoded)) < 3 if len(decoded) > 5 else False
                is_empty = len(decoded.strip()) == 0

                status = ""
                if is_repetitive:
                    status = " [重复字符]"
                elif is_empty:
                    status = " [空输出]"

                print(f"  样本{i+1} [{class_name}]: '{decoded}'{status}")

    def get_debug_samples_dict(self, num_samples: int = 5, max_length: int = 50, temperature: float = 1.5) -> Dict[str, str]:
        """获取调试样本字典，用于日志记录

        Args:
            num_samples: 生成样本数量
            max_length: 最大生成长度
            temperature: 采样温度

        Returns:
            样本字典 {攻击类型: 生成样本}
        """
        self.model.eval()
        samples_dict = {}

        # 使用多种攻击类型进行生成测试
        test_labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=self.device)  # SQLi, XSS, CMDi, Overflow, XXE
        class_names = ['SQLi', 'XSS', 'CMDi', 'Overflow', 'XXE', 'SSI']

        with torch.no_grad():
            for i in range(min(num_samples, len(test_labels))):
                label = test_labels[i:i+1]  # 保持2D形状 [1, 1]

                # 生成样本
                generated = self.model.generate(
                    c=label,
                    num_samples=1,
                    max_length=max_length,
                    temperature=temperature
                )

                # 解码序列（使用visual_mode便于观察）
                sequence = generated[0, 0]  # [seq_len]
                decoded = self.metrics.decode_sequence(sequence, visual_mode=True)  # 视觉模式

                # 获取攻击类型名称
                class_name = class_names[int(label.item())] if int(label.item()) < len(class_names) else f"Class_{int(label.item())}"

                # 记录到字典
                samples_dict[class_name] = decoded

        return samples_dict


def create_default_config() -> Dict[str, Any]:
    """创建默认训练配置"""
    return {
        # 数据配置
        'data_path': 'Data/processed/processed_data.pt',
        'vocab_path': 'Data/processed/vocab.json',
        'output_dir': 'CVAE/checkpoints',

        # 模型配置
        'embed_dim': 128,
        'hidden_dim': 256,
        'latent_dim': 32,
        'condition_dim': 6,
        'num_layers': 2,

        # 训练配置
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'train_split': 0.8,
        'oversample': True,
        'num_workers': 0,
        'random_state': 42,

        # 🔥 训练策略配置（单一稳定增长版本）
        'use_amp': True,
        'temperature': 1.3,  # 进一步提高temperature增加随机性
        'kl_cycles': 1,  # 🔥 单一周期，稳定增长不重置！
        'kl_ratio': 0.6,  # 更快进入有效约束阶段
        'beta_max': 0.25,  # 🔥 提高约束力
        'delay_epochs': 20,  # 🔥 延迟20个epoch
        'metric_interval': 10,

        # 早停配置
        'early_stopping': True,
        'patience': 15
    }