#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVAE 训练工具
============

实现损失函数、KL退火策略和训练指标


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import math


class CVAELoss(nn.Module):
    """CVAE 损失函数

    实现严格的损失函数：L = L_Recon + β * L_KL
    L_Recon 使用 CrossEntropy
    L_KL 使用高斯分布解析解
    """

    def __init__(self):
        super(CVAELoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=2)  # 忽略 PAD token (index=2)

    def reconstruction_loss(self, decoder_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        重构损失 (CrossEntropy)

        Args:
            decoder_output: [batch_size, seq_len, vocab_size] 解码器输出
            target: [batch_size, seq_len] 目标序列

        Returns:
            重构损失值
        """
        batch_size, seq_len, vocab_size = decoder_output.shape

        # 重塑为适合 CrossEntropyLoss 的形状
        # decoder_output: [batch_size * seq_len, vocab_size]
        # target: [batch_size * seq_len]
        decoder_output_flat = decoder_output.view(-1, vocab_size)
        target_flat = target.view(-1)

        # 计算 CrossEntropy 损失
        recon_loss = self.cross_entropy(decoder_output_flat, target_flat)

        return recon_loss

    def kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor, seq_len: float = 120.0) -> torch.Tensor:
        """
        🔥 彻底释放KL散度 - 无任何截断保护

        公式：L_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        允许KL损失自由波动，让编码器感受真实压力

        Args:
            mu: [batch_size, latent_dim] 均值
            logvar: [batch_size, latent_dim] 对数方差
            seq_len: 平均序列长度（浮点数），用于归一化KL损失量级

        Returns:
            完全自由的KL散度损失值
        """
        # 🔥 使用原始的高斯分布解析解公式，不加任何限制
        # L_KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # 取batch平均
        kl_loss = torch.mean(kl_loss)

        # 🔥 彻底无保护的归一化 - 确保使用浮点数除法
        seq_len_tensor = torch.tensor(seq_len, dtype=torch.float32, device=mu.device)
        kl_loss = kl_loss / seq_len_tensor

        # 🔥 完全移除所有限制！让KL损失可以自由增长到任何值！
        # 不再有torch.clamp，不再有硬编码限制！
        # 如果KL损失达到10.0、20.0甚至更高，那就让它达到！

        # 🔥 调试信息：确保KL损失没有被截断
        # print(f"DEBUG: Raw KL loss: {kl_loss.item():.6f}, mu_mean: {mu.mean().item():.6f}, logvar_mean: {logvar.mean().item():.6f}")

        return kl_loss

    def forward(self, decoder_output: torch.Tensor, target: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        Args:
            decoder_output: [batch_size, seq_len, vocab_size] 解码器输出
            target: [batch_size, seq_len] 目标序列
            mu: [batch_size, latent_dim] 均值
            logvar: [batch_size, latent_dim] 对数方差
            beta: KL 损失权重

        Returns:
            包含各种损失的字典
        """
        # 计算重构损失
        recon_loss = self.reconstruction_loss(decoder_output, target)

        # 🔥 确保使用浮点数序列长度 - 彻底无保护KL计算
        seq_len = float(decoder_output.size(1))  # 确保是浮点数！

        # 计算 KL 散度损失（完全无保护版本）
        kl_loss = self.kl_divergence_loss(mu, logvar, seq_len=seq_len)

        # 总损失
        total_loss = recon_loss + beta * kl_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta': beta
        }


class CyclicalAnnealingSchedule:
    """ 🔥 配置驱动的KL退火策略

    所有参数通过配置字典传入，移除硬编码默认值
    支持完全自定义的KL退火策略
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 包含所有KL退火参数的配置字典
        """
        # 🔥 强制要求配置参数，移除硬编码默认值
        self.total_steps = config['total_steps']
        self.n_cycles = config['n_cycles']
        self.ratio = config['ratio']
        self.beta_max = config['beta_max']
        self.delay_epochs = config['delay_epochs']
        self.steps_per_epoch = config['steps_per_epoch']

        # 计算延迟步数
        self.delay_steps = self.delay_epochs * self.steps_per_epoch

        # 计算延迟后的有效步数
        effective_steps = self.total_steps - self.delay_steps
        if effective_steps <= 0:
            self.steps_per_cycle = 1
            self.rise_steps = 1
        else:
            # 计算每个周期的步数
            self.steps_per_cycle = effective_steps // self.n_cycles
            self.rise_steps = int(self.steps_per_cycle * self.ratio)

        # 🔥 日志记录配置
        self._log_config()

    def _log_config(self):
        """记录KL退火配置"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🔥 KL退火配置:")
        logger.info(f"   总步数: {self.total_steps}")
        logger.info(f"   周期数: {self.n_cycles}")
        logger.info(f"   上升比例: {self.ratio}")
        logger.info(f"   最大Beta: {self.beta_max}")
        logger.info(f"   延迟轮数: {self.delay_epochs}")
        logger.info(f"   每轮步数: {self.steps_per_epoch}")

    def get_beta(self, step: int) -> float:
        """
        获取当前步骤的 β 值（优化版本）

        Args:
            step: 当前训练步数

        Returns:
            当前 β 值 [0, beta_max]
        """
        if step >= self.total_steps:
            return self.beta_max

        # 延迟阶段：前delay_steps步保持beta=0
        if step < self.delay_steps:
            return 0.0

        # 减去延迟步数，使用有效步数计算
        effective_step = step - self.delay_steps
        effective_total = self.total_steps - self.delay_steps

        if effective_step < 0 or effective_total <= 0:
            return 0.0

        # 计算当前在哪个周期中
        cycle = effective_step // self.steps_per_cycle
        step_in_cycle = effective_step % self.steps_per_cycle

        if step_in_cycle < self.rise_steps:
            # 上升阶段：β 从 0 线性增长到 beta_max
            progress = step_in_cycle / self.rise_steps
            return self.beta_max * progress
        else:
            # 平台阶段：β 保持为 beta_max
            return self.beta_max

    def get_beta_tensor(self, step: torch.Tensor) -> torch.Tensor:
        """
        获取 β 值的张量版本

        Args:
            step: 当前步数张量

        Returns:
            β 值张量
        """
        if isinstance(step, torch.Tensor):
            device = step.device
            step = step.item()
        else:
            device = torch.device('cpu')

        beta = self.get_beta(step)
        return torch.tensor(beta, device=device, dtype=torch.float32)


class CVAEMetrics:
    """CVAE 训练指标计算

    计算 Reconstruction Accuracy 和 Validity Rate
    """

    def __init__(self, vocab: Dict[str, int], pad_idx: int = 2, sos_idx: int = 0, eos_idx: int = 1, unk_idx: int = 3):
        """
        Args:
            vocab: 词表字典 {char: idx}
            pad_idx: PAD token 索引
            sos_idx: SOS token 索引
            eos_idx: EOS token 索引
            unk_idx: UNK token 索引（动态获取）
        """
        self.vocab = vocab
        self.idx_to_char = {idx: char for char, idx in vocab.items()}
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx  # 🔥 动态获取UNK索引

    def decode_sequence(self, sequence: torch.Tensor, visual_mode: bool = False) -> str:
        """
        🔥 重构：将索引序列解码为字符串，区分真实评估和视觉调试

        Args:
            sequence: [seq_len] 索引序列
            visual_mode: 是否为视觉模式（影响UNK处理方式）

        Returns:
            解码后的字符串
        """
        chars = []
        unk_count = 0
        total_chars = 0

        for idx in sequence:
            # 🔥 关键修复：强制转换为整数，解决Tensor匹配导致的"全问号"显示Bug
            idx_val = idx.item() if hasattr(idx, 'item') else idx

            if idx_val == self.eos_idx:
                break
            if idx_val not in [self.pad_idx, self.sos_idx]:
                total_chars += 1
                if idx_val == self.unk_idx:  # UNK token (使用动态索引)
                    unk_count += 1

                    if visual_mode:
                        # 🔥 视觉模式：随机替换为常见字符，便于观察
                        common_chars = ['a', 'e', 'i', 'o', 'u', '1', '0', '=', '\'', '"', ' ', '(', ')', '*', '+']
                        import random
                        if random.random() < 0.7:  # 70%概率替换为常见字符
                            chars.append(random.choice(common_chars))
                        else:
                            chars.append('?')  # 30%概率显示为?
                    else:
                        # 🔥 评估模式：保持UNK为?，确保指标客观性
                        chars.append('?')  # 严格保留UNK标识
                elif idx_val in self.idx_to_char:
                    chars.append(self.idx_to_char[idx_val])
                else:
                    chars.append('?')  # 真正的未知字符

        # 🔥 如果UNK字符占比过高，返回失败标记（仅在评估模式下）
        if not visual_mode and total_chars > 0 and unk_count / total_chars > 0.8:
            return '[GENERATION_FAILED]'  # 标记生成失败的样本

        return ''.join(chars)

    def reconstruction_accuracy(self, decoder_output: torch.Tensor, target: torch.Tensor) -> float:
        """
        🔥 修复：计算重构准确率，兼容2D/3D输入和序列长度不匹配问题

        Args:
            decoder_output:
                - 3D: [batch_size, seq_len, vocab_size] 解码器输出logits
                - 2D: [batch_size, seq_len] 已经是token_ids
            target: [batch_size, seq_len] 目标序列

        Returns:
            重构准确率 (0-1)
        """
        # 🔥 关键修复：检查输入维度，兼容2D/3D
        if decoder_output.dim() == 3:
            # 3D输入：logits，需要argmax
            predicted = torch.argmax(decoder_output, dim=-1)  # [batch_size, seq_len]
        elif decoder_output.dim() == 2:
            # 2D输入：已经是token_ids
            predicted = decoder_output
        else:
            raise ValueError(f"decoder_output维度错误: {decoder_output.dim()}D，期望2D或3D")

        # 🔥 处理序列长度不匹配问题
        pred_len = predicted.size(1) if predicted.dim() > 1 else predicted.size(0)
        target_len = target.size(1)

        if pred_len != target_len:
            # 截取到最小长度
            min_len = min(pred_len, target_len)
            predicted = predicted[:, :min_len]  # [batch_size, min_len]
            target = target[:, :min_len]  # [batch_size, min_len]

        # 计算每个位置的正确性
        correct = (predicted == target).float()

        # 创建掩码，忽略 PAD token
        mask = (target != self.pad_idx).float()

        # 计算准确率
        total_tokens = mask.sum()
        correct_tokens = (correct * mask).sum()

        if total_tokens == 0:
            return 1.0

        accuracy = (correct_tokens / total_tokens).item()
        return accuracy

    def validity_rate(self, decoder_output: torch.Tensor, target: torch.Tensor) -> float:
        """
        🔥 修复：计算有效载荷比例，兼容2D/3D输入

        一个载荷被认为是有效的，如果：
        1. 包含有效的语法结构
        2. 长度合理 (至少3个字符，去除特殊token后)
        3. 包含实际内容 (不只是特殊token)

        Args:
            decoder_output:
                - 3D: [batch_size, seq_len, vocab_size] 解码器输出logits
                - 2D: [batch_size, seq_len] 已经是token_ids
            target: [batch_size, seq_len] 目标序列

        Returns:
            有效载荷比例 (0-1)
        """
        # 🔥 关键修复：检查输入维度，兼容2D/3D
        if decoder_output.dim() == 3:
            # 3D输入：logits，需要argmax
            predicted = torch.argmax(decoder_output, dim=-1)  # [batch_size, seq_len]
        elif decoder_output.dim() == 2:
            # 2D输入：已经是token_ids
            predicted = decoder_output
        else:
            raise ValueError(f"decoder_output维度错误: {decoder_output.dim()}D，期望2D或3D")

        batch_size = predicted.size(0)
        valid_count = 0

        for i in range(batch_size):
            # 解码预测序列 - 🔥 使用评估模式，不使用visual_mode
            pred_seq = predicted[i]
            decoded = self.decode_sequence(pred_seq, visual_mode=False)  # 强制评估模式

            # 检查有效性
            if self._is_valid_payload(decoded):
                valid_count += 1

        validity_rate = valid_count / batch_size
        return validity_rate

    def _is_valid_payload(self, payload: str) -> bool:
        """
        检查载荷是否有效

        Args:
            payload: 解码后的载荷字符串

        Returns:
            是否有效
        """
        # 长度检查
        if len(payload) < 3:
            return False

        # 检查是否包含实际内容
        if payload.strip() == '':
            return False

        # 检查是否全是未知字符
        if payload.count('?') > len(payload) * 0.8:
            return False

        # 检查基本语法合理性（简单检查）
        # 对于 SQLi，应该包含一些常见字符
        sql_chars = ["'", '"', '=', '<', '>', ' ', ';', ',', '(', ')', '*', '/', '-']
        if any(char in payload for char in sql_chars):
            return True

        # 对于 XSS，应该包含标签相关字符
        xss_chars = ['<', '>', '/', '\\', '&']
        if any(char in payload for char in xss_chars):
            return True

        # 包含字母数字也算有效
        if any(char.isalnum() for char in payload):
            return True

        return False

    def calculate_metrics(self, decoder_output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        计算所有指标

        Args:
            decoder_output: [batch_size, seq_len, vocab_size] 解码器输出
            target: [batch_size, seq_len] 目标序列

        Returns:
            包含所有指标的字典
        """
        recon_acc = self.reconstruction_accuracy(decoder_output, target)
        validity = self.validity_rate(decoder_output, target)

        return {
            'reconstruction_accuracy': recon_acc,
            'validity_rate': validity
        }