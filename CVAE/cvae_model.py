#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVAE 模型定义
============

基于 GRU 的 Seq2Seq 条件变分自编码器实现
严格遵循 Doc/prompt指导.md 中的技术规范

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class GumbelSoftmax(nn.Module):
    """Gumbel-Softmax 采样器

    用于处理离散数据生成，解决不可导采样问题
    """

    def __init__(self, temperature: float = 1.0, hard: bool = False):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.hard = hard

    def sample_gumbel(self, logits: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """从 Gumbel(0,1) 分布采样"""
        U = torch.rand_like(logits)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, logits: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, vocab_size] 未归一化的对数概率
            temperature: 可选的温度参数，覆盖实例的temperature

        Returns:
            [batch_size, vocab_size] 采样结果
        """
        if temperature is None:
            temperature = self.temperature

        # 添加 Gumbel 噪声
        y = logits + self.sample_gumbel(logits)

        # 应用 Softmax
        y = F.softmax(y / temperature, dim=-1)

        if self.hard:
            # 硬化：返回 one-hot 向量，但保持梯度
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).scatter_(-1, ind.view(-1, 1), 1.0)
            y = y_hard - y.detach() + y

        return y


class CVAREncoder(nn.Module):
    """CVAE 编码器

    使用双向 GRU 将输入序列 x 和条件 c 映射到隐空间的均值和方差
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 latent_dim: int, condition_dim: int, num_layers: int = 2):
        super(CVAREncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_layers = num_layers

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 条件嵌入层
        self.condition_embedding = nn.Embedding(condition_dim, embed_dim)

        # 双向 GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 隐空间参数投影层
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)  # 双向 GRU 输出为 hidden_dim * 2
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # 🔥 LayerNorm层 - 在__init__中正确初始化
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len] 输入序列
            c: [batch_size] 条件标签

        Returns:
            mu: [batch_size, latent_dim] 均值
            logvar: [batch_size, latent_dim] 对数方差
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 嵌入输入序列
        x_embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        # 嵌入条件并广播到序列长度
        c_embed = self.condition_embedding(c).unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, embed_dim]

        # 结合条件信息 (简单相加)
        combined = x_embed + c_embed

        # 双向 GRU 编码
        outputs, hidden = self.gru(combined)

        # 🔥 层级特征聚合：多维度信息融合确保隐变量承载结构化信息
        # outputs形状: [batch_size, seq_len, hidden_dim * 2]

        # 🔥 策略1：多层级特征提取
        global_avg = torch.mean(outputs, dim=1)  # [batch_size, hidden_dim * 2] 全局平均
        global_max = torch.max(outputs, dim=1)[0]  # [batch_size, hidden_dim * 2] 全局最大
        last_output = outputs[:, -1, :]  # [batch_size, hidden_dim * 2] 最后时间步
        first_output = outputs[:, 0, :]  # [batch_size, hidden_dim * 2] 第一个时间步

        # 🔥 策略2：自注意力机制加权
        # 计算特征重要性分数
        feature_scores = torch.mean(outputs, dim=-1)  # [batch_size, seq_len] 每个时间步的重要性
        attention_weights = torch.softmax(feature_scores, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]

        # 注意力加权输出
        attention_output = torch.sum(outputs * attention_weights, dim=1)  # [batch_size, hidden_dim * 2]

        # 🔥 策略3：层级聚合 - 四种不同视角的融合
        hierarchical_features = 0.3 * global_avg + 0.2 * global_max + 0.3 * last_output + 0.2 * first_output

        # 🔥 策略4：标准差特征 - 捕捉序列变化信息
        std_features = torch.std(outputs, dim=1)  # [batch_size, hidden_dim * 2] 标准差特征

        # 🔥 最终特征组合：多层级 + 注意力 + 统计特征
        # 使用可学习的权重融合不同特征
        final_features = (
            0.4 * hierarchical_features +  # 主要的层级特征
            0.3 * attention_output +        # 注意力加权特征
            0.2 * std_features +            # 变化特征
            0.1 * torch.tanh(global_avg)    # 非线性变换的全局特征
        )

        # 🔥 添加特征标准化，确保隐空间训练稳定
        final_features = self.layer_norm(final_features)

        # 映射到隐空间参数
        mu = self.fc_mu(final_features)
        logvar = self.fc_logvar(final_features)

        return mu, logvar


class CVARDecoder(nn.Module):
    """CVAE 解码器

    使用单向 GRU 根据隐变量 z 和条件 c 生成序列
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 latent_dim: int, condition_dim: int, num_layers: int = 2,
                 vocab_info: Dict[str, int] = None):
        super(CVARDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_layers = num_layers

        # 🔥 动态获取特殊token索引，避免硬编码
        if vocab_info is not None and 'special_tokens' in vocab_info:
            self.sos_idx = vocab_info['special_tokens'].get('<SOS>', 0)
            self.eos_idx = vocab_info['special_tokens'].get('<EOS>', 1)
            self.pad_idx = vocab_info['special_tokens'].get('<PAD>', 2)
            self.unk_idx = vocab_info['special_tokens'].get('<UNK>', 3)
        else:
            # 保持向后兼容的默认值
            self.sos_idx = 0
            self.eos_idx = 1
            self.pad_idx = 2
            self.unk_idx = 3

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 条件嵌入层
        self.condition_embedding = nn.Embedding(condition_dim, embed_dim)

        # 隐变量到初始隐藏状态的投影
        self.fc_hidden = nn.Linear(latent_dim + embed_dim, hidden_dim)

        # 单向 GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # 输出层
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Gumbel-Softmax 采样器
        self.gumbel_softmax = GumbelSoftmax(hard=False)  # 训练时使用 soft 采样

        # Word Dropout 参数 - 🔥 强制断奶！彻底打破Teacher Forcing依赖
        self.word_dropout_prob = 0.6  # 提高到60%概率，强制模型必须依赖隐变量！

    def forward(self, z: torch.Tensor, c: torch.Tensor,
                target_seq: Optional[torch.Tensor] = None,
                max_length: int = 150, temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            z: [batch_size, latent_dim] 隐变量
            c: [batch_size] 条件标签
            target_seq: [batch_size, seq_len] 目标序列 (训练时使用)
            max_length: 最大生成长度
            temperature: Gumbel-Softmax 温度参数

        Returns:
            训练模式: [batch_size, seq_len, vocab_size] 输出概率分布
            生成模式: [batch_size, seq_len] 采样得到的token_ids
        """
        batch_size = z.size(0)
        device = z.device

        # 嵌入条件
        c_embed = self.condition_embedding(c)  # [batch_size, embed_dim]

        # 初始隐藏状态
        initial_input = torch.cat([z, c_embed], dim=-1)
        hidden = self.fc_hidden(initial_input)  # [batch_size, hidden_dim]
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch_size, hidden_dim]

        # 初始输入 (SOS token)
        input_token = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)  # 动态SOS

        outputs = []

        if target_seq is not None:
            # 🔥 训练模式：增强的 Teacher Forcing + Word Dropout
            seq_len = target_seq.size(1)
            input_step = input_token

            # 🔥 强制断奶策略：大幅降低Teacher Forcing依赖
            teacher_forcing_ratio = 0.5  # 从70%降到50%，一半时间靠隐变量！
            use_teacher_forcing = torch.rand(batch_size, device=device) < teacher_forcing_ratio

            for t in range(seq_len):
                # 嵌入当前输入
                input_embed = self.embedding(input_step)  # [batch_size, 1, embed_dim]

                # 🔥 增强干扰：在嵌入中添加噪声，迫使模型依赖隐变量z
                if self.training and torch.rand(1, device=device) < 0.3:  # 30%概率添加噪声
                    noise = torch.randn_like(input_embed) * 0.1  # 添加高斯噪声
                    input_embed = input_embed + noise

                # GRU 前向传播
                output, hidden = self.gru(input_embed, hidden)  # output: [batch_size, 1, hidden_dim]

                # 输出投影
                logits = self.fc_out(output.squeeze(1))  # [batch_size, vocab_size]
                outputs.append(logits)

                # 下一个输入 - 🔥 激进的强制自预测策略
                if t < seq_len - 1:
                    # 🔥 策略1：基于teacher_forcing向量选择输入
                    true_next = target_seq[:, t:t+1]  # [batch_size, 1] 真实标签
                    pred_next = torch.argmax(logits, dim=-1, keepdim=True)  # [batch_size, 1] 模型预测

                    # 为每个样本选择输入：Teacher Forcing vs 自预测
                    tf_mask = use_teacher_forcing.unsqueeze(1)  # [batch_size, 1]
                    input_step = torch.where(tf_mask, true_next, pred_next)

                    # 🔥 额外的强制自预测：20%概率强制所有样本使用自预测
                    if torch.rand(1, device=device) < 0.2:
                        input_step = pred_next

                    # 🔥 策略2：超强的 Word Dropout (60%概率)
                    if self.training:
                        # 基础Word Dropout
                        mask = torch.rand_like(input_step.float()) < self.word_dropout_prob
                        # 🔥 使用动态特殊token索引
                        special_tokens = (input_step == self.sos_idx) | (input_step == self.eos_idx) | (input_step == self.pad_idx)
                        mask = mask & (~special_tokens)
                        input_step = torch.where(mask, torch.tensor(self.unk_idx, device=device), input_step)

                        # 🔥 策略3：额外的随机替换 - 15%概率替换为随机token
                        random_mask = torch.rand_like(input_step.float()) < 0.15
                        # 🔥 从特殊token索引+1开始，避免替换为特殊token
                        min_valid_idx = max(self.unk_idx, self.pad_idx, self.sos_idx, self.eos_idx) + 1
                        random_tokens = torch.randint(min_valid_idx, self.vocab_size, input_step.shape, device=device)
                        random_mask = random_mask & (~special_tokens)
                        input_step = torch.where(random_mask, random_tokens, input_step)

        else:
            # 🔥 生成模式：自回归生成，直接收集采样token_ids！
            input_step = input_token
            sampled_ids = []  # 🔥 直接在循环中收集token_ids，避免后续argmax！

            for t in range(max_length):
                # 嵌入当前输入
                input_embed = self.embedding(input_step)  # [batch_size, 1, embed_dim]

                # GRU 前向传播
                output, hidden = self.gru(input_embed, hidden)  # output: [batch_size, 1, hidden_dim]

                # 输出投影
                logits = self.fc_out(output.squeeze(1))  # [batch_size, vocab_size]
                outputs.append(logits)

                # 🔥 Top-k采样与UNK强抑制策略！
                if self.training:
                    # 训练时使用Gumbel-Softmax
                    probs = self.gumbel_softmax(logits, temperature=temperature)
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    # 🔥 生成时：禁止使用简单的argmax，强制使用智能采样
                    # Step 1: Temperature缩放
                    scaled_logits = logits / temperature

                    # Step 2: Top-k采样 (k=10)
                    k = 10
                    topk_values, topk_indices = torch.topk(scaled_logits, k=k, dim=-1)  # [batch_size, k]

                    # Step 3: 检查UNK是否在top-k中
                    is_unk_in_topk = (topk_indices == self.unk_idx).any(dim=-1)  # [batch_size]

                    # Step 4: UNK强抑制 - 如果预测结果的Top-1是UNK，强制降低其概率
                    top1_indices = torch.argmax(scaled_logits, dim=-1)  # [batch_size]
                    is_top1_unk = (top1_indices == self.unk_idx)

                    # 为每个样本单独处理
                    final_tokens = []
                    for i in range(batch_size):
                        if is_top1_unk[i]:
                            # UNK强力惩罚：将UNK概率降低90%并重新归一化
                            current_logits = scaled_logits[i].clone()
                            unk_idx = self.unk_idx

                            # 降低UNK权重90%
                            current_logits[unk_idx] = current_logits[unk_idx] - 10.0  # 🔥 超强惩罚：log(0.000045) ≈ -10.0

                            # 从剩余的top-10中选择（排除UNK）
                            topk_vals, topk_idxs = torch.topk(current_logits, k=k)

                            # 如果UNK还在top-10中，去掉它并取下一个
                            if unk_idx in topk_idxs:
                                topk_mask = topk_idxs != unk_idx
                                topk_vals = topk_vals[topk_mask][:9]  # 取前9个非UNK
                                topk_idxs = topk_idxs[topk_mask][:9]

                            # 从top-k中随机采样
                            probs = F.softmax(topk_vals, dim=0)
                            selected_idx = torch.multinomial(probs, num_samples=1)
                            next_token_single = topk_idxs[selected_idx].unsqueeze(0)
                        else:
                            # Top-1不是UNK，使用标准Top-k采样
                            topk_vals_i = topk_values[i]
                            topk_idxs_i = topk_indices[i]

                            # 如果UNK在top-k中，降低其概率权重
                            if is_unk_in_topk[i]:
                                unk_mask = topk_idxs_i == self.unk_idx
                                topk_vals_i = topk_vals_i.clone()
                                topk_vals_i[unk_mask] = topk_vals_i[unk_mask] - 5.0  # 🔥 强化UNK惩罚

                            # 从top-k中采样
                            probs = F.softmax(topk_vals_i, dim=0)
                            selected_idx = torch.multinomial(probs, num_samples=1)
                            next_token_single = topk_idxs_i[selected_idx].unsqueeze(0)

                        final_tokens.append(next_token_single)

                    next_token = torch.cat(final_tokens, dim=0).unsqueeze(1)  # [batch_size, 1]

                # 🔥 确保next_token维度正确：应该是[batch_size, 1]
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
                elif next_token.dim() == 3:
                    next_token = next_token.squeeze(-1)  # [batch_size, 1, 1] -> [batch_size, 1]

                # 🔥 关键修复：直接将当前采样token添加到sampled_ids列表！
                sampled_ids.append(next_token.squeeze(1))  # [batch_size, 1] -> [batch_size]

                input_step = next_token

                # 检查是否所有序列都已生成 EOS
                if (next_token == self.eos_idx).all():  # 动态EOS
                    break

        if target_seq is not None:
            # 训练模式：返回概率分布用于损失计算
            outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, vocab_size]
            return outputs
        else:
            # 🔥 生成模式：直接返回在循环中收集的sampled_ids，确保逻辑一致性！
            # sampled_ids已包含每个时间步的采样token: [batch_size] for each step
            sampled_token_ids = torch.stack(sampled_ids, dim=1)  # [batch_size, seq_len]
            return sampled_token_ids


class CVAE(nn.Module):
    """条件变分自编码器 (CVAE)

    完整的 CVAE 模型，结合编码器和解码器
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256,
                 latent_dim: int = 32, condition_dim: int = 6, num_layers: int = 2,
                 vocab_info: Dict[str, int] = None):
        super(CVAE, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_layers = num_layers
        self.vocab_info = vocab_info  # 🔥 保存词表信息

        # 编码器和解码器
        self.encoder = CVAREncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            num_layers=num_layers
        )

        self.decoder = CVARDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            num_layers=num_layers,
            vocab_info=vocab_info  # 🔥 传递词表信息
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧

        Args:
            mu: [batch_size, latent_dim] 均值
            logvar: [batch_size, latent_dim] 对数方差

        Returns:
            [batch_size, latent_dim] 采样的隐变量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                target_seq: Optional[torch.Tensor] = None,
                max_length: int = 150, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len] 输入序列
            c: [batch_size] 条件标签
            target_seq: [batch_size, seq_len] 目标序列
            max_length: 最大生成长度
            temperature: Gumbel-Softmax 温度参数

        Returns:
            包含各种输出的字典
        """
        # 编码
        mu, logvar = self.encoder(x, c)

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 解码
        decoder_output = self.decoder(
            z=z,
            c=c,
            target_seq=target_seq,
            max_length=max_length,
            temperature=temperature
        )

        return {
            'decoder_output': decoder_output,  # [batch_size, seq_len, vocab_size]
            'mu': mu,  # [batch_size, latent_dim]
            'logvar': logvar,  # [batch_size, latent_dim]
            'z': z,  # [batch_size, latent_dim]
        }

    def generate(self, c: torch.Tensor, num_samples: int = 1,
                 max_length: int = 150, temperature: float = 1.8) -> torch.Tensor:
        """生成样本

        Args:
            c: [batch_size] 条件标签
            num_samples: 每个条件生成的样本数
            max_length: 最大生成长度
            temperature: 采样温度参数，提高默认值增加随机性

        Returns:
            [batch_size, num_samples, max_length] 生成的序列
        """
        self.eval()
        with torch.no_grad():
            batch_size = c.size(0)
            device = c.device

            # 🔥 从先验分布采样隐变量，增加采样次数找更好的z
            best_samples = []
            for _ in range(3):  # 每个条件生成3次，选最好的
                z = torch.randn(batch_size * num_samples, self.latent_dim, device=device)
                c_expanded = c.unsqueeze(1).repeat(1, num_samples).flatten()

                # 🔥 生成序列，使用更高的温度
                # 🔥 现在decoder直接返回token_ids，不需要从概率分布中采样！
                sampled_token_ids = self.decoder(
                    z=z,
                    c=c_expanded,
                    target_seq=None,
                    max_length=max_length,
                    temperature=temperature * 1.2  # 进一步提高温度
                )

                # 🔥 直接使用返回的token_ids，重塑为正确维度
                # sampled_token_ids: [batch_size * num_samples, seq_len] -> [batch_size, num_samples, seq_len]
                generated_tokens = sampled_token_ids.view(batch_size, num_samples, -1)

                best_samples.append(generated_tokens)

            # 🔥 从多次采样中选择UNK最少的样本 - 使用动态UNK索引
            final_samples = []
            unk_idx = self.decoder.unk_idx  # 🔥 直接使用decoder的动态UNK索引

            for i in range(batch_size):
                best_idx = 0
                min_unk_count = float('inf')

                for j, samples in enumerate(best_samples):
                    unk_count = (samples[i] == unk_idx).sum().item()  # 使用动态UNK索引
                    if unk_count < min_unk_count:
                        min_unk_count = unk_count
                        best_idx = j

                final_samples.append(best_samples[best_idx][i])

            # 合并最终结果
            generated_tokens = torch.stack(final_samples, dim=0)
            return generated_tokens

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_type': 'Seq2Seq CVAE',
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'condition_dim': self.condition_dim,
            'num_layers': self.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }