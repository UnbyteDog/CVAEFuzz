#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVAE 数据加载器
==============

实现数据加载、过采样和批处理功能
支持从 processed_data.pt 和 vocab.json 加载数据

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-18
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import random


class CVADDataset(Dataset):
    """CVAE 训练数据集

    支持过采样来缓解类别不平衡问题
    """

    def __init__(self, data_path: str, vocab_path: str,
                 oversample: bool = True, random_state: int = 42):
        """
        Args:
            data_path: processed_data.pt 文件路径
            vocab_path: vocab.json 文件路径
            oversample: 是否启用过采样
            random_state: 随机种子
        """
        self.data_path = Path(data_path)
        self.vocab_path = Path(vocab_path)
        self.oversample = oversample
        self.random_state = random_state

        # 设置随机种子
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # 加载数据和词表
        self.data, self.labels, self.vocab = self._load_data()
        self.original_length = len(self.data)

        # 过采样处理
        if self.oversample:
            self.data, self.labels = self._apply_oversampling()

        print(f"📊 数据集加载完成：")
        print(f"   原始样本数：{self.original_length}")
        print(f"   过采样后样本数：{len(self.data)}")
        print(f"   词表大小：{len(self.vocab['char_to_idx'])}")

        # 统计各类别分布
        self._print_class_distribution()

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """加载预处理数据和词表"""
        print(f"🔄 正在加载数据：{self.data_path}")

        # 加载张量数据
        data_tensor = torch.load(self.data_path, weights_only=False)
        print(f"✅ 数据张量加载完成：{data_tensor.shape}")

        # 加载词表
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        print(f"✅ 词表加载完成，大小：{vocab_data['vocab_size']}")

        # 加载标签映射
        label_mapping_path = self.vocab_path.parent / 'label_mapping.json'
        if label_mapping_path.exists():
            with open(label_mapping_path, 'r', encoding='utf-8') as f:
                label_mapping = json.load(f)
            print(f"✅ 标签映射加载完成")
        else:
            # 使用默认标签映射
            label_mapping = {
                'SQLi': 0,
                'XSS': 1,
                'CMDi': 2,
                'Overflow': 3,
                'XXE': 4,
                'SSI': 5
            }
            print(f"⚠️ 使用默认标签映射")

        # 加载数据集统计信息
        stats_path = self.vocab_path.parent / 'dataset_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                dataset_stats = json.load(f)

            # 从统计信息中获取实际类别分布
            attack_distribution = dataset_stats.get('attack_distribution', {})
            total_samples = dataset_stats.get('total_samples', data_tensor.shape[0])

            print(f"📊 从统计信息加载类别分布：{attack_distribution}")
        else:
            # 使用默认分布
            attack_distribution = {
                'SQLi': 759,
                'XSS': 6711,
                'CMDi': 439,
                'Overflow': 49,
                'XXE': 105,
                'SSI': 18
            }
            total_samples = data_tensor.shape[0]
            print(f"⚠️ 使用默认类别分布")

        # 创建标签
        labels = []
        for attack_type, count in attack_distribution.items():
            if attack_type in label_mapping:
                class_id = label_mapping[attack_type]
                labels.extend([class_id] * count)
                print(f"   {attack_type}: {count} 样本 -> 类别 {class_id}")

        # 确保标签数量与数据样本数匹配
        if len(labels) != total_samples:
            print(f"⚠️ 标签数量({len(labels)})与数据样本数({total_samples})不匹配，进行调整")
            if len(labels) < total_samples:
                # 补充最后一个类别的标签
                last_class_id = labels[-1] if labels else 0
                labels.extend([last_class_id] * (total_samples - len(labels)))
            else:
                # 截断多余的标签
                labels = labels[:total_samples]

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        print(f"✅ 标签创建完成：{len(labels)} 个标签")

        return data_tensor, labels_tensor, vocab_data

    def _apply_oversampling(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用过采样来平衡类别分布"""
        print("🔄 正在应用过采样...")

        # 统计各类别数量
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        class_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}

        # 找到最多的类别数量
        max_count = max(class_counts.values())
        print(f"📈 最大类别样本数：{max_count}")

        # 为每个类别过采样到最大数量
        oversampled_data = []
        oversampled_labels = []

        for class_id in unique_labels:
            class_id = int(class_id)
            class_mask = (self.labels == class_id)
            class_data = self.data[class_mask]
            class_labels = self.labels[class_mask]

            current_count = len(class_data)
            needed_count = max_count - current_count

            if needed_count > 0:
                # 随机采样现有样本
                indices = torch.randint(0, current_count, (needed_count,))
                additional_data = class_data[indices]
                additional_labels = class_labels[indices]

                # 合并原始数据和过采样数据
                class_data = torch.cat([class_data, additional_data], dim=0)
                class_labels = torch.cat([class_labels, additional_labels], dim=0)

            oversampled_data.append(class_data)
            oversampled_labels.append(class_labels)

            print(f"   类别 {class_id}: {current_count} -> {len(class_data)} 样本")

        # 合并所有类别的数据
        final_data = torch.cat(oversampled_data, dim=0)
        final_labels = torch.cat(oversampled_labels, dim=0)

        # 随机打乱数据
        permutation = torch.randperm(len(final_data))
        final_data = final_data[permutation]
        final_labels = final_labels[permutation]

        return final_data, final_labels

    def _print_class_distribution(self):
        """打印类别分布统计"""
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        total = len(self.labels)

        print("📋 类别分布：")
        class_names = ['SQLi', 'XSS', 'CMDi', 'Overflow', 'XXE', 'SSI']

        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            label_int = int(label)
            percentage = (count / total) * 100
            class_name = class_names[label_int] if label_int < len(class_names) else f"Class_{label_int}"
            print(f"   {class_name:>8}: {count:>6} ({percentage:>5.1f}%)")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (sequence, label): 序列和对应的标签
        """
        sequence = self.data[idx]
        label = self.labels[idx]

        return sequence, label

    def get_vocab_info(self) -> Dict:
        """获取词表信息"""
        return {
            'vocab_size': self.vocab['vocab_size'],
            'char_to_idx': self.vocab['char_to_idx'],
            'special_tokens': self.vocab['special_tokens'],
            'max_length': self.vocab.get('max_length', 150)
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        计算类别权重，用于平衡损失函数

        Returns:
            [num_classes] 类别权重
        """
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        num_classes = len(unique_labels)

        # 计算权重：1 / (类别频率)
        weights = torch.zeros(num_classes)
        for label, count in zip(unique_labels, counts):
            weights[int(label)] = total_samples / (num_classes * count)

        # 归一化权重
        weights = weights / weights.sum() * num_classes

        return weights


def create_data_loaders(data_path: str, vocab_path: str,
                       batch_size: int = 32, train_split: float = 0.8,
                       oversample: bool = True, num_workers: int = 0,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    创建训练和验证数据加载器

    Args:
        data_path: 数据文件路径
        vocab_path: 词表文件路径
        batch_size: 批大小
        train_split: 训练集比例
        oversample: 是否启用过采样
        num_workers: 数据加载工作进程数
        random_state: 随机种子

    Returns:
        (train_loader, val_loader, vocab_info): 训练加载器、验证加载器和词表信息
    """
    print(f"🚀 开始创建数据加载器...")

    # 创建数据集
    dataset = CVADDataset(
        data_path=data_path,
        vocab_path=vocab_path,
        oversample=oversample,
        random_state=random_state
    )

    # 划分训练集和验证集
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size

    # 创建随机索引
    indices = torch.randperm(total_size)

    # 划分索引
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 创建子数据集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 获取词表信息
    vocab_info = dataset.get_vocab_info()

    print(f"✅ 数据加载器创建完成：")
    print(f"   训练样本数：{len(train_dataset)}")
    print(f"   验证样本数：{len(val_dataset)}")
    print(f"   批大小：{batch_size}")
    print(f"   训练批次数：{len(train_loader)}")
    print(f"   验证批次数：{len(val_loader)}")

    return train_loader, val_loader, vocab_info


def load_vocab(vocab_path: str) -> Dict:
    """
    加载词表文件

    Args:
        vocab_path: 词表文件路径

    Returns:
        词表字典
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    return vocab


def sample_from_dataset(data_path: str, vocab_path: str,
                       num_samples: int = 10, random_state: int = 42) -> List[Dict]:
    """
    从数据集中随机采样样本用于调试

    Args:
        data_path: 数据文件路径
        vocab_path: 词表文件路径
        num_samples: 采样数量
        random_state: 随机种子

    Returns:
        采样的样本列表
    """
    # 加载数据
    data_tensor = torch.load(data_path, weights_only=False)
    vocab = load_vocab(vocab_path)

    # 创建索引映射
    idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}

    # 随机采样
    torch.manual_seed(random_state)
    indices = torch.randperm(len(data_tensor))[:num_samples]

    samples = []
    for idx in indices:
        sequence = data_tensor[idx]
        decoded = []

        for token in sequence:
            if token == 1:  # EOS
                break
            if token not in [0, 2]:  # 不是 SOS 或 PAD
                if int(token) in idx_to_char:
                    decoded.append(idx_to_char[int(token)])
                else:
                    decoded.append('?')

        payload = ''.join(decoded)
        samples.append({
            'index': int(idx),
            'sequence': sequence.tolist(),
            'payload': payload,
            'length': len(payload)
        })

    return samples