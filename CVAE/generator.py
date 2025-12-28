#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVAE 载荷生成与特征提取器 (增强版 - 支持全选)
============================================

基于训练好的CVAE模型进行大规模载荷采样和隐空间特征提取
支持条件生成、批量处理、多攻击类型和数据清洗功能

作者：老王 (暴躁技术流)
版本：2.0 (全选功能版)
日期：2025-12-19
"""

import os
import sys
import json
import torch
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import logging

# 添加CVAE模块到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cvae_model import CVAE

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CVAEGenerator:
    """CVAE载荷生成器 (增强版)

    负责从训练好的CVAE模型批量生成攻击载荷
    支持多攻击类型和全选功能
    """

    # 攻击类型映射
    ATTACK_TYPES = {
        'SQLi': 0,
        'XSS': 1,
        'CMDi': 2,
        'Overflow': 3,
        'XXE': 4,
        'SSI': 5,
    }

    # 反向映射
    ID_TO_ATTACK = {v: k for k, v in ATTACK_TYPES.items()}

    def __init__(self, model_path: str, vocab_path: str, device: str = 'auto'):
        """初始化生成器

        Args:
            model_path: 训练好的CVAE模型路径
            vocab_path: 词表文件路径
            device: 计算设备 ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        self.vocab_path = Path(vocab_path)

        # 自动选择设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"使用设备: {self.device}")

        # 加载词表和模型
        self.vocab_data = self._load_vocab()
        self.model = self._load_model()

        # 生成统计信息
        self.generation_stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'failed_samples': 0,
            'attack_distribution': defaultdict(int)
        }

    def _load_vocab(self) -> Dict:
        """加载词表文件"""
        logger.info(f"加载词表: {self.vocab_path}")

        if not self.vocab_path.exists():
            raise FileNotFoundError(f"词表文件不存在: {self.vocab_path}")

        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        logger.info(f"词表加载完成，大小: {vocab_data['vocab_size']}")
        return vocab_data

    def _load_model(self) -> CVAE:
        """加载训练好的CVAE模型"""
        logger.info(f"加载CVAE模型: {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 从词表获取模型参数
        vocab_info = {
            'special_tokens': self.vocab_data['special_tokens']
        }

        model = CVAE(
            vocab_size=self.vocab_data['vocab_size'],
            embed_dim=128,  # 默认参数，应与训练时一致
            hidden_dim=256,
            latent_dim=32,
            condition_dim=6,
            num_layers=2,
            vocab_info=vocab_info
        )

        # 加载模型权重
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 处理不同的保存格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        logger.info(f"CVAE模型加载完成")
        model_info = model.get_model_info()
        logger.info(f"模型参数: {model_info['total_parameters']:,}")

        return model

    def decode_payloads(self, token_ids: torch.Tensor) -> List[str]:
        """将token IDs解码为文本载荷

        Args:
            token_ids: [batch_size, seq_len] token ID张量

        Returns:
            解码后的文本载荷列表
        """
        batch_size = token_ids.size(0)
        payloads = []

        # 获取反向词表
        idx_to_char = {int(k): v for k, v in self.vocab_data['idx_to_char'].items()}

        special_tokens = set(self.vocab_data['special_tokens'].values())

        for i in range(batch_size):
            tokens = token_ids[i].cpu().numpy()

            # 解码为字符
            chars = []
            for token_id in tokens:
                if token_id in idx_to_char:
                    char = idx_to_char[token_id]
                    # 跳过特殊token，但保留UNK用于标识生成失败
                    if char == '<PAD>':
                        continue
                    elif char == '<SOS>':
                        continue
                    elif char == '<EOS>':
                        break
                    elif char == '<UNK>':
                        chars.append('?')  # 替换字符
                    else:
                        chars.append(char)
                else:
                    chars.append('?')  # 未知token

            decoded_text = ''.join(chars)
            payloads.append(decoded_text)

        return payloads

    def generate_payloads(self,
                         attack_types: str | List[str] = 'ALL',
                         num_samples: int = 10000,
                         temperature: float = 1.8,
                         max_length: int = 150,
                         batch_size: int = 500) -> Tuple[List[str], List[Dict]]:
        """大规模生成攻击载荷

        Args:
            attack_types: 攻击类型 ('SQLi', 'XSS', 'CMDi', 'ALL', or list)
            num_samples: 每种攻击类型生成的样本数量
            temperature: 采样温度参数，控制随机性
            max_length: 最大生成长度
            batch_size: 批处理大小

        Returns:
            (payloads, metadata): 生成的载荷列表和元数据
        """
        # 处理攻击类型参数
        if attack_types == 'ALL' or attack_types == 'all':
            target_attack_types = list(self.ATTACK_TYPES.keys())
            logger.info(f"开始生成全类型载荷，类型: {', '.join(target_attack_types)}")
            total_samples = num_samples * len(target_attack_types)
        elif isinstance(attack_types, str):
            target_attack_types = [attack_types]
            total_samples = num_samples
        else:
            target_attack_types = attack_types
            total_samples = num_samples * len(target_attack_types)

        logger.info(f"温度参数: {temperature}, 批大小: {batch_size}")
        logger.info(f"总计划生成样本数: {total_samples}")

        all_payloads = []
        all_metadata = []

        # 为每种攻击类型生成载荷
        with torch.no_grad():
            for attack_type in target_attack_types:
                if attack_type not in self.ATTACK_TYPES:
                    logger.warning(f"跳过不支持的攻击类型: {attack_type}")
                    continue

                attack_label = self.ATTACK_TYPES[attack_type]
                remaining_samples = num_samples
                batch_count = 0

                logger.info(f"开始生成 {attack_type} 载荷，数量: {num_samples}")

                while remaining_samples > 0:
                    current_batch_size = min(batch_size, remaining_samples)
                    batch_count += 1

                    logger.info(f"{attack_type} 批次 {batch_count}: {current_batch_size} 个样本")

                    # 创建条件标签张量
                    conditions = torch.full((current_batch_size,), attack_label,
                                          dtype=torch.long, device=self.device)

                    try:
                        # CVAE生成
                        generated_tokens = self.model.generate(
                            c=conditions,
                            num_samples=1,
                            max_length=max_length,
                            temperature=temperature
                        )  # [batch_size, 1, seq_len]

                        # 移除num_samples维度
                        generated_tokens = generated_tokens.squeeze(1)  # [batch_size, seq_len]

                        # 解码为文本
                        batch_payloads = self.decode_payloads(generated_tokens)

                        # 创建元数据
                        batch_metadata = []
                        for i, payload in enumerate(batch_payloads):
                            metadata = {
                                'id': len(all_payloads) + i,
                                'type': attack_type,
                                'label': attack_label,
                                'payload': payload,
                                'length': len(payload),
                                'batch_id': batch_count,
                                'generation_success': True
                            }
                            batch_metadata.append(metadata)

                        all_payloads.extend(batch_payloads)
                        all_metadata.extend(batch_metadata)

                        remaining_samples -= current_batch_size

                        if batch_count % 5 == 0:
                            completed_for_type = len([m for m in all_metadata if m['type'] == attack_type])
                            logger.info(f"{attack_type} 已完成 {completed_for_type}/{num_samples} 个样本")

                    except Exception as e:
                        logger.error(f"{attack_type} 批次 {batch_count} 生成失败: {e}")
                        # 为失败的批次创建占位符
                        for i in range(current_batch_size):
                            failed_payload = "[GENERATION_FAILED]"
                            failed_metadata = {
                                'id': len(all_payloads) + i,
                                'type': attack_type,
                                'label': attack_label,
                                'payload': failed_payload,
                                'length': len(failed_payload),
                                'batch_id': batch_count,
                                'generation_success': False,
                                'error': str(e)
                            }
                            all_payloads.append(failed_payload)
                            all_metadata.append(failed_metadata)

                        remaining_samples -= current_batch_size

        logger.info(f"所有载荷生成完成！总计: {len(all_payloads)} 个样本")

        # 统计各类型数量
        type_stats = {}
        for meta in all_metadata:
            attack_type = meta.get('type', 'Unknown')
            if attack_type not in type_stats:
                type_stats[attack_type] = 0
            type_stats[attack_type] += 1

        logger.info(f"各类型分布:")
        for attack_type, count in type_stats.items():
            percentage = (count / len(all_metadata)) * 100
            logger.info(f"   {attack_type:>8}: {count:>6} 条 ({percentage:>5.1f}%)")

        return all_payloads, all_metadata

    def get_embeddings(self, payloads: List[str], metadata: List[Dict]) -> Tuple[np.ndarray, List[bool]]:
        """将生成的载荷通过编码器映射回隐空间

        Args:
            payloads: 载荷文本列表
            metadata: 载荷元数据列表，包含真实的标签信息

        Returns:
            (embeddings, valid_mask): 隐空间向量和有效性掩码
        """

        # 确保payloads和metadata长度一致
        if len(payloads) != len(metadata):
            raise ValueError(f"payloads和metadata长度不一致: {len(payloads)} vs {len(metadata)}")
        logger.info(f"提取 {len(payloads)} 个载荷的隐空间特征（使用真实标签）")

        embeddings = []
        valid_mask = []

        # 获取词表映射
        char_to_idx = self.vocab_data['char_to_idx']
        max_length = self.vocab_data['max_length']

        # 编码载荷为token IDs，同时记录有效样本的索引
        encoded_payloads = []
        valid_sample_indices = []  # 记录有效样本的原始索引

        for i, payload in enumerate(payloads):
            if payload == "[GENERATION_FAILED]" or len(payload.strip()) < 3:
                # 无效载荷，创建零向量
                encoded_payloads.append(None)
                continue

            valid_sample_indices.append(i)

            # 编码为token序列
            encoded = [char_to_idx.get('<SOS>', 0)]

            for char in payload[:max_length-2]:
                if char in char_to_idx:
                    encoded.append(char_to_idx[char])
                else:
                    encoded.append(char_to_idx.get('<UNK>', 3))

            encoded.append(char_to_idx.get('<EOS>', 1))

            # 填充到固定长度
            while len(encoded) < max_length:
                encoded.append(char_to_idx.get('<PAD>', 2))

            encoded_payloads.append(encoded[:max_length])

        # 批量提取隐空间特征
        batch_size = 500
        with torch.no_grad():
            for batch_start in range(0, len(encoded_payloads), batch_size):
                batch_end = min(batch_start + batch_size, len(encoded_payloads))
                batch_encoded = encoded_payloads[batch_start:batch_end]

                # 获取这个批次在原始数据中的索引范围
                original_indices = range(batch_start, batch_end)

                # 过滤有效样本并获取对应的元数据和标签
                valid_items = []
                for j, (enc, orig_idx) in enumerate(zip(batch_encoded, original_indices)):
                    if enc is not None and metadata[orig_idx].get('generation_success', True):
                        # 使用真实的标签信息
                        label = metadata[orig_idx].get('label', 0)  # 默认SQLi=0
                        valid_items.append({
                            'encoded': enc,
                            'label': label,
                            'batch_local_idx': j,  # 在当前批次中的位置
                            'original_idx': orig_idx  # 在原始数据中的位置
                        })

                if valid_items:
                    # 创建有效的批次数据
                    valid_encoded = [item['encoded'] for item in valid_items]
                    valid_labels = [item['label'] for item in valid_items]
                    valid_tensors = torch.tensor(valid_encoded, dtype=torch.long, device=self.device)

                    # 创建真实的条件标签
                    batch_conditions = torch.tensor(valid_labels, dtype=torch.long, device=self.device)

                    try:
                        # 通过编码器获取隐空间参数，使用真实标签
                        mu, logvar = self.model.encoder(valid_tensors, batch_conditions)

                        # 使用均值作为隐向量
                        batch_embeddings = mu.cpu().numpy()

                        # 填充到结果数组
                        batch_embeddings_full = np.zeros((len(batch_encoded), 32))  # latent_dim=32
                        batch_valid_mask = [False] * len(batch_encoded)

                        for item, embedding in zip(valid_items, batch_embeddings):
                            batch_local_idx = item['batch_local_idx']
                            batch_embeddings_full[batch_local_idx] = embedding
                            batch_valid_mask[batch_local_idx] = True

                        embeddings.extend(batch_embeddings_full.tolist())
                        valid_mask.extend(batch_valid_mask)

                    except Exception as e:
                        logger.warning(f"批次 {batch_start//batch_size} 特征提取失败: {e}")
                        # 为整个批次创建零向量
                        for _ in range(len(batch_encoded)):
                            embeddings.append(np.zeros(32).tolist())
                            valid_mask.append(False)
                else:
                    # 整个批次都是无效样本
                    for _ in range(len(batch_encoded)):
                        embeddings.append(np.zeros(32).tolist())
                        valid_mask.append(False)

                if (i // batch_size) % 10 == 0:
                    logger.info(f"已处理 {min(batch_end, len(encoded_payloads))}/{len(encoded_payloads)} 个特征")

        embeddings_array = np.array(embeddings)
        valid_mask_array = np.array(valid_mask)

        valid_count = np.sum(valid_mask_array)
        logger.info(f"特征提取完成！有效特征: {valid_count}/{len(payloads)} ({valid_count/len(payloads)*100:.1f}%)")

        return embeddings_array, valid_mask_array

    def clean_payloads(self, payloads: List[str], metadata: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """清洗生成的载荷

        Args:
            payloads: 原始载荷列表
            metadata: 载荷元数据

        Returns:
            (cleaned_payloads, cleaned_metadata): 清洗后的载荷和元数据
        """
        logger.info(f"开始清洗 {len(payloads)} 个载荷")

        cleaned_payloads = []
        cleaned_metadata = []

        for i, (payload, meta) in enumerate(zip(payloads, metadata)):
            # 检查是否生成失败
            if payload == "[GENERATION_FAILED]":
                meta['generation_success'] = False
                meta['failure_reason'] = 'generation_failed'
                continue

            # 检查长度
            if len(payload.strip()) < 3:
                meta['generation_success'] = False
                meta['failure_reason'] = 'too_short'
                continue

            # 检查是否全是未知字符
            if payload.count('?') / len(payload) > 0.5:
                meta['generation_success'] = False
                meta['failure_reason'] = 'too_many_unk'
                continue

            # 通过所有检查
            cleaned_payloads.append(payload)
            cleaned_metadata.append(meta)

        logger.info(f"载荷清洗完成！有效载荷: {len(cleaned_payloads)}/{len(payloads)}")
        logger.info(f"无效载荷: {len(payloads) - len(cleaned_payloads)} 个")

        return cleaned_payloads, cleaned_metadata

    def save_generated_data(self,
                          payloads: List[str],
                          metadata: List[Dict],
                          output_dir: str) -> None:
        """保存生成的数据

        Args:
            payloads: 载荷列表
            metadata: 载荷元数据
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存原始载荷
        raw_file = output_path / "raw_payloads.txt"
        with open(raw_file, 'w', encoding='utf-8') as f:
            for payload in payloads:
                f.write(payload + '\n')
        logger.info(f"原始载荷已保存: {raw_file}")

        # 保存元数据
        meta_file = output_path / "payload_metadata.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"载荷元数据已保存: {meta_file}")

        # 保存生成统计信息
        self.generation_stats['total_samples'] = len(payloads)
        self.generation_stats['valid_samples'] = sum(1 for meta in metadata if meta.get('generation_success', True))
        self.generation_stats['failed_samples'] = len(payloads) - self.generation_stats['valid_samples']

        for meta in metadata:
            if meta.get('generation_success', True):
                self.generation_stats['attack_distribution'][meta['type']] += 1

        stats_file = output_path / "generation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            # 转换defaultdict为普通dict以便JSON序列化
            stats_to_save = dict(self.generation_stats)
            stats_to_save['attack_distribution'] = dict(stats_to_save['attack_distribution'])
            json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
        logger.info(f"生成统计已保存: {stats_file}")

        logger.info(f"所有数据已保存到: {output_path}")


def main():
    """主函数 - 支持命令行调用"""
    parser = argparse.ArgumentParser(
        description="CVAE载荷生成器 (支持全选功能)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python generator_new.py --attack-type ALL --num-samples 5000 --output-dir ./generated
  python generator_new.py --attack-type SQLi --num-samples 10000
  python generator_new.py --attack-type XSS,CMDi --num-samples 3000
        """
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='CVAE/checkpoints/cvae_final.pth',
        help='CVAE模型路径 (默认: CVAE/checkpoints/cvae_final.pth)'
    )

    parser.add_argument(
        '--vocab-path',
        type=str,
        default='Data/processed/vocab.json',
        help='词表文件路径 (默认: Data/processed/vocab.json)'
    )

    parser.add_argument(
        '--attack-type',
        type=str,
        default='ALL',
        help='攻击类型 (默认: ALL，支持: SQLi, XSS, CMDi, Overflow, XXE, SSI, ALL 或逗号分隔的组合)'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=5000,
        help='每种攻击类型生成样本数量 (默认: 5000)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.8,
        help='采样温度参数 (默认: 1.8)'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=150,
        help='最大载荷长度 (默认: 150)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help='批处理大小 (默认: 500)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='Data/generated',
        help='输出目录 (默认: Data/generated)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='计算设备 (默认: auto)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出信息'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CVAE载荷生成器 (支持全选功能)")
    print("=" * 80)

    try:
        # 处理攻击类型参数
        if ',' in args.attack_type:
            attack_types = [t.strip() for t in args.attack_type.split(',')]
        else:
            attack_types = args.attack_type

        # 初始化生成器
        generator = CVAEGenerator(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            device=args.device
        )

        # 生成载荷
        payloads, metadata = generator.generate_payloads(
            attack_types=attack_types,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_length=args.max_length,
            batch_size=args.batch_size
        )

        # 清洗载荷
        cleaned_payloads, cleaned_metadata = generator.clean_payloads(payloads, metadata)

        # 提取隐空间特征
        embeddings, valid_mask = generator.get_embeddings(cleaned_payloads, cleaned_metadata)

        # 保存隐空间特征
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        embeddings_file = output_path / "latent_embeddings.npy"
        np.save(embeddings_file, embeddings)
        logger.info(f"隐空间特征已保存: {embeddings_file}")

        valid_mask_file = output_path / "valid_mask.npy"
        np.save(valid_mask_file, valid_mask)
        logger.info(f"有效性掩码已保存: {valid_mask_file}")

        # 保存生成的数据
        generator.save_generated_data(cleaned_payloads, cleaned_metadata, args.output_dir)

        print("\n" + "=" * 80)
        print("载荷生成任务完成！")
        print("=" * 80)
        print(f"生成统计:")
        print(f"   总样本数: {len(payloads)}")
        print(f"   有效样本: {len(cleaned_payloads)}")
        print(f"   成功率: {len(cleaned_payloads)/len(payloads)*100:.1f}%")
        print(f"   隐空间特征: {embeddings.shape}")
        print(f"   有效特征: {np.sum(valid_mask)}")

    except Exception as e:
        logger.error(f"载荷生成失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()