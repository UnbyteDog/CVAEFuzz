#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块
================================

模块功能：将原始Web攻击载荷转换为神经网络可处理的数值张量
技术路径：字符级分词 + 序列标准化 + 词表构建 + 质量分析

"""

import json
import os
import sys
import io
import argparse
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter, defaultdict
import pickle
import torch
import numpy as np
from pathlib import Path

# 解决Windows中文显示问题
if sys.platform.startswith('win'):
    # 设置UTF-8编码输出
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class AdvancedCharTokenizer:
    """高级字符级分词器

    专门处理Web攻击载荷的分词器，支持扩展字符集以提升覆盖率
    解决传统Word-level分词在代码类数据上的OOV问题
    """

    def __init__(self, vocab_size: int = 256, extended_chars: bool = True):
        """初始化分词器

        Args:
            vocab_size: 词表大小，默认256个字符
            extended_chars: 是否启用扩展字符集，包含常用Unicode字符
        """
        self.vocab_size = vocab_size
        self.extended_chars = extended_chars
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.special_tokens = {
            '<SOS>': 0,    # 序列开始标记
            '<EOS>': 1,    # 序列结束标记
            '<PAD>': 2,    # 填充标记
            '<UNK>': 3     # 未知字符标记
        }

        # 字符统计
        self.char_frequency = Counter()
        self.uncovered_chars = set()

        # 初始化词表
        self._build_vocab()

    def _build_vocab(self) -> None:
        """构建扩展字符集词表

        包含ASCII可打印字符 + 常用Unicode字符 + Web攻击常见特殊字符
        """
        # 基础ASCII可打印字符 (32-126)
        base_chars = [chr(i) for i in range(32, 127)]

        # 控制字符
        control_chars = ['\t', '\n', '\r', '\f', '\v']

        # 扩展字符集，针对Web攻击载荷中的常见字符
        extended_chars = []
        if self.extended_chars:
            extended_chars = [
                # 扩展ASCII字符 (128-255) - 欧洲语言字符
                '¡', '¢', '£', '¤', '¥', '¦', '§', '¨', '©', 'ª', '«', '¬', '®', '¯',
                '°', '±', '²', '³', '´', 'µ', '¶', '·', '¸', '¹', 'º', '»', '¼', '½',
                '¾', '¿', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë',
                'Ì', 'Í', 'Î', 'Ï', 'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø', 'Ù',
                'Ú', 'Û', 'Ü', 'Ý', 'Þ', 'ß', 'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç',
                'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò', 'ó', 'ô', 'õ',
                'ö', '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ',

                # Unicode特殊字符 (Web攻击中常见)
                '\u00a0',  # 不换行空格
                '\u200b',  # 零宽空格
                '\u200c',  # 零宽非连字符
                '\u200d',  # 零宽连字符
                '\ufeff',  # 零宽非断空格
                '\u2060',  # 单词连接符

                # 数学符号和特殊字符
                '…', '–', '—', ''', ''', '"', '"', '•', '‰', '‹', '›',

                # 其他Web攻击常见字符
                'Š', 'š', 'Ž', 'ž', 'Œ', 'œ', 'Ÿ', 'ƒ', 'ˆ', '˜',
                '€', '™', '∞', '≠', '≤', '≥', '∂', '∆', '∇', '∏',
                '∑', '∫', 'π', 'Ω', 'α', 'β', 'γ', 'δ', 'ε', 'ζ',
                'η', 'θ', 'λ', 'μ', 'ξ', 'ρ', 'σ', 'τ', 'φ', 'χ', 'ψ', 'ω',

                # 中文字符 (中文攻击载荷)
                '的', '是', '在', '了', '和', '有', '我', '你', '他', '这',
                '个', '一', '不', '会', '就', '说', '要', '可', '以', '来',

                # 日文字符 (日本攻击载荷)
                'あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ',
                'ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク', 'ケ', 'コ',

                # 韩文字符 (韩国攻击载荷)
                '가', '나', '다', '라', '마', '바', '사', '아', '자', '차',

                # 阿拉伯字符
                'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر',

                # 西里尔字符 (俄语)
                'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'К',
                'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'к',
            ]

        # 合并所有字符
        all_chars = base_chars + control_chars + extended_chars

        # 添加特殊Token到词表开头
        vocab_list = list(self.special_tokens.keys()) + all_chars

        # 限制词表大小
        if len(vocab_list) > self.vocab_size:
            vocab_list = vocab_list[:self.vocab_size]
            print(f"⚠️ 词表过大，截断到 {self.vocab_size} 个字符")

        # 构建双向映射
        for idx, char in enumerate(vocab_list):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char

        print(f"✅ 词表构建完成，总大小：{len(self.char_to_idx)}")
        print(f"📝 特殊Token数量：{len(self.special_tokens)}")
        print(f"🔤 ASCII可打印字符：{len(base_chars)}")
        print(f"⌨️  控制字符：{len(control_chars)}")
        print(f"🌐 扩展字符数量：{len(extended_chars)}")
        print(f"🎯 实际使用字符：{len(all_chars)}")

    def analyze_text(self, texts: List[str]) -> Dict[str, any]:
        """分析文本中的字符分布

        Args:
            texts: 待分析的文本列表

        Returns:
            字符分布分析结果
        """
        print(f"\n🔍 开始字符分布分析...")

        # 统计字符频率
        self.char_frequency.clear()
        self.uncovered_chars.clear()

        total_chars = 0
        covered_chars = 0

        for text in texts:
            for char in text:
                total_chars += 1
                self.char_frequency[char] += 1
                if char in self.char_to_idx:
                    covered_chars += 1
                else:
                    self.uncovered_chars.add(char)

        coverage = (covered_chars / total_chars * 100) if total_chars > 0 else 0

        print(f"📊 字符分布分析结果：")
        print(f"   总字符数：{total_chars}")
        print(f"   词表覆盖字符数：{covered_chars}")
        print(f"   未覆盖字符数：{len(self.uncovered_chars)}")
        print(f"   词表覆盖率：{coverage:.3f}%")

        # 显示最常见的字符
        most_common = self.char_frequency.most_common(20)
        print(f"\n📈 最常见的20个字符：")
        for char, freq in most_common:
            char_repr = repr(char)
            print(f"   {char_repr:>6}: {freq:>6} 次")

        # 显示未覆盖的字符
        if self.uncovered_chars:
            print(f"\n⚠️ 未覆盖的字符（{len(self.uncovered_chars)}个）：")
            for char in sorted(self.uncovered_chars):
                print(f"   {repr(char)} (U+{ord(char):04X})")

        return {
            'total_chars': total_chars,
            'covered_chars': covered_chars,
            'uncovered_chars': len(self.uncovered_chars),
            'coverage_rate': coverage,
            'most_common': most_common,
            'uncovered_list': list(self.uncovered_chars)
        }

    def encode(self, text: str, max_length: int = 150) -> List[int]:
        """将文本编码为索引序列

        Args:
            text: 待编码的文本
            max_length: 最大序列长度

        Returns:
            编码后的索引序列
        """
        # 添加起始和结束标记
        encoded = [self.special_tokens['<SOS>']]

        # 编码文本内容
        for char in text[:max_length-2]:  # 为SOS和EOS预留位置
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                encoded.append(self.special_tokens['<UNK>'])

        # 添加结束标记
        encoded.append(self.special_tokens['<EOS>'])

        # 后缀填充到max_length
        while len(encoded) < max_length:
            encoded.append(self.special_tokens['<PAD>'])

        return encoded[:max_length]

    def decode(self, indices: List[int]) -> str:
        """将索引序列解码为文本

        Args:
            indices: 索引序列

        Returns:
            解码后的文本
        """
        chars = []
        for idx in indices:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                # 跳过特殊Token（除了UNK）
                if char not in ['<SOS>', '<EOS>', '<PAD>']:
                    if char == '<UNK>':
                        chars.append('�')  # 未知字符用替换符号表示
                    else:
                        chars.append(char)
            else:
                chars.append('�')  # 无效索引也用替换符号表示

        return ''.join(chars)

    def get_vocab_stats(self) -> Dict[str, any]:
        """获取词表统计信息

        Returns:
            词表统计信息
        """
        return {
            'vocab_size': len(self.char_to_idx),
            'special_tokens': self.special_tokens,
            'extended_mode': self.extended_chars,
            'char_frequency': dict(self.char_frequency.most_common(50)),
            'uncovered_count': len(self.uncovered_chars)
        }


class PayloadDataset:
    """Web攻击载荷数据集处理器

    支持6种攻击类型的加载和预处理，包含数据质量分析
    """

    # 支持的攻击类型映射
    ATTACK_TYPES = {
        'SQLi': 0,
        'XSS': 1,
        'CMDi': 2,
        'Overflow': 3,
        'XXE': 4,
        'SSI': 5,
        'XML': 4,  # XML类型映射到XXE
    }

    # 文件名到攻击类型的映射
    FILE_MAPPING = {
        'sqli.jsonl': 'SQLi',
        'xss.jsonl': 'XSS',
        'cmdi.jsonl': 'CMDi',
        'overflow.jsonl': 'Overflow',
        'xml.jsonl': 'XXE',      # xml.jsonl包含XXE攻击
        'ssi.jsonl': 'SSI'
    }

    def __init__(self, data_dir: str, max_length: int = 150, vocab_size: int = 256):
        """初始化数据集处理器

        Args:
            data_dir: 训练数据目录路径
            max_length: 最大序列长度
            vocab_size: 词表大小
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.tokenizer = AdvancedCharTokenizer(vocab_size=vocab_size, extended_chars=True)
        self.payloads = []      # 存储载荷文本
        self.labels = []        # 存储载荷标签
        self.attack_counts = {}  # 各类型攻击数量统计

        # 数据质量统计
        self.quality_stats = {
            'empty_payloads': 0,
            'duplicate_payloads': 0,
            'avg_payload_length': 0,
            'max_payload_length': 0,
            'min_payload_length': float('inf'),
            'total_chars': 0
        }

        print(f"🎯 初始化数据集处理器 (增强版)")
        print(f"📁 数据目录：{self.data_dir}")
        print(f"📏 最大序列长度：{self.max_length}")
        print(f"📚 词表大小：{vocab_size}")

    def _quality_check(self, payload: str) -> bool:
        """数据质量检查

        Args:
            payload: 待检查的载荷

        Returns:
            是否通过质量检查
        """
        # 检查空载荷
        if not payload or not payload.strip():
            self.quality_stats['empty_payloads'] += 1
            return False

        # 检查载荷长度
        payload_len = len(payload)
        self.quality_stats['total_chars'] += payload_len
        self.quality_stats['max_payload_length'] = max(self.quality_stats['max_payload_length'], payload_len)
        self.quality_stats['min_payload_length'] = min(self.quality_stats['min_payload_length'], payload_len)

        return True

    def load_data(self) -> Tuple[List[str], List[int]]:
        """加载所有训练数据

        Returns:
            (payloads, labels): 载荷列表和对应标签列表
        """
        print(f"\n🚀 开始加载训练数据 (增强版)...")

        total_samples = 0
        seen_payloads = set()  # 用于去重

        # 遍历所有训练文件
        for filename, attack_type in self.FILE_MAPPING.items():
            file_path = self.data_dir / filename

            if not file_path.exists():
                print(f"⚠️ 警告：文件不存在 {file_path}")
                continue

            print(f"📖 正在读取：{filename} -> {attack_type}")

            samples_in_file = 0
            duplicates_in_file = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            payload = data.get('payload', '').strip()
                            data_type = data.get('type', '')

                            # 数据质量检查
                            if not self._quality_check(payload):
                                continue

                            # 验证攻击类型匹配
                            if data_type != attack_type and data_type not in self.ATTACK_TYPES:
                                print(f"⚠️ {filename}:{line_num} 未知攻击类型：{data_type}")
                                continue

                            # 去重检查
                            if payload in seen_payloads:
                                duplicates_in_file += 1
                                self.quality_stats['duplicate_payloads'] += 1
                                continue

                            seen_payloads.add(payload)

                            # 添加到数据集
                            self.payloads.append(payload)
                            self.labels.append(self.ATTACK_TYPES[attack_type])
                            samples_in_file += 1

                        except json.JSONDecodeError as e:
                            print(f"⚠️ {filename}:{line_num} JSON解析错误：{e}")
                            continue

            except Exception as e:
                print(f"❌ 读取文件失败 {file_path}：{e}")
                continue

            # 记录该类型攻击数量
            self.attack_counts[attack_type] = samples_in_file
            total_samples += samples_in_file

            print(f"✅ {filename} 加载完成：{samples_in_file} 条载荷 (去重后)")
            if duplicates_in_file > 0:
                print(f"   跳过重复载荷：{duplicates_in_file} 条")

        # 计算平均长度
        if self.payloads:
            total_length = sum(len(p) for p in self.payloads)
            self.quality_stats['avg_payload_length'] = total_length / len(self.payloads)
        else:
            self.quality_stats['avg_payload_length'] = 0
            self.quality_stats['min_payload_length'] = 0

        print(f"\n🎊 数据加载完成！")
        print(f"📊 有效样本数：{total_samples}")
        print(f"📈 各类型分布：")
        for attack_type, count in self.attack_counts.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"   {attack_type:>8}: {count:>6} 条 ({percentage:>5.1f}%)")

        # 显示质量统计
        print(f"\n📋 数据质量统计：")
        print(f"   空载荷：{self.quality_stats['empty_payloads']} 条")
        print(f"   重复载荷：{self.quality_stats['duplicate_payloads']} 条")
        print(f"   平均长度：{self.quality_stats['avg_payload_length']:.1f} 字符")
        print(f"   最大长度：{self.quality_stats['max_payload_length']} 字符")
        print(f"   最小长度：{self.quality_stats['min_payload_length']} 字符")

        return self.payloads, self.labels

    def preprocess(self) -> torch.Tensor:
        """预处理所有数据

        将文本载荷转换为数值张量矩阵 X ∈ R^(N × L_max)

        Returns:
            处理后的数据张量
        """
        print(f"\n🔄 开始数据预处理 (增强版)...")

        if not self.payloads:
            raise ValueError("❌ 没有加载任何数据，请先调用 load_data()")

        # 首先进行字符分布分析
        char_analysis = self.tokenizer.analyze_text(self.payloads)

        # 编码所有载荷
        encoded_data = []
        print(f"🔤 正在编码 {len(self.payloads)} 条载荷...")

        # 批量处理优化
        batch_size = 1000
        for batch_start in range(0, len(self.payloads), batch_size):
            batch_end = min(batch_start + batch_size, len(self.payloads))
            batch_payloads = self.payloads[batch_start:batch_end]

            for i, payload in enumerate(batch_payloads):
                if (batch_start + i + 1) % 1000 == 0 or (batch_start + i) == 0:
                    print(f"   进度：{batch_start + i + 1}/{len(self.payloads)} ({(batch_start + i + 1)/len(self.payloads)*100:.1f}%)")

                encoded = self.tokenizer.encode(payload, self.max_length)
                encoded_data.append(encoded)

        # 转换为PyTorch张量
        print("🔥 正在转换为PyTorch张量...")
        data_tensor = torch.tensor(encoded_data, dtype=torch.long)

        print(f"✅ 预处理完成！")
        print(f"📐 输出张量形状：{data_tensor.shape}")
        print(f"🔢 数据类型：{data_tensor.dtype}")

        return data_tensor

    def calculate_vocabulary_coverage(self) -> float:
        """计算词表覆盖率

        Returns:
            词表覆盖率百分比（0-100）
        """
        print(f"\n🔍 计算词表覆盖率 (增强版)...")

        # 使用tokenizer的分析结果
        if not self.tokenizer.char_frequency:
            self.tokenizer.analyze_text(self.payloads)

        total_chars = sum(self.tokenizer.char_frequency.values())
        covered_chars = sum(freq for char, freq in self.tokenizer.char_frequency.items()
                          if char in self.tokenizer.char_to_idx)
        uncovered_chars = total_chars - covered_chars

        coverage = (covered_chars / total_chars * 100) if total_chars > 0 else 0

        print(f"📊 词表覆盖率分析结果：")
        print(f"   总字符数：{total_chars}")
        print(f"   覆盖字符数：{covered_chars}")
        print(f"   未覆盖字符数：{uncovered_chars}")
        print(f"   覆盖率：{coverage:.3f}%")

        # 检查是否达到目标覆盖率
        target_coverage = 99.9
        if coverage >= target_coverage:
            print(f"🎯 达标！词表覆盖率 {coverage:.3f}% >= {target_coverage}%")
        else:
            print(f"⚠️ 未达标！词表覆盖率 {coverage:.3f}% < {target_coverage}%")
            print("💡 建议：")
            if self.tokenizer.uncovered_chars:
                print("   1. 将未覆盖字符添加到扩展字符集")
                print("   2. 增加词表大小")
                print("   3. 考虑使用字符替换策略")

        return coverage

    def save_processed_data(self, output_dir: str, data_tensor: torch.Tensor) -> None:
        """保存处理后的数据 (增强版)

        Args:
            output_dir: 输出目录
            data_tensor: 处理后的数据张量
        """
        print(f"\n💾 保存处理后的数据 (增强版)...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存数据张量
        data_file = output_path / "processed_data.pt"
        torch.save(data_tensor, data_file)
        print(f"✅ 数据张量已保存：{data_file}")

        # 保存词表
        vocab_file = output_path / "vocab.json"
        vocab_data = {
            'char_to_idx': self.tokenizer.char_to_idx,
            'idx_to_char': {str(k): v for k, v in self.tokenizer.idx_to_char.items()},
            'special_tokens': self.tokenizer.special_tokens,
            'vocab_size': len(self.tokenizer.char_to_idx),
            'max_length': self.max_length,
            'stats': self.tokenizer.get_vocab_stats()
        }

        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 词表已保存：{vocab_file}")

        # 保存标签映射
        label_file = output_path / "label_mapping.json"
        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump(self.ATTACK_TYPES, f, ensure_ascii=False, indent=2)
        print(f"✅ 标签映射已保存：{label_file}")

        # 保存数据集统计信息 (增强版)
        stats_file = output_path / "dataset_stats.json"
        stats = {
            'total_samples': len(self.payloads),
            'attack_distribution': self.attack_counts,
            'vocabulary_coverage': self.calculate_vocabulary_coverage(),
            'tensor_shape': list(data_tensor.shape),
            'max_length': self.max_length,
            'quality_stats': self.quality_stats,
            'char_analysis': {
                'total_chars': self.quality_stats['total_chars'],
                'uncovered_chars': len(self.tokenizer.uncovered_chars),
                'uncovered_list': list(self.tokenizer.uncovered_chars)[:20]  # 只保存前20个
            }
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ 统计信息已保存：{stats_file}")

        # 保存样本数据 (用于调试)
        samples_file = output_path / "sample_payloads.json"
        sample_data = []
        for i in range(min(20, len(self.payloads))):
            attack_type = list(self.ATTACK_TYPES.keys())[list(self.ATTACK_TYPES.values()).index(self.labels[i])]
            sample_data.append({
                'id': i,
                'type': attack_type,
                'label': self.labels[i],
                'payload': self.payloads[i],
                'encoded': data_tensor[i].tolist()[:20],  # 只保存前20个token
                'length': len(self.payloads[i])
            })

        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 样本数据已保存：{samples_file}")

        print(f"\n🎉 所有数据文件已保存到：{output_path}")


def main():
    """主函数 - 支持命令行调用"""
    parser = argparse.ArgumentParser(
        description="CVDBFuzz 数据预处理工具 (增强版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python main.py --preprocess
  python main.py --preprocess --data-dir ./custom_data --output-dir ./output --vocab-size 512
        """
    )

    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='执行数据预处理'
    )

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

    args = parser.parse_args()

    if not args.preprocess:
        parser.print_help()
        return

    print("=" * 80)
    print("🎯 CVDBFuzz 数据预处理工具 (增强版)")
    print("=" * 80)

    try:
        # 初始化数据集处理器
        dataset = PayloadDataset(
            data_dir=args.data_dir,
            max_length=args.max_length,
            vocab_size=args.vocab_size
        )

        # 加载原始数据
        payloads, labels = dataset.load_data()

        # 数据预处理
        data_tensor = dataset.preprocess()

        # 保存处理后的数据
        dataset.save_processed_data(args.output_dir, data_tensor)

        print("\n" + "=" * 80)
        print("🎉 数据预处理任务完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 预处理失败：{e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()