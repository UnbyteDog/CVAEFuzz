#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVAE 模块
========

基于 GRU 的 Seq2Seq 条件变分自编码器实现
用于 CVDBFuzz 项目的智能载荷生成

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-18
"""

from .cvae_model import CVAE, GumbelSoftmax
from .training_utils import CVAELoss, CyclicalAnnealingSchedule, CVAEMetrics
from .data_loader import CVADDataset, create_data_loaders
from .trainer import CVAETrainer, create_default_config

__version__ = "1.0"
__author__ = "老王 (暴躁技术流)"

__all__ = [
    # 模型类
    'CVAE',
    'GumbelSoftmax',

    # 训练工具
    'CVAELoss',
    'CyclicalAnnealingSchedule',
    'CVAEMetrics',

    # 数据处理
    'CVADDataset',
    'create_data_loaders',

    # 训练器
    'CVAETrainer',
    'create_default_config'
]