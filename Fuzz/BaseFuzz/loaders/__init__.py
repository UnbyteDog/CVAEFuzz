#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaseFuzz Loaders - 载荷管理包
==============================

包含载荷加载和变异相关模块。

核心模块：
- payload_manager: 载荷管理中心
- transformer: 深度变异引擎

"""

from Fuzz.BaseFuzz.loaders.payload_manager import PayloadManager
from Fuzz.BaseFuzz.loaders.transformer import PayloadTransformer, mutate_payload

__all__ = [
    'PayloadManager',
    'PayloadTransformer',
    'mutate_payload',
]
