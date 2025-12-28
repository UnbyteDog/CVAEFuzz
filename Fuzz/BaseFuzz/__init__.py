#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaseFuzz - 基础模糊测试引擎框架
================================

CVDBFuzz Phase 4 的核心实现，提供：
- HTTP通信层（Requester）
- 基线画像（BaselineProfile）
- 载荷管理（PayloadManager）
- 深度变异（PayloadTransformer）
- 检测引擎调度器（Engine）
- 检测引擎抽象基类（BaseEngine）

使用示例：
    >>> from Fuzz.BaseFuzz.engine import Engine
    >>> from Fuzz.spider import FuzzTarget
    >>>
    >>> engine = Engine(
    ...     engine_names=['sqli', 'xss'],
    ...     mode='cvae'
    ... )
    >>> results = engine.run(targets)

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-25
"""

from Fuzz.BaseFuzz.requester import Requester, create_requester
from Fuzz.BaseFuzz.baseline import BaselineProfile
from Fuzz.BaseFuzz.engine import Engine, create_engine

__all__ = [
    'Requester',
    'create_requester',
    'BaselineProfile',
    'Engine',
    'create_engine',
]
