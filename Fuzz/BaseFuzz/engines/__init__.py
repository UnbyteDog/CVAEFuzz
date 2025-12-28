#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaseFuzz Engines - 漏洞检测引擎包
==================================

包含所有漏洞检测引擎的实现。

可用引擎：
- base_engine: 检测引擎抽象基类
- sqli_engine: SQL注入检测引擎（待实现）
- xss_engine: 跨站脚本检测引擎（待实现）

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-25
"""

from Fuzz.BaseFuzz.engines.base_engine import BaseEngine, VulnerabilityEntry

# 动态导入引擎（如果已实现）
try:
    from Fuzz.BaseFuzz.engines.sqli_engine import SQLiEngine
    SQLI_AVAILABLE = True
except ImportError:
    SQLiEngine = None
    SQLI_AVAILABLE = False

try:
    from Fuzz.BaseFuzz.engines.xss_engine import XSSEngine
    XSS_AVAILABLE = True
except ImportError:
    XSSEngine = None
    XSS_AVAILABLE = False

__all__ = [
    'BaseEngine',
    'VulnerabilityEntry',
    'SQLiEngine',
    'XSSEngine',
]
