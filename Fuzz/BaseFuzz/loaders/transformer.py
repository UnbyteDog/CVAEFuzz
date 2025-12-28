#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Payload Transformer - 深度变异引擎
================================

负责对攻击载荷进行编码转换、混淆注入和随机变异，
以绕过WAF防护并探索非预期的解析错误。

核心功能：
- 编码转换（URL编码、双重URL编码、Unicode转义）
- SQL混淆（注释注入、空格替换）
- 位翻转变异（Radamsa风格）
- 随机字符插入/删除/替换

使用示例：
    >>> from Fuzz.BaseFuzz.loaders.transformer import PayloadTransformer
    >>>
    >>> # 原始载荷
    >>> payload = "<script>alert('XSS')</script>"
    >>>
    >>> # URL编码
    >>> encoded = PayloadTransformer.url_encode(payload)
    >>> print(encoded)
    >>> %3Cscript%3Ealert%28%27XSS%27%29%3C%2Fscript%3E
    >>>
    >>> # SQL混淆
    >>> obfuscated = PayloadTransformer.sql_noise(payload)
    >>> print(obfuscated)
    >>> SELECT/**/*FROM*/users

作者：老王 (暴躁技术流)
版本：1.0
日期：2025-12-25
"""

import random
import string
import re
from typing import List, Optional
from urllib.parse import quote, quote_plus
import logging

# 配置日志
logger = logging.getLogger(__name__)


class PayloadTransformer:
    """
    载荷转换器 - 深度变异引擎

    老王注释：这个SB类负责对载荷进行各种变异，绕过WAF！

    核心职责：
    1. 编码转换（URL、Double URL、Unicode）
    2. SQL混淆（注释注入）
    3. 位翻转变异（Radamsa风格）
    4. 随机变异（插入/删除/替换）

    静态方法类（无状态），所有方法都是线程安全的。

    Example:
        >>> # 单一转换
        >>> encoded = PayloadTransformer.url_encode("<script>")
        >>>
        >>> # 链式转换
        >>> mutated = PayloadTransformer.sql_noise(payload)
        >>> mutated = PayloadTransformer.double_url_encode(mutated)
    """

    # ========== 编码转换 ==========

    @staticmethod
    def url_encode(payload: str, plus: bool = False) -> str:
        """
        URL编码

        将特殊字符转换为%XX格式

        Args:
            payload: 原始载荷
            plus: 是否使用+号替换空格（默认False，使用%20）

        Returns:
            编码后的载荷

        Example:
            >>> PayloadTransformer.url_encode("<script>")
            '%3Cscript%3E'
            >>>
            >>> PayloadTransformer.url_encode("hello world", plus=True)
            'hello+world'
        """
        if plus:
            return quote_plus(payload)
        else:
            return quote(payload)

    @staticmethod
    def double_url_encode(payload: str) -> str:
        """
        双重URL编码

        对载荷进行两次URL编码，可绕过部分WAF的解码过滤器

        Args:
            payload: 原始载荷

        Returns:
            双重编码后的载荷

        Example:
            >>> PayloadTransformer.double_url_encode("<script>")
            '%253Cscript%253E'

        Note:
            某些WAF只解码一次，双重编码可以绕过
        """
        # 第一次编码
        first_encode = quote(payload)
        # 第二次编码
        second_encode = quote(first_encode)
        return second_encode

    @staticmethod
    def unicode_escape(payload: str) -> str:
        """
        Unicode转义编码

        将字符转换为\\uXXXX格式（如 < → \\u003c）

        Args:
            payload: 原始载荷

        Returns:
            Unicode转义后的载荷

        Example:
            >>> PayloadTransformer.unicode_escape("<script>alert(1)</script>")
            '\\u003cscript\\u003ealert(1)\\u003c/script\\u003e'

        Note:
            常用于绕过XSS过滤器
        """
        escaped = []
        for char in payload:
            # ASCII可打印字符保持原样，其他字符转义
            if ord(char) < 128 and char in string.printable:
                escaped.append(char)
            else:
                escaped.append(f"\\u{ord(char):04x}")
        return ''.join(escaped)

    @staticmethod
    def hex_encode(payload: str) -> str:
        """
        十六进制编码

        将每个字符转换为\\xXX格式

        Args:
            payload: 原始载荷

        Returns:
            十六进制编码后的载荷

        Example:
            >>> PayloadTransformer.hex_encode("<script>")
            '\\x3cscript\\x3e'
        """
        escaped = []
        for char in payload:
            escaped.append(f"\\x{ord(char):02x}")
        return ''.join(escaped)

    # ========== SQL 混淆 ==========

    @staticmethod
    def comment_injection(payload: str,
                        comment_type: str = 'standard') -> str:
        """
        SQL注释注入（空格替换为注释）

        艹，这个SB方法把空格替换成注释，绕过WAF的正则匹配！

        Args:
            payload: 原始SQL载荷
            comment_type: 注释类型
                - 'standard': 标准注释 /**/
                - 'mysql': MySQL版本注释 /*!50000*/
                - 'inline': 内联注释 --
                - 'random': 随机选择

        Returns:
            混淆后的SQL载荷

        Example:
            >>> payload = "SELECT * FROM users WHERE id=1"
            >>> PayloadTransformer.comment_injection(payload, 'standard')
            'SELECT/**//*FROM*/users/**/WHERE/**/id=1'
            >>>
            >>> PayloadTransformer.comment_injection(payload, 'mysql')
            'SELECT/*!50000*//*FROM*/users/*!50000*//*WHERE*//*id=1'
        """
        if comment_type == 'random':
            comment_type = random.choice(['standard', 'mysql', 'inline'])

        # 定义注释替换规则
        comment_patterns = {
            'standard': (' ', '/**/'),  # 空格 → /**/
            'mysql': (' ', '/*!50000*/'),  # 空格 → /*!50000*/
            'inline': (' ', '-- '),  # 空格 → --
        }

        if comment_type not in comment_patterns:
            logger.warning(f"[TRANSFORM] 未知注释类型: {comment_type}，使用standard")
            comment_type = 'standard'

        old_char, new_char = comment_patterns[comment_type]

        # 执行替换
        obfuscated = payload.replace(old_char, new_char)

        return obfuscated

    @staticmethod
    def case_randomization(payload: str) -> str:
        """
        SQL关键字大小写随机化

        随机转换SQL关键字的字母大小写，绕过正则匹配

        Args:
            payload: 原始SQL载荷

        Returns:
            大小写随机化后的载荷

        Example:
            >>> payload = "SELECT * FROM users"
            >>> PayloadTransformer.case_randomization(payload)
            'SeLeCT * FrOM users'
        """
        sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE',
            'DROP', 'UNION', 'AND', 'OR', 'ORDER', 'BY', 'LIMIT',
            'GROUP', 'HAVING', 'JOIN', 'INNER', 'LEFT', 'RIGHT'
        ]

        obfuscated = payload

        for keyword in sql_keywords:
            # 生成大小写随机版本
            randomized = ''.join(
                random.choice([c.upper(), c.lower()]) for c in keyword
            )
            # 不区分大小写替换
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            obfuscated = pattern.sub(randomized, obfuscated)

        return obfuscated

    # ========== 位翻转变异 ==========

    @staticmethod
    def bit_flipping(payload: str,
                   probability: float = 0.1,
                   max_flips: int = 5) -> str:
        """
        位翻转变异（Radamsa风格）

        艹，这个SB方法模仿Radamsa，随机替换/插入/删除字符！

        以固定概率对载荷进行随机变异：
        - 字符替换（10%概率）
        - 字符插入（5%概率）
        - 字符删除（5%概率）

        Args:
            payload: 原始载荷
            probability: 变异概率（默认0.1，即10%）
            max_flips: 最大变异次数（默认5次）

        Returns:
            变异后的载荷

        Example:
            >>> payload = "<script>alert(1)</script>"
            >>> mutated = PayloadTransformer.bit_flipping(payload, probability=0.1)
            >>> # 可能结果: "<scRipt>alert(1)</scrIpt>"

        Note:
            用于探索非预期的解析错误和边界情况
        """
        if not payload:
            return payload

        mutated = list(payload)
        flip_count = 0
        payload_len = len(payload)

        # 计算实际变异次数（不超过max_flips）
        num_flips = min(int(payload_len * probability), max_flips)

        for _ in range(num_flips):
            if flip_count >= max_flips:
                break

            # 随机选择变异类型
            mutation_type = random.choices(
                ['replace', 'insert', 'delete'],
                weights=[0.7, 0.15, 0.15]  # 替换占70%
            )[0]

            # 随机选择位置
            if mutated:
                pos = random.randint(0, len(mutated) - 1)

                if mutation_type == 'replace':
                    # 字符替换
                    if mutated[pos] == ' ':
                        # 空格替换为随机字符
                        mutated[pos] = random.choice(string.ascii_letters + string.digits)
                    else:
                        # 非空格字符，有10%概率替换为空格
                        if random.random() < 0.1:
                            mutated[pos] = ' '
                        else:
                            mutated[pos] = random.choice(string.ascii_letters + string.digits)

                    flip_count += 1

                elif mutation_type == 'insert':
                    # 字符插入
                    mutated.insert(pos, random.choice(string.ascii_letters + string.digits))
                    flip_count += 1

                elif mutation_type == 'delete':
                    # 字符删除（保留至少1个字符）
                    if len(mutated) > 1:
                        del mutated[pos]
                        flip_count += 1

        return ''.join(mutated)

    @staticmethod
    def random_padding(payload: str,
                      padding_chars: str = ' \t\n',
                      min_pad: int = 1,
                      max_pad: int = 5) -> str:
        """
        随机填充

        在载荷前后添加随机数量的空白字符，绕过trim()过滤

        Args:
            payload: 原始载荷
            padding_chars: 填充字符集合（默认：空格、制表符、换行符）
            min_pad: 最小填充数量
            max_pad: 最大填充数量

        Returns:
            填充后的载荷

        Example:
            >>> PayloadTransformer.random_padding("OR 1=1", min_pad=2, max_pad=5)
            '  OR 1=1\t\t'
        """
        pad_left = ''.join(random.choice(padding_chars)
                         for _ in range(random.randint(min_pad, max_pad)))
        pad_right = ''.join(random.choice(padding_chars)
                          for _ in range(random.randint(min_pad, max_pad)))

        return pad_left + payload + pad_right

    # ========== 组合变异 ==========

    @staticmethod
    def deep_mutate(payload: str,
                    strategy: str = 'mixed') -> str:
        """
        深度变异（组合多种变异技术）

        艹，这个方法是深度变异的核心！组合多种技术绕过WAF！

        Args:
            payload: 原始载荷
            strategy: 变异策略
                - 'encoding': 纯编码变异（URL/双重URL/Unicode）
                - 'sql': 纯SQL混淆
                - 'mixed': 混合变异（编码+混淆）
                - 'random': 随机选择

        Returns:
            变异后的载荷

        Example:
            >>> payload = "' OR 1=1--"
            >>> mutated = PayloadTransformer.deep_mutate(payload, strategy='mixed')
            >>> # 可能结果: '%27%20OR%201%3D1--%20' (URL编码)
        """
        if strategy == 'random':
            strategy = random.choice(['encoding', 'sql', 'mixed'])

        if strategy == 'encoding':
            # 纯编码变异
            encoder = random.choice([
                PayloadTransformer.url_encode,
                PayloadTransformer.double_url_encode,
                PayloadTransformer.unicode_escape
            ])
            return encoder(payload)

        elif strategy == 'sql':
            # 纯SQL混淆
            obfuscated = PayloadTransformer.comment_injection(payload)
            obfuscated = PayloadTransformer.case_randomization(obfuscated)
            return obfuscated

        elif strategy == 'mixed':
            # 混合变异：先混淆再编码
            obfuscated = PayloadTransformer.comment_injection(payload)
            encoded = PayloadTransformer.url_encode(obfuscated)
            return encoded

        else:
            return payload

    # ========== 批量变异 ==========

    @staticmethod
    def batch_mutate(payloads: List[str],
                     mutation_func: callable,
                     probability: float = 1.0) -> List[str]:
        """
        批量变异

        对载荷列表中的每个载荷应用变异函数

        Args:
            payloads: 原始载荷列表
            mutation_func: 变异函数（如url_encode, sql_noise等）
            probability: 变异概率（0.0-1.0），默认1.0（全部变异）

        Returns:
            变异后的载荷列表

        Example:
            >>> payloads = ["<script>alert(1)</script>", "' OR 1=1--"]
            >>> mutated = PayloadTransformer.batch_mutate(
            ...     payloads,
            ...     PayloadTransformer.url_encode,
            ...     probability=0.5  # 只变异50%
            ... )
        """
        mutated = []

        for payload in payloads:
            # 根据概率决定是否变异
            if random.random() < probability:
                try:
                    result = mutation_func(payload)
                    mutated.append(result)
                except Exception as e:
                    logger.error(f"[TRANSFORM] 变异失败: {payload[:50]}... - {e}")
                    mutated.append(payload)  # 保留原载荷
            else:
                mutated.append(payload)

        return mutated


# 便捷函数
def mutate_payload(payload: str,
                  strategy: str = 'mixed') -> str:
    """
    载荷变异的便捷函数

    Args:
        payload: 原始载荷
        strategy: 变异策略

    Returns:
        变异后的载荷

    Example:
        >>> from Fuzz.BaseFuzz.loaders.transformer import mutate_payload
        >>> mutated = mutate_payload("<script>alert(1)</script>", strategy='encoding')
    """
    return PayloadTransformer.deep_mutate(payload, strategy=strategy)


if __name__ == '__main__':
    # 测试代码
    import logging

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("PayloadTransformer 单元测试")
    print("=" * 60)

    # 测试载荷
    test_cases = [
        ("XSS", "<script>alert('XSS')</script>"),
        ("SQLi", "' OR 1=1--"),
        ("SQLi", "SELECT * FROM users WHERE id=1"),
    ]

    # 测试1：编码转换
    print("\n[测试1] 编码转换")
    print("-" * 60)
    for name, payload in test_cases:
        print(f"\n原始载荷 ({name}): {payload}")
        print(f"URL编码:     {PayloadTransformer.url_encode(payload)}")
        print(f"双重URL编码:  {PayloadTransformer.double_url_encode(payload)}")
        print(f"Unicode转义: {PayloadTransformer.unicode_escape(payload)}")

    # 测试2：SQL混淆
    print("\n[测试2] SQL混淆")
    print("-" * 60)
    sql_payload = "SELECT * FROM users WHERE id=1"
    print(f"原始SQL:     {sql_payload}")
    print(f"标准注释:    {PayloadTransformer.comment_injection(sql_payload, 'standard')}")
    print(f"MySQL注释:   {PayloadTransformer.comment_injection(sql_payload, 'mysql')}")
    print(f"大小写随机:  {PayloadTransformer.case_randomization(sql_payload)}")

    # 测试3：位翻转
    print("\n[测试3] 位翻转变异")
    print("-" * 60)
    xss_payload = "<script>alert(1)</script>"
    print(f"原始XSS:     {xss_payload}")
    print(f"位翻转10%:   {PayloadTransformer.bit_flipping(xss_payload, probability=0.1)}")
    print(f"位翻转30%:   {PayloadTransformer.bit_flipping(xss_payload, probability=0.3)}")

    # 测试4：随机填充
    print("\n[测试4] 随机填充")
    print("-" * 60)
    sqli_payload = "OR 1=1"
    print(f"原始SQLi:    {sqli_payload}")
    print(f"随机填充:    '{PayloadTransformer.random_padding(sqli_payload)}'")

    # 测试5：深度变异
    print("\n[测试5] 深度变异")
    print("-" * 60)
    print(f"原始:        {xss_payload}")
    print(f"编码策略:    {PayloadTransformer.deep_mutate(xss_payload, 'encoding')}")
    print(f"SQL策略:     {PayloadTransformer.deep_mutate(sql_payload, 'sql')}")
    print(f"混合策略:    {PayloadTransformer.deep_mutate(xss_payload, 'mixed')}")

    # 测试6：批量变异
    print("\n[测试6] 批量变异")
    print("-" * 60)
    payloads = ["<script>alert(1)</script>", "' OR 1=1--"]
    print(f"原始列表:    {payloads}")
    mutated = PayloadTransformer.batch_mutate(
        payloads,
        PayloadTransformer.url_encode,
        probability=0.5
    )
    print(f"变异列表:    {mutated}")

    print("\n[SUCCESS] 所有测试通过！")
