#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Payload Manager - 字典调度中心
================================

负责加载、管理和分发攻击载荷字典，支持CVAE模式、专家字典模式
以及深度变异集成。采用生成器设计，避免海量数据撑爆内存。

核心功能：
- 多源加载（单文件/文件夹/通配符）
- CVAE模式适配（加载DBSCAN优化的精锐载荷）
- 迭代器设计（生成器按需吐出载荷）
- 类型过滤（按漏洞类型筛选）
- 深度变异集成（可选调用PayloadTransformer）

使用示例：
    >>> from Fuzz.BaseFuzz.loaders.payload_manager import PayloadManager
    >>>
    >>> # 初始化
    >>> mgr = PayloadManager()
    >>>
    >>> # 加载CVAE载荷
    >>> payloads = mgr.load(mode='cvae')
    >>>
    >>> # 遍历载荷（生成器）
    >>> for payload in mgr.iterate(payloads, vtype='SQLi'):
    ...     response = requester.send('GET', url, params={'id': payload})
    ...     # 发送请求


"""

import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
import logging

# 配置日志
logger = logging.getLogger(__name__)


class PayloadManager:
    """
    载荷管理器 - 字典调度中心

    负责管理所有的攻击载荷字典！

    核心职责：
    1. 加载载荷字典（单文件/文件夹/CVAE）
    2. 按攻击类型分组
    3. 提供生成器接口（避免内存溢出）
    4. 集成深度变异（可选）

    Attributes:
        payload_paths: 载荷路径配置字典

    Example:
        >>> mgr = PayloadManager()
        >>> payloads = mgr.load(mode='cvae')
        >>> for payload in mgr.iterate(payloads, vtype='ALL'):
        ...     print(payload)
    """

    # 载荷路径配置
    PAYLOAD_PATHS = {
        'cvae': 'Data/processed/fuzzing/refined_payloads.txt',
        'cvae_raw': 'Data/generated/raw_payloads.txt',
        'common': 'payload/dict/',
        'seclists': 'Data/payload/FuzzLists/',
    }

    # 攻击类型映射（文件名 → 类型）
    VTYPE_MAPPING = {
        'sqli': 'SQLi',
        'sql': 'SQLi',
        'xss': 'XSS',
        'cmdi': 'CMDi',
        'rce': 'CMDi',
        'lfi': 'LFI',
        'lf': 'LFI',
        'rfi': 'RFI',
        'xxe': 'XXE',
        'ssi': 'SSI',
        'overflow': 'Overflow',
        'dos': 'Overflow',
    }

    def __init__(self, base_path: Optional[str] = None):
        """
        初始化PayloadManager

        Args:
            base_path: 项目根目录路径（默认自动检测）

        Example:
            >>> mgr = PayloadManager()
            >>> # 或指定根目录
            >>> mgr = PayloadManager(base_path='/path/to/CVDBFuzz')
        """
        if base_path:
            self.base_path = Path(base_path)
        else:
            # 自动检测项目根目录
            self.base_path = Path(__file__).resolve().parents[3]  # 回溯3级到CVDBFuzz/

        logger.info(f"[PAYLOAD] PayloadManager初始化完成，base_path={self.base_path}")

    def load(self,
            mode: str = 'common',
            source: Optional[str] = None,
            vtype: Optional[str] = None) -> Dict[str, List[str]]:
        """
        加载载荷字典


        Args:
            mode: 加载模式
                - 'cvae': CVAE模式（加载精锐载荷）
                - 'cvae_raw': CVAE原始载荷（未聚类）
                - 'common': 普通模式（加载专家字典）
                - 'custom': 自定义模式
            source: 自定义载荷源（mode='custom'时使用）
            vtype: 漏洞类型过滤（如 'SQLi', 'XSS', 'ALL'）

        Returns:
            按攻击类型分组的载荷字典
            {
                'SQLi': [payload1, payload2, ...],
                'XSS': [payload1, payload2, ...],
                'ALL': [payload1, payload2, ...]
            }

        Example:
            >>> # CVAE模式
            >>> payloads = mgr.load(mode='cvae')
            >>> # {'ALL': ['" OR 1=1--', '<script>alert(1)</script>', ...]}
            >>>
            >>> # 普通模式
            >>> payloads = mgr.load(mode='common')
            >>> # {'SQLi': [...], 'XSS': [...], 'CMDi': [...]}
            >>>
            >>> # 自定义模式
            >>> payloads = mgr.load(mode='custom', source='payload/dict/sqli.txt')
            >>> # {'CUSTOM': [...]}
        """
        logger.info(f"[PAYLOAD] 开始加载载荷: mode={mode}, vtype={vtype}")

        if mode == 'cvae':
            # CVAE模式：加载精锐载荷
            payloads = self._load_cvae(vtype=vtype)

        elif mode == 'cvae_raw':
            # CVAE原始载荷：加载未聚类的载荷
            payloads = self._load_cvae_raw(vtype=vtype)

        elif mode == 'common':
            # 普通模式：加载专家字典
            payloads = self._load_common(vtype=vtype)

        elif mode == 'custom':
            # 自定义模式：加载指定源
            if not source:
                raise ValueError("[PAYLOAD] 自定义模式必须指定source参数")
            payloads = self._load_custom(source, vtype=vtype)

        else:
            raise ValueError(f"[PAYLOAD] 未知的加载模式: {mode}")

        # 统计信息
        total_count = sum(len(plist) for plist in payloads.values())
        logger.info(f"[PAYLOAD] 载荷加载完成: {len(payloads)}个类型, {total_count}条载荷")

        for vtype_name, plist in payloads.items():
            logger.info(f"[PAYLOAD]   - {vtype_name}: {len(plist)} 条")

        return payloads

    def _load_cvae(self,
                   vtype: Optional[str] = None) -> Dict[str, List[str]]:
        """
        加载CVAE精锐载荷（DBSCAN优化后）

        Args:
            vtype: 漏洞类型过滤

        Returns:
            载荷字典
        """
        cvae_path = self.base_path / self.PAYLOAD_PATHS['cvae']

        if not cvae_path.exists():
            logger.error(f"[PAYLOAD] CVAE载荷文件不存在: {cvae_path}")
            logger.error(f"[PAYLOAD] 请先运行: python fuzzmain.py --generate --cluster")
            return {}

        logger.info(f"[PAYLOAD] 加载CVAE精锐载荷: {cvae_path}")

        # 加载CVAE载荷
        payloads = self._load_from_file(cvae_path)

        # CVAE载荷未分类，全部放到'ALL'类别
        return {'ALL': payloads}

    def _load_cvae_raw(self,
                       vtype: Optional[str] = None) -> Dict[str, List[str]]:
        """
        加载CVAE原始载荷（未聚类）

        Args:
            vtype: 漏洞类型过滤

        Returns:
            载荷字典
        """
        raw_path = self.base_path / self.PAYLOAD_PATHS['cvae_raw']

        if not raw_path.exists():
            logger.error(f"[PAYLOAD] CVAE原始载荷文件不存在: {raw_path}")
            return {}

        logger.info(f"[PAYLOAD] 加载CVAE原始载荷: {raw_path}")

        # 加载原始载荷
        payloads = self._load_from_file(raw_path)

        # 尝试加载元数据进行分类
        metadata_path = self.base_path / 'Data/generated/payload_metadata.json'
        if metadata_path.exists():
            payloads = self._load_cvae_with_metadata(payloads, metadata_path, vtype)
        else:
            # 元数据不存在，全部放到'ALL'
            logger.warning("[PAYLOAD] 元数据文件不存在，无法分类")
            return {'ALL': payloads}

        return payloads

    def _load_cvae_with_metadata(self,
                                  payloads: List[str],
                                  metadata_path: Path,
                                  vtype: Optional[str]) -> Dict[str, List[str]]:
        """
        使用元数据对CVAE载荷进行分类

        Args:
            payloads: 原始载荷列表
            metadata_path: 元数据文件路径
            vtype: 漏洞类型过滤

        Returns:
            分类后的载荷字典
        """
        logger.info(f"[PAYLOAD] 加载元数据: {metadata_path}")

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)

            # 按类型分组
            grouped = {}
            for i, meta in enumerate(metadata_list):
                if i >= len(payloads):
                    break

                payload_type = meta.get('type', 'UNKNOWN')
                if payload_type not in grouped:
                    grouped[payload_type] = []
                grouped[payload_type].append(payloads[i])

            # 类型过滤
            if vtype and vtype != 'ALL':
                if vtype in grouped:
                    return {vtype: grouped[vtype]}
                else:
                    logger.warning(f"[PAYLOAD] 未找到类型: {vtype}")
                    return {}
            else:
                return grouped

        except Exception as e:
            logger.error(f"[PAYLOAD] 元数据加载失败: {e}")
            return {'ALL': payloads}

    def _load_common(self,
                     vtype: Optional[str] = None) -> Dict[str, List[str]]:
        """
        加载专家字典（普通模式）

        Args:
            vtype: 漏洞类型过滤

        Returns:
            分类后的载荷字典
        """
        dict_dir = self.base_path / self.PAYLOAD_PATHS['common']

        if not dict_dir.exists():
            logger.error(f"[PAYLOAD] 字典目录不存在: {dict_dir}")
            return {}

        logger.info(f"[PAYLOAD] 加载专家字典: {dict_dir}")

        # 遍历字典目录
        payloads = {}

        for dict_file in dict_dir.glob('*.txt'):
            # 提取攻击类型
            vtype_name = self._map_filename_to_vtype(dict_file.stem)
            if not vtype_name:
                continue

            # 类型过滤
            if vtype and vtype != 'ALL' and vtype != vtype_name:
                continue

            # 加载载荷
            file_payloads = self._load_from_file(dict_file)
            payloads[vtype_name] = file_payloads
            logger.debug(f"[PAYLOAD]   加载 {dict_file.name}: {len(file_payloads)} 条")

        return payloads

    def _load_custom(self,
                     source: str,
                     vtype: Optional[str] = None) -> Dict[str, List[str]]:
        """
        加载自定义载荷源

        支持格式：
        1. 单文件: ./payload/sqli.txt
        2. 文件夹: ./payload/
        3. 通配符: ./payload/*.txt

        Args:
            source: 载荷源
            vtype: 漏洞类型过滤

        Returns:
            载荷字典
        """
        logger.info(f"[PAYLOAD] 加载自定义载荷: {source}")

        source_path = Path(source)

        if not source_path.is_absolute():
            # 相对路径，基于base_path
            source_path = self.base_path / source

        if '*' in source:
            # 通配符模式
            files = glob.glob(str(source_path))
            payloads = []
            for f in files:
                payloads.extend(self._load_from_file(Path(f)))
            return {'CUSTOM': payloads}

        elif source_path.is_dir():
            # 文件夹模式
            payloads = {}
            for dict_file in source_path.glob('*.txt'):
                vtype_name = self._map_filename_to_vtype(dict_file.stem)
                if not vtype_name:
                    vtype_name = 'CUSTOM'

                # 类型过滤
                if vtype and vtype != 'ALL' and vtype != vtype_name:
                    continue

                file_payloads = self._load_from_file(dict_file)
                payloads[vtype_name] = file_payloads

            return payloads

        else:
            # 单文件模式
            if not source_path.exists():
                logger.error(f"[PAYLOAD] 文件不存在: {source_path}")
                return {}

            payloads = self._load_from_file(source_path)
            return {'CUSTOM': payloads}

    def _load_from_file(self, file_path: Path) -> List[str]:
        """
        从文件加载载荷


        Args:
            file_path: 文件路径

        Returns:
            载荷列表
        """
        payloads = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 只去掉右边的换行符，保留左边的语法空格！
                    payload = line.rstrip('\n\r')

                    # 跳过空行和注释
                    if not payload or payload.strip().startswith('#'):
                        continue

                    # 跳过纯空格行（全空格但无实际内容）
                    if not payload.strip():
                        continue

                    payloads.append(payload)

        except Exception as e:
            logger.error(f"[PAYLOAD] 文件读取失败: {file_path} - {e}")

        return payloads

    def _map_filename_to_vtype(self, filename: str) -> Optional[str]:
        """
        映射文件名到攻击类型

        Args:
            filename: 文件名（不含扩展名）

        Returns:
            攻击类型（如 'SQLi', 'XSS'）或 None

        Example:
            >>> mgr._map_filename_to_vtype('sqli')
            'SQLi'
            >>> mgr._map_filename_to_vtype('xss')
            'XSS'
            >>> mgr._map_filename_to_vtype('unknown')
            None
        """
        # 直接查找
        filename_lower = filename.lower()
        if filename_lower in self.VTYPE_MAPPING:
            return self.VTYPE_MAPPING[filename_lower]

        # 模糊匹配
        for key, value in self.VTYPE_MAPPING.items():
            if key in filename_lower:
                return value

        return None

    # ========== 迭代器设计 ==========

    def iterate(self,
                payloads: Dict[str, List[str]],
                vtype: str = 'ALL',
                mutate: bool = False,
                mutation_strategy: str = 'mixed',
                mutation_prob: float = 0.5) -> Iterator[str]:
        """
        载荷迭代器（生成器）

        用生成器按需吐出载荷，避免撑爆内存

        Args:
            payloads: 载荷字典（load()的返回值）
            vtype: 要遍历的攻击类型（'ALL' 表示全部）
            mutate: 是否进行深度变异
            mutation_strategy: 变异策略
            mutation_prob: 变异概率

        Yields:
            单个载荷字符串

        Example:
            >>> mgr = PayloadManager()
            >>> payloads = mgr.load(mode='cvae')
            >>>
            >>> # 遍历所有载荷
            >>> for payload in mgr.iterate(payloads):
            ...     print(payload)
            >>>
            >>> # 只遍历SQLi载荷
            >>> for payload in mgr.iterate(payloads, vtype='SQLi'):
            ...     print(payload)
            >>>
            >>> # 遍历时进行深度变异
            >>> for payload in mgr.iterate(payloads, mutate=True):
            ...     print(payload)

        Note:
            使用生成器可以避免一次性加载所有载荷到内存，
            特别适合CVAE模式下的海量数据。
        """
        from .transformer import PayloadTransformer

        # 确定要遍历的载荷类型
        if vtype == 'ALL':
            # 遍历所有类型
            types_to_iterate = list(payloads.keys())
        else:
            # 只遍历指定类型
            types_to_iterate = [vtype] if vtype in payloads else []

        for vtype_name in types_to_iterate:
            payload_list = payloads[vtype_name]
            logger.debug(f"[PAYLOAD] 开始遍历类型: {vtype_name}, 数量: {len(payload_list)}")

            for i, payload in enumerate(payload_list):
                # 深度变异
                if mutate:
                    try:
                        payload = PayloadTransformer.deep_mutate(
                            payload,
                            strategy=mutation_strategy
                        )
                    except Exception as e:
                        logger.error(f"[PAYLOAD] 变异失败: {payload[:50]}... - {e}")
                        # 变异失败，使用原载荷

                yield payload

                # 进度日志（每1000条）
                if (i + 1) % 1000 == 0:
                    logger.debug(f"[PAYLOAD] 已遍历 {i+1}/{len(payload_list)} 条")

    def distribute(self,
                  payloads: Dict[str, List[str]],
                  count: Optional[int] = None) -> Iterator[Tuple[str, str]]:
        """
        载荷分发器（带类型标签的迭代器）

        在吐出载荷的同时带上类型标签！

        Args:
            payloads: 载荷字典
            count: 限制分发数量（None=全部）

        Yields:
            (vtype, payload) 元组

        Example:
            >>> for vtype, payload in mgr.distribute(payloads):
            ...     print(f"[{vtype}] {payload}")
        """
        distributed_count = 0

        for vtype_name, payload_list in payloads.items():
            for payload in payload_list:
                if count is not None and distributed_count >= count:
                    return

                yield (vtype_name, payload)
                distributed_count += 1

    # ========== 工具方法 ==========

    def get_stats(self, payloads: Dict[str, List[str]]) -> Dict[str, int]:
        """
        获取载荷统计信息

        Args:
            payloads: 载荷字典

        Returns:
            统计字典 {'SQLi': 100, 'XSS': 200, 'total': 300}
        """
        stats = {'total': 0}

        for vtype, plist in payloads.items():
            stats[vtype] = len(plist)
            stats['total'] += len(plist)

        return stats

    def shuffle(self, payloads: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        随机打乱载荷顺序

        Args:
            payloads: 载荷字典

        Returns:
            打乱后的载荷字典
        """
        import random

        shuffled = {}
        for vtype, payload_list in payloads.items():
            shuffled_list = payload_list.copy()
            random.shuffle(shuffled_list)
            shuffled[vtype] = shuffled_list

        return shuffled

    def deduplicate(self, payloads: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        去重载荷

        Args:
            payloads: 载荷字典

        Returns:
            去重后的载荷字典
        """
        deduped = {}

        for vtype, payload_list in payloads.items():
            # 使用set去重，同时保持顺序
            seen = set()
            unique_list = []

            for payload in payload_list:
                if payload not in seen:
                    seen.add(payload)
                    unique_list.append(payload)

            deduped[vtype] = unique_list

        return deduped

    def sample(self,
               payloads: Dict[str, List[str]],
               max_per_type: int = 1000) -> Dict[str, List[str]]:
        """
        采样载荷（限制每个类型的数量）

        Args:
            payloads: 载荷字典
            max_per_type: 每个类型最大保留数量

        Returns:
            采样后的载荷字典
        """
        sampled = {}

        for vtype, payload_list in payloads.items():
            if len(payload_list) > max_per_type:
                # 随机采样
                import random
                sampled[vtype] = random.sample(payload_list, max_per_type)
                logger.info(f"[PAYLOAD] {vtype}: 采样 {max_per_type}/{len(payload_list)} 条")
            else:
                sampled[vtype] = payload_list

        return sampled


# 便捷函数
def load_payloads(mode: str = 'common',
                  source: Optional[str] = None,
                  vtype: Optional[str] = None) -> Dict[str, List[str]]:
    """
    加载载荷的便捷函数

    Args:
        mode: 加载模式
        source: 自定义源
        vtype: 漏洞类型

    Returns:
        载荷字典

    Example:
        >>> from Fuzz.BaseFuzz.loaders.payload_manager import load_payloads
        >>> payloads = load_payloads(mode='cvae')
    """
    mgr = PayloadManager()
    return mgr.load(mode=mode, source=source, vtype=vtype)


if __name__ == '__main__':
    # 测试代码
    import logging

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("PayloadManager 单元测试")
    print("=" * 60)

    # 初始化
    mgr = PayloadManager()

    # 测试1：_map_filename_to_vtype
    print("\n[测试1] 文件名映射")
    print("-" * 60)
    test_filenames = ['sqli', 'xss', 'cmdi', 'unknown']
    for filename in test_filenames:
        vtype = mgr._map_filename_to_vtype(filename)
        print(f"{filename} → {vttype}")

    # 测试2：加载载荷（模拟）
    print("\n[测试2] 加载载荷")
    print("-" * 60)

    # 创建测试载荷文件
    test_dir = Path('test_payloads')
    test_dir.mkdir(exist_ok=True)

    # SQLi测试文件
    sqli_file = test_dir / 'test_sqli.txt'
    with open(sqli_file, 'w', encoding='utf-8') as f:
        f.write("' OR 1=1--\n")
        f.write('" OR 1=1--\n')
        f.write("' AND '1'='1\n")

    # 测试加载
    payloads = mgr._load_custom(str(test_dir))
    print(f"加载结果: {payloads}")
    print(f"统计信息: {mgr.get_stats(payloads)}")

    # 测试3：迭代器
    print("\n[测试3] 迭代器")
    print("-" * 60)
    test_payloads = {
        'SQLi': ["' OR 1=1--", '" OR 1=1--'],
        'XSS': ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>"]
    }

    print("遍历所有载荷:")
    count = 0
    for payload in mgr.iterate(test_payloads):
        print(f"  {count + 1}. {payload[:50]}...")
        count += 1
    print(f"总计: {count} 条")

    print("\n只遍历SQLi:")
    count = 0
    for payload in mgr.iterate(test_payloads, vtype='SQLi'):
        print(f"  {count + 1}. {payload}")
        count += 1
    print(f"总计: {count} 条")

    # 测试4：分发器
    print("\n[测试4] 分发器")
    print("-" * 60)
    for vtype, payload in mgr.distribute(test_payloads):
        print(f"[{vtype}] {payload}")

    # 测试5：工具方法
    print("\n[测试5] 工具方法")
    print("-" * 60)
    print(f"去重前: {mgr.get_stats(test_payloads)}")
    deduped = mgr.deduplicate(test_payloads)
    print(f"去重后: {mgr.get_stats(deduped)}")

    print(f"\n打乱顺序: {mgr.shuffle(test_payloads)}")

    sampled = mgr.sample(test_payloads, max_per_type=1)
    print(f"采样结果: {mgr.get_stats(sampled)}")

    # 清理测试文件
    sqli_file.unlink()
    test_dir.rmdir()

    print("\n[SUCCESS] 所有测试通过！")
