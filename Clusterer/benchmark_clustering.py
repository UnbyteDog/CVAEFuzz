#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVAE聚类器基准测试工具 (Benchmark Clustering)
==============================================

用于验证CVAEClusterer优化效果的基准测试工具，包括：
- 标签权重注入效果验证
- PCA降维预处理效果验证
- 聚类质量评估（噪声比例、簇纯度、轮廓系数）

使用场景：
1. 算法修改后的效果验证
2. 参数调优的基准测试
3. 生产环境问题调试
4. 新手使用示例参考

运行示例：
  # 标准基准测试
  python benchmark_clustering.py

  # 参数调优测试
  python benchmark_clustering.py --label-weight 5.0 --samples 1000

  # 快速性能测试
  python benchmark_clustering.py --quiet --no-viz

"""

import numpy as np
import json
import sys
import os
import argparse
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from clusterer import CVAEClusterer


def generate_test_data(n_samples=200, latent_dim=32, n_classes=6):
    """生成测试数据

    Args:
        n_samples: 样本数量
        latent_dim: 隐空间维度
        n_classes: 类别数量

    Returns:
        embeddings, payloads, metadata
    """
    print(f"[DEBUG] 生成测试数据 (N={n_samples}, dim={latent_dim}, classes={n_classes})")

    np.random.seed(42)

    # 生成隐空间特征（每个类别有不同均值）
    embeddings = []
    labels = []
    payloads = []
    metadata = []

    class_names = ['SQLi', 'XSS', 'CMDi', 'Overflow', 'XXE', 'SSI']

    for class_id in range(n_classes):
        # 每个类别的样本数
        n_class_samples = n_samples // n_classes

        # 为每个类别生成不同均值的特征
        class_mean = np.random.randn(latent_dim) * 2
        class_embeddings = np.random.randn(n_class_samples, latent_dim) * 0.5 + class_mean

        for i in range(n_class_samples):
            embeddings.append(class_embeddings[i])
            labels.append(class_id)

            # 生成测试载荷
            payload = f"{class_names[class_id]}_payload_{i:03d}"
            payloads.append(payload)

            # 生成元数据
            meta = {
                "id": len(embeddings) - 1,
                "label": class_id,
                "type": class_names[class_id],
                "attack_vector": class_names[class_id].lower(),
                "severity": np.random.choice(["high", "medium", "low"])
            }
            metadata.append(meta)

    embeddings = np.array(embeddings)
    return embeddings, payloads, metadata


def test_clustering(label_weight=8.0, n_samples=300, verbose=True, generate_viz=True):
    """测试优化后的聚类器

    Args:
        label_weight: 标签权重因子
        n_samples: 测试样本数量
        verbose: 是否显示详细信息
        generate_viz: 是否生成可视化图像
    """
    print("=" * 80)
    print("CVAE聚类器优化效果测试")
    print("=" * 80)

    # 生成测试数据
    embeddings, payloads, metadata = generate_test_data(n_samples=n_samples, latent_dim=32, n_classes=6)

    print(f"[OK] 测试数据生成完成:")
    print(f"   样本数量: {len(embeddings)}")
    print(f"   特征维度: {embeddings.shape[1]}")
    print(f"   类别分布: {np.bincount([meta['label'] for meta in metadata])}")

    # 初始化聚类器（使用重构优化的参数）
    if verbose:
        print(f"\n[START] 初始化重构聚类器 (label_weight={label_weight})...")
    clusterer = CVAEClusterer(
        embeddings=embeddings,
        payloads=payloads,
        metadata=metadata,
        label_weight=label_weight  # 超高权重强制分离不同类型
    )

    # 执行聚类
    print("\n[CLUSTERING] 执行重构聚类（标准化+PCA16维+超高位权重）...")
    results = clusterer.perform_clustering()
    # eps=None 自动寻找最优参数
    # min_samples=None 使用默认值3

    print(f"[RESULTS] 聚类结果:")
    print(f"   检测到簇数量: {results['n_clusters']}")
    print(f"   噪声点数量: {results['n_noise']}")
    print(f"   噪声点比例: {results['n_noise']/len(payloads)*100:.1f}%")
    print(f"   有效聚类率: {(1-results['n_noise']/len(payloads))*100:.1f}%")

    if results['silhouette_score'] is not None:
        print(f"   轮廓系数: {results['silhouette_score']:.3f}")

    # 分析每个簇的类别纯度
    print("\n[ANALYSIS] 分析簇的类别纯度...")
    cluster_labels = results['labels']
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue

        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # 统计该簇中各类别的分布
        cluster_label_distribution = {}
        for idx in cluster_indices:
            true_label = metadata[idx]['label']
            cluster_label_distribution[true_label] = cluster_label_distribution.get(true_label, 0) + 1

        # 计算主要类别比例
        dominant_count = max(cluster_label_distribution.values())
        purity = dominant_count / sum(cluster_label_distribution.values())

        print(f"   簇 {cluster_id}: {len(cluster_indices)}个样本, 纯度{purity:.1%}, 主类别{max(cluster_label_distribution, key=cluster_label_distribution.get)}")

    # 筛选精锐载荷
    print("\n[REFINEMENT] 筛选精锐载荷...")
    refined = clusterer.select_refined_payloads(samples_per_cluster=3, keep_all_noise=True)

    print(f"   精锐载荷数量: {len(refined)}")
    print(f"   压缩比例: {(1-len(refined)/len(payloads))*100:.1f}%")

    # 统计精锐载荷的类别分布
    refined_dist = {}
    for item in refined:
        cluster_id = item['cluster_id']
        refined_dist[cluster_id] = refined_dist.get(cluster_id, 0) + 1

    print(f"   精锐载荷簇分布: {dict(sorted(refined_dist.items()))}")

    # 可视化
    if generate_viz:
        print("\n[VISUALIZATION] 生成可视化图...")
        try:
            clusterer.reduce_dimensions(method='tsne')
            clusterer.visualize_clusters(method='tsne', save_path="test_clustering_results.png")
            print("   [OK] 可视化图已保存: test_clustering_results.png")
        except Exception as e:
            print(f"   [ERROR] 可视化失败: {e}")
    else:
        print("\n[VISUALIZATION] 跳过可视化生成")

    print("\n" + "=" * 80)
    print("测试完成！聚类器优化效果验证完毕")
    print("=" * 80)

    return clusterer, results


def main():
    """主函数 - 支持命令行参数"""
    parser = argparse.ArgumentParser(
        description="CVAE聚类器基准测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 标准测试
  python benchmark_clustering.py

  # 测试不同标签权重
  python benchmark_clustering.py --label-weight 5.0
  python benchmark_clustering.py --label-weight 10.0

  # 大规模测试
  python benchmark_clustering.py --samples 1000 --quiet

  # 仅返回结果用于脚本
  python benchmark_clustering.py --quiet --no-viz
        """
    )

    parser.add_argument(
        '--label-weight',
        type=float,
        default=8.0,
        help='标签权重因子 (默认: 8.0)'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=300,
        help='测试样本数量 (默认: 300)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式，仅显示关键结果'
    )

    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='不生成可视化图像'
    )

    args = parser.parse_args()

    try:
        clusterer, results = test_clustering(
            label_weight=args.label_weight,
            n_samples=args.samples,
            verbose=not args.quiet,
            generate_viz=not args.no_viz
        )

        if args.quiet:
            # 静默模式：仅输出关键指标
            print(f"RESULTS: clusters={results['n_clusters']}, "
                  f"noise_ratio={results['n_noise']/args.samples*100:.1f}%, "
                  f"silhouette={results.get('silhouette_score', 'N/A')}")

        return clusterer, results

    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()