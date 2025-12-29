#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DBSCAN聚类器与隐空间可视化工具
=============================

基于生成的隐空间特征进行DBSCAN密度聚类
实现降维可视化、质心提取和精锐载荷筛选功能


"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from scipy.spatial.distance import cdist
import argparse

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CVAEClusterer:
    """隐空间聚类器

    基于DBSCAN算法对生成的隐空间向量进行聚类分析
    支持多种降维方法和精锐载荷筛选策略
    """

    def __init__(self, embeddings: np.ndarray, payloads: List[str], metadata: List[Dict],
                 valid_mask: Optional[np.ndarray] = None, label_weight: float = 15.0):
        """初始化聚类器

        Args:
            embeddings: [N, latent_dim] 隐空间特征矩阵
            payloads: 对应的载荷文本列表
            metadata: 载荷元数据列表
            valid_mask: [N] 有效性掩码，标记哪些特征是有效的
            label_weight: 标签权重因子，用于增强不同类型载荷的分离度 (建议15.0-20.0，强力类型隔离)
        """
        self.embeddings = embeddings
        self.payloads = payloads
        self.metadata = metadata
        self.valid_mask = valid_mask if valid_mask is not None else np.ones(len(embeddings), dtype=bool)
        self.label_weight = label_weight

        # 特征增强：标签权重注入
        self.enhanced_embeddings = self._create_label_enhanced_embeddings()

        # 确保数据一致性
        assert len(embeddings) == len(payloads) == len(metadata), "输入数据长度不一致"
        if valid_mask is not None:
            assert len(valid_mask) == len(embeddings), "有效性掩码长度不匹配"

        # 过滤有效样本（使用增强特征）
        self.valid_indices = np.where(self.valid_mask)[0]
        self.valid_embeddings = self.enhanced_embeddings[self.valid_mask]  # 使用增强特征
        self.valid_payloads = [payloads[i] for i in self.valid_indices]
        self.valid_metadata = [metadata[i] for i in self.valid_indices]

        logger.info(f"开始聚类器初始化")
        logger.info(f"   总样本数: {len(embeddings)}")
        logger.info(f"   有效样本: {len(self.valid_embeddings)}")
        logger.info(f"   隐空间维度: {embeddings.shape[1]}")

        # 聚类结果
        self.clustering_results = {}
        self.reduced_embeddings = {}
        self.refined_payloads = []

    def _create_label_enhanced_embeddings(self) -> np.ndarray:
        """创建标签增强的特征向量

        将原始32维隐向量与6维带权重的One-hot标签向量拼接，
        形成38维复合特征向量，强制不同类型载荷在空间上分离。

        Returns:
            [N, 38] 标签增强特征矩阵
        """
        logger.info(f"开始标签权重注入 (权重因子: {self.label_weight})")

        # 从metadata中提取标签并转换为One-hot编码
        labels = []
        for meta in self.metadata:
            if 'label' in meta:
                label = int(meta['label'])
            else:
                # 如果没有label字段，尝试从type字段推断
                attack_type = meta.get('type', 'SQLi')
                type_to_label = {
                    'SQLi': 0, 'XSS': 1, 'CMDi': 2,
                    'Overflow': 3, 'XXE': 4, 'SSI': 5
                }
                label = type_to_label.get(attack_type, 0)
            labels.append(label)

        labels = np.array(labels)

        # 创建One-hot编码 (6维)
        n_samples = len(labels)
        n_classes = 6
        one_hot_labels = np.zeros((n_samples, n_classes))
        one_hot_labels[np.arange(n_samples), labels] = 1

        # 应用标签权重
        weighted_one_hot = one_hot_labels * self.label_weight

        # 拼接原始特征和加权标签特征
        enhanced_embeddings = np.hstack([self.embeddings, weighted_one_hot])

        logger.info(f"   原始特征维度: {self.embeddings.shape[1]}")
        logger.info(f"   标签特征维度: {weighted_one_hot.shape[1]}")
        logger.info(f"   增强后维度: {enhanced_embeddings.shape[1]}")

        return enhanced_embeddings

    def find_optimal_eps(self, k: int = 5, method: str = 'knee', embeddings: Optional[np.ndarray] = None) -> float:
        """使用K-距离图寻找最优eps参数

        Args:
            k: K距离的k值，通常设为min_samples-1
            method: 寻找方法 ('knee', 'percentile')
            embeddings: 可选的特征矩阵，如果为None则使用valid_embeddings

        Returns:
            最优的eps值
        """
        logger.info(f"寻找最优eps参数 (k={k}, method={method})")

        # 使用指定的特征矩阵或默认的有效特征
        target_embeddings = embeddings if embeddings is not None else self.valid_embeddings

        if len(target_embeddings) < k:
            logger.warning(f"样本数不足，使用默认eps=1.0")
            return 1.0

        # 计算k-距离
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(target_embeddings)
        distances, _ = nbrs.kneighbors(target_embeddings)
        k_distances = distances[:, k]  # 距离第k个最近邻居的距离

        # 排序
        sorted_distances = np.sort(k_distances)[::-1]

        if method == 'knee':
            # 肘部法：寻找曲率最大的点
            # 简化的肘部检测：寻找二阶差分最大的点
            second_diff = np.diff(sorted_distances, 2)
            knee_idx = np.argmax(second_diff) + 1
            optimal_eps = sorted_distances[knee_idx]

        elif method == 'percentile':
            # 百分位数法：使用95%分位数
            optimal_eps = np.percentile(sorted_distances, 95)

        else:
            raise ValueError(f"未知的eps寻找方法: {method}")

        logger.info(f"最优eps值: {optimal_eps:.4f}")
        return optimal_eps

    def perform_clustering(self, eps: float = None, min_samples: int = None) -> Dict:
        """执行DBSCAN聚类

        Args:
            eps: 邻域半径，如果为None则自动寻找
            min_samples: 核心点最小样本数，如果为None则设为2*latent_dim

        Returns:
            聚类结果字典
        """
        logger.info(f"开始DBSCAN聚类")

        # 关键修复：重新设计预处理管道
        logger.info(f"开始正确的预处理管道（隐向量标准化+标签权重保留）")

        # 第1步：拆分增强特征为原始隐向量和加权标签向量
        original_embeddings = self.valid_embeddings[:, :32]  # 前32列：原始隐向量
        weighted_labels = self.valid_embeddings[:, 32:]      # 后6列：加权标签向量

        logger.info(f"   原始隐向量维度: {original_embeddings.shape}")
        logger.info(f"   加权标签向量维度: {weighted_labels.shape}")
        logger.info(f"   标签权重范围: [{np.min(weighted_labels):.1f}, {np.max(weighted_labels):.1f}]")

        # 第2步：仅对原始隐向量执行标准化（绝不触碰标签权重！）
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(original_embeddings)

        # 保存Scaler模型以便后续使用
        self.scaler_model = scaler

        logger.info(f"   隐向量标准化完成:")
        logger.info(f"   标准化前均值: {np.mean(original_embeddings, axis=0)[:3]}...")
        logger.info(f"   标准化前标准差: {np.std(original_embeddings, axis=0)[:3]}...")
        logger.info(f"   标准化后均值: {np.mean(scaled_embeddings, axis=0)[:3]}...")
        logger.info(f"   标准化后标准差: {np.std(scaled_embeddings, axis=0)[:3]}...")

        # 第3步：将标准化隐向量与原始权重标签重新拼接（保持标签权重效果！）
        embeddings_processed = np.hstack([scaled_embeddings, weighted_labels])

        logger.info(f"   拼接后特征维度: {embeddings_processed.shape}")
        logger.info(f"   标签权重保持完整: {np.mean(embeddings_processed[:, 32:], axis=0)}")

        # 第4步：PCA降维预处理（从38维降到12维，保留更多标签主导结构）
        logger.info(f"开始PCA降维预处理（保留标签主导结构）")
        pca_components = 12
        pca = PCA(n_components=pca_components, random_state=42)
        embeddings_for_clustering = pca.fit_transform(embeddings_processed)

        # 保存PCA模型以便后续使用
        self.pca_model = pca

        logger.info(f"   PCA降维: 38维 -> {pca_components}维")
        logger.info(f"   方差解释比例: {np.sum(pca.explained_variance_ratio_):.3f}")
        logger.info(f"   前5个成分解释比例: {pca.explained_variance_ratio_[:5]}")
        logger.info(f"   标签特征在PCA中的影响力得到保留！")

        # 参数设置（基于降维后的空间）
        if eps is None:
            eps = self.find_optimal_eps(k=min_samples or 3, embeddings=embeddings_for_clustering)
        if min_samples is None:
            min_samples = 3  # 固定为较小值，适合低维空间

        logger.info(f"聚类参数: eps={eps:.4f}, min_samples={min_samples}")

        # 在12维PCA空间上执行DBSCAN（标签权重主导的空间）
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        cluster_labels = dbscan.fit_predict(embeddings_for_clustering)

        # 分析聚类结果
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        noise_ratio = n_noise/len(cluster_labels)*100

        logger.info(f"🎯 聚类结果分析:")
        logger.info(f"   簇数量: {n_clusters}")
        logger.info(f"   噪声点: {n_noise} ({noise_ratio:.1f}%)")
        logger.info(f"   有效聚类率: {100-noise_ratio:.1f}%")

        # 友好提示
        if n_clusters == 0:
            logger.warning(f"未检测到任何聚类！建议：")
            logger.warning(f"  1. 增大eps参数（当前: {eps:.3f}）")
            logger.warning(f"  2. 减小min_samples参数（当前: {min_samples}）")
            logger.warning(f"  3. 检查数据质量和多样性")

        # 计算轮廓系数（仅当簇数>1且噪声点不过多时）
        silhouette_avg = None
        if n_clusters > 1 and n_noise < len(cluster_labels) * 0.5:
            try:
                silhouette_avg = silhouette_score(
                    embeddings_for_clustering[cluster_labels != -1],  # 使用降维后的特征
                    cluster_labels[cluster_labels != -1]
                )
                logger.info(f"   轮廓系数: {silhouette_avg:.3f}")
            except Exception as e:
                logger.warning(f"轮廓系数计算失败: {e}")

        # 保存聚类结果（确保Python原生类型）
        self.clustering_results = {
            'labels': cluster_labels,
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise),
            'eps': float(eps) if eps is not None else None,
            'min_samples': int(min_samples) if min_samples is not None else None,
            'silhouette_score': float(silhouette_avg) if silhouette_avg is not None else None,
            'cluster_info': {}
        }

        # 分析每个簇的详细信息
        for label in unique_labels:
            if label == -1:
                continue  # 噪声点稍后处理

            cluster_mask = cluster_labels == label
            cluster_size = np.sum(cluster_mask)
            cluster_embeddings = embeddings_for_clustering[cluster_mask]  # 使用降维后的特征

            # 计算簇的质心（在降维空间中）
            centroid = np.mean(cluster_embeddings, axis=0)

            # 计算簇的统计信息（确保JSON序列化兼容性）
            cluster_info = {
                'label': int(label),
                'size': int(cluster_size),
                'centroid': [float(x) for x in centroid.tolist()],  # 12维质心，确保float类型
                'indices': [int(x) for x in self.valid_indices[cluster_mask].tolist()],  # 确保int类型
                'avg_distance_to_centroid': float(np.mean(cdist([centroid], cluster_embeddings)[0]))
            }

            self.clustering_results['cluster_info'][int(label)] = cluster_info
            logger.info(f"   簇 {label}: {cluster_size} 个样本")

        return self.clustering_results

    def reduce_dimensions(self, method: str = 'tsne', n_components: int = 2, **kwargs) -> np.ndarray:
        """降维处理用于可视化

        Args:
            method: 降维方法 ('tsne', 'pca')
            n_components: 降维后的维度
            **kwargs: 降维算法的额外参数

        Returns:
            降维后的特征矩阵
        """
        logger.info(f"开始降维处理 (方法: {method})")

        if method == 'tsne':
            # t-SNE降维（带PCA优化）
            n_samples = len(self.valid_embeddings)
            n_features = self.valid_embeddings.shape[1]

            # 当样本数超过2000时，先用PCA降维加速
            if n_samples > 2000:
                pca_components = min(50, n_features, n_samples // 4)
                logger.info(f"样本数量较多({n_samples})，先用PCA降维到{pca_components}维加速t-SNE")

                pca = PCA(n_components=pca_components, random_state=42)
                embeddings_for_tsne = pca.fit_transform(self.valid_embeddings)

                # 显示PCA预处理信息
                explained_variance = pca.explained_variance_ratio_
                logger.info(f"   PCA预处理解释方差比例: {sum(explained_variance[:10]):.3f}")
                logger.info(f"   累计解释方差: {np.cumsum(explained_variance)[-1]:.3f}")
            else:
                embeddings_for_tsne = self.valid_embeddings

            perplexity = kwargs.get('perplexity', min(30, len(embeddings_for_tsne) - 1))
            n_iter = kwargs.get('n_iter', 1000)

            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                max_iter=n_iter,
                random_state=42,
                verbose=1
            )
            reduced_embeddings = tsne.fit_transform(embeddings_for_tsne)

        elif method == 'pca':
            # PCA降维
            pca = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = pca.fit_transform(self.valid_embeddings)

            # 显示解释方差比例
            explained_variance = pca.explained_variance_ratio_
            logger.info(f"   PCA解释方差比例: {explained_variance}")
            logger.info(f"   累计解释方差: {np.cumsum(explained_variance)[-1]:.3f}")

        else:
            raise ValueError(f"未知的降维方法: {method}")

        self.reduced_embeddings[method] = reduced_embeddings
        logger.info(f"降维完成: {reduced_embeddings.shape}")
        return reduced_embeddings

    def visualize_clusters(self, method: str = 'tsne', save_path: str = None,
                          figsize: Tuple[int, int] = (12, 8)) -> None:
        """可视化聚类结果

        Args:
            method: 使用的降维方法
            save_path: 保存路径，如果为None则不保存
            figsize: 图像大小
        """
        logger.info(f"🎨 开始绘制聚类可视化图")

        if not self.clustering_results:
            raise ValueError("请先执行聚类分析")

        if method not in self.reduced_embeddings:
            self.reduce_dimensions(method)

        reduced_embeddings = self.reduced_embeddings[method]
        cluster_labels = self.clustering_results['labels']

        # 创建颜色映射
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        # 创建图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 左图：所有点，按簇着色
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            if label == -1:
                # 噪声点用红色星星标记，更醒目
                ax1.scatter(
                    reduced_embeddings[mask, 0],
                    reduced_embeddings[mask, 1],
                    c='red',
                    marker='*',
                    s=80,  # 增大尺寸
                    alpha=0.8,  # 增加透明度
                    edgecolors='darkred',  # 添加边缘
                    linewidths=1,
                    label=f'噪声点 ({np.sum(mask)}个)',
                    zorder=10  # 确保在最上层
                )
                # 添加噪声点总数标注
                noise_count = np.sum(mask)
                if noise_count > 0:
                    # 计算噪声点的中心位置
                    center_x = np.mean(reduced_embeddings[mask, 0])
                    center_y = np.mean(reduced_embeddings[mask, 1])
                    ax1.annotate(f'噪声点: {noise_count}',
                               xy=(center_x, center_y),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               fontsize=9, fontweight='bold')
            else:
                ax1.scatter(
                    reduced_embeddings[mask, 0],
                    reduced_embeddings[mask, 1],
                    c=[colors[i]],
                    s=50,
                    alpha=0.7,
                    label=f'簇 {label} ({np.sum(mask)}个)'
                )

        ax1.set_xlabel(f'{method.upper()} 第1维')
        ax1.set_ylabel(f'{method.upper()} 第2维')
        ax1.set_title(f'CVAE隐空间聚类结果 ({method.upper()}降维)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 右图：簇分布统计
        cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
        cluster_names = ['噪声点' if label == -1 else f'簇 {label}' for label in unique_labels]

        bars = ax2.bar(range(len(cluster_names)), cluster_sizes, color=colors)
        ax2.set_xlabel('簇')
        ax2.set_ylabel('样本数量')
        ax2.set_title('簇分布统计')
        ax2.set_xticks(range(len(cluster_names)))
        ax2.set_xticklabels(cluster_names, rotation=45)

        # 在柱子上显示数值
        for bar, size in zip(bars, cluster_sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(cluster_sizes),
                    f'{size}', ha='center', va='bottom')

        plt.tight_layout()

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图已保存: {save_path}")

        plt.show()

    def select_refined_payloads(self, samples_per_cluster: int = 5, keep_all_noise: bool = True) -> List[Dict]:
        """筛选精锐载荷

        Args:
            samples_per_cluster: 每个簇保留的样本数（质心附近）
            keep_all_noise: 是否保留所有噪声点

        Returns:
            精锐载荷列表
        """
        logger.info(f"开始筛选精锐载荷")
        logger.info(f"   每簇保留样本数: {samples_per_cluster}")
        logger.info(f"   保留所有噪声点: {keep_all_noise}")

        if not self.clustering_results:
            raise ValueError("请先执行聚类分析")

        refined_payloads = []
        cluster_labels = self.clustering_results['labels']

        # 处理每个簇
        for cluster_id, cluster_info in self.clustering_results['cluster_info'].items():
            cluster_indices = cluster_info['indices']
            centroid = np.array(cluster_info['centroid'])

            # 修复索引映射问题：直接从cluster_info获取indices，这些已经是全局索引
            # 获取对应的增强特征
            cluster_mask = np.isin(self.valid_indices, cluster_indices)
            cluster_enhanced_embeddings = self.valid_embeddings[cluster_mask]

            # 应用正确的预处理管道（与perform_clustering完全一致）
            if hasattr(self, 'scaler_model') and hasattr(self, 'pca_model'):
                # 拆分特征：原始隐向量 + 加权标签
                original_part = cluster_enhanced_embeddings[:, :32]
                label_part = cluster_enhanced_embeddings[:, 32:]

                # 仅对隐向量部分标准化，保持标签权重不变
                scaled_original = self.scaler_model.transform(original_part)

                # 重新拼接并应用PCA
                cluster_processed = np.hstack([scaled_original, label_part])
                cluster_embeddings_pca = self.pca_model.transform(cluster_processed)
            elif hasattr(self, 'pca_model'):
                # 仅有PCA模型的情况（向后兼容）
                cluster_embeddings_pca = self.pca_model.transform(cluster_enhanced_embeddings)
            else:
                cluster_embeddings_pca = cluster_enhanced_embeddings

            # 计算每个样本到质心的距离（质心已经在PCA空间中）
            distances = cdist([centroid], cluster_embeddings_pca)[0]

            # 选择距离质心最近的样本
            n_select = min(samples_per_cluster, len(cluster_indices))
            selected_local_indices = np.argsort(distances)[:n_select]

            # 确保索引映射正确
            valid_cluster_indices = self.valid_indices[cluster_mask]

            for i, local_idx in enumerate(selected_local_indices):
                global_idx = int(valid_cluster_indices[local_idx])
                refined_payload = {
                    'id': int(global_idx),
                    'payload': self.payloads[global_idx],
                    'metadata': self.metadata[global_idx],
                    'cluster_id': int(cluster_id),
                    'distance_to_centroid': float(distances[local_idx]),
                    'selection_reason': 'centroid_close'
                }
                refined_payloads.append(refined_payload)

            logger.info(f"   簇 {cluster_id}: 选择 {n_select} 个质心样本")

        # 处理噪声点
        if keep_all_noise:
            noise_mask = cluster_labels == -1
            noise_indices = self.valid_indices[noise_mask]

            for idx in noise_indices:
                global_idx = int(idx)
                refined_payload = {
                    'id': global_idx,
                    'payload': self.payloads[global_idx],
                    'metadata': self.metadata[global_idx],
                    'cluster_id': -1,
                    'distance_to_centroid': float('inf'),
                    'selection_reason': 'noise_outlier'
                }
                refined_payloads.append(refined_payload)

            logger.info(f"   噪声点: 保留 {len(noise_indices)} 个异常样本")

        self.refined_payloads = refined_payloads

        # 统计信息
        total_selected = len(refined_payloads)
        total_original = len(self.valid_embeddings)
        reduction_ratio = (total_original - total_selected) / total_original

        logger.info(f"精锐载荷筛选完成！")
        logger.info(f"   原始样本: {total_original}")
        logger.info(f"   筛选后: {total_selected}")
        logger.info(f"   压缩比例: {reduction_ratio*100:.1f}%")

        return refined_payloads

    def save_refined_payloads(self, output_path: str) -> None:
        """保存精锐载荷到文件

        Args:
            output_path: 输出文件路径
        """
        if not self.refined_payloads:
            raise ValueError("请先执行精锐载荷筛选")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 保存为txt格式（仅载荷文本）
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in self.refined_payloads:
                f.write(item['payload'] + '\n')

        logger.info(f"精锐载荷已保存: {output_file}")
        logger.info(f"   总数: {len(self.refined_payloads)} 个")

    def save_clustering_results(self, output_dir: str) -> None:
        """保存完整的聚类分析结果

        Args:
            output_dir: 输出目录
        """
        def convert_numpy_types(obj):
            """递归转换NumPy类型为Python原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存聚类结果（处理numpy数组序列化）
        results_for_saving = convert_numpy_types(self.clustering_results)

        results_file = output_path / "clustering_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_for_saving, f, ensure_ascii=False, indent=2)
        logger.info(f"聚类结果已保存: {results_file}")

        # 保存降维结果
        for method, embeddings in self.reduced_embeddings.items():
            embeddings_file = output_path / f"reduced_embeddings_{method}.npy"
            np.save(embeddings_file, embeddings)
            logger.info(f"降维结果已保存: {embeddings_file}")

        # 保存精锐载荷详细信息
        if self.refined_payloads:
            refined_file = output_path / "refined_payloads.json"
            # 确保精锐载荷数据也是JSON可序列化的
            refined_payloads_for_saving = convert_numpy_types(self.refined_payloads)
            with open(refined_file, 'w', encoding='utf-8') as f:
                json.dump(refined_payloads_for_saving, f, ensure_ascii=False, indent=2)
            logger.info(f"精锐载荷详情已保存: {refined_file}")

        logger.info(f"所有分析结果已保存到: {output_path}")


# 使用示例：
#
# # 初始化聚类器（已优化：标签权重注入 + PCA降维预处理）
# clusterer = CVAEClusterer(
#     embeddings=latent_features,      # [N, 32] 原始隐空间特征
#     payloads=payloads,               # [N] 载荷文本列表
#     metadata=metadata,               # [N] 包含label字段的元数据
#     label_weight=8.0                 # 标签权重（更高权重强制分离不同类型）
# )
#
# # 执行优化后的聚类（自动：标签增强 -> PCA降维 -> DBSCAN聚类）
# results = clusterer.perform_clustering()
#
# # 筛选精锐载荷（每个簇保留质心附近5个样本 + 所有噪声点）
# refined = clusterer.select_refined_payloads(samples_per_cluster=5, keep_all_noise=True)
#
# print(f"聚类效果: {results['n_clusters']}个簇，噪声比例{results['n_noise']/len(payloads)*100:.1f}%")



def main():
    """主函数 - 支持命令行调用"""
    parser = argparse.ArgumentParser(
        description="CVAE隐空间聚类器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python clusterer.py --embeddings Data/generated/latent_embeddings.npy
                     --payloads Data/generated/raw_payloads.txt
                     --metadata Data/generated/payload_metadata.json
        """
    )

    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='隐空间特征文件路径 (.npy)'
    )

    parser.add_argument(
        '--payloads',
        type=str,
        required=True,
        help='载荷文件路径 (.txt)'
    )

    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='载荷元数据文件路径 (.json)'
    )

    parser.add_argument(
        '--valid-mask',
        type=str,
        help='有效性掩码文件路径 (.npy)'
    )

    parser.add_argument(
        '--eps',
        type=float,
        help='DBSCAN的eps参数（自动寻找如果未指定）'
    )

    parser.add_argument(
        '--min-samples',
        type=int,
        help='DBSCAN的min_samples参数'
    )

    parser.add_argument(
        '--samples-per-cluster',
        type=int,
        default=5,
        help='每个簇保留的样本数 (默认: 5)'
    )

    parser.add_argument(
        '--keep-noise',
        action='store_true',
        default=False,
        help='保留所有噪声点 (默认: False)'
    )

    parser.add_argument(
        '--cluster',
        action='store_true',
        default=True,
        help='执行聚类分析 (默认: True)'
    )

    parser.add_argument(
        '--reduction-method',
        type=str,
        default='tsne',
        choices=['tsne', 'pca'],
        help='降维方法 (默认: tsne)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='Data/clustered',
        help='输出目录 (默认: Data/clustered)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='生成可视化图像'
    )

    parser.add_argument(
        '--label-weight',
        type=float,
        default=8.0,
        help='标签权重因子，用于增强不同类型载荷的分离度 (默认: 8.0)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CVAE隐空间聚类器")
    print("=" * 80)

    try:
        # 加载数据
        logger.info("📂 加载数据文件...")

        # 加载隐空间特征
        embeddings = np.load(args.embeddings)
        logger.info(f"   隐空间特征: {embeddings.shape}")

        # 加载载荷
        with open(args.payloads, 'r', encoding='utf-8') as f:
            payloads = [line.strip() for line in f.readlines()]
        logger.info(f"   载荷数量: {len(payloads)}")

        # 加载元数据
        with open(args.metadata, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"   元数据数量: {len(metadata)}")

        # 加载有效性掩码
        valid_mask = None
        if args.valid_mask:
            valid_mask = np.load(args.valid_mask)
            logger.info(f"   有效性掩码: {valid_mask.shape}")

        # 确保数据长度一致
        min_len = min(len(embeddings), len(payloads), len(metadata))
        embeddings = embeddings[:min_len]
        payloads = payloads[:min_len]
        metadata = metadata[:min_len]
        if valid_mask is not None:
            valid_mask = valid_mask[:min_len]

        # 初始化聚类器
        clusterer = CVAEClusterer(embeddings, payloads, metadata, valid_mask, label_weight=args.label_weight)

        # 执行聚类
        clustering_results = clusterer.perform_clustering(
            eps=args.eps,
            min_samples=args.min_samples
        )

        # 降维
        clusterer.reduce_dimensions(method=args.reduction_method)

        # 可视化
        if args.visualize:
            viz_path = Path(args.output_dir) / "clustering_visualization.png"
            clusterer.visualize_clusters(
                method=args.reduction_method,
                save_path=str(viz_path)
            )

        # 筛选精锐载荷
        refined_payloads = clusterer.select_refined_payloads(
            samples_per_cluster=args.samples_per_cluster,
            keep_all_noise=args.keep_noise
        )

        # 保存结果
        clusterer.save_clustering_results(args.output_dir)

        # 保存精锐载荷到指定位置
        refined_path = Path(args.output_dir).parent / "fuzzing" / "refined_payloads.txt"
        clusterer.save_refined_payloads(str(refined_path))

        print("\n" + "=" * 80)
        print("聚类分析任务完成！")
        print("=" * 80)
        print(f"聚类结果:")
        print(f"   簇数量: {clustering_results['n_clusters']}")
        print(f"   噪声点: {clustering_results['n_noise']}")
        print(f"   精锐载荷: {len(refined_payloads)}")
        print(f"   压缩比例: {(1 - len(refined_payloads)/len(payloads))*100:.1f}%")

    except Exception as e:
        logger.error(f"❌ 聚类分析失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()