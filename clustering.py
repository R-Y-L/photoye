#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - DBSCAN 人脸聚类模块

使用 DBSCAN 算法对人脸特征向量进行聚类，
相比 Union-Find 能更好地处理噪声点（离群人脸）
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize


def cluster_faces_dbscan(
    face_embeddings: List[Tuple[int, np.ndarray]],
    eps: float = 0.7,
    min_samples: int = 2,
    metric: str = 'cosine'
) -> Dict[str, any]:
    """
    使用 DBSCAN 算法对人脸特征向量进行聚类
    
    Args:
        face_embeddings: List of (face_id, embedding) tuples
        eps: DBSCAN 的邻域半径参数（余弦距离）
            - 余弦距离范围是 [0, 2]，0 表示完全相同，2 表示完全相反
            - 0.5: 严格，只有高度相似的人脸聚在一起
            - 0.7: 适中，推荐值
            - 1.0: 宽松，可能会把不同人聚在一起
        min_samples: 形成簇的最小样本数
            - 2: 两张人脸就可以形成一个簇
            - 3: 至少需要3张人脸
        metric: 距离度量方式
            - 'cosine': 余弦距离（推荐，对人脸特征更适合）
            - 'euclidean': 欧氏距离
    
    Returns:
        Dict containing:
            - 'clusters': Dict[int, List[int]] - {cluster_id: [face_ids]}
            - 'noise_ids': List[int] - 被标记为噪声的 face_ids
            - 'n_clusters': int - 聚类数量
            - 'n_noise': int - 噪声点数量
    """
    if not face_embeddings:
        return {
            'clusters': {},
            'noise_ids': [],
            'n_clusters': 0,
            'n_noise': 0
        }
    
    # 提取 face_ids 和 embeddings
    face_ids = [fid for fid, _ in face_embeddings]
    embeddings = np.array([emb for _, emb in face_embeddings])
    
    # L2 归一化（对于余弦距离很重要）
    embeddings = normalize(embeddings, norm='l2')
    
    print(f"DBSCAN 聚类: {len(face_ids)} 个人脸, eps={eps}, min_samples={min_samples}")
    
    # 执行 DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = dbscan.fit_predict(embeddings)
    
    # 整理结果
    clusters = {}
    noise_ids = []
    
    for i, label in enumerate(labels):
        face_id = face_ids[i]
        if label == -1:
            # 噪声点
            noise_ids.append(face_id)
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(face_id)
    
    n_clusters = len(clusters)
    n_noise = len(noise_ids)
    
    print(f"DBSCAN 结果: {n_clusters} 个聚类, {n_noise} 个噪声点")
    
    # 打印聚类详情
    for cluster_id, members in clusters.items():
        print(f"  Cluster {cluster_id}: {len(members)} 个人脸")
    
    return {
        'clusters': clusters,
        'noise_ids': noise_ids,
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }


def find_optimal_eps(
    embeddings: np.ndarray,
    k: int = 5,
    plot: bool = False
) -> float:
    """
    使用 K-距离图找到最佳 eps 参数
    
    Args:
        embeddings: 特征向量数组
        k: 用于计算 k-最近邻距离的 k 值
        plot: 是否绘制 K-距离图
    
    Returns:
        建议的 eps 值
    """
    from sklearn.neighbors import NearestNeighbors
    
    # L2 归一化
    embeddings = normalize(embeddings, norm='l2')
    
    # 计算 k 最近邻距离
    k = min(k, len(embeddings) - 1)
    if k < 1:
        return 0.5  # 默认值
    
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    
    # 取第 k 个邻居的距离（跳过自身）
    k_distances = distances[:, k]
    k_distances = np.sort(k_distances)
    
    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(k_distances)
            plt.xlabel('Points sorted by distance')
            plt.ylabel(f'{k}-NN Distance')
            plt.title('K-Distance Graph for DBSCAN eps selection')
            plt.grid(True)
            plt.show()
        except ImportError:
            print("matplotlib 未安装，跳过绘图")
    
    # 使用肘部法则自动选择 eps
    # 简单方法：取中位数附近的值
    suggested_eps = float(np.percentile(k_distances, 90))
    
    print(f"建议的 eps 值: {suggested_eps:.4f}")
    return suggested_eps


def merge_clusters(
    clusters: Dict[int, List[int]],
    embeddings_dict: Dict[int, np.ndarray],
    merge_threshold: float = 0.4
) -> Dict[int, List[int]]:
    """
    合并相似的聚类（可选的后处理步骤）
    
    Args:
        clusters: {cluster_id: [face_ids]}
        embeddings_dict: {face_id: embedding}
        merge_threshold: 合并阈值（余弦距离）
    
    Returns:
        合并后的聚类
    """
    if len(clusters) <= 1:
        return clusters
    
    # 计算每个聚类的中心向量
    cluster_centers = {}
    for cid, face_ids in clusters.items():
        embs = [embeddings_dict[fid] for fid in face_ids if fid in embeddings_dict]
        if embs:
            center = np.mean(embs, axis=0)
            center = center / (np.linalg.norm(center) + 1e-8)
            cluster_centers[cid] = center
    
    # 查找需要合并的聚类对
    merged = set()
    merge_map = {}  # old_cid -> new_cid
    
    cluster_ids = list(cluster_centers.keys())
    for i, cid1 in enumerate(cluster_ids):
        if cid1 in merged:
            continue
        for cid2 in cluster_ids[i+1:]:
            if cid2 in merged:
                continue
            
            # 计算中心向量的余弦距离
            dist = 1 - np.dot(cluster_centers[cid1], cluster_centers[cid2])
            if dist < merge_threshold:
                # 合并 cid2 到 cid1
                merge_map[cid2] = cid1
                merged.add(cid2)
    
    # 应用合并
    if merge_map:
        new_clusters = {}
        for cid, face_ids in clusters.items():
            target_cid = merge_map.get(cid, cid)
            while target_cid in merge_map:
                target_cid = merge_map[target_cid]
            
            if target_cid not in new_clusters:
                new_clusters[target_cid] = []
            new_clusters[target_cid].extend(face_ids)
        
        print(f"合并了 {len(merged)} 个聚类")
        return new_clusters
    
    return clusters


if __name__ == "__main__":
    # 简单测试
    print("DBSCAN 聚类模块测试")
    
    # 生成模拟数据：3个人，每人3张人脸 + 2个噪声
    np.random.seed(42)
    
    # 人1的特征
    person1_base = np.random.randn(512).astype(np.float32)
    person1_base /= np.linalg.norm(person1_base)
    
    # 人2的特征
    person2_base = np.random.randn(512).astype(np.float32)
    person2_base /= np.linalg.norm(person2_base)
    
    # 人3的特征
    person3_base = np.random.randn(512).astype(np.float32)
    person3_base /= np.linalg.norm(person3_base)
    
    face_embeddings = []
    
    # 人1的3张人脸（添加少量噪声）
    for i in range(3):
        noise = np.random.randn(512).astype(np.float32) * 0.05
        emb = person1_base + noise
        emb /= np.linalg.norm(emb)
        face_embeddings.append((i + 1, emb))
    
    # 人2的3张人脸
    for i in range(3):
        noise = np.random.randn(512).astype(np.float32) * 0.05
        emb = person2_base + noise
        emb /= np.linalg.norm(emb)
        face_embeddings.append((i + 4, emb))
    
    # 人3的3张人脸
    for i in range(3):
        noise = np.random.randn(512).astype(np.float32) * 0.05
        emb = person3_base + noise
        emb /= np.linalg.norm(emb)
        face_embeddings.append((i + 7, emb))
    
    # 2个噪声点（随机特征）
    for i in range(2):
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        face_embeddings.append((i + 10, emb))
    
    # 执行聚类
    result = cluster_faces_dbscan(face_embeddings, eps=0.7, min_samples=2)
    
    print(f"\n期望: 3 个聚类, 2 个噪声")
    print(f"实际: {result['n_clusters']} 个聚类, {result['n_noise']} 个噪声")
    
    if result['n_clusters'] == 3 and result['n_noise'] == 2:
        print("✅ 测试通过!")
    else:
        print("⚠️ 测试结果与期望不符，可能需要调整 eps 参数")
