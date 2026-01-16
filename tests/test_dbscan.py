#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - DBSCAN 聚类单元测试
"""

import unittest
import os
import numpy as np
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

from clustering import cluster_faces_dbscan, find_optimal_eps, merge_clusters


class TestDBSCANClustering(unittest.TestCase):
    """测试 DBSCAN 聚类功能"""
    
    def setUp(self):
        """生成测试数据"""
        np.random.seed(42)
        
        # 生成3个人的模拟人脸特征
        self.person1_base = np.random.randn(512).astype(np.float32)
        self.person1_base /= np.linalg.norm(self.person1_base)
        
        self.person2_base = np.random.randn(512).astype(np.float32)
        self.person2_base /= np.linalg.norm(self.person2_base)
        
        self.person3_base = np.random.randn(512).astype(np.float32)
        self.person3_base /= np.linalg.norm(self.person3_base)
    
    def _generate_face_embeddings(self, n_per_person: int = 3, n_noise: int = 2):
        """生成模拟人脸 embedding"""
        face_embeddings = []
        face_id = 1
        
        # 人1
        for _ in range(n_per_person):
            noise = np.random.randn(512).astype(np.float32) * 0.05
            emb = self.person1_base + noise
            emb /= np.linalg.norm(emb)
            face_embeddings.append((face_id, emb))
            face_id += 1
        
        # 人2
        for _ in range(n_per_person):
            noise = np.random.randn(512).astype(np.float32) * 0.05
            emb = self.person2_base + noise
            emb /= np.linalg.norm(emb)
            face_embeddings.append((face_id, emb))
            face_id += 1
        
        # 人3
        for _ in range(n_per_person):
            noise = np.random.randn(512).astype(np.float32) * 0.05
            emb = self.person3_base + noise
            emb /= np.linalg.norm(emb)
            face_embeddings.append((face_id, emb))
            face_id += 1
        
        # 噪声
        for _ in range(n_noise):
            emb = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            face_embeddings.append((face_id, emb))
            face_id += 1
        
        return face_embeddings
    
    def test_basic_clustering(self):
        """测试基本聚类功能"""
        face_embeddings = self._generate_face_embeddings(n_per_person=3, n_noise=2)
        
        result = cluster_faces_dbscan(face_embeddings, eps=0.7, min_samples=2)
        
        self.assertIn('clusters', result)
        self.assertIn('noise_ids', result)
        self.assertIn('n_clusters', result)
        self.assertIn('n_noise', result)
        
        # 应该有3个聚类
        self.assertEqual(result['n_clusters'], 3)
        # 应该有2个噪声
        self.assertEqual(result['n_noise'], 2)
    
    def test_empty_input(self):
        """测试空输入"""
        result = cluster_faces_dbscan([], eps=0.7, min_samples=2)
        
        self.assertEqual(result['n_clusters'], 0)
        self.assertEqual(result['n_noise'], 0)
        self.assertEqual(result['clusters'], {})
        self.assertEqual(result['noise_ids'], [])
    
    def test_single_face(self):
        """测试单个人脸（应该成为噪声）"""
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        face_embeddings = [(1, emb)]
        
        result = cluster_faces_dbscan(face_embeddings, eps=0.7, min_samples=2)
        
        self.assertEqual(result['n_clusters'], 0)
        self.assertEqual(result['n_noise'], 1)
        self.assertEqual(result['noise_ids'], [1])
    
    def test_strict_eps(self):
        """测试严格的 eps 值"""
        face_embeddings = self._generate_face_embeddings(n_per_person=3, n_noise=0)
        
        # 非常严格的 eps，可能导致更少的聚类
        result = cluster_faces_dbscan(face_embeddings, eps=0.3, min_samples=2)
        
        # 结果应该是合理的（0-3个聚类）
        self.assertGreaterEqual(result['n_clusters'], 0)
        self.assertLessEqual(result['n_clusters'], 3)
    
    def test_loose_eps(self):
        """测试宽松的 eps 值"""
        face_embeddings = self._generate_face_embeddings(n_per_person=3, n_noise=0)
        
        # 非常宽松的 eps，可能将所有人聚在一起
        result = cluster_faces_dbscan(face_embeddings, eps=1.5, min_samples=2)
        
        # 所有人脸都应该在聚类中（没有噪声）
        self.assertGreaterEqual(result['n_clusters'], 1)
    
    def test_cluster_members(self):
        """测试聚类成员正确性"""
        face_embeddings = self._generate_face_embeddings(n_per_person=3, n_noise=2)
        
        result = cluster_faces_dbscan(face_embeddings, eps=0.7, min_samples=2)
        
        # 检查所有人脸都被分配
        all_assigned = set()
        for face_ids in result['clusters'].values():
            all_assigned.update(face_ids)
        all_assigned.update(result['noise_ids'])
        
        expected_ids = set(fid for fid, _ in face_embeddings)
        self.assertEqual(all_assigned, expected_ids)
    
    def test_large_dataset(self):
        """测试较大数据集"""
        np.random.seed(123)
        
        # 10个人，每人5张人脸
        face_embeddings = []
        face_id = 1
        for person_idx in range(10):
            base = np.random.randn(512).astype(np.float32)
            base /= np.linalg.norm(base)
            
            for _ in range(5):
                noise = np.random.randn(512).astype(np.float32) * 0.05
                emb = base + noise
                emb /= np.linalg.norm(emb)
                face_embeddings.append((face_id, emb))
                face_id += 1
        
        # 加10个噪声
        for _ in range(10):
            emb = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            face_embeddings.append((face_id, emb))
            face_id += 1
        
        result = cluster_faces_dbscan(face_embeddings, eps=0.7, min_samples=2)
        
        # 应该有接近10个聚类
        self.assertGreaterEqual(result['n_clusters'], 8)
        self.assertLessEqual(result['n_clusters'], 12)
        
        # 噪声应该在合理范围
        self.assertGreaterEqual(result['n_noise'], 5)


class TestFindOptimalEps(unittest.TestCase):
    """测试 eps 自动选择"""
    
    def test_find_eps(self):
        """测试 eps 自动选择"""
        np.random.seed(42)
        
        # 生成一些聚类数据
        embeddings = []
        for _ in range(3):  # 3个人
            base = np.random.randn(512).astype(np.float32)
            base /= np.linalg.norm(base)
            for _ in range(5):
                noise = np.random.randn(512).astype(np.float32) * 0.05
                emb = base + noise
                emb /= np.linalg.norm(emb)
                embeddings.append(emb)
        
        X = np.array(embeddings)
        
        suggested_eps = find_optimal_eps(X, k=3, plot=False)
        
        # 建议的 eps 应该在合理范围
        self.assertGreater(suggested_eps, 0.0)
        self.assertLess(suggested_eps, 2.0)


class TestMergeClusters(unittest.TestCase):
    """测试聚类合并"""
    
    def test_no_merge_needed(self):
        """测试不需要合并的情况"""
        np.random.seed(42)
        
        clusters = {0: [1, 2, 3], 1: [4, 5, 6]}
        embeddings_dict = {}
        
        # 两个不同的人
        base1 = np.random.randn(512).astype(np.float32)
        base1 /= np.linalg.norm(base1)
        base2 = np.random.randn(512).astype(np.float32)
        base2 /= np.linalg.norm(base2)
        
        for fid in [1, 2, 3]:
            embeddings_dict[fid] = base1 + np.random.randn(512).astype(np.float32) * 0.01
        for fid in [4, 5, 6]:
            embeddings_dict[fid] = base2 + np.random.randn(512).astype(np.float32) * 0.01
        
        merged = merge_clusters(clusters, embeddings_dict, merge_threshold=0.3)
        
        # 不应该合并
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
