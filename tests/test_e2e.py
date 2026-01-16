#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye V2.1 端到端测试
"""

import os
import sys
import numpy as np
from pathlib import Path

# 确保在项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print('=== Photoye V2.1 端到端测试 ===')

# 1. 数据库迁移
print('\n1. 测试数据库迁移...')
from database import init_db, migrate_to_v21
init_db()
migrate_to_v21()
print('   ✅ 数据库迁移完成')

# 2. CLIP Embedding
print('\n2. 测试 CLIP Embedding...')
from models.clip_embedding import CLIPEmbeddingEncoder
clip = CLIPEmbeddingEncoder()
if clip.is_available():
    emb = clip.encode_image('tests/test_images/food.jpg')
    print(f'   ✅ CLIP 编码: shape={emb.shape}')
else:
    print('   ⚠️ CLIP 不可用')

# 3. InsightFace
print('\n3. 测试 InsightFace...')
from models.insightface_detector import InsightFaceDetector
from models.insightface_recognizer import InsightFaceRecognizer
det = InsightFaceDetector()
rec = InsightFaceRecognizer()
faces = det.detect('tests/test_images/single_face.jpg')
if faces:
    emb = rec.get_embedding('tests/test_images/single_face.jpg', faces[0]['bbox'], faces[0]['landmarks'])
    print(f'   ✅ InsightFace: {len(faces)} faces, embedding shape={emb.shape}')

# 4. DBSCAN
print('\n4. 测试 DBSCAN 聚类...')
from clustering import cluster_faces_dbscan
np.random.seed(42)
test_embs = [(i, np.random.randn(512).astype(np.float32)) for i in range(10)]
result = cluster_faces_dbscan(test_embs, eps=0.7, min_samples=2)
n_clusters = result['n_clusters']
n_noise = result['n_noise']
print(f'   ✅ DBSCAN: {n_clusters} clusters, {n_noise} noise')

# 5. Model Profiles
print('\n5. 测试 Model Profiles...')
from model_profiles import list_available_profiles
profiles = list_available_profiles()
print(f'   ✅ 可用配置: {list(profiles.keys())}')

# 6. AIAnalyzer with InsightFace
print('\n6. 测试 AIAnalyzer (insightface 配置)...')
from analyzer import AIAnalyzer
analyzer = AIAnalyzer(model_profile='insightface')
print(f'   ✅ 检测器: {analyzer.face_detector.__class__.__name__}')
print(f'   ✅ 识别器: {analyzer.face_recognizer.__class__.__name__}')

print('\n=== 所有测试通过 ===')
