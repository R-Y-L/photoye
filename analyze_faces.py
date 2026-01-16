#!/usr/bin/env python3
"""临时脚本：分析人脸相似度并测试聚类效果"""

import numpy as np
from analyzer import AIAnalyzer
from sklearn.metrics.pairwise import cosine_distances
from clustering import cluster_faces_dbscan
import os

# 初始化分析器
analyzer = AIAnalyzer()

# 测试图片
test_images = [
    'tests/test_images/group_photo.jpg',
    'tests/test_images/single_face.jpg',
]

all_faces = []  # (face_id, embedding, source_file)

print("=== 提取人脸特征 ===")
face_id_counter = 1

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"文件不存在: {img_path}")
        continue
    
    filename = os.path.basename(img_path)
    faces = analyzer.detect_faces(img_path)
    
    if not faces:
        print(f"{filename}: 没有检测到人脸")
        continue
    
    print(f"\n{filename}: 检测到 {len(faces)} 个人脸")
    
    for i, face in enumerate(faces):
        embedding = analyzer.get_face_embedding(
            img_path,
            face['bbox'],
            face.get('landmarks')
        )
        
        if embedding is not None:
            all_faces.append((face_id_counter, embedding, f"{filename}#{i+1}"))
            print(f"  人脸 {face_id_counter}: bbox={face['bbox']}, conf={face.get('confidence', 0):.3f}")
            face_id_counter += 1

print(f"\n=== 共提取 {len(all_faces)} 个人脸特征 ===")

if len(all_faces) >= 2:
    # 计算相似度矩阵
    embeddings = np.array([emb for _, emb, _ in all_faces])
    distances = cosine_distances(embeddings)
    
    print("\n=== 人脸间余弦距离矩阵 ===")
    print("(距离越小越相似, 0=相同, <0.5=很可能同一人, >0.7=不同人)")
    
    # 打印标题
    print("\n      ", end="")
    for _, _, name in all_faces:
        print(f"{name[:10]:>12}", end="")
    print()
    
    for i, (_, _, name_i) in enumerate(all_faces):
        print(f"{name_i[:6]:>6}", end="")
        for j in range(len(all_faces)):
            dist = distances[i][j]
            # 标记相似的
            marker = " *" if dist < 0.5 and i != j else ""
            print(f"{dist:>10.4f}{marker}", end="")
        print()
    
    # 使用新的 eps=0.6 测试
    print("\n=== 使用 eps=0.6 的聚类效果（新参数） ===")
    
    face_embeddings = [(fid, emb) for fid, emb, _ in all_faces]
    result = cluster_faces_dbscan(face_embeddings, eps=0.6, min_samples=2)
    
    print(f"\n聚类结果: {result['n_clusters']} 个簇, {result['n_noise']} 个噪声")
    
    for cluster_id, face_ids in result['clusters'].items():
        members = [all_faces[fid-1][2] for fid in face_ids]
        print(f"  簇{cluster_id} (同一人): {members}")
    
    noise_names = [all_faces[fid-1][2] for fid in result['noise_ids']]
    if noise_names:
        print(f"  噪声 (各自独立): {noise_names}")
    
    print("\n=== 最终人物创建 ===")
    print(f"将创建 {result['n_clusters']} 个聚类人物 + {result['n_noise']} 个路人 = {result['n_clusters'] + result['n_noise']} 个人物")

