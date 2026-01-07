#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLIP Embedding 编码器测试"""

import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_clip_encoder_init():
    """测试 CLIP 编码器初始化"""
    from models.clip_embedding import CLIPEmbeddingEncoder
    
    encoder = CLIPEmbeddingEncoder()
    assert encoder.is_available(), "CLIP 编码器应该可用"
    assert encoder.embedding_dim == 512, "Embedding 维度应为 512"
    print("✅ test_clip_encoder_init passed")


def test_image_encoding():
    """测试图像编码"""
    from models.clip_embedding import CLIPEmbeddingEncoder
    
    encoder = CLIPEmbeddingEncoder()
    
    # 使用测试图片
    test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))]
    
    assert len(test_images) > 0, "需要测试图片"
    
    for img_name in test_images[:3]:
        img_path = os.path.join(test_images_dir, img_name)
        embedding = encoder.encode_image(img_path)
        
        assert embedding is not None, f"图像编码失败: {img_name}"
        assert embedding.shape == (512,), f"Embedding 形状错误: {embedding.shape}"
        
        # 检查 L2 归一化
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01, f"Embedding 未归一化: norm={norm}"
        
        print(f"  ✓ {img_name}: shape={embedding.shape}, norm={norm:.4f}")
    
    print("✅ test_image_encoding passed")


def test_text_encoding():
    """测试文本编码"""
    from models.clip_embedding import CLIPEmbeddingEncoder
    
    encoder = CLIPEmbeddingEncoder()
    
    queries = [
        "a beautiful landscape photo",
        "a group photo of friends",
        "delicious food on a table",
        "a cute dog",
    ]
    
    for query in queries:
        embedding = encoder.encode_text(query)
        
        assert embedding is not None, f"文本编码失败: {query}"
        assert embedding.shape == (512,), f"Embedding 形状错误: {embedding.shape}"
        
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01, f"Embedding 未归一化: norm={norm}"
        
        print(f"  ✓ '{query[:30]}...': norm={norm:.4f}")
    
    print("✅ test_text_encoding passed")


def test_similarity_search():
    """测试相似度搜索"""
    from models.clip_embedding import CLIPEmbeddingEncoder
    
    encoder = CLIPEmbeddingEncoder()
    
    # 编码测试图片
    test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))]
    
    embeddings = []
    image_names = []
    for img_name in test_images:
        img_path = os.path.join(test_images_dir, img_name)
        emb = encoder.encode_image(img_path)
        if emb is not None:
            embeddings.append(emb)
            image_names.append(img_name)
    
    embeddings = np.stack(embeddings)
    
    # 测试不同查询
    test_queries = [
        ("landscape.jpg", "a beautiful landscape photo"),
        ("food.jpg", "delicious food"),
        ("animal.jpg", "a cute animal"),
    ]
    
    for expected_img, query in test_queries:
        if expected_img not in image_names:
            print(f"  ⚠️ 跳过: {expected_img} 不存在")
            continue
        
        query_emb = encoder.encode_text(query)
        results = encoder.search(query_emb, embeddings, top_k=3)
        
        print(f"\n  查询: '{query}'")
        for idx, sim in results:
            print(f"    {image_names[idx]}: {sim:.3f}")
    
    print("\n✅ test_similarity_search passed")


def test_database_integration():
    """测试数据库集成"""
    import database
    
    # 迁移数据库
    database.migrate_to_v21()
    
    # 测试 embedding 更新函数
    from models.clip_embedding import CLIPEmbeddingEncoder
    
    encoder = CLIPEmbeddingEncoder()
    test_embedding = encoder.encode_text("test")
    
    # 这里只测试函数存在性，不实际写入
    assert hasattr(database, 'update_photo_embedding'), "缺少 update_photo_embedding"
    assert hasattr(database, 'get_photos_without_embedding'), "缺少 get_photos_without_embedding"
    assert hasattr(database, 'search_photos_by_embedding'), "缺少 search_photos_by_embedding"
    
    print("✅ test_database_integration passed")


if __name__ == "__main__":
    print("=" * 50)
    print("CLIP Embedding 测试")
    print("=" * 50)
    
    test_clip_encoder_init()
    test_image_encoding()
    test_text_encoding()
    test_similarity_search()
    test_database_integration()
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
