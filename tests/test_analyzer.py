#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye AI分析模块测试脚本
用于测试AIAnalyzer类的功能
"""

import sys
import os
import tempfile
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import AIAnalyzer


def create_test_image(filename):
    """创建测试图片文件"""
    with open(filename, 'w') as f:
        f.write("fake image data")


def test_ai_analyzer_initialization():
    """测试AIAnalyzer初始化"""
    print("测试AIAnalyzer初始化...")
    
    # 测试默认路径初始化
    analyzer1 = AIAnalyzer()
    assert analyzer1.models_path == "./models"
    print("✓ 默认路径初始化成功")
    
    # 测试自定义路径初始化
    custom_path = "./custom_models"
    analyzer2 = AIAnalyzer(custom_path)
    assert analyzer2.models_path == custom_path
    print("✓ 自定义路径初始化成功")


def test_face_detection():
    """测试人脸检测功能"""
    print("\n测试人脸检测功能...")
    
    analyzer = AIAnalyzer()
    
    # 创建测试图片
    test_image = "test_face_detection.jpg"
    create_test_image(test_image)
    
    # 检测人脸
    faces = analyzer.detect_faces(test_image)
    
    # 验证结果
    assert isinstance(faces, list)
    for face in faces:
        assert 'bbox' in face
        assert 'confidence' in face
        assert len(face['bbox']) == 4
        assert 0 <= face['confidence'] <= 1
    
    print(f"✓ 人脸检测完成，检测到 {len(faces)} 个人脸")
    
    # 清理测试文件
    if os.path.exists(test_image):
        os.remove(test_image)


def test_face_embedding():
    """测试人脸特征提取功能"""
    print("\n测试人脸特征提取功能...")
    
    analyzer = AIAnalyzer()
    
    # 创建测试图片
    test_image = "test_embedding.jpg"
    create_test_image(test_image)
    
    # 提取特征向量
    bbox = [100, 100, 300, 300]
    embedding = analyzer.get_face_embedding(test_image, bbox)
    
    # 验证结果
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32
    
    print("✓ 人脸特征提取成功")
    
    # 清理测试文件
    if os.path.exists(test_image):
        os.remove(test_image)


def test_scene_classification():
    """测试场景分类功能"""
    print("\n测试场景分类功能...")
    
    analyzer = AIAnalyzer()
    
    # 创建测试图片
    test_image = "test_classification.jpg"
    create_test_image(test_image)
    
    # 场景分类
    classification = analyzer.classify_scene(test_image)
    
    # 验证结果
    assert isinstance(classification, dict)
    assert len(classification) > 0
    
    # 验证所有值都是概率（0-1之间）
    total_prob = 0
    for category, prob in classification.items():
        assert 0 <= prob <= 1
        total_prob += prob
    
    print(f"✓ 场景分类完成，识别出 {len(classification)} 个类别")
    
    # 清理测试文件
    if os.path.exists(test_image):
        os.remove(test_image)


def test_photo_analysis():
    """测试完整照片分析功能"""
    print("\n测试完整照片分析功能...")
    
    analyzer = AIAnalyzer()
    
    # 创建测试图片
    test_image = "test_analysis.jpg"
    create_test_image(test_image)
    
    # 完整分析
    result = analyzer.analyze_photo(test_image)
    
    # 验证结果
    assert result is not None
    assert 'image_path' in result
    assert 'faces' in result
    assert 'scene_classification' in result
    assert 'category' in result
    
    print(f"✓ 照片分析完成")
    print(f"  - 图片路径: {result['image_path']}")
    print(f"  - 检测到人脸数: {len(result['faces'])}")
    print(f"  - 场景分类类别数: {len(result['scene_classification'])}")
    print(f"  - 图片分类: {result['category']}")
    
    # 清理测试文件
    if os.path.exists(test_image):
        os.remove(test_image)


def test_batch_analysis():
    """测试批量分析功能"""
    print("\n测试批量分析功能...")
    
    analyzer = AIAnalyzer()
    
    # 创建测试图片
    test_images = []
    for i in range(3):
        filename = f"test_batch_{i}.jpg"
        create_test_image(filename)
        test_images.append(filename)
    
    # 批量分析
    def progress_callback(current, total):
        print(f"  进度: {current}/{total}")
    
    results = analyzer.batch_analyze(test_images, progress_callback)
    
    # 验证结果
    assert len(results) == len(test_images)
    
    print(f"✓ 批量分析完成，处理 {len(results)} 张图片")
    
    # 清理测试文件
    for image in test_images:
        if os.path.exists(image):
            os.remove(image)


def test_face_comparison():
    """测试人脸相似度比较功能"""
    print("\n测试人脸相似度比较功能...")
    
    analyzer = AIAnalyzer()
    
    # 创建两个随机特征向量
    emb1 = np.random.rand(512).astype(np.float32)
    emb2 = np.random.rand(512).astype(np.float32)
    
    # 计算相似度
    similarity = analyzer.compare_faces(emb1, emb2)
    
    # 验证结果
    assert 0 <= similarity <= 1
    
    print(f"✓ 人脸相似度计算完成，相似度: {similarity:.3f}")


def test_face_clustering():
    """测试人脸聚类功能"""
    print("\n测试人脸聚类功能...")
    
    analyzer = AIAnalyzer()
    
    # 创建随机特征向量
    embeddings = [np.random.rand(512).astype(np.float32) for _ in range(10)]
    
    # 执行聚类
    clusters = analyzer.cluster_faces(embeddings)
    
    # 验证结果
    assert isinstance(clusters, list)
    
    total_clustered = sum(len(cluster) for cluster in clusters)
    assert total_clustered <= len(embeddings)
    
    print(f"✓ 人脸聚类完成，形成 {len(clusters)} 个聚类")


def main():
    """主测试函数"""
    print("=" * 60)
    print("Photoye AI分析模块测试")
    print("=" * 60)
    
    try:
        test_ai_analyzer_initialization()
        test_face_detection()
        test_face_embedding()
        test_scene_classification()
        test_photo_analysis()
        test_batch_analysis()
        test_face_comparison()
        test_face_clustering()
        
        print("\n" + "=" * 60)
        print("所有测试通过! AI分析模块功能正常。")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())