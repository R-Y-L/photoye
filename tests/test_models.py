#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye 模型诊断脚本
逐一测试各个模型的输入输出，快速定位问题

使用方式:
  python test_models.py <image_path> [model_profile]
  
示例:
  python test_models.py "D:/Pictures/test.jpg" balanced
  python test_models.py "D:/Pictures/test.jpg" accuracy
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

def test_face_detection(image_path: str):
    """测试人脸检测模型"""
    print("\n" + "="*60)
    print("测试 1: 人脸检测 (YuNet)")
    print("="*60)
    
    try:
        from models.opencv_yunet_detector import OpenCVYuNetDetector
        
        detector = OpenCVYuNetDetector()
        if detector.detector is None:
            print("❌ YuNet 模型未加载，请检查模型文件路径")
            return None
        
        print(f"✅ YuNet 模型已加载")
        print(f"📷 输入图片: {image_path}")
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图片: {image_path}")
            return None
        
        h, w = image.shape[:2]
        print(f"📊 图片尺寸: {w}x{h}")
        
        # 检测
        results = detector.detect(image_path)
        
        print(f"\n🎯 检测结果: 发现 {len(results)} 个人脸")
        for i, face in enumerate(results, 1):
            bbox = face.get('bbox', [])
            confidence = face.get('confidence', 0)
            landmarks = face.get('landmarks', [])
            print(f"  人脸 {i}:")
            print(f"    - BBox: {bbox}")
            print(f"    - 置信度: {confidence:.4f}")
            if landmarks:
                print(f"    - 关键点数: {len(landmarks)}")
        
        return results
        
    except Exception as e:
        print(f"❌ 人脸检测错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_face_recognition(image_path: str, faces: list):
    """测试人脸识别模型"""
    print("\n" + "="*60)
    print("测试 2: 人脸识别 (Dlib / SFace)")
    print("="*60)
    
    if not faces:
        print("⚠️  没有检测到人脸，跳过识别测试")
        return None
    
    try:
        from models.dlib_detector import DlibFaceRecognizer
        
        recognizer = DlibFaceRecognizer()
        if recognizer.recognizer is None:
            print("❌ Dlib 识别模型未加载，请检查模型文件路径")
            return None
        
        print(f"✅ Dlib 识别模型已加载")
        
        embeddings = []
        for i, face in enumerate(faces, 1):
            bbox = face.get('bbox', [])
            landmarks = face.get('landmarks')
            
            embedding = recognizer.get_embedding(image_path, bbox, landmarks)
            
            if embedding is not None:
                embeddings.append(embedding)
                print(f"\n  人脸 {i} embedding:")
                print(f"    - 维度: {embedding.shape}")
                print(f"    - 范围: [{embedding.min():.4f}, {embedding.max():.4f}]")
                print(f"    - 均值: {embedding.mean():.4f}")
            else:
                print(f"\n  人脸 {i}: 提取失败")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ 人脸识别错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_scene_classification(image_path: str, model_profile: str = "balanced"):
    """测试场景分类模型"""
    print("\n" + "="*60)
    print(f"测试 3: 场景分类 (MobileNetV2 | 模型档位: {model_profile})")
    print("="*60)
    
    try:
        from analyzer import AIAnalyzer
        
        analyzer = AIAnalyzer(model_profile=model_profile)
        
        print(f"✅ 分析器已初始化")
        print(f"   - 检测模型: {analyzer.detector_type} -> {analyzer.face_detector.__class__.__name__}")
        print(f"   - 识别模型: {analyzer.recognizer_type} -> {analyzer.face_recognizer.__class__.__name__}")
        print(f"   - 分类模型: {analyzer.classifier_type} -> {analyzer.scene_classifier.__class__.__name__}")
        
        # 测试分类器
        if isinstance(analyzer.scene_classifier, str):
            print(f"⚠️  分类器是占位符: {analyzer.scene_classifier}")
            return None
        
        print(f"\n📷 输入图片: {image_path}")
        
        # 读取图片验证
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图片")
            return None
        
        # 调用分类
        results = analyzer.classify_scene(image_path)
        
        print(f"\n🎯 分类结果:")
        if results:
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            for i, (category, score) in enumerate(sorted_results, 1):
                print(f"  {i}. {category}: {score:.4f}")
        else:
            print("  (空结果)")
        
        return results
        
    except Exception as e:
        print(f"❌ 场景分类错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_openclip_classification(image_path: str):
    """测试 OpenCLIP 零样本分类"""
    print("\n" + "="*60)
    print("测试 4: OpenCLIP 零样本分类")
    print("="*60)
    
    try:
        from models.openclip_zero_shot import OpenCLIPZeroShotClassifier
        
        classifier = OpenCLIPZeroShotClassifier()
        
        if not classifier.loaded:
            print("❌ OpenCLIP 模型未加载")
            return None
        
        print(f"✅ OpenCLIP 模型已加载")
        print(f"   - Vision 模型: {classifier.vision_model_path}")
        print(f"   - Text 模型: {classifier.text_model_path}")
        print(f"   - Tokenizer: {classifier.tokenizer_path}")
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图片")
            return None
        
        print(f"\n📷 输入图片: {image_path}")
        h, w = image.shape[:2]
        print(f"📊 图片尺寸: {w}x{h}")
        
        # 分类
        results = classifier.classify(image_path)
        
        print(f"\n🎯 分类结果:")
        if results:
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            for i, (category, score) in enumerate(sorted_results, 1):
                print(f"  {i}. {category}: {score:.4f}")
        else:
            print("  (空结果)")
        
        return results
        
    except Exception as e:
        print(f"❌ OpenCLIP 分类错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_full_pipeline(image_path: str, model_profile: str = "balanced"):
    """测试完整分析流程"""
    print("\n" + "="*60)
    print("测试 5: 完整分析流程")
    print("="*60)
    
    try:
        from analyzer import AIAnalyzer
        
        analyzer = AIAnalyzer(model_profile=model_profile)
        
        result = analyzer.analyze_photo(image_path)
        
        if result:
            print(f"✅ 分析完成")
            print(f"\n结果摘要:")
            print(f"  - 最终分类: {result.get('category', 'N/A')}")
            print(f"  - 检测人脸数: {len(result.get('faces', []))}")
            print(f"  - 场景分类: {result.get('scene_classification', {})}")
        else:
            print(f"❌ 分析失败")
        
        return result
        
    except Exception as e:
        print(f"❌ 完整流程错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    if len(sys.argv) < 2:
        print("使用方式: python test_models.py <image_path> [model_profile]")
        print("示例: python test_models.py 'D:/Pictures/test.jpg' balanced")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_profile = sys.argv[2] if len(sys.argv) > 2 else "balanced"
    
    # 验证文件存在
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        sys.exit(1)
    
    print(f"\n🔍 Photoye 模型诊断工具")
    print(f"测试图片: {image_path}")
    print(f"模型档位: {model_profile}")
    
    # 逐个测试
    print("\n" + "#"*60)
    print("# 第一步: 测试人脸检测")
    print("#"*60)
    faces = test_face_detection(image_path)
    
    if faces:
        print("\n" + "#"*60)
        print("# 第二步: 测试人脸识别")
        print("#"*60)
        embeddings = test_face_recognition(image_path, faces)
    
    print("\n" + "#"*60)
    print("# 第三步: 测试 MobileNetV2 分类")
    print("#"*60)
    classification = test_scene_classification(image_path, model_profile)
    
    print("\n" + "#"*60)
    print("# 第四步: 测试 OpenCLIP 零样本分类")
    print("#"*60)
    openclip_result = test_openclip_classification(image_path)
    
    print("\n" + "#"*60)
    print("# 第五步: 完整分析流程")
    print("#"*60)
    full_result = test_full_pipeline(image_path, model_profile)
    
    print("\n" + "="*60)
    print("✅ 诊断完成")
    print("="*60)
    print("\n诊断建议:")
    if not faces:
        print("  ⚠️  未检测到人脸 -> 检查 YuNet 模型或图片内容")
    if not classification or all(v < 0.3 for v in classification.values()):
        print("  ⚠️  分类置信度过低 -> 检查 MobileNetV2 模型或输入预处理")
    if not openclip_result or all(v < 0.3 for v in openclip_result.values()):
        print("  ⚠️  OpenCLIP 置信度过低 -> 检查 OpenCLIP 模型或提示词")
    print("\n更多信息请查看上方详细输出。")


if __name__ == "__main__":
    main()
