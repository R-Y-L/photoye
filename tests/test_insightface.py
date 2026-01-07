#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - InsightFace 模型单元测试

测试 RetinaFace 检测器和 ArcFace 识别器
"""

import unittest
import os
import numpy as np
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

from models.insightface_detector import InsightFaceDetector
from models.insightface_recognizer import InsightFaceRecognizer


class TestInsightFaceDetector(unittest.TestCase):
    """测试 InsightFace RetinaFace 检测器"""
    
    @classmethod
    def setUpClass(cls):
        """初始化检测器（只加载一次）"""
        cls.detector = InsightFaceDetector()
        cls.test_images_dir = PROJECT_ROOT / "tests" / "test_images"
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.app)
    
    def test_detect_single_face(self):
        """测试单人脸检测"""
        image_path = str(self.test_images_dir / "single_face.jpg")
        if not os.path.exists(image_path):
            self.skipTest(f"测试图片不存在: {image_path}")
        
        faces = self.detector.detect(image_path)
        
        self.assertIsInstance(faces, list)
        self.assertGreaterEqual(len(faces), 1, "应该检测到至少1个人脸")
        
        # 检查返回格式
        face = faces[0]
        self.assertIn("bbox", face)
        self.assertIn("confidence", face)
        self.assertIn("landmarks", face)
        
        # 检查 bbox 格式
        bbox = face["bbox"]
        self.assertEqual(len(bbox), 4)
        self.assertLess(bbox[0], bbox[2], "x1 应该小于 x2")
        self.assertLess(bbox[1], bbox[3], "y1 应该小于 y2")
        
        # 检查 landmarks 格式 (5点)
        landmarks = face["landmarks"]
        self.assertIsNotNone(landmarks)
        self.assertEqual(len(landmarks), 5, "应该有5个关键点")
        for pt in landmarks:
            self.assertEqual(len(pt), 2, "每个关键点应该有x,y坐标")
    
    def test_detect_group_photo(self):
        """测试多人脸检测"""
        image_path = str(self.test_images_dir / "group_photo.jpg")
        if not os.path.exists(image_path):
            self.skipTest(f"测试图片不存在: {image_path}")
        
        faces = self.detector.detect(image_path)
        
        self.assertIsInstance(faces, list)
        self.assertGreater(len(faces), 1, "群组照片应该检测到多个人脸")
    
    def test_detect_no_face(self):
        """测试无人脸图片"""
        image_path = str(self.test_images_dir / "landscape.jpg")
        if not os.path.exists(image_path):
            self.skipTest(f"测试图片不存在: {image_path}")
        
        faces = self.detector.detect(image_path)
        
        self.assertIsInstance(faces, list)
        # 风景图片应该没有人脸（或很少）
    
    def test_detect_nonexistent_file(self):
        """测试不存在的文件"""
        faces = self.detector.detect("/nonexistent/path/image.jpg")
        self.assertEqual(faces, [])


class TestInsightFaceRecognizer(unittest.TestCase):
    """测试 InsightFace ArcFace 识别器"""
    
    @classmethod
    def setUpClass(cls):
        """初始化检测器和识别器"""
        cls.detector = InsightFaceDetector()
        cls.recognizer = InsightFaceRecognizer()
        cls.test_images_dir = PROJECT_ROOT / "tests" / "test_images"
    
    def test_recognizer_initialization(self):
        """测试识别器初始化"""
        self.assertIsNotNone(self.recognizer)
        self.assertIsNotNone(self.recognizer.session)
    
    def test_embedding_extraction(self):
        """测试特征向量提取"""
        image_path = str(self.test_images_dir / "single_face.jpg")
        if not os.path.exists(image_path):
            self.skipTest(f"测试图片不存在: {image_path}")
        
        # 先检测人脸
        faces = self.detector.detect(image_path)
        self.assertGreater(len(faces), 0, "需要检测到人脸")
        
        face = faces[0]
        
        # 提取特征向量
        embedding = self.recognizer.get_embedding(
            image_path,
            face["bbox"],
            face["landmarks"]
        )
        
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (512,), "应该是512维特征向量")
        
        # 检查L2归一化
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=4, msg="特征向量应该L2归一化")
    
    def test_embedding_without_landmarks(self):
        """测试无关键点时的特征提取"""
        image_path = str(self.test_images_dir / "single_face.jpg")
        if not os.path.exists(image_path):
            self.skipTest(f"测试图片不存在: {image_path}")
        
        faces = self.detector.detect(image_path)
        self.assertGreater(len(faces), 0)
        
        face = faces[0]
        
        # 不传入 landmarks
        embedding = self.recognizer.get_embedding(
            image_path,
            face["bbox"],
            landmarks=None
        )
        
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (512,))
    
    def test_embedding_similarity(self):
        """测试同一人脸的相似度"""
        image_path = str(self.test_images_dir / "group_photo.jpg")
        if not os.path.exists(image_path):
            self.skipTest(f"测试图片不存在: {image_path}")
        
        faces = self.detector.detect(image_path)
        if len(faces) < 2:
            self.skipTest("需要至少2个人脸进行相似度测试")
        
        # 提取两个不同人的特征
        emb1 = self.recognizer.get_embedding(
            image_path, faces[0]["bbox"], faces[0]["landmarks"]
        )
        emb2 = self.recognizer.get_embedding(
            image_path, faces[1]["bbox"], faces[1]["landmarks"]
        )
        
        self.assertIsNotNone(emb1)
        self.assertIsNotNone(emb2)
        
        # 计算余弦相似度
        similarity = InsightFaceRecognizer.cosine_similarity(emb1, emb2)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)
        
        print(f"\n两个不同人脸的相似度: {similarity:.4f}")
        # 不同人的相似度通常 < 0.4
        self.assertLess(similarity, 0.7, "不同人的相似度应该较低")


class TestInsightFaceIntegration(unittest.TestCase):
    """InsightFace 集成测试"""
    
    def test_analyzer_with_insightface_profile(self):
        """测试分析器使用 insightface 配置"""
        from analyzer import AIAnalyzer
        
        analyzer = AIAnalyzer(model_profile="insightface")
        
        self.assertEqual(analyzer.detector_type, "insightface")
        self.assertEqual(analyzer.recognizer_type, "insightface")
        
        # 检测器和识别器类型
        self.assertIsInstance(analyzer.face_detector, InsightFaceDetector)
        self.assertIsInstance(analyzer.face_recognizer, InsightFaceRecognizer)
    
    def test_analyzer_with_best_profile(self):
        """测试分析器使用 best 配置"""
        from analyzer import AIAnalyzer
        
        analyzer = AIAnalyzer(model_profile="best")
        
        self.assertEqual(analyzer.detector_type, "insightface")
        self.assertEqual(analyzer.recognizer_type, "insightface")


if __name__ == "__main__":
    unittest.main(verbosity=2)
