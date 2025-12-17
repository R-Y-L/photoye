#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - AI分析模块
负责所有计算机视觉模型的推理和分析工作

版本: 1.0 (阶段2)
"""

import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from model_profiles import MODEL_PROFILES, DEFAULT_MODEL_PROFILE, MODEL_PROFILE_ENV_VAR


class AIAnalyzer:
    """
    AI分析器类 - 封装所有AI模型推理功能
    
    支持插件化的模型切换
    实现场景分类优先，再进行人脸识别的两阶段分析流程
    """
    
    def __init__(self, models_path: str = "./models", 
                 detector_type: Optional[str] = None,
                 recognizer_type: Optional[str] = None,
                 classifier_type: Optional[str] = None,
                 model_profile: Optional[str] = None):
        """
        初始化AI分析器
        
        Args:
            models_path: 模型文件存放路径
            detector_type: 人脸检测模型类型 ("yunet", "dlib", "yolov8")
            recognizer_type: 人脸识别模型类型 ("dlib", "arcface", "sface")
            classifier_type: 场景分类模型类型 ("mobilenetv2", "resnet", "openclip")
            model_profile: 模型配置档位 ("speed", "balanced", "accuracy" 等)
                - 如果未显式指定，将读取环境变量 PHOTOYE_MODEL_PROFILE
                - 若依然为空，则使用 model_profiles.DEFAULT_MODEL_PROFILE
        """
        self.models_path = models_path

        env_profile = os.getenv(MODEL_PROFILE_ENV_VAR)
        resolved_profile = (model_profile or env_profile or DEFAULT_MODEL_PROFILE).lower()
        if resolved_profile not in MODEL_PROFILES:
            print(f"⚠️ 未识别的模型配置 {resolved_profile}, 将使用默认 {DEFAULT_MODEL_PROFILE}")
            resolved_profile = DEFAULT_MODEL_PROFILE

        profile_cfg = MODEL_PROFILES[resolved_profile]

        def _normalize(choice: Optional[str], fallback_key: str) -> str:
            return (choice or fallback_key).lower()

        self.model_profile = resolved_profile
        self.detector_type = _normalize(detector_type, profile_cfg["detector"])
        self.recognizer_type = _normalize(recognizer_type, profile_cfg["recognizer"])
        self.classifier_type = _normalize(classifier_type, profile_cfg["classifier"])

        # 初始化模型
        self.face_detector = self._init_detector(self.detector_type)
        self.face_recognizer = self._init_recognizer(self.recognizer_type)
        self.scene_classifier = self._init_classifier(self.classifier_type)
        
        print(f"AI分析器初始化完成 (配置: {self.model_profile})")
        print(f"人脸检测模型: {self.detector_type}")
        print(f"人脸识别模型: {self.recognizer_type}")
        print(f"场景分类模型: {self.classifier_type}")
    
    def _init_detector(self, detector_type: str):
        """初始化人脸检测模型"""
        if detector_type == "yunet":
            from models.opencv_yunet_detector import OpenCVYuNetDetector
            return OpenCVYuNetDetector()
        elif detector_type == "dlib":
            from models.dlib_detector import DlibFaceDetector
            return DlibFaceDetector()
        elif detector_type == "yolov8":
            # YOLOv8实现
            print("使用YOLOv8人脸检测模型（占位符）")
            return "YOLOv8-Face 模型占位符"
        else:
            # 默认使用模拟模型
            print("使用默认人脸检测模型（占位符）")
            return "人脸检测模型占位符"
    
    def _init_recognizer(self, recognizer_type: str):
        """初始化人脸识别模型"""
        if recognizer_type == "dlib":
            from models.dlib_detector import DlibFaceRecognizer

            return DlibFaceRecognizer()
        elif recognizer_type == "sface":
            from models.opencv_sface_recognizer import OpenCVSFaceRecognizer

            return OpenCVSFaceRecognizer()
        elif recognizer_type == "arcface":
            # ArcFace实现
            print("使用ArcFace人脸识别模型（占位符）")
            return "ArcFace 模型占位符"
        else:
            # 默认使用模拟模型
            print("使用默认人脸识别模型（占位符）")
            return "人脸识别模型占位符"
    
    def _init_classifier(self, classifier_type: str):
        """初始化场景分类模型"""
        if classifier_type in ("mobilenetv2", "mobilenetv3"):
            from models.mobilenetv2_classifier import MobileNetV2SceneClassifier

            return MobileNetV2SceneClassifier()
        elif classifier_type == "openclip":
            from models.openclip_zero_shot import OpenCLIPZeroShotClassifier

            return OpenCLIPZeroShotClassifier()
        elif classifier_type == "resnet":
            # ResNet实现
            print("使用ResNet场景分类模型（占位符）")
            return "ResNet 模型占位符"
        else:
            # 默认使用模拟模型
            print("使用默认场景分类模型（占位符）")
            return "场景分类模型占位符"
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        检测图片中的人脸
        
        Args:
            image_path: 图片文件路径
        
        Returns:
            人脸检测结果列表，每个元素包含边界框和置信度
            格式: [{'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
        """
        print(f"检测人脸: {os.path.basename(image_path)}")
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            return []
        
        # 如果使用真实模型
        if isinstance(self.face_detector, str):
            # 当前阶段使用模拟结果
            # 模拟检测到1-3个人脸
            import random
            num_faces = random.randint(0, 3)
            mock_results = []
            for i in range(num_faces):
                mock_results.append({
                    'bbox': [random.randint(50, 200), random.randint(50, 200), 
                             random.randint(300, 500), random.randint(300, 500)],
                    'confidence': random.uniform(0.7, 0.95)
                })
            
            print(f"检测到 {num_faces} 个人脸（模拟结果）")
            return mock_results
        else:
            # 使用真实模型进行检测
            return self.face_detector.detect(image_path)
    
    def get_face_embedding(self, image_path: str, bbox: List[int], landmarks: Optional[List[List[int]]] = None) -> Optional[np.ndarray]:
        """
        获取人脸特征向量
        
        Args:
            image_path: 图片文件路径
            bbox: 人脸边界框 [x1, y1, x2, y2]
        
        Returns:
            人脸特征向量，如果失败返回None
        """
        print(f"提取人脸特征: {os.path.basename(image_path)}, bbox={bbox}")
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            return None
        
        # 如果使用真实模型
        if isinstance(self.face_recognizer, str):
            # 当前阶段使用模拟结果
            mock_embedding = np.random.rand(512).astype(np.float32)
            print("生成模拟人脸特征向量")
            return mock_embedding
        else:
            # 使用真实模型进行识别
            return self.face_recognizer.get_embedding(image_path, bbox, landmarks)
    
    def classify_scene(self, image_path: str) -> Dict[str, float]:
        """
        对图片进行场景分类
        
        Args:
            image_path: 图片文件路径
        
        Returns:
            分类结果字典，包含各类别的置信度
            格式: {'风景': 0.8, '建筑': 0.1, '室内': 0.05, ...}
        """
        print(f"场景分类: {os.path.basename(image_path)}")
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            return {}
        
        # 如果使用真实模型
        if isinstance(self.scene_classifier, str):
            # 当前阶段使用模拟结果
            import random
            categories = ['风景', '建筑', '动物', '文档', '室内', '美食']
            weights = [random.random() for _ in categories]
            total_weight = sum(weights)
            normalized_weights = [w/total_weight for w in weights]
            
            mock_classification = dict(zip(categories, normalized_weights))
            return mock_classification
        else:
            # 使用真实模型进行分类
            return self.scene_classifier.classify(image_path)
    
    def analyze_photo(self, image_path: str) -> Dict:
        """
        对单张照片进行完整分析（两阶段流程）
        第一阶段：场景分类，筛选出可能包含人脸的图片
        第二阶段：对筛选出的图片进行人脸识别
        
        Args:
            image_path: 图片文件路径
        
        Returns:
            完整的分析结果字典
        """
        print(f"分析照片: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            return None
        
        result = {
            'image_path': image_path,
            'faces': [],
            'scene_classification': {},
            'category': '未分类'
        }
        
        try:
            # 第一阶段：场景分类
            classification = self.classify_scene(image_path)
            result['scene_classification'] = classification
            
            # 根据场景分类决定是否需要人脸识别
            # 例如，只有在人物相关的分类中才进行人脸识别
            need_face_recognition = self._should_detect_faces(classification)
            
            if need_face_recognition:
                # 第二阶段：人脸检测和识别
                faces = self.detect_faces(image_path)
                
                # 为每个人脸提取特征向量
                for face in faces:
                    embedding = self.get_face_embedding(image_path, face['bbox'], face.get('landmarks'))
                    if embedding is not None:
                        face['embedding'] = embedding
                        result['faces'].append(face)
                
                # 根据人脸数量确定最终分类
                face_count = len(result['faces'])
                if face_count == 1:
                    result['category'] = '单人照'
                elif face_count > 1:
                    result['category'] = '合照'
                else:
                    # 有场景分类但未检测到人脸，使用场景分类结果
                    best_scene = max(classification.items(), key=lambda x: x[1])
                    result['category'] = best_scene[0]
            else:
                # 不需要人脸识别，直接使用场景分类结果
                best_scene = max(classification.items(), key=lambda x: x[1])
                result['category'] = best_scene[0]
            
            print(f"分析完成: 分类={result['category']}, 人脸数={len(result['faces'])}")
            return result
            
        except Exception as e:
            print(f"分析照片失败: {e}")
            return None
    
    def _should_detect_faces(self, classification: Dict[str, float]) -> bool:
        """
        根据场景分类结果判断是否需要进行人脸识别
        """
        # 定义需要人脸识别的场景类别
        face_needed_categories = ['单人照', '合照', '人物', '肖像']
        
        # 检查最高概率的分类是否需要人脸识别
        if classification:
            best_category = max(classification.items(), key=lambda x: x[1])[0]
            return best_category in face_needed_categories
        
        # 默认进行人脸识别
        return True
    
    def batch_analyze(self, image_paths: List[str], progress_callback=None) -> List[Dict]:
        """
        批量分析多张照片
        
        Args:
            image_paths: 图片路径列表
            progress_callback: 进度回调函数，接收(current, total)参数
        
        Returns:
            分析结果列表
        """
        print(f"批量分析 {len(image_paths)} 张照片")
        
        results = []
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            result = self.analyze_photo(image_path)
            if result:
                results.append(result)
            
            # 调用进度回调
            if progress_callback:
                progress_callback(i + 1, total)
        
        print(f"批量分析完成: {len(results)}/{total}")
        return results
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        比较两个人脸特征向量的相似度
        
        Args:
            embedding1: 第一个人脸特征向量
            embedding2: 第二个人脸特征向量
        
        Returns:
            相似度分数 (0-1, 越高越相似)
        """
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
    
    def cluster_faces(self, embeddings: List[np.ndarray], threshold: float = 0.6) -> List[List[int]]:
        """
        对人脸特征向量进行聚类
        
        Args:
            embeddings: 人脸特征向量列表
            threshold: 相似度阈值，超过此值的被认为是同一人
        
        Returns:
            聚类结果，每个子列表包含属于同一人的embedding索引
        """
        print(f"聚类 {len(embeddings)} 个人脸特征")
        
        if not embeddings:
            return []
        
        # 在实际实现中，这里会使用更高效的聚类算法
        # 如DBSCAN或层次聚类
        
        # 简单的相似度聚类
        clusters = []
        used = set()
        
        for i, emb1 in enumerate(embeddings):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            
            for j, emb2 in enumerate(embeddings):
                if j <= i or j in used:
                    continue
                
                similarity = self.compare_faces(emb1, emb2)
                if similarity > threshold:
                    cluster.append(j)
                    used.add(j)
            
            clusters.append(cluster)
        
        print(f"聚类完成: {len(clusters)} 个聚类")
        return clusters


def main():
    """
    主函数 - 用于独立测试AI分析模块
    """
    print("=" * 50)
    print("Photoye AI分析模块测试 (阶段2)")
    print("=" * 50)
    
    # 创建AI分析器实例
    analyzer = AIAnalyzer()
    
    # 模拟测试图片路径
    test_image = "test_photo.jpg"
    
    # 创建一个测试图片文件
    with open(test_image, "w") as f:
        f.write("fake image data")
    
    print(f"\n测试单张照片分析...")
    result = analyzer.analyze_photo(test_image)
    if result:
        print(f"分析结果: {result['category']}, 人脸数: {len(result['faces'])}")
    
    print(f"\n测试人脸相似度比较...")
    emb1 = np.random.rand(512).astype(np.float32)
    emb2 = np.random.rand(512).astype(np.float32)
    similarity = analyzer.compare_faces(emb1, emb2)
    print(f"相似度: {similarity:.3f}")
    
    print(f"\n测试人脸聚类...")
    embeddings = [np.random.rand(512).astype(np.float32) for _ in range(10)]
    clusters = analyzer.cluster_faces(embeddings)
    print(f"聚类结果: {len(clusters)} 个聚类")
    
    # 清理测试文件
    if os.path.exists(test_image):
        os.remove(test_image)
    
    print("\nAI分析模块测试完成！")


if __name__ == "__main__":
    main()