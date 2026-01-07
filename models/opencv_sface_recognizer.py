#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - OpenCV SFace 人脸识别模型适配器

依赖: opencv-contrib-python >= 4.7 (提供 FaceRecognizerSF)
模型: face_recognition_sface_2021dec.onnx (由 download_models.py 下载)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .model_interfaces import FaceRecognizer

MODEL_DIR = Path(__file__).resolve().parent / "models"


class OpenCVSFaceRecognizer(FaceRecognizer):
    """使用 OpenCV SFace 生成特征向量（128维）"""

    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path) if model_path else MODEL_DIR / "face_recognition_sface_2021dec.onnx"
        self.recognizer = None

        try:
            import cv2

            if not self.model_path.exists():
                print(f"⚠️ 未找到 SFace 模型: {self.model_path}")
                return

            self.recognizer = cv2.FaceRecognizerSF_create(str(self.model_path), "")
            print(f"✅ 加载 SFace 模型: {self.model_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ 加载 SFace 模型失败: {exc}")
            self.recognizer = None

    def get_embedding(self, image_path: str, bbox: List[int], landmarks: Optional[List[List[int]]] = None) -> Optional[np.ndarray]:
        """
        提取人脸特征向量
        
        Args:
            image_path: 图片路径
            bbox: 人脸边界框 [x1, y1, x2, y2]
            landmarks: 人脸关键点（5点），来自 YuNet 检测结果
        
        Returns:
            128维特征向量，已L2归一化
        """
        try:
            import cv2
            import os

            if not os.path.exists(image_path):
                print(f"图片文件不存在: {image_path}")
                return None

            if self.recognizer is None:
                return self._mock_embedding()

            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                return None

            x1, y1, x2, y2 = bbox
            
            # 确保坐标在图像范围内
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if landmarks and len(landmarks) == 5:
                # 使用 YuNet 的 landmarks 进行对齐
                # SFace.alignCrop 期望的格式: [x, y, w, h, l0x, l0y, l1x, l1y, ..., l4x, l4y, score]
                face_w = x2 - x1
                face_h = y2 - y1
                detect = [float(x1), float(y1), float(face_w), float(face_h)]
                for px, py in landmarks:
                    detect.extend([float(px), float(py)])
                detect.append(1.0)  # score
                detect_np = np.array([detect], dtype=np.float32)
                
                try:
                    aligned_face = self.recognizer.alignCrop(image, detect_np)
                    feature = self.recognizer.feature(aligned_face)
                except Exception as e:
                    print(f"对齐裁剪失败，使用直接裁剪: {e}")
                    aligned_face = self._simple_crop(image, x1, y1, x2, y2)
                    feature = self.recognizer.feature(aligned_face)
            else:
                # 没有 landmarks，使用简单裁剪
                aligned_face = self._simple_crop(image, x1, y1, x2, y2)
                feature = self.recognizer.feature(aligned_face)
            
            # L2 归一化
            feature = feature.flatten()
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            
            return feature.astype(np.float32)

        except Exception as exc:  # noqa: BLE001
            print(f"SFace 提取特征失败: {exc}")
            import traceback
            traceback.print_exc()
            return self._mock_embedding()
    
    def _simple_crop(self, image, x1: int, y1: int, x2: int, y2: int):
        """简单裁剪并缩放到 112x112"""
        import cv2
        
        face_img = image[y1:y2, x1:x2]
        if face_img.size == 0:
            # 如果裁剪区域无效，返回一个空白图像
            return np.zeros((112, 112, 3), dtype=np.uint8)
        
        # 缩放到 SFace 期望的输入尺寸
        aligned_face = cv2.resize(face_img, (112, 112))
        return aligned_face

    @staticmethod
    def _mock_embedding() -> np.ndarray:
        """返回模拟特征向量"""
        mock_embedding = np.random.rand(128).astype(np.float32)
        # L2 归一化
        mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
        print("返回模拟特征向量 (SFace 模型缺失)")
        return mock_embedding
    
    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        比较两个特征向量的相似度
        
        Returns:
            余弦相似度 (0-1, 越高越相似)
        """
        # 由于已经L2归一化，直接点积就是余弦相似度
        similarity = np.dot(embedding1.flatten(), embedding2.flatten())
        return float(max(0, min(1, similarity)))  # 限制在 [0, 1] 范围
