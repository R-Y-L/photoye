#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - InsightFace ArcFace 人脸识别模型适配器

使用 insightface 库提供的 ArcFace 识别器 (w600k_mbf.onnx)
特点:
- 512维特征向量（比 SFace 的 128维更精确）
- 基于 MobileFaceNet 架构，适合 CPU 推理
- 需要配合 5点关键点进行人脸对齐
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .model_interfaces import FaceRecognizer

MODEL_DIR = Path(__file__).resolve().parent / "models"

# ArcFace 标准对齐模板 (112x112)
# 5点关键点: 左眼中心、右眼中心、鼻尖、左嘴角、右嘴角
ARCFACE_DST = np.array([
    [38.2946, 51.6963],   # 左眼
    [73.5318, 51.5014],   # 右眼
    [56.0252, 71.7366],   # 鼻尖
    [41.5493, 92.3655],   # 左嘴角
    [70.7299, 92.2041],   # 右嘴角
], dtype=np.float32)


class InsightFaceRecognizer(FaceRecognizer):
    """InsightFace ArcFace 人脸识别模型适配器 (512维特征向量)"""

    def __init__(self, model_path: str | None = None):
        """
        初始化 ArcFace 识别器
        
        Args:
            model_path: 模型路径，默认使用 models/w600k_mbf.onnx
        """
        self.model_path = Path(model_path) if model_path else MODEL_DIR / "w600k_mbf.onnx"
        self.session = None
        self.input_name = None
        self.output_name = None
        
        try:
            import onnxruntime as ort
            
            if not self.model_path.exists():
                print(f"⚠️ 未找到 ArcFace 模型: {self.model_path}")
                return
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"✅ 加载 ArcFace 模型: {self.model_path} (512维特征向量)")
            
        except Exception as e:
            print(f"⚠️ 加载 ArcFace 模型失败: {e}")
            self.session = None

    def get_embedding(
        self,
        image_path: str,
        bbox: List[int],
        landmarks: Optional[List[List[int]]] = None
    ) -> Optional[np.ndarray]:
        """
        获取人脸特征向量
        
        Args:
            image_path: 图片文件路径
            bbox: 人脸边界框 [x1, y1, x2, y2]
            landmarks: 5点关键点 [[x1,y1], [x2,y2], ...]，用于对齐
            
        Returns:
            512维特征向量 (L2归一化)，失败返回 None
        """
        try:
            import cv2
            import os
            
            if not os.path.exists(image_path):
                print(f"图片文件不存在: {image_path}")
                return None
            
            if self.session is None:
                print("ArcFace 模型未加载")
                return None
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                return None
            
            return self._extract_embedding(image, bbox, landmarks)
            
        except Exception as e:
            print(f"ArcFace 特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_embedding_from_image(
        self,
        image: np.ndarray,
        bbox: List[int],
        landmarks: Optional[List[List[int]]] = None
    ) -> Optional[np.ndarray]:
        """
        从图像数组获取人脸特征向量
        
        Args:
            image: BGR格式的图像数组
            bbox: 人脸边界框 [x1, y1, x2, y2]
            landmarks: 5点关键点
            
        Returns:
            512维特征向量
        """
        if self.session is None:
            return None
        
        return self._extract_embedding(image, bbox, landmarks)

    def _extract_embedding(
        self,
        image: np.ndarray,
        bbox: List[int],
        landmarks: Optional[List[List[int]]] = None
    ) -> Optional[np.ndarray]:
        """
        核心特征提取逻辑
        """
        try:
            import cv2
            
            # 对齐人脸
            if landmarks and len(landmarks) == 5:
                # 使用仿射变换对齐到标准模板
                aligned = self._align_face(image, landmarks)
            else:
                # 没有关键点，使用简单裁剪缩放
                aligned = self._simple_crop(image, bbox)
            
            if aligned is None:
                return None
            
            # 预处理：BGR -> RGB, HWC -> CHW, 归一化
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            aligned = aligned.astype(np.float32)
            aligned = (aligned - 127.5) / 127.5  # 归一化到 [-1, 1]
            aligned = aligned.transpose(2, 0, 1)  # HWC -> CHW
            aligned = np.expand_dims(aligned, axis=0)  # 添加 batch 维度
            
            # 推理
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: aligned}
            )
            
            embedding = outputs[0].flatten()
            
            # L2 归一化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None

    def _align_face(self, image: np.ndarray, landmarks: List[List[int]]) -> Optional[np.ndarray]:
        """
        使用 5 点关键点对齐人脸到 112x112
        
        Args:
            image: BGR图像
            landmarks: 5点关键点 [[x,y], ...]
            
        Returns:
            对齐后的 112x112 人脸图像
        """
        try:
            import cv2
            
            src_pts = np.array(landmarks, dtype=np.float32)
            
            # 使用相似变换（保持比例的仿射变换）
            # 计算变换矩阵
            M = self._estimate_affine_transform(src_pts, ARCFACE_DST)
            
            if M is None:
                return None
            
            # 应用变换
            aligned = cv2.warpAffine(
                image, M, (112, 112),
                borderValue=(0, 0, 0)
            )
            
            return aligned
            
        except Exception as e:
            print(f"人脸对齐失败: {e}")
            return None

    def _estimate_affine_transform(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        估计仿射变换矩阵 (使用 Umeyama 算法)
        
        Args:
            src_pts: 源点坐标 (5, 2)
            dst_pts: 目标点坐标 (5, 2)
            
        Returns:
            2x3 仿射变换矩阵
        """
        try:
            import cv2
            
            # 使用 OpenCV 的 estimateAffinePartial2D 计算相似变换
            # 这会保持宽高比
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            
            if M is None:
                # 回退到完整仿射变换
                M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
            
            return M
            
        except Exception as e:
            print(f"变换矩阵估计失败: {e}")
            return None

    def _simple_crop(self, image: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        简单裁剪缩放（没有关键点时使用）
        
        Args:
            image: BGR图像
            bbox: [x1, y1, x2, y2]
            
        Returns:
            112x112 人脸图像
        """
        try:
            import cv2
            
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            
            # 确保坐标有效
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            face = image[y1:y2, x1:x2]
            
            # 缩放到 112x112
            aligned = cv2.resize(face, (112, 112))
            
            return aligned
            
        except Exception as e:
            print(f"简单裁剪失败: {e}")
            return None

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个特征向量的余弦相似度
        
        Args:
            emb1, emb2: L2归一化的特征向量
            
        Returns:
            相似度 [-1, 1]，越接近1越相似
        """
        return float(np.dot(emb1, emb2))
