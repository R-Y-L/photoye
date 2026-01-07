#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - InsightFace RetinaFace 人脸检测模型适配器

使用 insightface 库提供的 RetinaFace 检测器 (det_500m.onnx)
特点:
- 比 YuNet 更准确的人脸检测
- 自带 5 点关键点检测 (左眼、右眼、鼻尖、左嘴角、右嘴角)
- 支持更大的人脸尺度范围
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .model_interfaces import FaceDetector

MODEL_DIR = Path(__file__).resolve().parent / "models"


class InsightFaceDetector(FaceDetector):
    """InsightFace RetinaFace 人脸检测模型适配器"""

    def __init__(self, model_path: str | None = None, det_size: tuple = (640, 640)):
        """
        初始化 InsightFace 检测器
        
        Args:
            model_path: 模型路径（未使用，保留接口兼容性）
            det_size: 检测尺寸，默认 (640, 640)
        """
        self.det_size = det_size
        self.app = None
        self._init_error = None
        
        try:
            from insightface.app import FaceAnalysis
            
            # 使用 buffalo_sc 模型包（最小，适合 CPU）
            self.app = FaceAnalysis(
                name='buffalo_sc',
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection']  # 只加载检测模块
            )
            self.app.prepare(ctx_id=0, det_size=self.det_size)
            print(f"✅ 加载 InsightFace RetinaFace 检测器 (det_size={self.det_size})")
            
        except ImportError as e:
            self._init_error = f"insightface 库未安装: {e}"
            print(f"⚠️ {self._init_error}")
        except Exception as e:
            self._init_error = f"InsightFace 初始化失败: {e}"
            print(f"⚠️ {self._init_error}")

    def detect(self, image_path: str) -> List[Dict]:
        """
        检测图片中的人脸
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            人脸检测结果列表，每个元素包含：
            - bbox: [x1, y1, x2, y2] 边界框
            - confidence: 置信度 float
            - landmarks: [[x1,y1], [x2,y2], ...] 5点关键点
        """
        print(f"使用 InsightFace RetinaFace 检测人脸: {image_path}")
        
        if self.app is None:
            print(f"检测器未初始化: {self._init_error}")
            return []
        
        try:
            import cv2
            import os
            
            if not os.path.exists(image_path):
                print(f"图片文件不存在: {image_path}")
                return []
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                return []
            
            # 检测人脸
            faces = self.app.get(image)
            
            results = []
            for face in faces:
                # bbox: [x1, y1, x2, y2]
                bbox = face.bbox.astype(int).tolist()
                
                # 确保坐标在图像范围内
                h, w = image.shape[:2]
                bbox = [
                    max(0, bbox[0]),
                    max(0, bbox[1]),
                    min(w, bbox[2]),
                    min(h, bbox[3])
                ]
                
                # 置信度 (det_score)
                confidence = float(face.det_score) if hasattr(face, 'det_score') else 0.9
                
                # 5点关键点: kps shape (5, 2)
                landmarks = None
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = face.kps.astype(int).tolist()
                
                results.append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "landmarks": landmarks
                })
            
            print(f"检测到 {len(results)} 个人脸")
            return results
            
        except Exception as e:
            print(f"InsightFace 人脸检测出错: {e}")
            import traceback
            traceback.print_exc()
            return []

    def detect_with_image(self, image: np.ndarray) -> List[Dict]:
        """
        检测图片中的人脸（直接接受图像数组）
        
        Args:
            image: BGR格式的图像数组
            
        Returns:
            人脸检测结果列表
        """
        if self.app is None:
            return []
        
        try:
            faces = self.app.get(image)
            
            results = []
            h, w = image.shape[:2]
            
            for face in faces:
                bbox = face.bbox.astype(int).tolist()
                bbox = [
                    max(0, bbox[0]),
                    max(0, bbox[1]),
                    min(w, bbox[2]),
                    min(h, bbox[3])
                ]
                
                confidence = float(face.det_score) if hasattr(face, 'det_score') else 0.9
                
                landmarks = None
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = face.kps.astype(int).tolist()
                
                results.append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "landmarks": landmarks
                })
            
            return results
            
        except Exception as e:
            print(f"InsightFace 人脸检测出错: {e}")
            return []
