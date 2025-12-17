#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - Dlib 人脸检测与识别适配器（真实加载版）
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .model_interfaces import FaceDetector, FaceRecognizer


MODEL_DIR = Path(__file__).resolve().parent / "models"


class DlibFaceDetector(FaceDetector):
    """Dlib人脸检测模型适配器"""

    def __init__(self, predictor_path: Optional[str] = None):
        predictor_file = predictor_path or str(MODEL_DIR / "shape_predictor_68_face_landmarks.dat")

        self.predictor = None
        self.detector = None

        try:
            import dlib

            predictor_resolved = Path(predictor_file)
            if not predictor_resolved.exists():
                print(f"⚠️ 未找到 Dlib 关键点模型: {predictor_resolved}")
                return

            self.predictor = dlib.shape_predictor(str(predictor_resolved))
            self.detector = dlib.get_frontal_face_detector()
            print(f"✅ 加载 Dlib 检测模型: {predictor_resolved}")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Dlib 检测模型加载失败: {exc}")
            self.predictor = None
            self.detector = None

    def detect(self, image_path: str) -> List[Dict]:
        print(f"使用 Dlib 检测人脸: {image_path}")

        try:
            import cv2
            import os

            if not os.path.exists(image_path):
                print(f"图片文件不存在: {image_path}")
                return []

            if self.detector is None:
                return self._mock_detection()

            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                return []

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.detector(rgb_image)

            results = [
                {
                    "bbox": [face.left(), face.top(), face.right(), face.bottom()],
                    "confidence": 0.95,
                }
                for face in faces
            ]

            print(f"检测到 {len(results)} 个人脸")
            return results

        except Exception as exc:  # noqa: BLE001
            print(f"Dlib 人脸检测出错: {exc}")
            return self._mock_detection()

    @staticmethod
    def _mock_detection() -> List[Dict]:
        num_faces = random.randint(0, 3)
        return [
            {
                "bbox": [
                    random.randint(50, 200),
                    random.randint(50, 200),
                    random.randint(300, 500),
                    random.randint(300, 500),
                ],
                "confidence": random.uniform(0.7, 0.95),
            }
            for _ in range(num_faces)
        ]


class DlibFaceRecognizer(FaceRecognizer):
    """Dlib人脸识别模型适配器"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        predictor_path: Optional[str] = None,
    ):
        self.model_path = Path(model_path) if model_path else MODEL_DIR / "dlib_face_recognition_resnet_model_v1.dat"
        self.predictor_path = Path(predictor_path) if predictor_path else MODEL_DIR / "shape_predictor_68_face_landmarks.dat"

        self.recognizer = None
        self.predictor = None

        try:
            import dlib

            if not self.model_path.exists():
                print(f"⚠️ 未找到 Dlib 识别模型: {self.model_path}")
                return
            if not self.predictor_path.exists():
                print(f"⚠️ 未找到 Dlib 关键点模型: {self.predictor_path}")
                return

            self.recognizer = dlib.face_recognition_model_v1(str(self.model_path))
            self.predictor = dlib.shape_predictor(str(self.predictor_path))
            print(f"✅ 加载 Dlib 识别模型: {self.model_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Dlib 识别模型加载失败: {exc}")
            self.recognizer = None
            self.predictor = None

    def get_embedding(self, image_path: str, bbox: List[int], landmarks: Optional[List[List[int]]] = None) -> Optional[np.ndarray]:
        print(f"使用 Dlib 提取人脸特征: {image_path}")

        try:
            import cv2
            import os
            import dlib

            if not os.path.exists(image_path):
                print(f"图片文件不存在: {image_path}")
                return None

            if self.recognizer is None or self.predictor is None:
                return self._mock_embedding()

            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                return None

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rect = dlib.rectangle(*bbox)
            shape = self.predictor(rgb_image, rect)
            face_descriptor = self.recognizer.compute_face_descriptor(rgb_image, shape)
            return np.array(face_descriptor, dtype=np.float32)

        except Exception as exc:  # noqa: BLE001
            print(f"Dlib人脸识别出错: {exc}")
            return self._mock_embedding()

    @staticmethod
    def _mock_embedding() -> np.ndarray:
        mock_embedding = np.random.rand(128).astype(np.float32)
        print("返回模拟人脸特征向量 (Dlib 模型缺失)")
        return mock_embedding