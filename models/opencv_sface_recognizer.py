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
    """使用 OpenCV SFace 生成特征向量"""

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

            if landmarks:
                # YuNet 输出的格式: [x, y, w, h, l0x, l0y, ..., l4y, score]
                w = x2 - x1
                h = y2 - y1
                detect = [float(x1), float(y1), float(w), float(h)]
                for px, py in landmarks:
                    detect.extend([float(px), float(py)])
                detect.append(1.0)
                detect_np = np.array([detect], dtype=np.float32)
                aligned_face = self.recognizer.alignCrop(image, detect_np)
                feature = self.recognizer.feature(aligned_face)
            else:
                face_img = image[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
                if face_img.size == 0:
                    print("无效的人脸裁剪区域")
                    return None

                aligned_face = cv2.resize(face_img, (112, 112))
                feature = self.recognizer.feature(aligned_face)
            return feature.astype(np.float32)

        except Exception as exc:  # noqa: BLE001
            print(f"SFace 提取特征失败: {exc}")
            return self._mock_embedding()

    @staticmethod
    def _mock_embedding() -> np.ndarray:
        mock_embedding = np.random.rand(128).astype(np.float32)
        print("返回模拟特征向量 (SFace 模型缺失)")
        return mock_embedding
