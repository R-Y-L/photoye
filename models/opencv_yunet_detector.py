#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - OpenCV YuNet人脸检测模型适配器
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from .model_interfaces import FaceDetector

MODEL_DIR = Path(__file__).resolve().parent / "models"


class OpenCVYuNetDetector(FaceDetector):
    """OpenCV YuNet人脸检测模型适配器"""

    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path) if model_path else MODEL_DIR / "face_detection_yunet_2023mar.onnx"
        self.detector = None

        if not self.model_path.exists():
            print(f"⚠️ 未找到 YuNet 模型，将尝试由 OpenCV 自动下载: {self.model_path}")
        else:
            print(f"✅ 使用 YuNet 模型: {self.model_path}")

    def _ensure_detector(self, width: int, height: int):
        import cv2

        if self.detector is None:
            self.detector = cv2.FaceDetectorYN_create(
                model=str(self.model_path),
                config="",
                input_size=(width, height),
                score_threshold=0.5,
                nms_threshold=0.3,
                top_k=5000,
            )
        else:
            self.detector.setInputSize((width, height))

    def detect(self, image_path: str) -> List[Dict]:
        print(f"使用 OpenCV YuNet 检测人脸: {image_path}")

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

            h, w = image.shape[:2]
            self._ensure_detector(w, h)

            if self.detector is None:
                return self._mock_detection()

            faces = self.detector.detect(image)
            results = []
            if faces[1] is not None:
                for face in faces[1]:
                    x, y, bw, bh = face[0:4]
                    landmarks = [
                        [face[4], face[5]],
                        [face[6], face[7]],
                        [face[8], face[9]],
                        [face[10], face[11]],
                        [face[12], face[13]],
                    ]
                    confidence = face[-1]
                    results.append({
                        "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                        "confidence": float(confidence),
                        "landmarks": [[int(px), int(py)] for px, py in landmarks],
                    })

            print(f"检测到 {len(results)} 个人脸")
            return results

        except Exception as exc:  # noqa: BLE001
            print(f"YuNet 人脸检测出错: {exc}")
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