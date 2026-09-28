#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 自适应人像感知引擎 (V3.0)

基于纯 ONNX 运行时实现高精度、零外部编译的人脸检测、关键点定位、
几何姿态（水平偏航角 Yaw 与垂直俯仰角 Pitch）评估，以及 512 维特征抽取。
符合 zlog/ARCHITECTURE.md 第 3.1 节规范。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .face_library import normalize_vector


@dataclass
class DetectedFace:
    """人脸检测结果对象"""
    bbox: List[int]                     # 边界框: [x1, y1, x2, y2]
    landmarks: List[List[float]]        # 5 点关键点: 左眼、右眼、鼻尖、左嘴角、右嘴角
    embedding: np.ndarray               # 512 维 ArcFace 归一化特征向量
    confidence: float                   # 检测置信度 (0.0 ~ 1.0)
    yaw_ratio: float                    # 水平偏航比率 R_yaw
    pitch_ratio: float                  # 垂直俯仰比率 R_pitch
    pose: str                           # 姿态类别: "front", "profile", "pitch"


class FaceEngine:
    """自适应人像感知引擎

    特性:
    - 纯 ONNX 运行时: 默认采用 buffalo_sc (det_500m + w600k_mbf)，开箱即用
    - 跨平台中文路径安全: 底层基于 cv2.imdecode，消除 Windows 路径编码问题
    - 双向几何姿态计算: 自动分辨平视正脸、大角度侧脸与大幅度仰俯姿态
    """

    def __init__(
        self,
        model_name: str = "buffalo_sc",
        det_size: Tuple[int, int] = (640, 640),
        providers: Optional[List[str]] = None,
    ) -> None:
        """初始化人像感知引擎。

        Args:
            model_name: 模型包名称，默认 'buffalo_sc'
            det_size: 人脸检测输入尺寸，默认 (640, 640)
            providers: ONNX Runtime 执行提供商，默认 ['CPUExecutionProvider']
        """
        self.model_name = model_name
        self.det_size = det_size
        self.providers = providers or ["CPUExecutionProvider"]
        self.app = None
        self._init_engine()

    def _init_engine(self) -> None:
        """加载 InsightFace ONNX 模型。"""
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name=self.model_name,
            providers=self.providers,
        )
        self.app.prepare(ctx_id=0, det_size=self.det_size)

    @staticmethod
    def read_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """安全读取包含中文或特殊字符路径的图像文件。"""
        path_str = str(image_path)
        if not os.path.exists(path_str):
            return None
        data = np.fromfile(path_str, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    @staticmethod
    def evaluate_pose(landmarks: np.ndarray) -> Tuple[float, float, str]:
        """根据 5 点关键点评估人脸的水平偏航比率与垂直俯仰比率。

        返回: (yaw_ratio, pitch_ratio, pose_category)
        - front: 正脸或微侧脸
        - profile: 大角度侧脸
        - pitch: 大角度抬头或低头
        """
        # landmarks: 0:左眼, 1:右眼, 2:鼻尖, 3:左嘴角, 4:右嘴角
        kps = np.asarray(landmarks, dtype=np.float32)

        # 1. 水平偏航角评估 (Yaw Ratio)
        d_left = abs(float(kps[2][0] - kps[0][0]))
        d_right = abs(float(kps[1][0] - kps[2][0]))
        yaw_ratio = d_left / (d_right + 1e-6)

        # 2. 垂直俯仰角评估 (Pitch Ratio)
        eye_center_y = float(kps[0][1] + kps[1][1]) / 2.0
        mouth_center_y = float(kps[3][1] + kps[4][1]) / 2.0
        nose_y = float(kps[2][1])
        d_eye_nose = abs(nose_y - eye_center_y)
        d_nose_mouth = abs(mouth_center_y - nose_y)
        pitch_ratio = d_eye_nose / (d_nose_mouth + 1e-6)

        # 判定姿态类别 (依照 ARCHITECTURE.md 3.1 节规范)
        is_yaw_profile = (yaw_ratio >= 1.8 or yaw_ratio <= 0.55)
        is_pitch_extreme = (pitch_ratio <= 0.35 or pitch_ratio >= 1.6)

        if is_pitch_extreme:
            pose_category = "pitch"
        elif is_yaw_profile:
            pose_category = "profile"
        else:
            pose_category = "front"

        return yaw_ratio, pitch_ratio, pose_category

    def detect_faces(self, image_path: Union[str, Path]) -> List[DetectedFace]:
        """输入图片文件路径，完成人脸检测、姿态判别与 512 维特征抽取。"""
        img = self.read_image(image_path)
        if img is None:
            return []

        raw_faces = self.app.get(img)
        detected_list: List[DetectedFace] = []

        for face in raw_faces:
            bbox = face.bbox.astype(int).tolist()
            conf = float(face.det_score)
            kps = face.kps  # 5x2 array

            # 计算姿态
            yaw_ratio, pitch_ratio, pose = self.evaluate_pose(kps)

            # 512 维特征向量 L2 归一化
            norm_embedding = normalize_vector(face.embedding)

            detected_list.append(
                DetectedFace(
                    bbox=bbox,
                    landmarks=kps.tolist(),
                    embedding=norm_embedding,
                    confidence=conf,
                    yaw_ratio=yaw_ratio,
                    pitch_ratio=pitch_ratio,
                    pose=pose,
                )
            )

        return detected_list
