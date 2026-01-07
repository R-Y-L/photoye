#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - AI模型模块

提供以下模型适配器:
- 人脸检测: OpenCVYuNetDetector, InsightFaceDetector
- 人脸识别: OpenCVSFaceRecognizer, InsightFaceRecognizer  
- 场景分类: MobileNetV2Classifier, MobileNetV3Classifier
- 图像嵌入: CLIPEmbeddingEncoder, DLibDetector
"""

from .model_interfaces import FaceDetector, FaceRecognizer, SceneClassifier
from .opencv_yunet_detector import OpenCVYuNetDetector
from .opencv_sface_recognizer import OpenCVSFaceRecognizer
from .insightface_detector import InsightFaceDetector
from .insightface_recognizer import InsightFaceRecognizer
from .mobilenetv2_classifier import MobileNetV2SceneClassifier
from .clip_embedding import CLIPEmbeddingEncoder

__all__ = [
    # 接口
    'FaceDetector',
    'FaceRecognizer', 
    'SceneClassifier',
    # 人脸检测
    'OpenCVYuNetDetector',
    'InsightFaceDetector',
    # 人脸识别
    'OpenCVSFaceRecognizer',
    'InsightFaceRecognizer',
    # 场景分类
    'MobileNetV2SceneClassifier',
    # 图像嵌入
    'CLIPEmbeddingEncoder',
]