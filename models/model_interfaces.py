#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - AI模型接口定义
定义统一的模型接口，支持不同模型的插件化接入
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np


class FaceDetector(ABC):
    """人脸检测模型接口"""
    
    @abstractmethod
    def detect(self, image_path: str) -> List[Dict]:
        """
        检测图片中的人脸
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            人脸检测结果列表，每个元素包含边界框和置信度
            格式: [{'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
        """
        pass


class FaceRecognizer(ABC):
    """人脸识别模型接口"""
    
    @abstractmethod
    def get_embedding(self, image_path: str, bbox: List[int], landmarks: Optional[List[List[int]]] = None) -> Optional[np.ndarray]:
        """
        获取人脸特征向量
        
        Args:
            image_path: 图片文件路径
            bbox: 人脸边界框 [x1, y1, x2, y2]
            landmarks: 可选的五点关键点 [[x1,y1], ...]
            
        Returns:
            人脸特征向量，如果失败返回None
        """
        pass


class SceneClassifier(ABC):
    """场景分类模型接口"""
    
    @abstractmethod
    def classify(self, image_path: str) -> Dict[str, float]:
        """
        对图片进行场景分类
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            分类结果字典，包含各类别的置信度
            格式: {'风景': 0.8, '建筑': 0.1, '室内': 0.05, ...}
        """
        pass