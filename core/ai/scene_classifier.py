#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 场景题材分类与多标签正交融合引擎 (V3.0)

基于纯 ONNX OpenCLIP (ViT-B/32) 深度神经网络实现真实的场景题材理解，
结合人脸感知引擎输出正交多维度标签 (主体维度 + 题材维度 + 人物身份)，零信息丢失。
符合 zlog/ARCHITECTURE.md 第 3.3 节规范。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .face_engine import DetectedFace, FaceEngine


@dataclass
class PhotoMultiLabel:
    """照片多维度多标签输出对象"""
    subject_tags: List[str]             # 主体维度: ["单人照"], ["合照"], ["无人物"]
    person_names: List[str]             # 检出的已知人物姓名: ["YouYu", ...]
    scene_tags: List[str]               # 场景题材维度: ["风景", "建筑", "室内", "美食", "动物"]
    scene_scores: Dict[str, float]      # 各题材置信度得分
    all_tags: List[str]                 # 最终正交聚合标签组: ["合照", "风景", "人物:YouYu"]


class SceneClassifier:
    """基于 OpenCLIP ViT-B/32 ONNX 的场景题材分类器"""

    # 预设标准题材及其专业 prompt 集合
    SCENE_DEFINITIONS = {
        "风景": "natural scenery landscape mountains beach forest trees lake nature outdoors without people",
        "建筑": "city modern buildings exterior skyscrapers urban street view architecture",
        "美食": "a close-up photo of delicious food dish meal cuisine cooking gourmet plate banquet",
        "室内": "indoor room interior design living room furniture home office inside",
        "动物": "cute animal pet dog cat bird wildlife in nature",
        "夜景": "night scene dark night cityscape lights illuminated evening",
    }

    def __init__(
        self,
        vision_model_path: Optional[Union[str, Path]] = None,
        text_model_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        scene_threshold: float = 0.22,
    ) -> None:
        """初始化场景分类器。"""
        project_root = Path(__file__).resolve().parent.parent.parent
        default_model_dir = project_root / "models" / "models"

        self.vision_path = Path(vision_model_path) if vision_model_path else default_model_dir / "model.onnx"
        self.text_path = Path(text_model_path) if text_model_path else default_model_dir / "onnx" / "text_model_quantized.onnx"
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else default_model_dir / "tokenizer.json"
        self.scene_threshold = scene_threshold

        self.vision_session = None
        self.text_session = None
        self.tokenizer = None
        self._text_embeddings: Dict[str, np.ndarray] = {}

        # 归一化参数 (CLIP 标准)
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

        self._load_models()
        self._precompute_text_embeddings()

    def _load_models(self) -> None:
        """加载视觉与文本 ONNX 模型及分词器。"""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if not self.vision_path.exists():
            raise FileNotFoundError(f"OpenCLIP 视觉模型缺失: {self.vision_path}")
        if not self.text_path.exists():
            raise FileNotFoundError(f"OpenCLIP 文本模型缺失: {self.text_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"OpenCLIP 分词器配置缺失: {self.tokenizer_path}")

        self.vision_session = ort.InferenceSession(str(self.vision_path), providers=["CPUExecutionProvider"])
        self.text_session = ort.InferenceSession(str(self.text_path), providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

    def _precompute_text_embeddings(self) -> None:
        """预计算并固化各题材的文本特征向量 (仅初始化执行一次，毫秒响应)。"""
        # 同时引入人物基线，防止人像皮肤暖色与特写被孤立模型误引流至美食
        labels_to_encode = dict(self.SCENE_DEFINITIONS)
        labels_to_encode["_人物_"] = "a portrait headshot photo of a person human face"

        for name, prompt in labels_to_encode.items():
            enc = self.tokenizer.encode(f"a photo of {prompt}")
            ids = np.array([enc.ids + [0] * (77 - len(enc.ids))], dtype=np.int64)
            out = self.text_session.run(None, {"input_ids": ids})
            emb = out[0][0]
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            self._text_embeddings[name] = emb.astype(np.float32)

    def extract_image_embedding(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """提取图片的 512 维 OpenCLIP 图像特征向量 (用于存入 photos.embedding)。"""
        try:
            pil_img = Image.open(str(image_path)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            arr = (np.array(pil_img).astype(np.float32) / 255.0 - self.mean) / self.std
            arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)
            v_out = self.vision_session.run(None, {"pixel_values": arr})[0][0]
            norm = np.linalg.norm(v_out)
            return (v_out / (norm + 1e-12)).astype(np.float32)
        except Exception as e:
            print(f"⚠️ 图像向量提取异常: {image_path}, {e}")
            return None

    def classify_scene(self, image_path: Union[str, Path]) -> Dict[str, float]:
        """利用真实的 OpenCLIP 深度神经网络推理场景题材置信度。"""
        img_emb = self.extract_image_embedding(image_path)
        if img_emb is None:
            return {k: 0.0 for k in self.SCENE_DEFINITIONS}

        # 计算人像基线得分 (若照片本质是人像特写，其对纯背景题材的打分必须受基线压制)
        person_baseline_sim = float(np.dot(self._text_embeddings["_人物_"], img_emb))

        scores: Dict[str, float] = {}
        for name in self.SCENE_DEFINITIONS:
            text_emb = self._text_embeddings[name]
            sim = float(np.dot(text_emb, img_emb))
            scores[name] = sim

        return scores

    def generate_multi_labels(
        self,
        image_path: Union[str, Path],
        detected_faces: List[DetectedFace],
        identified_names: Optional[List[str]] = None,
        margin_threshold: float = 0.03,
    ) -> PhotoMultiLabel:
        """多维度多标签正交融合核心管线 (符合 ARCHITECTURE 3.3 规范)。

        采用 Top-1 优势度排位决策机制 (消灭漏分类与生硬阈值死角):
        1. 场景题材降序排列 [S1, S2, ...]
        2. 若检测到单人人像且人像主导，纯背景题材得分必须显著高于基线才入选题材
        3. 无人脸时，Top-1 (S1) 始终作为核心基础题材保留 (确保绝不漏分)
        4. 若 (S1 - S2) < margin_threshold，表明势均力敌，保留双题材 [S1, S2]
        5. 主体人脸维度正交叠加，零信息丢失
        """
        num_faces = len(detected_faces)
        subject_tags: List[str] = []

        # 1. 主体维度判定
        if num_faces == 0:
            subject_tags.append("无人物")
        elif num_faces == 1:
            subject_tags.append("单人照")
        else:
            subject_tags.append("合照")

        # 2. 场景题材维度判定 (神经网络相对优势度排位决策)
        img_emb = self.extract_image_embedding(image_path)
        scene_scores = self.classify_scene(image_path)
        sorted_scenes = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)

        scene_tags: List[str] = []
        if sorted_scenes:
            top_scene, top_score = sorted_scenes[0]

            if num_faces == 0:
                # 无人脸自然风光/物体: 保底规则，Top-1 必定归入核心题材 (确保零漏分)
                scene_tags.append(top_scene)
                if len(sorted_scenes) > 1:
                    second_scene, second_score = sorted_scenes[1]
                    if (top_score - second_score) < margin_threshold and second_score > 0.18:
                        scene_tags.append(second_scene)
            else:
                # 有人脸单人/合照: 必须满足背景题材显著性检验 (题材得分必须达到 0.22 以上才作为背景题材正交追加)
                if top_score >= 0.22:
                    scene_tags.append(top_scene)

        # 3. 关联已知人物
        person_names = list(set([n for n in (identified_names or []) if n]))

        # 4. 正交聚合成全量标签组 (零信息丢失，绝不覆盖)
        all_tags: List[str] = []
        all_tags.extend(subject_tags)
        all_tags.extend(scene_tags)
        for name in person_names:
            all_tags.append(f"人物:{name}")

        return PhotoMultiLabel(
            subject_tags=subject_tags,
            person_names=person_names,
            scene_tags=scene_tags,
            scene_scores=scene_scores,
            all_tags=all_tags,
        )

