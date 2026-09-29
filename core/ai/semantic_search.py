#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 开放式语义搜索服务模块 (V3.0)

基于纯 ONNX 运行时实现纯本地、零网络依赖的自然语言以文搜图能力。
采用 Prompt Ensemble 多模板平均算法，结合 512 维向量矩阵余弦点积，
毫秒级响应用户的自然语言生活化找图诉求。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Prompt Ensemble 经典模板集合 (平均多模板有效消除单一 prompt 的语言偏差)
PROMPT_TEMPLATES: List[str] = [
    "a photo of {}",
    "a photograph of {}",
    "an image showing {}",
    "{} in a photo",
    "a picture of {}",
    "a clear photo of {}",
    "a photo depicting {}",
]

# 本地中英文常用生活词汇对齐词典 (零外部网络依赖，解决英文原版 CLIP 中文分词碎片化断层问题)
CHINESE_TO_ENGLISH_CONCEPT_MAP: Dict[str, str] = {
    # 自然风光
    "风景": "natural landscape scenery nature",
    "树林": "green trees and forest woods nature",
    "森林": "green forest trees nature",
    "海边": "ocean beach sea coast water sunset",
    "海滩": "sandy beach ocean coast water",
    "山": "mountains rocky mountain peak hills",
    "雪山": "snow mountain winter landscape",
    "蓝天": "blue sky clouds sunny weather",
    "日落": "sunset golden hour evening sky",
    "夕阳": "sunset dusk evening sun",
    # 人物与合照
    "人": "person people human portrait",
    "单人": "single person headshot portrait photo",
    "自拍": "selfie headshot close up portrait",
    "肖像": "portrait headshot of a person",
    "合照": "a group photo of friends gathering together",
    "合影": "group photo of people together",
    "两个人": "two people two friends together",
    "双人": "two people together photo",
    "朋友": "friends gathering together smiling",
    "家庭": "family members together photo",
    "集体照": "large crowd group of people team photo",
    # 场景与生活
    "美食": "delicious food dish meal cooking gourmet restaurant cuisine",
    "食物": "delicious food cuisine dish",
    "吃饭": "people dining eating meal food on table",
    "建筑": "city modern buildings exterior architecture urban",
    "大楼": "modern skyscrapers city buildings",
    "街景": "urban street view city roads buildings",
    "室内": "indoor room interior living room furniture home",
    "房间": "room interior design home",
    "夜景": "night scene dark night city lights illuminated",
    "宠物": "cute animal pet dog cat",
    "猫": "cute cat pet kitty",
    "狗": "cute dog pet puppy",
    "文档": "document paper text scan screenshot receipt",
}


@dataclass
class SearchResultItem:
    """语义搜索单项匹配结果"""
    identifier: Union[int, str]        # 照片 ID 或 文件路径
    similarity: float                  # 余弦相似度得分 (-1.0 ~ 1.0)
    extra_metadata: Optional[Dict] = None


class SemanticSearchEngine:
    """纯本地开放式语义搜索服务引擎"""

    def __init__(
        self,
        text_model_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """初始化语义搜索引擎。

        Args:
            text_model_path: OpenCLIP 文本编码器 ONNX 模型路径
            tokenizer_path: 分词器配置文件路径
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        default_model_dir = project_root / "models" / "models"

        self.text_path = Path(text_model_path) if text_model_path else default_model_dir / "onnx" / "text_model_quantized.onnx"
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else default_model_dir / "tokenizer.json"

        self.text_session = None
        self.tokenizer = None
        self._load_resources()

    def _load_resources(self) -> None:
        """加载 ONNX 文本推理会话与分词器。"""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if not self.text_path.exists():
            raise FileNotFoundError(f"OpenCLIP 文本编码模型缺失: {self.text_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"OpenCLIP 分词器配置文件缺失: {self.tokenizer_path}")

        self.text_session = ort.InferenceSession(str(self.text_path), providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

    def encode_text_single(self, text: str) -> np.ndarray:
        """将单条文本编码为 512 维 L2 归一化向量。"""
        enc = self.tokenizer.encode(text)
        # 裁剪并补齐至 CLIP 标准的 77 长度
        raw_ids = enc.ids[:77]
        padded_ids = raw_ids + [0] * (77 - len(raw_ids))
        input_ids = np.array([padded_ids], dtype=np.int64)

        outputs = self.text_session.run(None, {"input_ids": input_ids})
        vec = outputs[0][0]
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        return vec.astype(np.float32)

    def encode_query_ensemble(self, query: str) -> np.ndarray:
        """基于 Prompt Ensemble 与中英文双轨自适应对查询词进行多模板增强平均编码。"""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("查询文本不可为空")

        # 检查是否命中本地中文概念映射，若命中则自适应扩充为精准的高质量英文概念
        mapped_concepts: List[str] = []
        for zh_kw, en_expansion in CHINESE_TO_ENGLISH_CONCEPT_MAP.items():
            if zh_kw in clean_query:
                mapped_concepts.append(en_expansion)

        if mapped_concepts:
            effective_query = f"{' '.join(mapped_concepts[:2])} {clean_query}"
        else:
            effective_query = clean_query

        template_vectors: List[np.ndarray] = []
        for tmpl in PROMPT_TEMPLATES:
            prompt_text = tmpl.format(effective_query)
            vec = self.encode_text_single(prompt_text)
            template_vectors.append(vec)

        # 向量求和平均并再次执行 L2 归一化
        mean_vec = np.mean(template_vectors, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-12:
            mean_vec = mean_vec / norm
        return mean_vec.astype(np.float32)

    def search(
        self,
        query: str,
        image_embeddings: np.ndarray,
        identifiers: List[Union[int, str]],
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[SearchResultItem]:
        """对全库照片特征矩阵进行批量余弦相似度检索并降序排序。

        Args:
            query: 自然语言查询词 (如 "海滩日落风景", "两个人合照")
            image_embeddings: 照片库特征矩阵，形状为 [N, 512] (预先 L2 归一化)
            identifiers: 与矩阵行一一对应的照片标识列表 (ID 或路径)
            top_k: 最多返回的匹配结果数
            min_similarity: 最低相似度阈值

        Returns:
            按相似度从高到低排序的 SearchResultItem 列表
        """
        if len(image_embeddings) == 0 or len(identifiers) == 0:
            return []
        if len(image_embeddings) != len(identifiers):
            raise ValueError("特征矩阵行数必须与标识符列表长度完全一致")

        # 1. 提取查询语句的多模板 512 维向量
        q_vec = self.encode_query_ensemble(query)

        # 2. 纯单指令集矩阵乘法计算余弦相似度: [N, 512] @ [512] -> [N]
        similarities = np.dot(image_embeddings, q_vec)

        # 3. 排序并截取 Top-K
        k = min(top_k, len(similarities))
        if k <= 0:
            return []

        # 获取排序索引 (降序)
        sorted_indices = np.argsort(similarities)[::-1][:k]

        results: List[SearchResultItem] = []
        for idx in sorted_indices:
            score = float(similarities[idx])
            if score >= min_similarity:
                results.append(
                    SearchResultItem(
                        identifier=identifiers[idx],
                        similarity=score,
                    )
                )

        return results
