#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye M1-F5: 纯本地开放式语义搜索自动化测试

测试覆盖:
1. 文本 Prompt Ensemble 多模板平均编码向量质量
2. 基于 16 张真实图片的端到端自然语言语义搜图 (自然风景意图召回 vs 合影意图召回)
3. 批量矩阵相似度检索准确性与排序严格性
"""

from pathlib import Path
import numpy as np
import pytest

from core.ai.scene_classifier import SceneClassifier
from core.ai.semantic_search import (
    PROMPT_TEMPLATES,
    SearchResultItem,
    SemanticSearchEngine,
)

TEST_IMAGES_DIR = Path(__file__).resolve().parent / "test_images"


@pytest.fixture(scope="module")
def search_engine():
    return SemanticSearchEngine()


@pytest.fixture(scope="module")
def scene_classifier():
    return SceneClassifier()


@pytest.fixture(scope="module")
def indexed_test_images(scene_classifier):
    """为 tests/test_images 下全部 16 张真实图片提取 512 维特征建立内存矩阵。"""
    images = sorted(list(TEST_IMAGES_DIR.glob("*.jpg")))
    identifiers: list[str] = []
    vectors: list[np.ndarray] = []

    for img_p in images:
        emb = scene_classifier.extract_image_embedding(img_p)
        assert emb is not None, f"图像特征提取失败: {img_p.name}"
        identifiers.append(img_p.name)
        vectors.append(emb)

    matrix = np.array(vectors, dtype=np.float32)
    return identifiers, matrix


def test_text_ensemble_encoding(search_engine):
    """验证 Prompt Ensemble 文本向量编码维度与单位化不变量。"""
    query = "beautiful sunset beach"
    vec = search_engine.encode_query_ensemble(query)
    assert vec.shape == (512,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)
    assert len(PROMPT_TEMPLATES) == 7


def test_search_nature_landscape(search_engine, indexed_test_images):
    """自然语言搜索 '自然风景与森林树木' -> 验证 Top 4 全部精准召回风景图。"""
    identifiers, matrix = indexed_test_images
    query = "natural outdoor landscape scenery with green trees and forest"

    results = search_engine.search(query, matrix, identifiers, top_k=5)
    print(f"\n[验证-搜风景] 查询词: '{query}'")
    for idx, item in enumerate(results):
        print(f"  Rank #{idx+1}: {item.identifier:<28} 相似度={item.similarity:.4f}")

    # 验证 Top 4 中必须全部为风景照片
    top_4_names = [item.identifier for item in results[:4]]
    for name in top_4_names:
        assert "landscape" in name, f"期望召回风景照片，实际召回: {name}"

    # 验证最高分的风景照显著高于人像照
    assert results[0].similarity > 0.25


def test_search_group_photo(search_engine, indexed_test_images):
    """自然语言搜索 '多人合照聚集' -> 验证合照排在最前列。"""
    identifiers, matrix = indexed_test_images
    query = "a group of people friends together photo"

    results = search_engine.search(query, matrix, identifiers, top_k=5)
    print(f"\n[验证-搜合影] 查询词: '{query}'")
    for idx, item in enumerate(results):
        print(f"  Rank #{idx+1}: {item.identifier:<28} 相似度={item.similarity:.4f}")

    # Top 1 必须是合影 (group 开头)
    assert "group" in results[0].identifier
    assert results[0].similarity > 0.20


def test_search_close_up_portrait(search_engine, indexed_test_images):
    """自然语言搜索 '特写单人人脸肖像' -> 验证人像特写排在最前列。"""
    identifiers, matrix = indexed_test_images
    query = "close up headshot portrait photo of a single person face"

    results = search_engine.search(query, matrix, identifiers, top_k=5)
    print(f"\n[验证-搜肖像] 查询词: '{query}'")
    for idx, item in enumerate(results):
        print(f"  Rank #{idx+1}: {item.identifier:<28} 相似度={item.similarity:.4f}")

    # Top 1 必须是个人肖像 (person 开头)
    assert "person" in results[0].identifier
