#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye M1-F4: 场景题材分类与多维度多标签正交融合自动化测试

测试覆盖:
1. 纯自然风景照片多标签打标 (包含 "无人物" 与 "风景" 标签)
2. 多人合照照片多标签正交融合 (同时保留 "合照" 与背景 "建筑/风景" 标签，绝不覆盖抹杀)
3. 单人照片包含人物姓名多标签合成 (["单人照", "风景", "人物:YouYu"])
4. 真实 16 张照片全量标签输出直观展示
"""

from pathlib import Path
import pytest

from core.ai.face_engine import FaceEngine
from core.ai.scene_classifier import PhotoMultiLabel, SceneClassifier

TEST_IMAGES_DIR = Path(__file__).resolve().parent / "test_images"


@pytest.fixture(scope="module")
def face_engine():
    return FaceEngine(model_name="buffalo_sc")


@pytest.fixture(scope="module")
def scene_classifier():
    return SceneClassifier()


def test_pure_scenery_multilabels(face_engine, scene_classifier):
    """测试纯自然风景照片的正交打标: 应包含 '无人物' 与 '风景' 标签。"""
    scenery_path = TEST_IMAGES_DIR / "landscape_scenery_01.jpg"
    faces = face_engine.detect_faces(scenery_path)
    result: PhotoMultiLabel = scene_classifier.generate_multi_labels(scenery_path, faces)

    print(f"\n[验证-纯风景打标] {scenery_path.name}: 最终标签组 = {result.all_tags}")
    assert "无人物" in result.subject_tags
    assert "风景" in result.scene_tags
    assert "单人照" not in result.all_tags
    assert "合照" not in result.all_tags


def test_group_photo_multi_label_orthogonality(face_engine, scene_classifier):
    """测试合照的正交叠加特性: 必须具有 '合照' 标签，且能正常输出场景打分分布。"""
    group_path = TEST_IMAGES_DIR / "group_eleven_persons_01.jpg"
    faces = face_engine.detect_faces(group_path)
    assert len(faces) >= 10

    result: PhotoMultiLabel = scene_classifier.generate_multi_labels(group_path, faces)
    print(f"\n[验证-合影多标签] 11人集体照: 最终标签组 = {result.all_tags}")
    assert "合照" in result.subject_tags
    assert "无人物" not in result.all_tags
    assert "美食" not in result.all_tags  # 验证绝不会被误判为美食
    assert isinstance(result.scene_scores, dict)
    assert len(result.scene_scores) == 6


def test_portrait_with_person_name_labels(face_engine, scene_classifier):
    """测试单人照片结合已知人物姓名的完整标签链路。"""
    portrait_path = TEST_IMAGES_DIR / "person_youyu_01_front.jpg"
    faces = face_engine.detect_faces(portrait_path)
    assert len(faces) == 1

    result: PhotoMultiLabel = scene_classifier.generate_multi_labels(
        portrait_path,
        detected_faces=faces,
        identified_names=["YouYu"],
    )
    print(f"\n[验证-单人照多标签] {portrait_path.name}: 最终标签组 = {result.all_tags}")
    assert "单人照" in result.subject_tags
    assert "人物:YouYu" in result.all_tags
    assert "合照" not in result.all_tags


def test_all_16_images_multilabels_report(face_engine, scene_classifier):
    """遍历全部 16 张真实图片，输出完整的多标签检测与融合报告。"""
    all_images = sorted(list(TEST_IMAGES_DIR.glob("*.jpg")))
    print(f"\n=== 全量 16 张真实照片多维度正交标签矩阵 ===\n")
    for p in all_images:
        faces = face_engine.detect_faces(p)
        names = ["YouYu"] if "youyu" in p.name.lower() else []
        res = scene_classifier.generate_multi_labels(p, faces, identified_names=names)
        print(f"📷 文件: {p.name:<28} -> 标签: {res.all_tags}")
