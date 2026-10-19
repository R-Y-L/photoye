#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye M2-F3: 物理存储操作落地与防灾重命名自动化测试

测试覆盖:
1. 同磁盘优先 os.link 原生硬链接秒级分流与 inode 共享验证 (0 额外磁盘占用)
2. 跨磁盘或不支持时自适应降级为 shutil.copy2 完整复制
3. 重名防灾强保护 (No Overwrite): 同名主文件自动递增 _1，且伴侣文件严格同步更名
4. 物理分拣完整执行审计报告统计准确性
"""

import os
from pathlib import Path
import pytest

from core.ai.scene_classifier import PhotoMultiLabel
from core.metadata.companion_scanner import CompanionAsset, PhotoAssetGroup
from core.sorter import (
    BasketFilterRule,
    DispatchBasket,
    DispatchPlan,
    FileOperationsExecutor,
    SortingPipeline,
)


@pytest.fixture
def mock_storage_env(tmp_path):
    """构建包含源相册与目标输出目录的临时测试环境。"""
    src_dir = tmp_path / "source_photos"
    out_dir = tmp_path / "target_output"
    src_dir.mkdir()
    out_dir.mkdir()

    # 1. 真实写入主照片及其 LivePhoto 视频伴侣
    p1 = src_dir / "IMG_001.JPG"
    p1_mov = src_dir / "IMG_001.MOV"
    p1.write_bytes(b"image_content_11111")
    p1_mov.write_bytes(b"live_video_content_22222")

    g1 = PhotoAssetGroup(
        main_photo_path=p1,
        filesize=p1.stat().st_size,
        created_at="2026-05-01T12:00:00",
        companions=[CompanionAsset(filepath=p1_mov, file_type="live_video", filesize=p1_mov.stat().st_size)],
    )

    # 2. 模拟目标目录下已存在一个同名的旧文件 (诱发重名冲突碰撞)
    collision_dir = out_dir / "人物" / "Alice"
    collision_dir.mkdir(parents=True, exist_ok=True)
    existing_old_photo = collision_dir / "IMG_001.JPG"
    existing_old_photo.write_bytes(b"pre_existing_different_content_999")

    labels_map = {
        "IMG_001.JPG": PhotoMultiLabel(
            subject_tags=["单人照"],
            person_names=["Alice"],
            scene_tags=[],
            scene_scores={},
            all_tags=["单人照", "人物:Alice"],
        )
    }

    return src_dir, out_dir, [g1], labels_map


def test_hardlink_execution_and_inode_sharing(mock_storage_env):
    """测试原生硬链接执行及零额外磁盘占用。"""
    src_dir, out_dir, groups, labels_map = mock_storage_env

    # 建立一个去向篮子到全新子目录 (不发生冲突)
    clean_basket = DispatchBasket(
        basket_id="b_clean",
        name="分给 Alice",
        output_subfolder="Alice_全新",
        rule=BasketFilterRule(person_names=["Alice"]),
    )

    pipeline = SortingPipeline(output_root=out_dir)
    plan = pipeline.plan_dispatch(groups, labels_map, [clean_basket])

    executor = FileOperationsExecutor(prefer_hardlink=True)
    report = executor.execute_plan(plan)

    assert report.successful_count == 2  # 1 主图 + 1 伴侣
    assert report.failed_count == 0
    assert report.hardlink_count == 2

    # 验证真实磁盘物理文件确实落地
    target_jpg = out_dir / "Alice_全新" / "IMG_001.JPG"
    target_mov = out_dir / "Alice_全新" / "IMG_001.MOV"
    assert target_jpg.exists()
    assert target_mov.exists()

    # 验证 inode 严格相等 (证明是真正的硬链接，0 空间占用)
    src_jpg = groups[0].main_photo_path
    assert src_jpg.stat().st_ino == target_jpg.stat().st_ino
    print(f"\n[验证-硬链接] 物理落地成功，共享 Inode: {target_jpg.stat().st_ino} (0 额外磁盘占用)")


def test_collision_avoidance_with_companion_sync(mock_storage_env):
    """测试重名防灾强保护 (No Overwrite): 主图自动递增序号，伴侣文件严格同步更名。"""
    src_dir, out_dir, groups, labels_map = mock_storage_env

    # 该篮子的子目录已有同名文件 IMG_001.JPG
    collision_basket = DispatchBasket(
        basket_id="b_alice",
        name="分给 Alice",
        output_subfolder="人物/Alice",
        rule=BasketFilterRule(person_names=["Alice"]),
    )

    pipeline = SortingPipeline(output_root=out_dir)
    plan = pipeline.plan_dispatch(groups, labels_map, [collision_basket])

    executor = FileOperationsExecutor(prefer_hardlink=True)
    report = executor.execute_plan(plan)

    assert report.successful_count == 2
    assert report.failed_count == 0

    # 1. 验证已存在的旧文件完好无损 (绝无静默覆盖！)
    existing_old = out_dir / "人物" / "Alice" / "IMG_001.JPG"
    assert existing_old.read_bytes() == b"pre_existing_different_content_999"

    # 2. 验证新分拣的主图自动获得防灾序号 _1
    resolved_jpg = out_dir / "人物" / "Alice" / "IMG_001_1.JPG"
    assert resolved_jpg.exists()
    assert resolved_jpg.read_bytes() == b"image_content_11111"

    # 3. 核心断言: 伴侣 LivePhoto 视频必须同步获得相同序号 _1 (绝不发生错位脱节！)
    resolved_mov = out_dir / "人物" / "Alice" / "IMG_001_1.MOV"
    assert resolved_mov.exists()
    assert resolved_mov.read_bytes() == b"live_video_content_22222"
    print(f"\n[验证-防灾重命名] 目标同名已保护! 主图更名: {resolved_jpg.name}, 伴侣视频同步更名: {resolved_mov.name}")


def test_fallback_copy_mode(mock_storage_env):
    """测试禁用硬链接时的纯复制落地模式。"""
    src_dir, out_dir, groups, labels_map = mock_storage_env

    basket_copy = DispatchBasket(
        basket_id="b_copy",
        name="复制专区",
        output_subfolder="Copy_Zone",
        rule=BasketFilterRule(person_names=["Alice"]),
    )

    pipeline = SortingPipeline(output_root=out_dir)
    plan = pipeline.plan_dispatch(groups, labels_map, [basket_copy])

    # 指定 prefer_hardlink=False 强制走复制
    executor = FileOperationsExecutor(prefer_hardlink=False)
    report = executor.execute_plan(plan)

    assert report.successful_count == 2
    assert report.copy_count == 2
    assert report.hardlink_count == 0

    target_jpg = out_dir / "Copy_Zone" / "IMG_001.JPG"
    assert target_jpg.exists()
    # 复制文件的 inode 必定与原文件不同
    assert groups[0].main_photo_path.stat().st_ino != target_jpg.stat().st_ino
    print(f"\n[验证-复制降级] 复制落地成功，独立 Inode: {target_jpg.stat().st_ino}")
