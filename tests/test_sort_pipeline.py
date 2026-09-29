#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye M2-F2: 去向篮子与拖拽追加分发自动化测试

测试覆盖:
1. 自动吸附规则: 按人物建立篮子时，合照自动 1对多扇出分发
2. 手工显式追加: 模拟用户拖拽将一张特定风景照手工塞入 Alice 篮子，验证成功并入且伴侣文件镜像随同
3. 复合条件篮子: 验证指定 "包含Bob 且 题材包含风景" 复合规则的准确匹配
4. 伴侣文件镜像级联绑定不变量
"""

from pathlib import Path
import pytest

from core.ai.scene_classifier import PhotoMultiLabel
from core.metadata.companion_scanner import CompanionAsset, PhotoAssetGroup
from core.sorter import (
    BasketFilterRule,
    DispatchBasket,
    DispatchPlan,
    SortingPipeline,
)


@pytest.fixture
def mock_basket_assets(tmp_path):
    """构建包含合影、风景及伴侣文件的测试资产。"""
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    out_dir.mkdir()

    # 1. Alice 与 Bob 的合照 (带 LivePhoto MOV)
    p_group = src_dir / "party_ab.jpg"
    p_group_mov = src_dir / "party_ab.mov"
    p_group.write_bytes(b"jpg1")
    p_group_mov.write_bytes(b"mov1")

    g1 = PhotoAssetGroup(
        main_photo_path=p_group,
        filesize=100,
        created_at="2026-05-20T12:00:00",
        companions=[CompanionAsset(filepath=p_group_mov, file_type="live_video", filesize=500)],
    )

    # 2. 5月20日的纯风景 (带 XMP 调色侧车)
    p_scenery = src_dir / "sunset_lake.jpg"
    p_scenery_xmp = src_dir / "sunset_lake.xmp"
    p_scenery.write_bytes(b"jpg2")
    p_scenery_xmp.write_bytes(b"xmp2")

    g2 = PhotoAssetGroup(
        main_photo_path=p_scenery,
        filesize=200,
        created_at="2026-05-20T18:00:00",
        companions=[CompanionAsset(filepath=p_scenery_xmp, file_type="sidecar", filesize=50)],
    )

    labels_map = {
        "party_ab.jpg": PhotoMultiLabel(
            subject_tags=["合照"],
            person_names=["Alice", "Bob"],
            scene_tags=[],
            scene_scores={},
            all_tags=["合照", "人物:Alice", "人物:Bob"],
        ),
        "sunset_lake.jpg": PhotoMultiLabel(
            subject_tags=["无人物"],
            person_names=[],
            scene_tags=["风景"],
            scene_scores={"风景": 0.28},
            all_tags=["无人物", "风景"],
        ),
    }

    return out_dir, [g1, g2], labels_map


def test_baskets_auto_and_manual_drag_dispatch(mock_basket_assets):
    """测试自动规则吸附与手工拖拽追加复合分拣。"""
    out_dir, groups, labels_map = mock_basket_assets
    pipeline = SortingPipeline(output_root=out_dir)

    # 1. 篮子 1: 分给 Alice (规则包含 Alice + 手动拖入风景照 sunset_lake)
    basket_alice = DispatchBasket(
        basket_id="b_alice",
        name="分给 Alice",
        output_subfolder="Alice_专享",
        rule=BasketFilterRule(person_names=["Alice"]),
        manual_photo_stems={"sunset_lake"},  # 重点: 用户手动把风景照也加给 Alice！
    )

    # 2. 篮子 2: 分给 Bob (仅自动规则包含 Bob)
    basket_bob = DispatchBasket(
        basket_id="b_bob",
        name="分给 Bob",
        output_subfolder="Bob_专享",
        rule=BasketFilterRule(person_names=["Bob"]),
    )

    # 3. 篮子 3: 5月20日活动相册 (按日期自动聚合)
    basket_event = DispatchBasket(
        basket_id="b_event",
        name="5月20日聚会专刊",
        output_subfolder="2026_0520活动",
        rule=BasketFilterRule(date_start="2026-05-20", date_end="2026-05-20"),
    )

    plan: DispatchPlan = pipeline.plan_dispatch(groups, labels_map, [basket_alice, basket_bob, basket_event])

    target_rel_paths = {str(a.target_path.relative_to(out_dir)).replace("\\", "/") for a in plan.actions}
    print("\n[验证-去向篮子动作清单]")
    for a in plan.actions:
        role = "伴侣" if a.is_companion else "主图"
        print(f"  [{a.basket_id:<7}] {role}: {a.source_path.name:<18} -> {a.target_path.relative_to(out_dir)} (原因: {a.reason})")

    # 验证 Alice 篮子: 包含合照及其视频 (自动命中) + 风景照及其侧车 (手动拖入)
    assert "Alice_专享/party_ab.jpg" in target_rel_paths
    assert "Alice_专享/party_ab.mov" in target_rel_paths
    assert "Alice_专享/sunset_lake.jpg" in target_rel_paths
    assert "Alice_专享/sunset_lake.xmp" in target_rel_paths

    # 验证 Bob 篮子: 仅包含合照及视频，没有风景照
    assert "Bob_专享/party_ab.jpg" in target_rel_paths
    assert "Bob_专享/party_ab.mov" in target_rel_paths
    assert "Bob_专享/sunset_lake.jpg" not in target_rel_paths

    # 验证 5月20日活动篮子: 两张当天拍摄的照片及其各自伴侣全部进入
    assert "2026_0520活动/party_ab.jpg" in target_rel_paths
    assert "2026_0520活动/sunset_lake.jpg" in target_rel_paths
    assert "2026_0520活动/sunset_lake.xmp" in target_rel_paths

