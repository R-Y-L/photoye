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
from core.sorter.pipeline import DispatchAction
from core.sorter.transaction import undo_transaction


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


def make_plan(env):
    _, root, groups, labels = env
    basket = DispatchBasket("alice", "Alice", "人物/Alice", rule=BasketFilterRule(person_names=["Alice"]))
    return SortingPipeline(root).plan_dispatch(groups, labels, [basket])


@pytest.mark.parametrize("hardlink", [True, False])
def test_companion_collision_reserves_whole_group(mock_storage_env, hardlink):
    plan = make_plan(mock_storage_env)
    old = plan.actions[0].target_path.with_name("IMG_001_1.MOV")
    old.write_bytes(b"existing companion")
    report = FileOperationsExecutor(hardlink).execute_plan(plan)
    assert report.failed_count == 0 and report.successful_count == 2
    assert {r.target_path.stem for r in report.results} == {"IMG_001_2"}
    assert old.read_bytes() == b"existing companion"
    assert report.manifest_path.exists()
    assert undo_transaction(report.manifest_path).removed_count == 2
    assert old.read_bytes() == b"existing companion"


def test_missing_companion_leaves_no_partial_group(mock_storage_env):
    plan = make_plan(mock_storage_env)
    plan.actions[1].source_path.unlink()
    report = FileOperationsExecutor().execute_plan(plan)
    assert report.failed_count == 2 and report.successful_count == 0
    assert not plan.actions[0].target_path.with_stem("IMG_001_1").exists()
    assert all("源文件不存在" in r.error_message for r in report.results)


def test_second_create_failure_rolls_back_only_new_files(mock_storage_env, monkeypatch):
    plan = make_plan(mock_storage_env)
    executor = FileOperationsExecutor(False)
    create = executor._create_file

    def fail_companion(src, dst):
        if src.suffix == ".MOV":
            raise PermissionError("companion denied")
        return create(src, dst)

    monkeypatch.setattr(executor, "_create_file", fail_companion)
    report = executor.execute_plan(plan)
    assert report.failed_count == 2 and report.successful_count == 0
    assert not plan.actions[0].target_path.with_stem("IMG_001_1").exists()
    assert plan.actions[0].target_path.read_bytes() == b"pre_existing_different_content_999"
    assert undo_transaction(report.manifest_path).completed


def test_exclusive_copy_does_not_overwrite_racing_file(mock_storage_env, monkeypatch):
    plan = make_plan(mock_storage_env)
    executor = FileOperationsExecutor(False)
    create = executor._create_file
    raced = []

    def race(src, dst):
        dst.write_bytes(b"another application")
        raced.append(dst)
        return create(src, dst)

    monkeypatch.setattr(executor, "_create_file", race)
    report = executor.execute_plan(plan)
    assert report.failed_count == 2
    assert raced[0].read_bytes() == b"another application"


def test_cross_volume_link_failure_falls_back_to_copy(mock_storage_env, monkeypatch):
    import errno
    monkeypatch.setattr(os, "link", lambda *args: (_ for _ in ()).throw(OSError(errno.EXDEV, "cross-device")))
    report = FileOperationsExecutor().execute_plan(make_plan(mock_storage_env))
    assert report.copy_count == 2 and report.failed_count == 0


@pytest.mark.parametrize("subfolder", ["../escaped", "a/../../escaped", "C:/escaped", "/escaped", "\\\\server\\share", "D:relative"])
def test_pipeline_rejects_escaping_paths(mock_storage_env, subfolder):
    _, root, groups, labels = mock_storage_env
    basket = DispatchBasket("escape", "escape", subfolder, rule=BasketFilterRule())
    with pytest.raises(ValueError):
        SortingPipeline(root).plan_dispatch(groups, labels, [basket])


def test_executor_rejects_forged_outside_plan(mock_storage_env):
    src, root, groups, _ = mock_storage_env
    outside = src.parent / "escaped.JPG"
    report = FileOperationsExecutor().execute_plan(DispatchPlan(root, actions=[DispatchAction(groups[0].main_photo_path, outside, "escape")]))
    assert report.failed_count == 1 and report.manifest_path is None
    assert not outside.exists() and groups[0].main_photo_path.exists()


def test_mkdir_failure_is_reported(mock_storage_env, monkeypatch):
    from pathlib import Path
    plan = make_plan(mock_storage_env)
    mkdir = Path.mkdir

    def deny(path, *args, **kwargs):
        if path == plan.output_root:
            raise PermissionError("output denied")
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny)
    report = FileOperationsExecutor().execute_plan(plan)
    assert report.failed_count == 2 and "output denied" in report.errors[0]


def test_journal_record_failure_rolls_back_and_stops(mock_storage_env, monkeypatch):
    from core.sorter.transaction import TransactionJournal
    plan = make_plan(mock_storage_env)
    monkeypatch.setattr(TransactionJournal, "record_created", lambda *args: (_ for _ in ()).throw(OSError("audit full")))
    report = FileOperationsExecutor().execute_plan(plan)
    assert report.failed_count == 2 and report.errors
    assert not plan.actions[0].target_path.with_stem("IMG_001_1").exists()


def test_directory_audit_failure_cleans_new_empty_directories(mock_storage_env, monkeypatch):
    from core.sorter.transaction import TransactionJournal
    plan = make_plan(mock_storage_env)
    for action in plan.actions:
        action.target_path = plan.output_root / "new" / "nested" / action.target_path.name
    monkeypatch.setattr(TransactionJournal, "record_directories", lambda *args: (_ for _ in ()).throw(OSError("directory audit full")))
    report = FileOperationsExecutor().execute_plan(plan)
    assert report.failed_count == 2 and report.errors
    assert not (plan.output_root / "new").exists()
    assert undo_transaction(report.manifest_path).completed


def test_audit_failure_counts_unexecuted_same_source_distinct_targets(mock_storage_env, monkeypatch):
    from core.sorter.transaction import TransactionJournal
    plan = make_plan(mock_storage_env)
    source = plan.actions[0].source_path
    plan.actions = [DispatchAction(source, plan.output_root / name, "b") for name in ("one.jpg", "two.jpg")]
    monkeypatch.setattr(TransactionJournal, "record_created", lambda *args: (_ for _ in ()).throw(OSError("audit full")))
    report = FileOperationsExecutor().execute_plan(plan)
    assert report.total_actions == report.failed_count == len(report.results) == 2
    assert report.successful_count == 0 and "未执行" in report.results[1].error_message
    assert not (plan.output_root / "one.jpg").exists() and not (plan.output_root / "two.jpg").exists()


def test_rollback_keeps_externally_changed_new_file(mock_storage_env, monkeypatch):
    plan = make_plan(mock_storage_env)
    executor = FileOperationsExecutor(False)
    create = executor._create_file
    created = []

    def edit_then_fail(source, target):
        if created:
            created[0].write_bytes(b"external edit")
            raise PermissionError("companion failure")
        result = create(source, target)
        created.append(target)
        return result

    monkeypatch.setattr(executor, "_create_file", edit_then_fail)
    report = executor.execute_plan(plan)
    assert report.failed_count == 2 and "回退失败" in report.results[0].error_message
    assert created[0].read_bytes() == b"external edit"
    assert not undo_transaction(report.manifest_path).completed


def test_date_naming_groups_use_distinct_sources(mock_storage_env):
    src, root, groups, labels = mock_storage_env
    other = src / "OTHER.JPG"
    other.write_bytes(b"other photo")
    groups.append(PhotoAssetGroup(other, other.stat().st_size, groups[0].created_at))
    basket = DispatchBasket("dates", "dates", "dates", rule=BasketFilterRule(), naming_pattern="date")
    plan = SortingPipeline(root).plan_dispatch(groups, labels, [basket])
    report = FileOperationsExecutor().execute_plan(plan)
    assert report.successful_count == 3 and report.failed_count == 0
    main_targets = [r.target_path for r in report.results if not r.is_companion]
    assert len(set(main_targets)) == 2
    assert {p.read_bytes() for p in main_targets} == {groups[0].main_photo_path.read_bytes(), b"other photo"}


def test_failed_group_does_not_block_later_group_with_same_date(mock_storage_env, monkeypatch):
    src, root, groups, labels = mock_storage_env
    other = src / "OTHER.JPG"
    other.write_bytes(b"second photo")
    groups.append(PhotoAssetGroup(other, other.stat().st_size, groups[0].created_at))
    basket = DispatchBasket("dates", "dates", "new/date", rule=BasketFilterRule(), naming_pattern="date")
    plan = SortingPipeline(root).plan_dispatch(groups, labels, [basket])
    executor = FileOperationsExecutor(False)
    create = executor._create_file

    def fail_video(source, target):
        if source.suffix == ".MOV":
            raise PermissionError("video locked")
        return create(source, target)

    monkeypatch.setattr(executor, "_create_file", fail_video)
    report = executor.execute_plan(plan)
    assert report.failed_count == 2 and report.successful_count == 1 and not report.errors
    success = next(r for r in report.results if r.success)
    assert success.target_path.read_bytes() == b"second photo"
    assert success.target_path.stem.endswith("_1")
    undo = undo_transaction(report.manifest_path)
    assert undo.completed and undo.removed_count == 1 and undo.missing_count == 1
    assert not (root / "new").exists()
