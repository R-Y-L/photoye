#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye V3.0 智能分拣工坊交互式实战演练与检测工具"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ai.face_engine import FaceEngine
from core.ai.face_library import FaceLibrary
from core.ai.scene_classifier import SceneClassifier
from core.metadata.companion_scanner import CompanionScanner
from core.sorter.pipeline import BasketFilterRule, DispatchBasket, DispatchPlan, SortingPipeline
from data.database import create_person, get_connection, init_db


def run_sorting_workshop_demo():
    print("======================================================================")
    print("       Photoye V3.0 智能整理工坊 - 去向篮子与多路分流实战验收")
    print("======================================================================\n")

    test_dir = Path("tests/test_images").resolve()
    output_dir = Path("tests/mock_output_sorted").resolve()
    
    # 1. 扫描相册目录并自动绑定伴侣文件
    print(f"📂 [Step 1] 扫描原始相册目录: {test_dir.resolve()}")
    asset_groups = CompanionScanner.scan_directory(test_dir)
    print(f"   成功读入 {len(asset_groups)} 组相册资产 (包含主照片及伴侣视频/侧车/RAW)\n")

    # 2. AI 引擎分析多标签
    print("🧠 [Step 2] AI 多模态多维度打标与人物底库匹配...")
    db_conn = get_connection(":memory:")
    init_db(":memory:")
    cursor = db_conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS photos (id INTEGER PRIMARY KEY, filepath TEXT, filesize INTEGER, sha256 TEXT, created_at TEXT, category TEXT, embedding BLOB, latitude REAL, longitude REAL, location_name TEXT, status TEXT, added_at TIMESTAMP);
        CREATE TABLE IF NOT EXISTS companion_files (id INTEGER PRIMARY KEY, photo_id INTEGER, filepath TEXT, file_type TEXT, filesize INTEGER);
        CREATE TABLE IF NOT EXISTS persons (id INTEGER PRIMARY KEY, name TEXT UNIQUE, cover_face_id INTEGER, centroid_front BLOB, centroid_profile BLOB, face_count INTEGER, updated_at TIMESTAMP);
        CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY, photo_id INTEGER, person_id INTEGER, bbox TEXT, landmarks TEXT, embedding BLOB, confidence REAL, yaw_angle REAL, is_noise INTEGER, created_at TIMESTAMP);
        CREATE TABLE IF NOT EXISTS backup_ledger (id INTEGER PRIMARY KEY, photo_sha256 TEXT, target_platform TEXT, album_name TEXT, is_uploaded INTEGER, uploaded_at TIMESTAMP, created_at TIMESTAMP);
    """)
    db_conn.commit()

    face_engine = FaceEngine()
    face_lib = FaceLibrary(db_conn=db_conn, match_threshold=0.60)
    scene_classifier = SceneClassifier()

    # 初始化已知人物 YouYu
    f_front = face_engine.detect_faces(test_dir / "person_youyu_01_front.jpg")[0]
    f_prof = face_engine.detect_faces(test_dir / "person_youyu_05_profile.jpg")[0]
    create_person("YouYu", centroid_front=f_front.embedding, centroid_profile=f_prof.embedding, conn=db_conn)

    labels_map = {}
    for g in asset_groups:
        faces = face_engine.detect_faces(g.main_photo_path)
        names = []
        for f in faces:
            m = face_lib.match_face(f.embedding)
            if m.is_matched:
                names.append(m.person_name)
        multi = scene_classifier.generate_multi_labels(g.main_photo_path, faces, identified_names=names)
        labels_map[g.main_photo_path.name] = multi

    print("   AI 多标签识别完成！\n")

    # 3. 建立右侧去向篮子
    pipeline = SortingPipeline(output_root=output_dir)
    print("🧺 [Step 3] 构建右侧去向篮子 (模拟用户配置与拖拽追加):")

    # 篮子 1: 分给 YouYu (包含 YouYu 人像 + 手工拖入 1 张风景照片 landscape_scenery_01)
    basket_youyu = DispatchBasket(
        basket_id="basket_youyu",
        name="分给 YouYu 的合照与专属包",
        output_subfolder="发给_YouYu",
        rule=BasketFilterRule(person_names=["YouYu"]),
        manual_photo_stems={"landscape_scenery_01"},  # 用户手动拖入指定风景！
    )

    # 篮子 2: 纯风景集锦 (仅自动吸附风景题材)
    basket_scenery = DispatchBasket(
        basket_id="basket_scenery",
        name="精选大自然风景",
        output_subfolder="题材_风景",
        rule=BasketFilterRule(scene_tags=["风景"]),
    )

    # 篮子 3: 聚会合影包 (包含合照)
    basket_group = DispatchBasket(
        basket_id="basket_group",
        name="朋友聚会与集体合影",
        output_subfolder="聚会_合影",
        rule=BasketFilterRule(scene_tags=[], person_names=[]),  # 稍后加入
    )
    # 为合照特别设置
    basket_group.manual_photo_stems = {
        "group_eleven_persons_01",
        "group_two_persons_01",
        "group_two_persons_02",
    }

    baskets = [basket_youyu, basket_scenery, basket_group]
    for b in baskets:
        print(f"   • [{b.name}] -> 目标目录: {b.output_subfolder}")

    # 4. 运行管道规划分发
    print("\n⚙️ [Step 4] 执行多维分拣路由计算...")
    plan: DispatchPlan = pipeline.plan_dispatch(asset_groups, labels_map, baskets)

    print("---------------------------------------------------------------------------------------------")
    print(f"  {'篮子名称':<22} | {'文件类型':<6} | {'源文件':<28} -> {'分发目标相对路径':<32} | {'来源原因'}")
    print("---------------------------------------------------------------------------------------------")

    for a in plan.actions:
        role = "伴侣" if a.is_companion else "主图"
        # 找篮子名称
        b_name = [b.name for b in baskets if b.basket_id == a.basket_id][0]
        if len(b_name) > 18:
            b_name = b_name[:16] + ".."
        target_rel = str(a.target_path.relative_to(output_dir))
        if len(target_rel) > 30:
            target_rel = target_rel[:28] + ".."
        reason_desc = "手动拖入" if a.reason == "manual_drag" else "自动规则"
        print(f"  {b_name:<20} | {role:<6} | {a.source_path.name:<28} -> {target_rel:<32} | {reason_desc}")

    print("---------------------------------------------------------------------------------------------\n")
    print("📊 各去向篮子分发数量统计:")
    for b in baskets:
        print(f"   🧺 [{b.name}]: 汇集了 {plan.basket_item_counts.get(b.basket_id, 0)} 组相册资产")

    print("\n======================================================================")
    print("✅ 智能分拣工坊演练完成: 自动规则、手动拖入追加与伴侣级联全部打通！")
    print("======================================================================")


if __name__ == "__main__":
    run_sorting_workshop_demo()
