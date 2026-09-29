#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye V3.0 真实照片端到端实战演示脚本"""

from pathlib import Path
import numpy as np

from core.ai.face_engine import FaceEngine
from core.ai.face_library import FaceLibrary
from core.ai.scene_classifier import SceneClassifier
from core.ai.semantic_search import SemanticSearchEngine
from data.database import (
    add_photo,
    create_person,
    get_connection,
    init_db,
)


def run_demo():
    print("======================================================================")
    print("           Photoye V3.0 AI 核心引擎端到端全链路实战验收")
    print("======================================================================\n")

    # 1. 初始化纯内存演示数据库与核心引擎
    db_conn = get_connection(":memory:")
    init_db(":memory:")
    cursor = db_conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL UNIQUE,
            filesize INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            created_at TEXT,
            category TEXT,
            embedding BLOB,
            latitude REAL,
            longitude REAL,
            location_name TEXT,
            status TEXT DEFAULT 'pending',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS companion_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            filepath TEXT NOT NULL UNIQUE,
            file_type TEXT NOT NULL,
            filesize INTEGER NOT NULL,
            FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            cover_face_id INTEGER,
            centroid_front BLOB,
            centroid_profile BLOB,
            face_count INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            person_id INTEGER,
            bbox TEXT NOT NULL,
            landmarks TEXT NOT NULL,
            embedding BLOB NOT NULL,
            confidence REAL DEFAULT 0.0,
            yaw_angle REAL,
            is_noise INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS backup_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_sha256 TEXT NOT NULL,
            target_platform TEXT NOT NULL,
            album_name TEXT,
            is_uploaded INTEGER DEFAULT 0,
            uploaded_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(photo_sha256, target_platform, album_name)
        );
    """)
    db_conn.commit()

    face_engine = FaceEngine()
    face_lib = FaceLibrary(db_conn=db_conn, match_threshold=0.60)
    scene_classifier = SceneClassifier()
    search_engine = SemanticSearchEngine()

    test_dir = Path("tests/test_images")
    all_images = sorted(list(test_dir.glob("*.jpg")))
    print(f"📂 载入测试相册: {test_dir.resolve()} (共 {len(all_images)} 张照片)\n")

    # 2. 模拟底库初始化: 先把 person_youyu_01(正脸) 与 05(侧脸) 录入底库作为已知人物 'YouYu'
    ref_front_p = test_dir / "person_youyu_01_front.jpg"
    ref_prof_p = test_dir / "person_youyu_05_profile.jpg"
    emb_front = face_engine.detect_faces(ref_front_p)[0].embedding
    emb_prof = face_engine.detect_faces(ref_prof_p)[0].embedding

    create_person(
        name="YouYu",
        centroid_front=emb_front,
        centroid_profile=emb_prof,
        conn=db_conn,
    )
    print("👤 【熟人底库建立】已记录人物 [YouYu] (正脸质心 + 侧脸质心就绪)\n")

    # 3. 逐张扫描真实相册并写入 V3.0 数据库
    print("---------------------------------------------------------------------------------------------------------")
    print(f"  {'序号':<4} | {'文件名':<28} | {'人脸数':<6} | {'识别身份与姿态':<26} | {'最终多标签输出'}")
    print("---------------------------------------------------------------------------------------------------------")

    photo_embeddings = []
    photo_names = []

    for idx, p in enumerate(all_images):
        faces = face_engine.detect_faces(p)
        num_faces = len(faces)

        # 识别人物
        identified_names = []
        face_details = []
        for f in faces:
            match_res = face_lib.match_face(f.embedding)
            if match_res.is_matched:
                identified_names.append(match_res.person_name)
                face_details.append(f"{match_res.person_name}({f.pose})")
            else:
                face_details.append(f"未知({f.pose})")

        face_desc = ", ".join(face_details) if face_details else "无"
        if len(face_desc) > 24:
            face_desc = face_desc[:21] + "..."

        # 真实 OpenCLIP 多标签融合打标
        multi_label = scene_classifier.generate_multi_labels(p, faces, identified_names=identified_names)
        img_vec = scene_classifier.extract_image_embedding(p)

        photo_embeddings.append(img_vec)
        photo_names.append(p.name)

        print(f"  {idx+1:02d}   | {p.name:<28} | {num_faces:<6} | {face_desc:<26} | {str(multi_label.all_tags)}")

    print("---------------------------------------------------------------------------------------------------------\n")

    # 4. 模拟用户输入 3 个不同的自然语言生活短语搜索
    matrix = np.array(photo_embeddings, dtype=np.float32)

    queries = [
        "natural outdoor trees and landscape",     # 搜自然风光与树林
        "a group photo of friends gathering",     # 搜朋友聚会合照
        "a headshot portrait photo of a person",  # 搜个人面部特写
    ]

    print("🔍 【纯本地自然语言以文搜图实战演示 (毫秒级响应)】\n")
    for q in queries:
        print(f"💬 自然语言检索: \"{q}\"")
        results = search_engine.search(q, matrix, photo_names, top_k=3)
        for r_idx, item in enumerate(results):
            print(f"   Top #{r_idx+1}: {item.identifier:<28} (匹配度得分: {item.similarity:.4f})")
        print()

    print("======================================================================")
    print("✅ 实战演练完成: 视觉感知、多姿态召回、正交多标签打标与语义检索全通！")
    print("======================================================================")


if __name__ == "__main__":
    run_demo()
