#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye M1-F3: 自适应人像感知引擎端到端真实照片测试

测试覆盖:
1. 真实全量 16 张照片的检出与误报测试 (合影召回 vs 风景零误报)
2. 水平偏航角 (Yaw) 与垂直俯仰角 (Pitch) 姿态判别准确性
3. 真实 512 维特征抽取并与 FaceLibrary 联动比对
"""

from pathlib import Path
import numpy as np
import pytest

from core.ai.face_engine import FaceEngine
from core.ai.face_library import FaceLibrary, normalize_vector
from data.database import create_person, get_connection, init_db

TEST_IMAGES_DIR = Path(__file__).resolve().parent / "test_images"


@pytest.fixture(scope="module")
def face_engine():
    """全局初始化单例 FaceEngine。"""
    return FaceEngine(model_name="buffalo_sc")


def test_landscape_zero_false_positives(face_engine):
    """验证纯自然风景照片的人脸检出数必须为 0 (零误报)。"""
    landscapes = [
        "landscape_scenery_01.jpg",
        "landscape_scenery_02.jpg",
        "landscape_scenery_03.jpg",
        "landscape_scenery_04.jpg",
    ]
    for fname in landscapes:
        img_p = TEST_IMAGES_DIR / fname
        assert img_p.exists(), f"测试图片缺失: {fname}"
        faces = face_engine.detect_faces(img_p)
        print(f"\n[验证-风景] {fname}: 检出人脸数={len(faces)}")
        assert len(faces) == 0, f"风景图不应检出人脸: {fname}"


def test_group_photos_recall(face_engine):
    """验证真实双人合影与大合照的人脸召回能力。"""
    # 1. 双人照
    for fname in ["group_two_persons_01.jpg", "group_two_persons_02.jpg"]:
        faces = face_engine.detect_faces(TEST_IMAGES_DIR / fname)
        print(f"\n[验证-双人照] {fname}: 检出人脸数={len(faces)}")
        assert len(faces) == 2

    # 2. 11 人大集体照
    eleven_faces = face_engine.detect_faces(TEST_IMAGES_DIR / "group_eleven_persons_01.jpg")
    print(f"\n[验证-集体照] 11人集体照: 检出人脸数={len(eleven_faces)}")
    assert len(eleven_faces) >= 10  # 至少准确召回 10 人以上


def test_multi_pose_detection(face_engine):
    """验证真实多角度照片的偏航角 (Yaw) 与俯仰角 (Pitch) 姿态判别。"""
    # 1. 正脸样本判定为 'front'
    front_faces = face_engine.detect_faces(TEST_IMAGES_DIR / "person_youyu_01_front.jpg")
    assert len(front_faces) == 1
    assert front_faces[0].pose == "front"
    print(f"\n[验证-姿态] 正脸: Yaw={front_faces[0].yaw_ratio:.2f}, Pitch={front_faces[0].pitch_ratio:.2f} -> {front_faces[0].pose}")

    # 2. 大角度侧脸样本判定为 'profile'
    profile_faces = face_engine.detect_faces(TEST_IMAGES_DIR / "person_youyu_05_profile.jpg")
    assert len(profile_faces) == 1
    assert profile_faces[0].pose == "profile"
    print(f"[验证-姿态] 侧脸: Yaw={profile_faces[0].yaw_ratio:.2f}, Pitch={profile_faces[0].pitch_ratio:.2f} -> {profile_faces[0].pose}")

    # 3. 大幅度俯仰角样本判定为 'pitch'
    pitch_faces = face_engine.detect_faces(TEST_IMAGES_DIR / "person_youyu_07_pitch.jpg")
    assert len(pitch_faces) == 1
    assert pitch_faces[0].pose == "pitch"
    print(f"[验证-姿态] 俯仰: Yaw={pitch_faces[0].yaw_ratio:.2f}, Pitch={pitch_faces[0].pitch_ratio:.2f} -> {pitch_faces[0].pose}")


def test_real_photos_library_matching_e2e(face_engine):
    """端到端闭环测试: 提取真实特征送入 FaceLibrary 进行熟人跨角度命中与陌生人排除。"""
    conn = get_connection(":memory:")
    init_db(":memory:")
    cursor = conn.cursor()
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
    conn.commit()

    library = FaceLibrary(db_conn=conn, match_threshold=0.60)

    # 1. 用真实正脸与侧脸注册 YouYu 底库
    front_face = face_engine.detect_faces(TEST_IMAGES_DIR / "person_youyu_01_front.jpg")[0]
    prof_face = face_engine.detect_faces(TEST_IMAGES_DIR / "person_youyu_05_profile.jpg")[0]

    create_person(
        name="YouYu",
        centroid_front=front_face.embedding,
        centroid_profile=prof_face.embedding,
        conn=conn,
    )

    # 2. 盲测 YouYu 的其他侧脸照片
    test_profile = face_engine.detect_faces(TEST_IMAGES_DIR / "person_youyu_04_profile.jpg")[0]
    res_prof = library.match_face(test_profile.embedding)
    assert res_prof.is_matched is True
    assert res_prof.person_name == "YouYu"
    assert res_prof.matched_pose == "profile"
    print(f"\n[验证-端到端] YouYu 侧脸照片成功命中底库侧脸质心: 相似度={res_prof.similarity:.4f}")

    # 3. 盲测陌生合影中的面孔 (非 YouYu)
    group_faces = face_engine.detect_faces(TEST_IMAGES_DIR / "group_two_persons_02.jpg")
    for idx, gf in enumerate(group_faces):
        res_g = library.match_face(gf.embedding)
        assert res_g.is_matched is False
        print(f"[验证-端到端] 陌生照片人脸 #{idx+1} 正确被排除: 相似度={res_g.similarity:.4f}")
