#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye V3.0 数据持久层单元测试"""

import numpy as np
import pytest
from data.database import (
    add_companion_file,
    add_face,
    add_photo,
    array_to_blob,
    assign_face_to_person,
    blob_to_array,
    create_person,
    delete_person,
    get_all_photos,
    get_backup_records_by_sha256,
    get_companion_files_by_photo_id,
    get_connection,
    get_faces_by_photo_id,
    get_person_by_id,
    get_person_by_name,
    get_photo_by_id,
    get_photo_by_path,
    get_photo_by_sha256,
    get_unassigned_faces,
    init_db,
    list_all_persons,
    list_unbacked_photos,
    mark_backup_completed,
    mark_face_as_noise,
    record_backup_status,
    update_person_centroids,
    update_person_name,
    update_photo_embedding,
    update_photo_metadata,
)


@pytest.fixture
def db_conn():
    """提供基于内存的独立测试数据库连接。"""
    conn = get_connection(":memory:")
    init_db(":memory:")
    # 将表结构载入当前连接
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
    yield conn
    conn.close()


def test_blob_conversion():
    """测试特征向量与 BLOB 的序列化及反序列化。"""
    vec = np.random.randn(512).astype(np.float32)
    blob = array_to_blob(vec)
    restored = blob_to_array(blob)
    assert restored is not None
    assert np.allclose(vec, restored, atol=1e-6)


def test_photo_lifecycle(db_conn):
    """测试照片基础信息的增删改查。"""
    vec = np.ones(512, dtype=np.float32)
    photo_id = add_photo(
        filepath="/photos/2026/img_01.jpg",
        filesize=2048576,
        sha256="abc123sha256",
        created_at="2026-05-01T12:00:00",
        category="风景",
        embedding=vec,
        conn=db_conn,
    )
    assert photo_id is not None

    record = get_photo_by_id(photo_id, conn=db_conn)
    assert record["filepath"] == "/photos/2026/img_01.jpg"
    assert record["category"] == "风景"
    assert record["sha256"] == "abc123sha256"

    by_sha = get_photo_by_sha256("abc123sha256", conn=db_conn)
    assert by_sha["id"] == photo_id

    by_path = get_photo_by_path("/photos/2026/img_01.jpg", conn=db_conn)
    assert by_path["id"] == photo_id

    # 更新元数据
    update_photo_metadata(
        photo_id,
        category="单人照",
        latitude=30.25,
        longitude=120.15,
        location_name="杭州西湖",
        status="processed",
        conn=db_conn,
    )
    updated = get_photo_by_id(photo_id, conn=db_conn)
    assert updated["category"] == "单人照"
    assert updated["location_name"] == "杭州西湖"
    assert updated["status"] == "processed"

    # 查询列表
    all_photos = get_all_photos(category="单人照", conn=db_conn)
    assert len(all_photos) == 1


def test_companion_files(db_conn):
    """测试伴侣文件的关联与级联删除。"""
    photo_id = add_photo(
        filepath="/photos/live.jpg",
        filesize=10000,
        sha256="hash_live",
        conn=db_conn,
    )
    comp_id = add_companion_file(
        photo_id=photo_id,
        filepath="/photos/live.mov",
        file_type="live_video",
        filesize=500000,
        conn=db_conn,
    )
    assert comp_id is not None

    companions = get_companion_files_by_photo_id(photo_id, conn=db_conn)
    assert len(companions) == 1
    assert companions[0]["file_type"] == "live_video"

    # 验证级联删除
    cursor = db_conn.cursor()
    cursor.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
    db_conn.commit()

    companions_after = get_companion_files_by_photo_id(photo_id, conn=db_conn)
    assert len(companions_after) == 0


def test_person_and_face_centroids(db_conn):
    """测试多姿态质心持久化与人脸归属关联。"""
    front_vec = np.full(512, 0.5, dtype=np.float32)
    profile_vec = np.full(512, -0.5, dtype=np.float32)

    person_id = create_person(
        name="爱丽丝",
        centroid_front=front_vec,
        centroid_profile=profile_vec,
        face_count=2,
        conn=db_conn,
    )
    assert person_id > 0

    person = get_person_by_id(person_id, conn=db_conn)
    assert person["name"] == "爱丽丝"
    saved_front = blob_to_array(person["centroid_front"])
    assert np.allclose(saved_front, front_vec)

    # 添加照片与人脸
    photo_id = add_photo(
        filepath="/photos/alice.jpg",
        filesize=1000,
        sha256="hash_alice",
        conn=db_conn,
    )
    face_emb = np.full(512, 0.6, dtype=np.float32)
    face_id = add_face(
        photo_id=photo_id,
        bbox=[10, 20, 100, 120],
        landmarks=[[30, 40], [70, 40], [50, 60], [40, 80], [60, 80]],
        embedding=face_emb,
        confidence=0.98,
        yaw_angle=12.5,
        conn=db_conn,
    )
    assert face_id > 0

    # 查未关联人脸
    unassigned = get_unassigned_faces(conn=db_conn)
    assert len(unassigned) == 1

    # 关联到人物
    assign_face_to_person(face_id, person_id, conn=db_conn)
    unassigned_after = get_unassigned_faces(conn=db_conn)
    assert len(unassigned_after) == 0

    # 更新质心与名字
    new_front = np.full(512, 0.55, dtype=np.float32)
    update_person_centroids(person_id, centroid_front=new_front, additional_face_count=1, conn=db_conn)
    update_person_name(person_id, "爱丽丝 (工作)", conn=db_conn)

    updated_person = get_person_by_id(person_id, conn=db_conn)
    assert updated_person["name"] == "爱丽丝 (工作)"
    assert updated_person["face_count"] == 3


def test_backup_ledger(db_conn):
    """测试云备份台账的记录、完成标记与漏传对账。"""
    photo_sha = "sha256_for_cloud_test"
    add_photo(
        filepath="/photos/cloud_test.jpg",
        filesize=5000,
        sha256=photo_sha,
        conn=db_conn,
    )

    # 初始状态未上传任何平台，应该出现在未备份列表
    unbacked = list_unbacked_photos(conn=db_conn)
    assert any(p["sha256"] == photo_sha for p in unbacked)

    # 登记为 QQ 相册但未完成
    record_backup_status(
        photo_sha256=photo_sha,
        target_platform="QQ相册",
        album_name="2026云南",
        is_uploaded=0,
        conn=db_conn,
    )
    unbacked_target = list_unbacked_photos(
        target_platform="QQ相册",
        album_name="2026云南",
        conn=db_conn,
    )
    assert any(p["sha256"] == photo_sha for p in unbacked_target)

    # 标记完成
    mark_backup_completed(
        photo_sha256=photo_sha,
        target_platform="QQ相册",
        album_name="2026云南",
        uploaded_at="2026-05-02T10:00:00",
        conn=db_conn,
    )

    # 验证该平台下不再属于未备份
    unbacked_after = list_unbacked_photos(
        target_platform="QQ相册",
        album_name="2026云南",
        conn=db_conn,
    )
    assert not any(p["sha256"] == photo_sha for p in unbacked_after)

    # 验证全局已上传
    global_unbacked = list_unbacked_photos(conn=db_conn)
    assert not any(p["sha256"] == photo_sha for p in global_unbacked)
