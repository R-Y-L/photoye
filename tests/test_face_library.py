#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye M1-F2: 熟人底库与多姿态质心功能自动化与行为测试

测试内容涵盖:
1. 向量归一化、余弦相似度与质心数学计算
2. 正脸与侧脸双质心比对机制 (Multi-View Matching)
3. 在线平滑微调与熟人样本持续学习
4. 离散未知人脸增量 DBSCAN 聚类发现与噪声剔除
5. 注册新聚类人物并固化到 SQLite 底库
"""

import sqlite3
import numpy as np
import pytest

from core.ai.face_library import (
    FaceLibrary,
    compute_centroid,
    compute_similarity,
    normalize_vector,
)
from data.database import (
    add_face,
    add_photo,
    array_to_blob,
    create_person,
    get_connection,
    get_person_by_id,
    init_db,
)


@pytest.fixture
def memory_db():
    """提供纯内存的测试数据库连接并初始化 Schema。"""
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
    yield conn
    conn.close()


def test_vector_math():
    """测试特征向量基础数学计算。"""
    vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    vec_c = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    # 1. 归一化测试
    norm_c = normalize_vector(vec_c)
    assert np.isclose(np.linalg.norm(norm_c), 1.0)

    # 2. 相似度测试
    assert np.isclose(compute_similarity(vec_a, vec_b), 0.0)
    assert np.isclose(compute_similarity(vec_a, vec_a), 1.0)
    assert np.isclose(compute_similarity(vec_a, norm_c), 1.0 / np.sqrt(2.0))

    # 3. 质心测试
    centroid = compute_centroid([vec_a, vec_b])
    assert np.isclose(np.linalg.norm(centroid), 1.0)
    assert np.isclose(centroid[0], centroid[1])


def test_multiview_face_matching(memory_db):
    """测试多姿态（正脸与侧脸）熟人匹配机制。"""
    library = FaceLibrary(db_conn=memory_db, match_threshold=0.65)

    # 构造基础正脸基向量与侧脸基向量 (512维)
    np.random.seed(42)
    base_front = normalize_vector(np.random.randn(512).astype(np.float32))
    # 模拟侧脸向量: 与正脸有重合但偏离较大 (角度差使相似度降低至约 0.55)
    random_noise = normalize_vector(np.random.randn(512).astype(np.float32))
    base_profile = normalize_vector(0.55 * base_front + 0.835 * random_noise)

    # 写入人物 "小明" 同时拥有正脸质心和侧脸质心
    person_id = create_person(
        name="小明",
        centroid_front=base_front,
        centroid_profile=base_profile,
        face_count=2,
        conn=memory_db,
    )
    assert person_id > 0

    # 场景 1: 输入一张小明的新正脸照片 (单位化轻微扰动，相似度 > 0.9)
    noise_front = normalize_vector(np.random.randn(512).astype(np.float32))
    query_front = normalize_vector(base_front + 0.1 * noise_front)
    res1 = library.match_face(query_front)
    assert res1.is_matched is True
    assert res1.person_name == "小明"
    assert res1.matched_pose == "front"
    assert res1.similarity > 0.85
    print(f"\n[验证1-正脸命中] 相似度: {res1.similarity:.4f}, 命中姿态: {res1.matched_pose}")

    # 场景 2: 输入一张小明的大角度侧脸抓拍 (与正脸相似度不足 0.6，但与侧脸质心高达 0.88)
    noise_profile = normalize_vector(np.random.randn(512).astype(np.float32))
    query_profile = normalize_vector(base_profile + 0.1 * noise_profile)
    sim_to_front = compute_similarity(query_profile, base_front)
    assert sim_to_front < 0.65  # 证明如果只有单正脸模板，这张侧脸必漏！

    res2 = library.match_face(query_profile)
    assert res2.is_matched is True
    assert res2.person_name == "小明"
    assert res2.matched_pose == "profile"
    assert res2.similarity > 0.85
    print(f"[验证2-侧脸召回] 对正脸仅: {sim_to_front:.4f} (本应漏人) -> 命中侧脸质心: {res2.similarity:.4f} (成功召回!)")

    # 场景 3: 输入陌生人 (正交随机向量，相似度低)
    stranger_vec = normalize_vector(np.random.randn(512).astype(np.float32))
    res3 = library.match_face(stranger_vec)
    assert res3.is_matched is False
    assert res3.person_id is None
    print(f"[验证3-陌生人排除] 相似度: {res3.similarity:.4f}, 判定结果: 未匹配")


def test_centroid_moving_average_update(memory_db):
    """测试命中熟人后在线质心平滑移动平均微调。"""
    library = FaceLibrary(db_conn=memory_db, match_threshold=0.65, centroid_alpha=0.8)

    c_init = normalize_vector(np.array([1.0] + [0.0] * 511, dtype=np.float32))
    person_id = create_person(name="张三", centroid_front=c_init, conn=memory_db)

    # 模拟一张新照片
    photo_id = add_photo(filepath="/p/img.jpg", filesize=100, sha256="h1", conn=memory_db)
    new_sample = normalize_vector(np.array([0.9, 0.4] + [0.0] * 510, dtype=np.float32))
    face_id = add_face(
        photo_id=photo_id,
        bbox=[0, 0, 10, 10],
        landmarks=[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        embedding=new_sample,
        conn=memory_db,
    )

    # 汇入新样本
    success = library.enroll_face_sample(
        person_id=person_id,
        face_id=face_id,
        embedding=new_sample,
        is_profile=False,
    )
    assert success is True

    # 验证质心已微调: alpha=0.8, c_new = Normalize(0.8 * [1, 0, ...] + 0.2 * [0.9, 0.4, ...])
    updated_person = get_person_by_id(person_id, conn=memory_db)
    c_updated = np.frombuffer(updated_person["centroid_front"], dtype=np.float32)
    expected_dir = 0.8 * c_init + 0.2 * new_sample
    expected_centroid = normalize_vector(expected_dir)

    assert np.allclose(c_updated, expected_centroid, atol=1e-5)
    assert updated_person["face_count"] == 1
    print(f"\n[验证4-质心自演进] 质心已成功融合新样本，累计人脸计数: {updated_person['face_count']}")


def test_unknown_faces_clustering_and_registration(memory_db):
    """测试陌生人脸批次 DBSCAN 聚类、噪声识别与确认命名写入底库。"""
    library = FaceLibrary(db_conn=memory_db, match_threshold=0.65)

    # 构造两个不同人物的密集人脸簇，以及 1 个离群噪声点
    np.random.seed(100)
    center_a = normalize_vector(np.random.randn(512).astype(np.float32))
    center_b = normalize_vector(np.random.randn(512).astype(np.float32))
    outlier = normalize_vector(np.random.randn(512).astype(np.float32))

    # 人物 A 产生 3 张人脸 (同人轻微扰动，相似度 > 0.95)
    faces_a = [
        normalize_vector(center_a + 0.1 * normalize_vector(np.random.randn(512).astype(np.float32)))
        for _ in range(3)
    ]
    # 人物 B 产生 2 张人脸 (同人轻微扰动，相似度 > 0.95)
    faces_b = [
        normalize_vector(center_b + 0.1 * normalize_vector(np.random.randn(512).astype(np.float32)))
        for _ in range(2)
    ]

    raw_items = [
        {"id": 101, "embedding": faces_a[0]},
        {"id": 102, "embedding": faces_a[1]},
        {"id": 103, "embedding": faces_a[2]},
        {"id": 201, "embedding": faces_b[0]},
        {"id": 202, "embedding": faces_b[1]},
        {"id": 999, "embedding": outlier},  # 孤立路人甲
    ]

    cluster_result = library.cluster_unknown_faces(
        unassigned_faces=raw_items,
        eps=0.55,
        min_samples=2,
    )

    clusters = cluster_result["clusters"]
    centroids = cluster_result["cluster_centroids"]
    noise = cluster_result["noise_face_ids"]

    assert len(clusters) == 2  # 成功聚成两个簇
    assert 999 in noise        # 路人甲被准确识别为噪声
    print(f"\n[验证5-DBSCAN聚类] 成功聚合 {len(clusters)} 组新人物簇，离群噪声点: {noise}")

    # 用户界面确认命名第一个聚类簇为 "李雷"
    cluster_0_faces = clusters[0]
    person_id = library.register_new_person_from_cluster(
        name="李雷",
        face_ids=cluster_0_faces,
        centroid=centroids[0],
    )
    assert person_id > 0
    saved = get_person_by_id(person_id, conn=memory_db)
    assert saved["name"] == "李雷"
    assert saved["face_count"] == len(cluster_0_faces)
    print(f"[验证6-确认登记] 用户命名成功，新人物已写入熟人底库: ID={person_id}, 姓名={saved['name']}")
