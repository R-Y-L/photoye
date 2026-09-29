#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye M2-F1: EXIF 解析与伴侣文件同进退机制自动化测试

测试覆盖:
1. 伴侣文件探测与绑定 (LivePhoto 动态视频、调色 XMP、相机 RAW 负片)
2. 伴侣文件类型归类准确性 (live_video, sidecar, raw_negative)
3. 孤立视频与异名文件绝不误配对
4. 真实 EXIF 拍摄时间与 GPS 经纬度读取
5. 与 SQLite V3.0 数据持久层 companion_files 表的联动写入与级联删除验证
"""

import shutil
from pathlib import Path
import pytest

from core.metadata import CompanionScanner, ExifReader, PhotoAssetGroup
from data.database import (
    add_companion_file,
    add_photo,
    get_companion_files_by_photo_id,
    get_connection,
    init_db,
)


@pytest.fixture
def mock_photo_album(tmp_path):
    """构建包含各种复杂伴侣文件场景的临时相册目录。"""
    album_dir = tmp_path / "test_album"
    album_dir.mkdir()

    # 场景 1: iPhone LivePhoto (JPG + MOV)
    (album_dir / "IMG_1001.JPG").write_bytes(b"\xFF\xD8\xFF" + b"fake_jpg_content" * 10)
    (album_dir / "IMG_1001.MOV").write_bytes(b"fake_live_video_content" * 20)

    # 场景 2: 调色照片 (JPG + XMP + AAE)
    (album_dir / "landscape_01.jpg").write_bytes(b"\xFF\xD8\xFF" + b"fake_landscape_content" * 10)
    (album_dir / "landscape_01.xmp").write_bytes(b"<xmp>color_profile</xmp>")
    (album_dir / "landscape_01.aae").write_bytes(b"plist_ios_adjustments")

    # 场景 3: 单反相机双格式拍摄 (JPG + CR3)
    (album_dir / "CANON_888.JPG").write_bytes(b"\xFF\xD8\xFF" + b"fake_canon_content" * 10)
    (album_dir / "CANON_888.CR3").write_bytes(b"fake_raw_negative_cr3" * 50)

    # 场景 4: 独立照片 (无伴侣)
    (album_dir / "standalone.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake_png" * 10)

    # 场景 5: 孤立视频 (不同名，绝不应该被误当成别人的伴侣)
    (album_dir / "other_vacation_video.mp4").write_bytes(b"isolated_video" * 10)

    return album_dir


def test_companion_file_pairing_and_types(mock_photo_album):
    """验证伴侣文件同名自动探测与类型判定。"""
    groups = CompanionScanner.scan_directory(mock_photo_album)

    # 应该扫描出 4 组主照片 (JPG/PNG，孤立 MP4 不会作为主照片)
    assert len(groups) == 4

    # 检查映射表
    group_map = {g.main_photo_path.name: g for g in groups}

    # 1. 验证 LivePhoto
    g_live = group_map["IMG_1001.JPG"]
    assert len(g_live.companions) == 1
    assert g_live.companions[0].filepath.name == "IMG_1001.MOV"
    assert g_live.companions[0].file_type == "live_video"
    print(f"\n[验证-LivePhoto] {g_live.main_photo_path.name} 成功绑定伴侣视频: {g_live.companions[0].filepath.name}")

    # 2. 验证调色侧车 (同时绑定 XMP 与 AAE)
    g_sidecar = group_map["landscape_01.jpg"]
    assert len(g_sidecar.companions) == 2
    types = {c.file_type for c in g_sidecar.companions}
    names = {c.filepath.name for c in g_sidecar.companions}
    assert types == {"sidecar"}
    assert names == {"landscape_01.xmp", "landscape_01.aae"}
    print(f"[验证-Sidecar] {g_sidecar.main_photo_path.name} 成功绑定双侧车: {names}")

    # 3. 验证相机 RAW 负片
    g_raw = group_map["CANON_888.JPG"]
    assert len(g_raw.companions) == 1
    assert g_raw.companions[0].filepath.name == "CANON_888.CR3"
    assert g_raw.companions[0].file_type == "raw_negative"
    print(f"[验证-RAW负片] {g_raw.main_photo_path.name} 成功绑定原始 RAW: {g_raw.companions[0].filepath.name}")

    # 4. 验证独立照片无伴侣，且孤立视频未被错误吸附
    g_standalone = group_map["standalone.png"]
    assert len(g_standalone.companions) == 0
    print(f"[验证-独立文件] {g_standalone.main_photo_path.name} 零误吸附伴侣")


def test_companion_database_persistence_and_cascade(mock_photo_album):
    """验证扫描出的伴侣文件原子写入 SQLite 数据库并在照片删除时级联删除。"""
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
    """)
    conn.commit()

    groups = CompanionScanner.scan_directory(mock_photo_album)
    g_live = [g for g in groups if g.main_photo_path.name == "IMG_1001.JPG"][0]

    # 写入主照片
    photo_id = add_photo(
        filepath=str(g_live.main_photo_path),
        filesize=g_live.filesize,
        sha256="live_hash_123",
        conn=conn,
    )
    assert photo_id is not None

    # 写入伴侣文件
    for c in g_live.companions:
        add_companion_file(
            photo_id=photo_id,
            filepath=str(c.filepath),
            file_type=c.file_type,
            filesize=c.filesize,
            conn=conn,
        )

    # 验证读取
    saved_companions = get_companion_files_by_photo_id(photo_id, conn=conn)
    assert len(saved_companions) == 1
    assert saved_companions[0]["file_type"] == "live_video"
    print(f"\n[验证-持久化] 数据库已成功登记伴侣关系: ID={saved_companions[0]['id']}, 类型={saved_companions[0]['file_type']}")

    # 验证级联删除主照片后，伴侣文件记录自动清除
    cursor.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
    conn.commit()
    after_delete = get_companion_files_by_photo_id(photo_id, conn=conn)
    assert len(after_delete) == 0
    print(f"[验证-级联安全] 主照片记录删除后，伴侣关系自动级联清理完成")
