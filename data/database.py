#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 数据持久层模块 (V3.0)

负责与 SQLite 数据库进行交互，提供标准化的数据读写接口。
支持照片资产索引、伴侣文件关联、多姿态人脸质心底库与云备份台账。
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DB_NAME = "photoye_v3.db"


def get_db_path(custom_path: Optional[str] = None) -> str:
    """获取数据库文件路径。"""
    if custom_path:
        return custom_path
    return os.path.join(os.getcwd(), DB_NAME)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """获取 SQLite 数据库连接并开启外键支持。"""
    target_path = get_db_path(db_path)
    conn = sqlite3.connect(target_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def array_to_blob(array: Optional[np.ndarray]) -> Optional[bytes]:
    """将 numpy 浮点向量转换为二进制 BLOB 存储。"""
    if array is None:
        return None
    return np.ascontiguousarray(array, dtype=np.float32).tobytes()


def blob_to_array(blob: Optional[bytes], dim: int = 512) -> Optional[np.ndarray]:
    """将二进制 BLOB 还原为指定维度的 numpy 浮点向量。"""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


def init_db(db_path: Optional[str] = None) -> None:
    """初始化数据库并创建 V3.0 所需的数据表和索引。"""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()

        # 1. 照片基础资产表
        cursor.execute("""
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
            )
        """)

        # 2. 伴侣文件关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companion_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                filepath TEXT NOT NULL UNIQUE,
                file_type TEXT NOT NULL,
                filesize INTEGER NOT NULL,
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
            )
        """)

        # 3. 熟人底库表 (支持多姿态质心持久化)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                cover_face_id INTEGER,
                centroid_front BLOB,
                centroid_profile BLOB,
                face_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 4. 人脸特征表
        cursor.execute("""
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
            )
        """)

        # 5. 云备份台账表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backup_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_sha256 TEXT NOT NULL,
                target_platform TEXT NOT NULL,
                album_name TEXT,
                is_uploaded INTEGER DEFAULT 0,
                uploaded_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(photo_sha256, target_platform, album_name)
            )
        """)

        # 建立高效查询索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_sha256 ON photos(sha256)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_category ON photos(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_created_at ON photos(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_companion_photo_id ON companion_files(photo_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_sha256 ON backup_ledger(photo_sha256)")

        conn.commit()
    finally:
        conn.close()


# ==================== 照片资产操作 ====================

def add_photo(
    filepath: str,
    filesize: int,
    sha256: str,
    created_at: Optional[str] = None,
    category: Optional[str] = None,
    embedding: Optional[np.ndarray] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
    status: str = "pending",
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[int]:
    """添加单张照片记录到数据库。若文件已存在则返回已有记录 ID。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO photos (
                filepath, filesize, sha256, created_at, category,
                embedding, latitude, longitude, location_name, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(filepath) DO UPDATE SET
                filesize = excluded.filesize,
                sha256 = excluded.sha256,
                created_at = COALESCE(excluded.created_at, photos.created_at),
                category = COALESCE(excluded.category, photos.category),
                embedding = COALESCE(excluded.embedding, photos.embedding),
                latitude = COALESCE(excluded.latitude, photos.latitude),
                longitude = COALESCE(excluded.longitude, photos.longitude),
                location_name = COALESCE(excluded.location_name, photos.location_name)
            """,
            (
                filepath,
                filesize,
                sha256,
                created_at,
                category,
                array_to_blob(embedding),
                latitude,
                longitude,
                location_name,
                status,
            ),
        )
        db.commit()
        if cursor.lastrowid:
            return cursor.lastrowid
        cursor.execute("SELECT id FROM photos WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        return row["id"] if row else None
    finally:
        if should_close:
            db.close()


def get_photo_by_id(photo_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    """根据 ID 获取照片记录。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        if should_close:
            db.close()


def get_photo_by_sha256(sha256: str, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    """根据 SHA-256 哈希值获取照片记录。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM photos WHERE sha256 = ?", (sha256,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        if should_close:
            db.close()


def get_photo_by_path(filepath: str, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    """根据文件绝对路径获取照片记录。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM photos WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        if should_close:
            db.close()


def update_photo_metadata(
    photo_id: int,
    category: Optional[str] = None,
    created_at: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
    status: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """更新照片的元数据与状态。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        fields = []
        values = []
        if category is not None:
            fields.append("category = ?")
            values.append(category)
        if created_at is not None:
            fields.append("created_at = ?")
            values.append(created_at)
        if latitude is not None:
            fields.append("latitude = ?")
            values.append(latitude)
        if longitude is not None:
            fields.append("longitude = ?")
            values.append(longitude)
        if location_name is not None:
            fields.append("location_name = ?")
            values.append(location_name)
        if status is not None:
            fields.append("status = ?")
            values.append(status)

        if not fields:
            return True

        values.append(photo_id)
        cursor.execute(f"UPDATE photos SET {', '.join(fields)} WHERE id = ?", values)
        db.commit()
        return cursor.rowcount > 0
    finally:
        if should_close:
            db.close()


def update_photo_embedding(
    photo_id: int,
    embedding: np.ndarray,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """更新照片的 OpenCLIP 特征向量。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute(
            "UPDATE photos SET embedding = ? WHERE id = ?",
            (array_to_blob(embedding), photo_id),
        )
        db.commit()
        return cursor.rowcount > 0
    finally:
        if should_close:
            db.close()


def get_all_photos(
    category: Optional[str] = None,
    status: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """按条件查询所有照片记录。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        conditions = []
        params = []
        if category:
            conditions.append("category = ?")
            params.append(category)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor.execute(
            f"SELECT * FROM photos {where_clause} ORDER BY created_at DESC, id DESC",
            params,
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if should_close:
            db.close()


# ==================== 伴侣文件操作 ====================

def add_companion_file(
    photo_id: int,
    filepath: str,
    file_type: str,
    filesize: int,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[int]:
    """关联伴侣文件 (如 LivePhoto 视频、XMP 调色文件、RAW 负片)。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO companion_files (photo_id, filepath, file_type, filesize)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(filepath) DO UPDATE SET
                photo_id = excluded.photo_id,
                file_type = excluded.file_type,
                filesize = excluded.filesize
            """,
            (photo_id, filepath, file_type, filesize),
        )
        db.commit()
        return cursor.lastrowid
    finally:
        if should_close:
            db.close()


def get_companion_files_by_photo_id(
    photo_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """获取指定照片的所有伴侣文件。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM companion_files WHERE photo_id = ?", (photo_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if should_close:
            db.close()


# ==================== 熟人底库与多姿态质心 ====================

def create_person(
    name: str,
    centroid_front: Optional[np.ndarray] = None,
    centroid_profile: Optional[np.ndarray] = None,
    cover_face_id: Optional[int] = None,
    face_count: int = 0,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """在底库中登记人物并存储初始质心。若已存在则返回已有 ID。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO persons (name, cover_face_id, centroid_front, centroid_profile, face_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                name,
                cover_face_id,
                array_to_blob(centroid_front),
                array_to_blob(centroid_profile),
                face_count,
            ),
        )
        db.commit()
        if cursor.lastrowid:
            return cursor.lastrowid
        cursor.execute("SELECT id FROM persons WHERE name = ?", (name,))
        row = cursor.fetchone()
        return row["id"]
    finally:
        if should_close:
            db.close()


def get_person_by_id(person_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    """根据 ID 获取人物信息。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        if should_close:
            db.close()


def get_person_by_name(name: str, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    """根据姓名获取人物信息。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM persons WHERE name = ?", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        if should_close:
            db.close()


def list_all_persons(conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    """查询底库中所有已记录的人物。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM persons ORDER BY face_count DESC, id ASC")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if should_close:
            db.close()


def update_person_centroids(
    person_id: int,
    centroid_front: Optional[np.ndarray] = None,
    centroid_profile: Optional[np.ndarray] = None,
    additional_face_count: int = 0,
    cover_face_id: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """更新人物的正脸主质心、侧脸质心及样本累计计数。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        fields = ["face_count = face_count + ?", "updated_at = CURRENT_TIMESTAMP"]
        values: List[Any] = [additional_face_count]

        if centroid_front is not None:
            fields.append("centroid_front = ?")
            values.append(array_to_blob(centroid_front))
        if centroid_profile is not None:
            fields.append("centroid_profile = ?")
            values.append(array_to_blob(centroid_profile))
        if cover_face_id is not None:
            fields.append("cover_face_id = ?")
            values.append(cover_face_id)

        values.append(person_id)
        cursor.execute(f"UPDATE persons SET {', '.join(fields)} WHERE id = ?", values)
        db.commit()
        return cursor.rowcount > 0
    finally:
        if should_close:
            db.close()


def update_person_name(person_id: int, new_name: str, conn: Optional[sqlite3.Connection] = None) -> bool:
    """修改人物姓名。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute("UPDATE persons SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (new_name, person_id))
        db.commit()
        return cursor.rowcount > 0
    finally:
        if should_close:
            db.close()


def delete_person(person_id: int, conn: Optional[sqlite3.Connection] = None) -> bool:
    """删除指定人物记录，关联的人脸将自动重置 person_id 为 NULL。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        db.commit()
        return cursor.rowcount > 0
    finally:
        if should_close:
            db.close()


# ==================== 人脸数据操作 ====================

def add_face(
    photo_id: int,
    bbox: List[int],
    landmarks: List[List[float]],
    embedding: np.ndarray,
    confidence: float = 1.0,
    yaw_angle: Optional[float] = None,
    person_id: Optional[int] = None,
    is_noise: int = 0,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """添加检测到的人脸特征数据。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO faces (
                photo_id, person_id, bbox, landmarks, embedding,
                confidence, yaw_angle, is_noise
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                photo_id,
                person_id,
                json.dumps(bbox),
                json.dumps(landmarks),
                array_to_blob(embedding),
                confidence,
                yaw_angle,
                is_noise,
            ),
        )
        db.commit()
        return cursor.lastrowid
    finally:
        if should_close:
            db.close()


def get_faces_by_photo_id(photo_id: int, conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    """获取照片包含的所有人脸数据。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM faces WHERE photo_id = ? ORDER BY id ASC", (photo_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if should_close:
            db.close()


def get_unassigned_faces(conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    """获取所有尚未归属人物且未标记为噪声的人脸。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM faces WHERE person_id IS NULL AND is_noise = 0 ORDER BY id ASC")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if should_close:
            db.close()


def assign_face_to_person(face_id: int, person_id: int, conn: Optional[sqlite3.Connection] = None) -> bool:
    """将人脸关联至指定的人物。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute(
            "UPDATE faces SET person_id = ?, is_noise = 0 WHERE id = ?",
            (person_id, face_id),
        )
        db.commit()
        return cursor.rowcount > 0
    finally:
        if should_close:
            db.close()


def mark_face_as_noise(face_id: int, conn: Optional[sqlite3.Connection] = None) -> bool:
    """将无法辨识或孤立的人脸标记为噪声。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute("UPDATE faces SET is_noise = 1, person_id = NULL WHERE id = ?", (face_id,))
        db.commit()
        return cursor.rowcount > 0
    finally:
        if should_close:
            db.close()


# ==================== 云备份台账操作 ====================

def record_backup_status(
    photo_sha256: str,
    target_platform: str,
    album_name: Optional[str] = None,
    is_uploaded: int = 0,
    uploaded_at: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """登记或更新照片在目标云平台的备份状态。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO backup_ledger (
                photo_sha256, target_platform, album_name, is_uploaded, uploaded_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(photo_sha256, target_platform, album_name) DO UPDATE SET
                is_uploaded = excluded.is_uploaded,
                uploaded_at = COALESCE(excluded.uploaded_at, backup_ledger.uploaded_at)
            """,
            (photo_sha256, target_platform, album_name or "", is_uploaded, uploaded_at),
        )
        db.commit()
        return cursor.lastrowid
    finally:
        if should_close:
            db.close()


def get_backup_records_by_sha256(
    photo_sha256: str,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """查询指定照片在各平台的备份记录。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        cursor.execute(
            "SELECT * FROM backup_ledger WHERE photo_sha256 = ?",
            (photo_sha256,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if should_close:
            db.close()


def list_unbacked_photos(
    target_platform: Optional[str] = None,
    album_name: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """查询尚未完成备份的照片记录。若指定平台和相册，则筛查对应目标下的漏传照片。"""
    should_close = conn is None
    db = conn or get_connection()
    try:
        cursor = db.cursor()
        if target_platform:
            cursor.execute(
                """
                SELECT p.* FROM photos p
                LEFT JOIN backup_ledger bl
                    ON p.sha256 = bl.photo_sha256
                    AND bl.target_platform = ?
                    AND bl.album_name = ?
                WHERE bl.id IS NULL OR bl.is_uploaded = 0
                ORDER BY p.created_at DESC
                """,
                (target_platform, album_name or ""),
            )
        else:
            # 查出从未在任何平台完成上传的照片
            cursor.execute(
                """
                SELECT p.* FROM photos p
                LEFT JOIN backup_ledger bl
                    ON p.sha256 = bl.photo_sha256
                    AND bl.is_uploaded = 1
                WHERE bl.id IS NULL
                ORDER BY p.created_at DESC
                """
            )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if should_close:
            db.close()


def mark_backup_completed(
    photo_sha256: str,
    target_platform: str,
    album_name: Optional[str] = None,
    uploaded_at: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """将照片在指定平台相册下的备份状态标记为已完成。"""
    return bool(
        record_backup_status(
            photo_sha256=photo_sha256,
            target_platform=target_platform,
            album_name=album_name,
            is_uploaded=1,
            uploaded_at=uploaded_at,
            conn=conn,
        )
    )
