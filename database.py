#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - 数据库交互模块
负责与 SQLite 数据库进行所有交互，提供标准化的数据读写接口

版本: 2.1 (CLIP Embedding 支持)
"""

import sqlite3
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np


# 数据库文件名
DB_NAME = "photoye_library.db"
DB_VERSION = "2.1"


def get_db_path() -> str:
    """获取数据库文件路径"""
    # 数据库文件放在当前工作目录下
    return os.path.join(os.getcwd(), DB_NAME)


def get_connection() -> sqlite3.Connection:
    """获取数据库连接"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    
    # 启用外键约束
    conn.execute("PRAGMA foreign_keys = ON")
    
    # 设置行工厂，使查询结果可以像字典一样访问
    conn.row_factory = sqlite3.Row
    
    return conn


def init_db() -> None:
    """
    初始化数据库，创建所有必要的表
    """
    conn = get_connection()
    
    try:
        cursor = conn.cursor()
        
        # 创建照片表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT NOT NULL UNIQUE,          -- 文件绝对路径
                filesize INTEGER,                       -- 文件大小(字节)
                created_at TEXT,                        -- EXIF拍摄日期(ISO格式)
                category TEXT,                          -- 初步分类: 风景, 单人照, 合照
                embedding BLOB,                         -- CLIP 512维图像特征向量
                status TEXT DEFAULT 'pending',          -- 处理状态: pending, processing, done
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 添加到数据库的时间
            )
        """)
        
        # 创建人物表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,              -- 用户命名的人物姓名
                cover_face_id INTEGER,                  -- 可选用作封面的人脸ID
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建人脸数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,              -- 关联的照片ID
                person_id INTEGER,                      -- 关联的人物ID (未命名时为NULL, -1为噪声点)
                bbox TEXT NOT NULL,                     -- 边界框坐标 (JSON: [x1,y1,x2,y2])
                landmarks TEXT,                         -- 5点关键点坐标 (JSON)
                embedding BLOB NOT NULL,                -- 512维人脸特征向量
                confidence REAL DEFAULT 0.0,            -- 检测置信度
                is_noise INTEGER DEFAULT 0,             -- DBSCAN噪声标记
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
            )
        """)
        
        # 创建标签表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建照片与标签关联表 (多对多)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS photo_tags (
                photo_id INTEGER,
                tag_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (photo_id, tag_id),
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            )
        """)
        
        # 创建索引以提升查询性能
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_category ON photos(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photo_tags_photo_id ON photo_tags(photo_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photo_tags_tag_id ON photo_tags(tag_id)")
        
        conn.commit()
        print(f"数据库初始化成功: {get_db_path()}")
        print(f"表结构: photos(临时,每次关闭时清空), persons/faces/tags(持久)")
        
    except sqlite3.Error as e:
        print(f"数据库初始化失败: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def clear_temp_photos() -> None:
    """
    清空临时照片表（photos 及其关联的临时数据）
    保留 persons, faces, tags 等持久数据供下次使用
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM photos")
        cursor.execute("DELETE FROM photo_tags WHERE photo_id NOT IN (SELECT id FROM photos)")
        conn.commit()
        print(f"已清空临时照片数据")
    except sqlite3.Error as e:
        print(f"清空临时照片失败: {e}")
        conn.rollback()
    finally:
        conn.close()


def cleanup_on_exit() -> None:
    """
    程序退出时的清理操作：
    1. 清空临时照片表
    2. 保留人脸embedding和人物身份信息供下次使用
    """
    print("正在清理临时数据...")
    clear_temp_photos()
    print("临时数据已清理，人脸和人物信息已保留")


def add_photo(filepath: str, filesize: int = None, created_at: str = None) -> Optional[int]:
    """
    添加照片记录到数据库
    
    Args:
        filepath: 文件绝对路径
        filesize: 文件大小(字节)
        created_at: 拍摄日期(ISO格式字符串)
    
    Returns:
        新插入记录的ID，如果插入失败返回None
    """
    conn = get_connection()
    
    try:
        cursor = conn.cursor()
        
        # 如果没有提供文件大小，尝试获取
        if filesize is None and os.path.exists(filepath):
            filesize = os.path.getsize(filepath)
        
        cursor.execute("""
            INSERT INTO photos (filepath, filesize, created_at, status)
            VALUES (?, ?, ?, 'pending')
        """, (filepath, filesize, created_at))
        
        conn.commit()
        photo_id = cursor.lastrowid
        print(f"添加照片记录: ID={photo_id}, Path={filepath}")
        return photo_id
        
    except sqlite3.IntegrityError:
        # 文件已存在于数据库中
        print(f"照片已存在于数据库中: {filepath}")
        return None
    except sqlite3.Error as e:
        print(f"添加照片记录失败: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()


def add_photos_batch(filepaths: List[str], batch_size: int = 100) -> List[int]:
    """
    批量添加照片记录到数据库
    
    Args:
        filepaths: 文件路径列表
        batch_size: 每批次处理的数量
    
    Returns:
        成功插入的照片ID列表
    """
    if not filepaths:
        return []
    
    conn = get_connection()
    inserted_ids = []
    
    try:
        cursor = conn.cursor()
        
        # 准备批量数据
        batch_data = []
        for filepath in filepaths:
            filesize = None
            if os.path.exists(filepath):
                filesize = os.path.getsize(filepath)
            batch_data.append((filepath, filesize, None, 'pending'))
        
        # 分批插入
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            try:
                cursor.executemany("""
                    INSERT OR IGNORE INTO photos (filepath, filesize, created_at, status)
                    VALUES (?, ?, ?, ?)
                """, batch)
                conn.commit()
            except sqlite3.Error as e:
                print(f"批量插入照片失败: {e}")
                conn.rollback()
        
        # 获取插入的ID（需要重新查询）
        placeholders = ','.join(['?' for _ in filepaths])
        cursor.execute(f"SELECT id FROM photos WHERE filepath IN ({placeholders})", filepaths)
        inserted_ids = [row[0] for row in cursor.fetchall()]
        
        print(f"批量添加照片: 成功 {len(inserted_ids)}/{len(filepaths)}")
        return inserted_ids
        
    except sqlite3.Error as e:
        print(f"批量添加照片失败: {e}")
        conn.rollback()
        return []
    finally:
        conn.close()


def get_photo_id_by_path(filepath: str) -> Optional[int]:
    """根据文件路径获取照片ID"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM photos WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        print(f"获取照片ID失败: {e}")
        return None
    finally:
        conn.close()


def get_photos_without_faces(library_path: str = None) -> List[Dict]:
    """
    获取尚未进行人脸检测的照片
    
    Args:
        library_path: 限制在某个目录下
    
    Returns:
        照片记录列表
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # 查找没有人脸记录且分类为人物相关的照片
        sql = """
            SELECT p.id, p.filepath, p.category
            FROM photos p
            LEFT JOIN faces f ON p.id = f.photo_id
            WHERE f.id IS NULL
            AND p.category IN ('单人照', '合照', '人物')
        """
        params = []
        
        if library_path:
            sql += " AND p.filepath LIKE ?"
            params.append(f"{library_path}%")
        
        cursor.execute(sql, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'filepath': row[1],
                'category': row[2]
            })
        
        return results
        
    except sqlite3.Error as e:
        print(f"获取未检测人脸照片失败: {e}")
        return []
    finally:
        conn.close()


def is_photo_exist(filepath: str) -> bool:
    """
    检查照片是否已存在于数据库中
    
    Args:
        filepath: 文件绝对路径
    
    Returns:
        True if exists, False otherwise
    """
    conn = get_connection()
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM photos WHERE filepath = ?", (filepath,))
        return cursor.fetchone() is not None
    except sqlite3.Error as e:
        print(f"检查照片存在性失败: {e}")
        return False
    finally:
        conn.close()


def get_photo_status(filepath: str) -> Optional[Tuple[int, str, Optional[str]]]:
    """返回指定路径的照片记录 (id, status, category)。"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, status, category FROM photos WHERE filepath = ?",
            (filepath,),
        )
        row = cursor.fetchone()
        if row:
            return row[0], row[1], row[2]
        return None
    except sqlite3.Error as e:
        print(f"获取照片状态失败: {e}")
        return None
    finally:
        conn.close()


def get_all_photos(
    status: str = None,
    categories: Optional[List[str]] = None,
    library_path: str = None,
    person_id: Optional[int] = None,
    has_faces: Optional[bool] = None,
    unlabeled_faces: bool = False,
) -> List[Dict]:
    """
    获取所有照片记录，支持按人物和人脸存在性过滤。
    
    注意：photos 表为临时表，仅在本次运行期间有效。

    Args:
        status: 可选的状态过滤器 ('pending', 'processing', 'done')
        categories: 可选的分类过滤器列表，支持多选（如 ['单人照', '合照']）
        library_path: 可选的库路径过滤器
        person_id: 若提供，则只返回包含指定人物的人脸的照片
        has_faces: True 只要含有人脸，False 仅无任何人脸
        unlabeled_faces: True 时返回含有未命名人脸的照片

    Returns:
        照片记录列表，每个记录是一个字典
    """
    conn = get_connection()

    try:
        cursor = conn.cursor()

        query = "SELECT DISTINCT p.* FROM photos p"
        joins = []
        conditions = []
        params: list = []

        if status:
            conditions.append("p.status = ?")
            params.append(status)

        if categories:
            placeholders = ",".join(["?"] * len(categories))
            conditions.append(f"p.category IN ({placeholders})")
            params.extend(categories)

        if library_path:
            conditions.append("p.filepath LIKE ?")
            params.append(library_path.replace('\\', '/') + '%')

        if person_id is not None:
            joins.append("JOIN faces f ON f.photo_id = p.id AND f.person_id = ?")
            params.append(person_id)
        elif unlabeled_faces:
            joins.append("JOIN faces f ON f.photo_id = p.id AND f.person_id IS NULL")
        elif has_faces is True:
            joins.append("JOIN faces f ON f.photo_id = p.id")
        elif has_faces is False:
            conditions.append("NOT EXISTS (SELECT 1 FROM faces f WHERE f.photo_id = p.id)")

        if joins:
            query += " " + " ".join(joins)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY p.added_at DESC"

        cursor.execute(query, params)

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    except sqlite3.Error as e:
        print(f"获取照片列表失败: {e}")
        return []
    finally:
        conn.close()


def get_unlabeled_faces(limit: int = 30) -> List[Dict]:
    """获取未命名人脸列表，附带照片路径与 bbox，便于前端展示命名。"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT f.id, f.photo_id, f.bbox, f.confidence, p.filepath
            FROM faces f
            JOIN photos p ON p.id = f.photo_id
            WHERE f.person_id IS NULL
            ORDER BY f.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        import json

        faces = []
        for row in rows:
            face = dict(row)
            face["bbox"] = json.loads(face["bbox"])
            faces.append(face)
        return faces
    except sqlite3.Error as e:
        print(f"获取未命名人脸失败: {e}")
        return []
    finally:
        conn.close()


def list_persons() -> List[Dict]:
    """返回已命名人物列表及人脸/照片计数。"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.id, p.name,
                   COUNT(f.id) AS face_count,
                   COUNT(DISTINCT f.photo_id) AS photo_count
            FROM persons p
            LEFT JOIN faces f ON f.person_id = p.id
            GROUP BY p.id, p.name
            ORDER BY p.name COLLATE NOCASE
            """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"获取人物列表失败: {e}")
        return []
    finally:
        conn.close()


def get_person_with_faces(person_id: int) -> Optional[Dict]:
    """
    获取指定人物的详细信息及其所有人脸记录
    
    Args:
        person_id: 人物ID
    
    Returns:
        包含人物信息和人脸列表的字典
    """
    import json
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # 获取人物基本信息
        cursor.execute("SELECT id, name, created_at FROM persons WHERE id = ?", (person_id,))
        person_row = cursor.fetchone()
        if not person_row:
            return None
        
        person = dict(person_row)
        
        # 获取该人物的所有人脸记录及对应照片路径
        cursor.execute("""
            SELECT f.id, f.photo_id, f.bbox, f.confidence, f.created_at,
                   p.filepath AS photo_filepath
            FROM faces f
            JOIN photos p ON p.id = f.photo_id
            WHERE f.person_id = ?
            ORDER BY f.created_at DESC
        """, (person_id,))
        
        faces = []
        for row in cursor.fetchall():
            face = dict(row)
            face['bbox'] = json.loads(face['bbox'])
            faces.append(face)
        
        person['faces'] = faces
        person['face_count'] = len(faces)
        person['photo_count'] = len(set(f['photo_id'] for f in faces))
        
        return person
    except sqlite3.Error as e:
        print(f"获取人物详情失败: {e}")
        return None
    finally:
        conn.close()


def get_all_persons_with_sample_faces(limit_faces: int = 4) -> List[Dict]:
    """
    获取所有人物及其样本人脸（用于人物视图展示）
    
    Args:
        limit_faces: 每个人物最多返回的人脸样本数
    
    Returns:
        人物列表，每个人物包含样本人脸信息
    """
    import json
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # 获取所有人物
        cursor.execute("""
            SELECT p.id, p.name, p.created_at,
                   COUNT(f.id) AS face_count,
                   COUNT(DISTINCT f.photo_id) AS photo_count
            FROM persons p
            LEFT JOIN faces f ON f.person_id = p.id
            GROUP BY p.id, p.name
            ORDER BY p.name COLLATE NOCASE
        """)
        
        persons = []
        for row in cursor.fetchall():
            person = dict(row)
            
            # 获取该人物的样本人脸
            cursor.execute("""
                SELECT f.id, f.photo_id, f.bbox, f.confidence,
                       ph.filepath AS photo_filepath
                FROM faces f
                JOIN photos ph ON ph.id = f.photo_id
                WHERE f.person_id = ?
                ORDER BY f.confidence DESC
                LIMIT ?
            """, (person['id'], limit_faces))
            
            sample_faces = []
            for face_row in cursor.fetchall():
                face = dict(face_row)
                face['bbox'] = json.loads(face['bbox'])
                sample_faces.append(face)
            
            person['sample_faces'] = sample_faces
            persons.append(person)
        
        return persons
    except sqlite3.Error as e:
        print(f"获取人物样本失败: {e}")
        return []
    finally:
        conn.close()


def delete_person(person_id: int) -> bool:
    """
    删除人物（人脸记录的person_id将被设为NULL）
    
    Args:
        person_id: 人物ID
    
    Returns:
        是否成功删除
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"删除人物失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def rename_person(person_id: int, new_name: str) -> bool:
    """
    重命名人物
    
    Args:
        person_id: 人物ID
        new_name: 新名称
    
    Returns:
        是否成功
    """
    if not new_name:
        return False
    
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE persons SET name = ? WHERE id = ?", (new_name, person_id))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"重命名人物失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def get_photos_by_person(person_id: int) -> List[Dict]:
    """
    获取包含指定人物人脸的所有照片
    
    Args:
        person_id: 人物ID
    
    Returns:
        照片列表
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT p.*
            FROM photos p
            JOIN faces f ON f.photo_id = p.id
            WHERE f.person_id = ?
            ORDER BY p.created_at DESC
        """, (person_id,))
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"获取人物照片失败: {e}")
        return []
    finally:
        conn.close()


def update_photo_status(photo_id: int, status: str, category: str = None) -> bool:
    """
    更新照片的处理状态和分类
    
    Args:
        photo_id: 照片ID
        status: 新状态 ('pending', 'processing', 'done')
        category: 可选的分类 ('风景', '单人照', '合照')
    
    Returns:
        True if successful, False otherwise
    """
    conn = get_connection()
    
    try:
        cursor = conn.cursor()
        
        if category:
            cursor.execute("""
                UPDATE photos SET status = ?, category = ?
                WHERE id = ?
            """, (status, category, photo_id))
        else:
            cursor.execute("""
                UPDATE photos SET status = ?
                WHERE id = ?
            """, (status, photo_id))
        
        conn.commit()
        
        if cursor.rowcount > 0:
            print(f"更新照片状态: ID={photo_id}, Status={status}, Category={category}")
            return True
        else:
            print(f"未找到照片记录: ID={photo_id}")
            return False
            
    except sqlite3.Error as e:
        print(f"更新照片状态失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def get_photos_count(library_path: str = None) -> Dict[str, int]:
    """
    获取照片数量统计信息
    
    Args:
        library_path: 可选的库路径过滤器
    
    Returns:
        包含各种统计数据的字典
    """
    conn = get_connection()
    
    try:
        cursor = conn.cursor()
        
        # 构造基础条件和参数
        base_conditions = []
        base_params: List = []

        if library_path:
            base_conditions.append("filepath LIKE ?")
            base_params.append(library_path.replace('\\', '/') + '%')

        def build_where(extra_conditions: Optional[List[str]] = None) -> str:
            conditions = list(base_conditions)
            if extra_conditions:
                conditions.extend(extra_conditions)
            if conditions:
                return " WHERE " + " AND ".join(conditions)
            return ""

        def build_params(extra_params: Optional[List] = None) -> List:
            params = list(base_params)
            if extra_params:
                params.extend(extra_params)
            return params

        # 总照片数
        query_total = "SELECT COUNT(*) FROM photos" + build_where()
        cursor.execute(query_total, build_params())
        total = cursor.fetchone()[0]

        # 按状态统计
        query_status = "SELECT status, COUNT(*) FROM photos" + build_where() + " GROUP BY status"
        cursor.execute(query_status, build_params())
        status_counts = dict(cursor.fetchall())

        # 按分类统计
        query_category = (
            "SELECT category, COUNT(*) FROM photos"
            + build_where(["category IS NOT NULL"])
            + " GROUP BY category"
        )
        cursor.execute(query_category, build_params())
        category_counts = dict(cursor.fetchall())

        # 人脸数量
        query_faces = (
            "SELECT COUNT(*) FROM faces WHERE photo_id IN (SELECT id FROM photos"
            + build_where()
            + ")"
        )
        cursor.execute(query_faces, build_params())
        faces_count = cursor.fetchone()[0]
        
        # 已命名人物数量
        query_persons = "SELECT COUNT(*) FROM persons"
        cursor.execute(query_persons)
        persons_count = cursor.fetchone()[0]
        
        return {
            'total': total,
            'status': status_counts,
            'category': category_counts,
            'faces': faces_count,
            'persons': persons_count
        }
        
    except sqlite3.Error as e:
        print(f"获取统计信息失败: {e}")
        return {}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 人脸标记与分类辅助
# ---------------------------------------------------------------------------


def get_or_create_person(name: str) -> Optional[int]:
    """获取或创建人物条目，返回人物ID。"""
    if not name:
        return None

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO persons (name) VALUES (?)", (name,))
        conn.commit()

        cursor.execute("SELECT id FROM persons WHERE name = ?", (name,))
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        print(f"获取/创建人物失败: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()


def assign_faces_to_person(face_ids: List[int], person_id: int) -> int:
    """批量将人脸记录关联到指定人物，返回更新条数。"""
    if not face_ids or person_id is None:
        return 0

    conn = get_connection()
    try:
        cursor = conn.cursor()
        placeholders = ",".join(["?"] * len(face_ids))
        params = [person_id] + face_ids
        cursor.execute(
            f"UPDATE faces SET person_id = ? WHERE id IN ({placeholders})",
            params,
        )
        conn.commit()
        return cursor.rowcount
    except sqlite3.Error as e:
        print(f"批量标记人脸失败: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def set_photo_category(photo_id: int, category: str) -> bool:
    """仅更新分类字段，保持状态不变。"""
    if not category:
        return False
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE photos SET category = ? WHERE id = ?", (category, photo_id))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"更新照片分类失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def add_face_data(photo_id: int, bbox: List[int], embedding: np.ndarray, confidence: float = 0.0) -> Optional[int]:
    """
    添加人脸数据到数据库
    
    Args:
        photo_id: 照片ID
        bbox: 人脸边界框坐标 [x1, y1, x2, y2]
        embedding: 512维人脸特征向量
        confidence: 检测置信度
    
    Returns:
        新插入记录的ID，如果插入失败返回None
    """
    conn = get_connection()
    
    try:
        cursor = conn.cursor()
        
        # 将bbox转换为JSON字符串
        import json
        bbox_json = json.dumps(bbox)
        
        # 将numpy数组转换为bytes
        embedding_blob = embedding.tobytes()
        
        cursor.execute("""
            INSERT INTO faces (photo_id, bbox, embedding, confidence)
            VALUES (?, ?, ?, ?)
        """, (photo_id, bbox_json, embedding_blob, confidence))
        
        conn.commit()
        face_id = cursor.lastrowid
        print(f"添加人脸记录: ID={face_id}, PhotoID={photo_id}")
        return face_id
        
    except sqlite3.Error as e:
        print(f"添加人脸记录失败: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()


def add_faces_batch(faces_data: List[Dict], batch_size: int = 50) -> int:
    """
    批量添加人脸数据到数据库
    
    Args:
        faces_data: 人脸数据列表，每个元素包含 {photo_id, bbox, embedding, confidence}
        batch_size: 每批次处理的数量
    
    Returns:
        成功插入的记录数
    """
    if not faces_data:
        return 0
    
    import json
    conn = get_connection()
    inserted_count = 0
    
    try:
        cursor = conn.cursor()
        
        # 准备批量数据
        batch_data = []
        for face in faces_data:
            bbox_json = json.dumps(face['bbox'])
            embedding_blob = face['embedding'].tobytes()
            batch_data.append((
                face['photo_id'],
                bbox_json,
                embedding_blob,
                face.get('confidence', 0.0)
            ))
        
        # 分批插入
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            try:
                cursor.executemany("""
                    INSERT INTO faces (photo_id, bbox, embedding, confidence)
                    VALUES (?, ?, ?, ?)
                """, batch)
                conn.commit()
                inserted_count += len(batch)
            except sqlite3.Error as e:
                print(f"批量插入人脸失败: {e}")
                conn.rollback()
        
        print(f"批量添加人脸: 成功 {inserted_count}/{len(faces_data)}")
        return inserted_count
        
    except sqlite3.Error as e:
        print(f"批量添加人脸失败: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def get_faces_by_photo_id(photo_id: int) -> List[Dict]:
    """
    根据照片ID获取所有人脸数据
    
    Args:
        photo_id: 照片ID
    
    Returns:
        人脸记录列表，每个记录是一个字典
    """
    conn = get_connection()
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, photo_id, person_id, bbox, embedding, confidence
            FROM faces 
            WHERE photo_id = ?
            ORDER BY id
        """, (photo_id,))
        
        rows = cursor.fetchall()
        faces = []
        
        import json
        for row in rows:
            face = dict(row)
            # 将bbox JSON字符串转回列表
            face['bbox'] = json.loads(face['bbox'])
            # 将embedding blob转回numpy数组
            face['embedding'] = np.frombuffer(face['embedding'], dtype=np.float32)
            faces.append(face)
        
        return faces
        
    except sqlite3.Error as e:
        print(f"获取人脸数据失败: {e}")
        return []
    finally:
        conn.close()


def cleanup_missing_files() -> int:
    """
    清理数据库中指向已不存在文件的记录
    
    Returns:
        清理的记录数量
    """
    conn = get_connection()
    
    try:
        cursor = conn.cursor()
        
        # 获取所有照片记录
        cursor.execute("SELECT id, filepath FROM photos")
        photos = cursor.fetchall()
        
        missing_ids = []
        for photo in photos:
            if not os.path.exists(photo['filepath']):
                missing_ids.append(photo['id'])
        
        # 删除缺失的照片记录(级联删除相关的人脸和标签记录)
        if missing_ids:
            placeholders = ','.join(['?'] * len(missing_ids))
            cursor.execute(f"DELETE FROM photos WHERE id IN ({placeholders})", missing_ids)
            conn.commit()
            
        print(f"清理了 {len(missing_ids)} 个缺失文件的记录")
        return len(missing_ids)
        
    except sqlite3.Error as e:
        print(f"清理缺失文件失败: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


# ============================================================
# 数据库迁移与 Embedding 操作 (V2.1)
# ============================================================

def migrate_to_v21() -> bool:
    """
    迁移数据库到 V2.1 版本:
    - photos 表添加 embedding 字段
    - faces 表添加 landmarks, is_noise 字段
    
    Returns:
        True if migration successful
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # 检查 photos.embedding 是否存在
        cursor.execute("PRAGMA table_info(photos)")
        photo_columns = [row['name'] for row in cursor.fetchall()]
        
        if 'embedding' not in photo_columns:
            print("迁移: 添加 photos.embedding 字段...")
            cursor.execute("ALTER TABLE photos ADD COLUMN embedding BLOB")
        
        # 检查 faces.landmarks, is_noise 是否存在
        cursor.execute("PRAGMA table_info(faces)")
        face_columns = [row['name'] for row in cursor.fetchall()]
        
        if 'landmarks' not in face_columns:
            print("迁移: 添加 faces.landmarks 字段...")
            cursor.execute("ALTER TABLE faces ADD COLUMN landmarks TEXT")
            
        if 'is_noise' not in face_columns:
            print("迁移: 添加 faces.is_noise 字段...")
            cursor.execute("ALTER TABLE faces ADD COLUMN is_noise INTEGER DEFAULT 0")
        
        conn.commit()
        print("数据库迁移到 V2.1 完成")
        return True
        
    except sqlite3.Error as e:
        print(f"数据库迁移失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def update_photo_embedding(photo_id: int, embedding: np.ndarray) -> bool:
    """
    更新照片的 CLIP embedding.
    
    Args:
        photo_id: 照片 ID
        embedding: 512 维 numpy 数组
    
    Returns:
        True if successful
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        embedding_blob = embedding.astype(np.float32).tobytes()
        cursor.execute(
            "UPDATE photos SET embedding = ? WHERE id = ?",
            (embedding_blob, photo_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"更新 embedding 失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def update_photo_embedding_by_path(filepath: str, embedding: np.ndarray) -> bool:
    """
    根据文件路径更新照片的 CLIP embedding.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        embedding_blob = embedding.astype(np.float32).tobytes()
        cursor.execute(
            "UPDATE photos SET embedding = ? WHERE filepath = ?",
            (embedding_blob, filepath)
        )
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"更新 embedding 失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def get_all_photo_embeddings() -> List[Tuple[int, str, Optional[np.ndarray]]]:
    """
    获取所有照片的 embedding.
    
    Returns:
        List of (photo_id, filepath, embedding) tuples.
        embedding is None if not yet computed.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filepath, embedding FROM photos")
        results = []
        for row in cursor.fetchall():
            photo_id = row['id']
            filepath = row['filepath']
            embedding = None
            if row['embedding']:
                embedding = np.frombuffer(row['embedding'], dtype=np.float32)
            results.append((photo_id, filepath, embedding))
        return results
    except sqlite3.Error as e:
        print(f"获取 embeddings 失败: {e}")
        return []
    finally:
        conn.close()


def get_photos_without_embedding() -> List[Tuple[int, str]]:
    """
    获取没有 embedding 的照片列表.
    
    Returns:
        List of (photo_id, filepath) tuples
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, filepath FROM photos WHERE embedding IS NULL"
        )
        return [(row['id'], row['filepath']) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"查询失败: {e}")
        return []
    finally:
        conn.close()


def batch_update_photo_embeddings(updates: List[Tuple[int, np.ndarray]]) -> int:
    """
    批量更新照片 embeddings.
    
    Args:
        updates: List of (photo_id, embedding) tuples
    
    Returns:
        Number of successfully updated records
    """
    if not updates:
        return 0
        
    conn = get_connection()
    try:
        cursor = conn.cursor()
        data = [
            (emb.astype(np.float32).tobytes(), pid)
            for pid, emb in updates
        ]
        cursor.executemany(
            "UPDATE photos SET embedding = ? WHERE id = ?",
            data
        )
        conn.commit()
        return cursor.rowcount
    except sqlite3.Error as e:
        print(f"批量更新失败: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def search_photos_by_embedding(
    query_embedding: np.ndarray,
    top_k: int = 20,
    threshold: float = 0.0
) -> List[Tuple[int, str, float]]:
    """
    通过 embedding 相似度搜索照片 (纯 SQLite 实现).
    
    注意: 对于大量照片，建议使用 FAISS 索引代替.
    
    Args:
        query_embedding: 查询向量 (512 维)
        top_k: 返回数量
        threshold: 最低相似度阈值
    
    Returns:
        List of (photo_id, filepath, similarity) tuples, sorted by similarity desc
    """
    all_photos = get_all_photo_embeddings()
    
    # 过滤有 embedding 的照片
    valid_photos = [(pid, path, emb) for pid, path, emb in all_photos if emb is not None]
    
    if not valid_photos:
        return []
    
    # 计算余弦相似度
    results = []
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    
    for photo_id, filepath, emb in valid_photos:
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        similarity = float(np.dot(query_norm, emb_norm))
        if similarity >= threshold:
            results.append((photo_id, filepath, similarity))
    
    # 按相似度排序
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results[:top_k]


def main():
    """
    主函数 - 用于独立测试数据库模块
    """
    print("=" * 50)
    print("Photoye 数据库模块测试")
    print("=" * 50)
    
    # 初始化数据库
    print("\n1. 初始化数据库...")
    init_db()
    
    # 迁移到 V2.1
    print("\n2. 迁移到 V2.1...")
    migrate_to_v21()
    
    # 显示数据库路径
    print(f"\n3. 数据库路径: {get_db_path()}")
    
    # 获取统计信息
    print("\n4. 数据库统计信息:")
    stats = get_photos_count()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n数据库模块测试完成！")


if __name__ == "__main__":
    main()