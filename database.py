#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - 数据库交互模块
负责与 SQLite 数据库进行所有交互，提供标准化的数据读写接口

版本: 1.0
日期: 2025年08月14日
"""

import sqlite3
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# 数据库文件名
DB_NAME = "photoye_library.db"


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
                person_id INTEGER,                      -- 关联的人物ID (未命名时为NULL)
                bbox TEXT NOT NULL,                     -- 边界框坐标 (JSON: [x1,y1,x2,y2])
                embedding BLOB NOT NULL,                -- 512维人脸特征向量
                confidence REAL DEFAULT 0.0,            -- 检测置信度
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
        
    except sqlite3.Error as e:
        print(f"数据库初始化失败: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


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


def get_all_photos(status: str = None, category: str = None) -> List[Dict]:
    """
    获取所有照片记录
    
    Args:
        status: 可选的状态过滤器 ('pending', 'processing', 'done')
        category: 可选的分类过滤器 ('风景', '单人照', '合照')
    
    Returns:
        照片记录列表，每个记录是一个字典
    """
    conn = get_connection()
    
    try:
        cursor = conn.cursor()
        
        query = "SELECT * FROM photos"
        params = []
        conditions = []
        
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY added_at DESC"
        
        cursor.execute(query, params)
        
        # 将Row对象转换为字典
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
        
    except sqlite3.Error as e:
        print(f"获取照片列表失败: {e}")
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


def get_photos_count() -> Dict[str, int]:
    """
    获取照片数量统计信息
    
    Returns:
        包含各种统计数据的字典
    """
    conn = get_connection()
    
    try:
        cursor = conn.cursor()
        
        # 总照片数
        cursor.execute("SELECT COUNT(*) FROM photos")
        total = cursor.fetchone()[0]
        
        # 按状态统计
        cursor.execute("SELECT status, COUNT(*) FROM photos GROUP BY status")
        status_counts = dict(cursor.fetchall())
        
        # 按分类统计
        cursor.execute("SELECT category, COUNT(*) FROM photos WHERE category IS NOT NULL GROUP BY category")
        category_counts = dict(cursor.fetchall())
        
        # 人脸数量
        cursor.execute("SELECT COUNT(*) FROM faces")
        faces_count = cursor.fetchone()[0]
        
        # 已命名人物数量
        cursor.execute("SELECT COUNT(*) FROM persons")
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
    
    # 显示数据库路径
    print(f"\n2. 数据库路径: {get_db_path()}")
    
    # 获取统计信息
    print("\n3. 数据库统计信息:")
    stats = get_photos_count()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n数据库模块测试完成！")


if __name__ == "__main__":
    main()