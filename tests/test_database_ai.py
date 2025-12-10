#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye 数据库AI功能测试脚本
用于测试AI分析结果的数据库存储功能
"""

import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db, add_photo, add_face_data, get_faces_by_photo_id


def test_ai_database_functions():
    """测试AI分析结果的数据库功能"""
    print("测试AI分析结果的数据库功能...")
    
    # 初始化数据库
    init_db()
    print("✓ 数据库初始化成功")
    
    # 添加测试照片
    test_photo_path = "/fake/path/ai_test_photo.jpg"
    photo_id = add_photo(test_photo_path)
    assert photo_id is not None, "应该成功添加照片"
    print(f"✓ 添加测试照片，ID: {photo_id}")
    
    # 添加人脸数据
    test_bbox = [100, 150, 300, 400]
    test_embedding = np.random.rand(512).astype(np.float32)
    test_confidence = 0.95
    
    face_id = add_face_data(photo_id, test_bbox, test_embedding, test_confidence)
    assert face_id is not None, "应该成功添加人脸数据"
    print(f"✓ 添加人脸数据，ID: {face_id}")
    
    # 获取人脸数据
    faces = get_faces_by_photo_id(photo_id)
    assert len(faces) == 1, "应该能获取到一个人脸记录"
    
    face = faces[0]
    assert face['photo_id'] == photo_id, "照片ID应该匹配"
    assert face['bbox'] == test_bbox, "边界框应该匹配"
    assert np.allclose(face['embedding'], test_embedding), "特征向量应该匹配"
    assert abs(face['confidence'] - test_confidence) < 1e-6, "置信度应该匹配"
    
    print("✓ 人脸数据存储和检索功能正常")
    print(f"  - 照片ID: {face['photo_id']}")
    print(f"  - 边界框: {face['bbox']}")
    print(f"  - 置信度: {face['confidence']}")
    print(f"  - 特征向量维度: {face['embedding'].shape}")


def main():
    """主测试函数"""
    print("=" * 60)
    print("Photoye 数据库AI功能测试")
    print("=" * 60)
    
    try:
        test_ai_database_functions()
        
        print("\n" + "=" * 60)
        print("所有数据库AI功能测试通过!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())