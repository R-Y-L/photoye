#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye 阶段1功能测试脚本
用于测试文件索引与数据库功能
"""

import sys
import os
import tempfile
import shutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db, add_photo, get_all_photos, is_photo_exist
from worker import ScanWorker
from PyQt6.QtCore import QCoreApplication


def create_test_directory():
    """创建测试目录和文件"""
    # 创建临时目录
    test_dir = tempfile.mkdtemp(prefix="photoye_test_")
    
    # 创建一些测试图片文件
    test_files = ["test1.jpg", "test2.png", "test3.jpeg"]
    for filename in test_files:
        filepath = os.path.join(test_dir, filename)
        with open(filepath, 'w') as f:
            f.write("fake image data")
    
    # 创建子目录和更多文件
    subdir = os.path.join(test_dir, "subdir")
    os.makedirs(subdir, exist_ok=True)
    subdir_files = ["sub_test1.jpg", "sub_test2.png"]
    for filename in subdir_files:
        filepath = os.path.join(subdir, filename)
        with open(filepath, 'w') as f:
            f.write("fake image data")
    
    return test_dir


def test_database_operations():
    """测试数据库操作"""
    print("测试数据库操作...")
    
    # 初始化数据库
    init_db()
    print("✓ 数据库初始化成功")
    
    # 添加测试照片
    test_photos = [
        "/fake/path/photo1.jpg",
        "/fake/path/photo2.png", 
        "/fake/path/subdir/photo3.jpeg"
    ]
    
    added_ids = []
    for photo_path in test_photos:
        photo_id = add_photo(photo_path)
        if photo_id is not None:
            added_ids.append(photo_id)
    
    print(f"✓ 添加了 {len(added_ids)} 张照片到数据库")
    
    # 检查照片是否存在
    for photo_path in test_photos:
        exists = is_photo_exist(photo_path)
        assert exists, f"照片应该存在于数据库中: {photo_path}"
    
    print("✓ 照片存在性检查通过")
    
    # 获取所有照片
    all_photos = get_all_photos()
    assert len(all_photos) >= len(test_photos), "应该能获取到所有添加的照片"
    
    print("✓ 获取所有照片功能正常")


def test_scan_worker():
    """测试扫描工作线程"""
    print("\n测试扫描工作线程...")
    
    # 创建Qt应用
    app = QCoreApplication([])
    
    # 创建测试目录
    test_dir = create_test_directory()
    print(f"创建测试目录: {test_dir}")
    
    # 创建扫描工作线程
    extensions = ['.jpg', '.jpeg', '.png']
    worker = ScanWorker(test_dir, extensions)
    
    # 记录扫描结果
    scan_results = {
        'progress_updates': [],
        'files_found': [],
        'completed': False,
        'total_files': 0
    }
    
    # 连接信号
    def on_progress(current, total):
        scan_results['progress_updates'].append((current, total))
        print(f"  扫描进度: {current}/{total}")
    
    def on_file_found(filepath):
        scan_results['files_found'].append(filepath)
        print(f"  发现文件: {os.path.basename(filepath)}")
    
    def on_scan_completed(total_files):
        scan_results['completed'] = True
        scan_results['total_files'] = total_files
        print(f"  扫描完成，共处理 {total_files} 个文件")
        app.quit()
    
    def on_error(error_msg):
        print(f"  扫描错误: {error_msg}")
        app.quit()
    
    worker.progress_updated.connect(on_progress)
    worker.file_found.connect(on_file_found)
    worker.scan_completed.connect(on_scan_completed)
    worker.error_occurred.connect(on_error)
    
    # 启动扫描
    worker.start()
    
    # 运行事件循环直到扫描完成
    app.exec()
    
    # 验证结果
    assert scan_results['completed'], "扫描应该完成"
    assert len(scan_results['files_found']) == 5, "应该发现5个测试文件"
    assert len(scan_results['progress_updates']) > 0, "应该有进度更新"
    
    # 清理测试目录
    shutil.rmtree(test_dir)
    print("✓ 测试目录清理完成")
    
    print("✓ 扫描工作线程功能正常")


def main():
    """主测试函数"""
    print("=" * 60)
    print("Photoye 阶段1功能测试")
    print("=" * 60)
    
    try:
        test_database_operations()
        test_scan_worker()
        
        print("\n" + "=" * 60)
        print("所有阶段1测试通过! 文件索引与数据库功能正常。")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())