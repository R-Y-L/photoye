#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye 阶段3功能测试脚本
用于测试完整分析流水线功能
"""

import sys
import os
import tempfile
import shutil
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db, add_photo, get_all_photos, get_faces_by_photo_id
from worker import ScanWorker
from PyQt6.QtCore import QCoreApplication


def create_test_directory():
    """创建测试目录和文件"""
    # 创建临时目录
    test_dir = tempfile.mkdtemp(prefix="photoye_stage3_test_")
    
    # 创建一些测试图片文件
    test_files = ["photo1.jpg", "photo2.png", "photo3.jpeg"]
    for filename in test_files:
        filepath = os.path.join(test_dir, filename)
        with open(filepath, 'w') as f:
            f.write("fake image data")
    
    return test_dir


def test_complete_analysis_pipeline():
    """测试完整分析流水线"""
    print("测试完整分析流水线...")
    
    # 创建Qt应用
    app = QCoreApplication([])
    
    # 初始化数据库
    init_db()
    print("✓ 数据库初始化成功")
    
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
        print(f"  发现并开始分析文件: {os.path.basename(filepath)}")
    
    def on_scan_completed(total_files):
        scan_results['completed'] = True
        scan_results['total_files'] = total_files
        print(f"  扫描和分析完成，共处理 {total_files} 个文件")
        app.quit()
    
    def on_error(error_msg):
        print(f"  扫描错误: {error_msg}")
        app.quit()
    
    worker.progress_updated.connect(on_progress)
    worker.file_found.connect(on_file_found)
    worker.scan_completed.connect(on_scan_completed)
    worker.error_occurred.connect(on_error)
    
    # 启动扫描和分析
    worker.start()
    
    # 运行事件循环直到扫描完成
    app.exec()
    
    # 验证结果
    assert scan_results['completed'], "扫描应该完成"
    assert len(scan_results['files_found']) == 3, "应该发现3个测试文件"
    assert len(scan_results['progress_updates']) > 0, "应该有进度更新"
    
    # 检查数据库中的分析结果
    photos = get_all_photos()
    analyzed_photos = [p for p in photos if p['status'] == 'done']
    print(f"  数据库中有 {len(analyzed_photos)} 张已完成分析的照片")
    
    # 检查人脸数据
    total_faces = 0
    for photo in analyzed_photos:
        photo_id = photo['id']
        faces = get_faces_by_photo_id(photo_id)
        total_faces += len(faces)
        print(f"  照片 '{os.path.basename(photo['filepath'])}' 检测到 {len(faces)} 个人脸")
    
    print(f"  总共检测到 {total_faces} 个人脸")
    
    # 清理测试目录
    shutil.rmtree(test_dir)
    print("✓ 测试目录清理完成")
    
    print("✓ 完整分析流水线功能正常")


def main():
    """主测试函数"""
    print("=" * 60)
    print("Photoye 阶段3功能测试")
    print("=" * 60)
    
    try:
        test_complete_analysis_pipeline()
        
        print("\n" + "=" * 60)
        print("所有阶段3测试通过! 完整分析流水线功能正常。")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())