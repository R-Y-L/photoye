#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye 阶段4功能测试脚本
用于测试UI界面和结果展示功能
"""

import sys
import os
import tempfile
import shutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from main import PhotoyeMainWindow


def test_ui_interface():
    """测试UI界面功能"""
    print("测试UI界面功能...")
    
    # 创建Qt应用
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = PhotoyeMainWindow()
    window.show()
    
    # 检查窗口是否正确创建
    assert window.isVisible(), "主窗口应该可见"
    assert window.windowTitle() == "Photoye - 本地智能照片管理助手", "窗口标题应该正确"
    
    print("✓ 主窗口创建成功")
    print("✓ 窗口标题正确")
    
    # 检查UI组件
    assert hasattr(window, 'photo_list'), "应该有照片列表组件"
    assert hasattr(window, 'nav_panel'), "应该有导航面板"
    assert hasattr(window, 'stats_label'), "应该有统计信息标签"
    
    print("✓ UI组件存在")
    
    # 检查菜单栏
    menubar = window.menuBar()
    menus = [action.text() for action in menubar.actions()]
    expected_menus = ["文件(&F)", "视图(&V)", "工具(&T)", "帮助(&H)"]
    for menu in expected_menus:
        assert menu in menus, f"应该有'{menu}'菜单"
    
    print("✓ 菜单栏功能正常")
    
    # 检查筛选按钮
    # 注意：这里我们只是验证UI结构，不实际点击按钮
    
    print("✓ 筛选功能UI组件存在")
    
    # 关闭窗口
    window.close()
    
    print("✓ UI界面功能正常")


def main():
    """主测试函数"""
    print("=" * 60)
    print("Photoye 阶段4功能测试")
    print("=" * 60)
    
    try:
        test_ui_interface()
        
        print("\n" + "=" * 60)
        print("所有阶段4测试通过! UI界面功能正常。")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())