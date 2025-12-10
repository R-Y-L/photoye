#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - 本地智能照片管理助手
主程序入口和用户界面

版本: 1.0
日期: 2025年08月14日
"""

import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QMenuBar, QStatusBar, QLabel, QSplitter, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon
from database import init_db
from worker import ScanWorker


class PhotoyeMainWindow(QMainWindow):
    """Photoye主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.scan_worker = None
        self.init_ui()
        self.init_database()
    
    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口基本属性
        self.setWindowTitle("Photoye - 本地智能照片管理助手")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局 - 水平分割器
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(main_splitter)
        
        # 左侧导航面板
        self.nav_panel = self.create_nav_panel()
        main_splitter.addWidget(self.nav_panel)
        
        # 右侧照片展示区
        self.photo_display = self.create_photo_display()
        main_splitter.addWidget(self.photo_display)
        
        # 设置分割器比例 (导航:展示 = 1:3)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.create_status_bar()
        
        # 设置窗口居中
        self.center_window()
    
    def create_nav_panel(self):
        """创建左侧导航面板"""
        nav_widget = QWidget()
        nav_widget.setFixedWidth(300)
        nav_widget.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-right: 1px solid #ddd;
            }
        """)
        
        layout = QVBoxLayout(nav_widget)
        
        # 导航标题
        nav_title = QLabel("导航与筛选")
        nav_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #e0e0e0;
                border-bottom: 1px solid #ccc;
            }
        """)
        layout.addWidget(nav_title)
        
        # 占位标签
        placeholder_label = QLabel("功能区域待开发\n\n将在后续阶段实现:\n• 视图切换\n• 人物列表\n• 分类筛选\n• 标签管理")
        placeholder_label.setStyleSheet("""
            QLabel {
                color: #666;
                padding: 20px;
                text-align: center;
            }
        """)
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder_label)
        
        layout.addStretch()
        
        return nav_widget
    
    def create_photo_display(self):
        """创建右侧照片展示区"""
        display_widget = QWidget()
        display_widget.setStyleSheet("""
            QWidget {
                background-color: white;
            }
        """)
        
        layout = QVBoxLayout(display_widget)
        
        # 展示区标题
        display_title = QLabel("照片展示区")
        display_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #f8f8f8;
                border-bottom: 1px solid #ddd;
            }
        """)
        layout.addWidget(display_title)
        
        # 占位标签
        placeholder_label = QLabel("照片展示功能待开发\n\n将在后续阶段实现:\n• 缩略图网格展示\n• 无限滚动加载\n• 详细信息查看\n• 双击高清预览")
        placeholder_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 14px;
                text-align: center;
            }
        """)
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder_label)
        
        return display_widget
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        # 选择照片库动作
        select_library_action = QAction("选择照片库(&S)", self)
        select_library_action.setShortcut("Ctrl+S")
        select_library_action.setStatusTip("选择要管理的照片文件夹")
        select_library_action.triggered.connect(self.select_library)
        file_menu.addAction(select_library_action)
        
        file_menu.addSeparator()
        
        # 退出动作
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("退出应用程序")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        
        # 图库视图动作
        gallery_action = QAction("图库视图(&G)", self)
        gallery_action.setStatusTip("切换到图库视图")
        gallery_action.triggered.connect(self.switch_to_gallery)
        view_menu.addAction(gallery_action)
        
        # 人物视图动作
        people_action = QAction("人物视图(&P)", self)
        people_action.setStatusTip("切换到人物管理视图")
        people_action.triggered.connect(self.switch_to_people)
        view_menu.addAction(people_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具(&T)")
        
        # 数据库信息动作
        db_info_action = QAction("数据库信息(&D)", self)
        db_info_action.setStatusTip("查看数据库状态信息")
        db_info_action.triggered.connect(self.show_db_info)
        tools_menu.addAction(db_info_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        # 关于动作
        about_action = QAction("关于 Photoye(&A)", self)
        about_action.setStatusTip("关于本应用程序")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 默认状态消息
        self.status_bar.showMessage("就绪 - 欢迎使用 Photoye 本地智能照片管理助手", 0)
    
    def center_window(self):
        """将窗口居中显示"""
        screen = QApplication.primaryScreen().availableGeometry()
        window = self.frameGeometry()
        center_point = screen.center()
        window.moveCenter(center_point)
        self.move(window.topLeft())
    
    def init_database(self):
        """初始化数据库"""
        try:
            init_db()
            self.status_bar.showMessage("数据库初始化成功", 3000)
        except Exception as e:
            self.status_bar.showMessage(f"数据库初始化失败: {str(e)}", 5000)
            print(f"数据库初始化错误: {e}")
    
    def select_library(self):
        """选择照片库"""
        # 打开目录选择对话框
        directory = QFileDialog.getExistingDirectory(
            self, 
            "选择照片库目录", 
            "", 
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
        )
        
        if directory:
            self.status_bar.showMessage(f"开始扫描目录: {directory}", 3000)
            self.start_scan(directory)
    
    def start_scan(self, directory):
        """开始扫描指定目录"""
        # 创建并启动扫描工作线程
        self.scan_worker = ScanWorker(directory)
        
        # 连接信号
        self.scan_worker.progress_updated.connect(self.on_scan_progress)
        self.scan_worker.file_found.connect(self.on_file_found)
        self.scan_worker.scan_completed.connect(self.on_scan_completed)
        self.scan_worker.error_occurred.connect(self.on_scan_error)
        
        # 启动线程
        self.scan_worker.start()
        
        self.status_bar.showMessage("正在扫描文件...")
    
    def on_scan_progress(self, current, total):
        """处理扫描进度更新"""
        self.status_bar.showMessage(f"已扫描 {current}/{total} 个文件")
    
    def on_file_found(self, filepath):
        """处理发现新文件"""
        filename = os.path.basename(filepath)
        print(f"发现新文件: {filename}")
    
    def on_scan_completed(self, total_files):
        """处理扫描完成"""
        self.status_bar.showMessage(f"扫描完成，共处理 {total_files} 个文件", 5000)
        self.scan_worker = None
    
    def on_scan_error(self, error_msg):
        """处理扫描错误"""
        self.status_bar.showMessage(f"扫描错误: {error_msg}", 5000)
        self.scan_worker = None
    
    def switch_to_gallery(self):
        """切换到图库视图(占位函数)"""
        self.status_bar.showMessage("图库视图功能将在阶段4实现", 3000)
    
    def switch_to_people(self):
        """切换到人物视图(占位函数)"""
        self.status_bar.showMessage("人物视图功能将在阶段5实现", 3000)
    
    def show_db_info(self):
        """显示数据库信息(占位函数)"""
        self.status_bar.showMessage("数据库信息功能待实现", 3000)
    
    def show_about(self):
        """显示关于信息(占位函数)"""
        from PyQt6.QtWidgets import QMessageBox
        
        QMessageBox.about(self, "关于 Photoye", 
            """
            <h3>Photoye - 本地智能照片管理助手</h3>
            <p><b>版本:</b> 1.0 (阶段1)</p>
            <p><b>日期:</b> 2025年08月14日</p>
            <br>
            <p>一款以隐私保护为核心、运行于本地的、智能化的照片与视频管理工具。</p>
            <br>
            <p><b>核心原则:</b></p>
            <p>• 本地优先 (Local-First)</p>
            <p>• 隐私至上 (Privacy-First)</p>
            <p>• 非破坏性操作 (Non-Destructive)</p>
            <p>• 用户友好 (User-Friendly)</p>
            """)


def main():
    """主函数"""
    # 创建QApplication实例
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("Photoye")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Photoye Team")
    
    # 创建并显示主窗口
    window = PhotoyeMainWindow()
    window.show()
    
    # 启动事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main()