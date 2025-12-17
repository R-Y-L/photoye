#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - 本地智能照片管理助手
主程序入口和用户界面

版本: 1.0 (阶段4)
"""

import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QMenuBar, QStatusBar, QLabel, QSplitter, 
                             QFileDialog, QListWidget, QListWidgetItem, QPushButton,
                             QButtonGroup, QGroupBox, QGridLayout, QLineEdit, QComboBox,
                             QMessageBox)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QPixmap, QImage
from database import (
    init_db,
    get_all_photos,
    get_faces_by_photo_id,
    get_or_create_person,
    assign_faces_to_person,
    set_photo_category,
)
from worker import ScanWorker


class PhotoyeMainWindow(QMainWindow):
    """Photoye主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.scan_worker = None
        self.current_filter = None
        self.current_library_path = None
        self.init_ui()
        self.init_database()
        self.load_photos()
    
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
        
        # 当前库信息
        library_info_group = QGroupBox("当前照片库")
        library_info_layout = QVBoxLayout(library_info_group)
        
        self.library_path_label = QLabel("未选择照片库")
        self.library_path_label.setWordWrap(True)
        self.library_path_label.setStyleSheet("""
            QLabel {
                padding: 5px;
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
        library_info_layout.addWidget(self.library_path_label)
        
        # 添加选择照片库按钮
        select_library_btn = QPushButton("选择照片库")
        select_library_btn.clicked.connect(self.select_library)
        library_info_layout.addWidget(select_library_btn)
        
        layout.addWidget(library_info_group)
        
        # 筛选器区域
        filter_group = QGroupBox("筛选器")
        filter_layout = QVBoxLayout(filter_group)
        
        # 全部照片按钮
        all_photos_btn = QPushButton("全部照片")
        all_photos_btn.clicked.connect(lambda: self.filter_photos(None))
        filter_layout.addWidget(all_photos_btn)
        
        # 分类筛选按钮
        categories = ["单人照", "合照", "风景", "建筑", "动物", "室内", "美食", "文档"]
        for category in categories:
            btn = QPushButton(category)
            btn.clicked.connect(lambda checked, c=category: self.filter_photos(c))
            filter_layout.addWidget(btn)
        
        layout.addWidget(filter_group)

        layout.addWidget(self.create_filter_tag_panel())
        
        # 统计信息
        self.stats_label = QLabel("照片总数: 0\n已分析: 0\n待处理: 0")
        self.stats_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.stats_label)
        
        layout.addStretch()
        
        return nav_widget

    def create_filter_tag_panel(self):
        """筛选和标记区域，集中分类过滤和人脸命名。"""
        panel = QGroupBox("分类筛选 / 标记")
        box = QVBoxLayout(panel)

        # 分类筛选
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("全部", userData=None)
        for cat in ["单人照", "合照", "风景", "建筑", "动物", "室内", "美食", "文档"]:
            self.filter_combo.addItem(cat, userData=cat)
        apply_filter_btn = QPushButton("应用筛选")
        apply_filter_btn.clicked.connect(self.apply_filter_from_combo)

        box.addWidget(QLabel("按分类筛选"))
        box.addWidget(self.filter_combo)
        box.addWidget(apply_filter_btn)

        # 分类修改
        self.set_category_combo = QComboBox()
        for cat in ["单人照", "合照", "风景", "建筑", "动物", "室内", "美食", "文档"]:
            self.set_category_combo.addItem(cat, userData=cat)
        set_cat_btn = QPushButton("将选中照片设为此分类")
        set_cat_btn.clicked.connect(self.update_selected_photo_category)

        box.addWidget(QLabel("手动修改分类"))
        box.addWidget(self.set_category_combo)
        box.addWidget(set_cat_btn)

        # 人脸标记
        self.person_input = QLineEdit()
        self.person_input.setPlaceholderText("输入人物名称")
        tag_btn = QPushButton("标记选中照片的人脸")
        tag_btn.clicked.connect(self.tag_faces_for_selection)

        box.addWidget(QLabel("人脸识别标记"))
        box.addWidget(self.person_input)
        box.addWidget(tag_btn)

        return panel
    
    def create_photo_display(self):
        """创建右侧照片展示区"""
        display_widget = QWidget()
        display_widget.setStyleSheet("""
            QWidget {
                background-color: white;
            }
        """)
        
        layout = QVBoxLayout(display_widget)
        
        # 展示区标题和工具栏
        header_layout = QHBoxLayout()
        
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
        header_layout.addWidget(display_title)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_photos)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # 照片列表
        self.photo_list = QListWidget()
        self.photo_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.photo_list.setIconSize(QSize(150, 150))
        self.photo_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.photo_list.setMovement(QListWidget.Movement.Static)
        self.photo_list.setSpacing(10)
        layout.addWidget(self.photo_list)
        
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
    
    def load_photos(self, category=None):
        """加载照片到界面"""
        # 获取照片数据
        if self.current_library_path:
            # 如果指定了当前库路径，则只加载该路径下的照片
            photos = get_all_photos(category=category, library_path=self.current_library_path)
        else:
            photos = get_all_photos(category=category)
        
        # 清空当前列表
        self.photo_list.clear()
        
        # 添加照片到列表
        for photo in photos:
            item = QListWidgetItem()
            item.setText(os.path.basename(photo['filepath']))
            
            # 创建缩略图
            pixmap = self.create_thumbnail(photo['filepath'])
            if pixmap:
                item.setIcon(QIcon(pixmap))
            else:
                # 如果无法创建缩略图，使用颜色占位符
                placeholder = QPixmap(150, 150)
                if photo['category'] == '风景':
                    placeholder.fill(Qt.GlobalColor.blue)
                elif photo['category'] == '单人照':
                    placeholder.fill(Qt.GlobalColor.magenta)
                elif photo['category'] == '合照':
                    placeholder.fill(Qt.GlobalColor.yellow)
                else:
                    placeholder.fill(Qt.GlobalColor.gray)
                item.setIcon(QIcon(placeholder))
                
            item.setData(Qt.ItemDataRole.UserRole, photo)  # 存储照片数据
            self.photo_list.addItem(item)
        
        # 更新统计信息
        self.update_stats()
    
    def create_thumbnail(self, image_path):
        """创建照片缩略图"""
        try:
            if not os.path.exists(image_path):
                return None
                
            # 尝试创建真实的缩略图
            image = QImage(image_path)
            if image.isNull():
                return None
                
            # 缩放到合适尺寸
            thumbnail = image.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, 
                                   Qt.TransformationMode.SmoothTransformation)
            return QPixmap.fromImage(thumbnail)
        except Exception as e:
            print(f"创建缩略图失败: {e}")
            return None
    
    def update_stats(self):
        """更新统计信息"""
        from database import get_photos_count
        if self.current_library_path:
            stats = get_photos_count(library_path=self.current_library_path)
        else:
            stats = get_photos_count()
        
        stats_text = f"照片总数: {stats.get('total', 0)}\n"
        stats_text += f"已分析: {stats.get('status', {}).get('done', 0)}\n"
        stats_text += f"待处理: {stats.get('status', {}).get('pending', 0)}"
        
        # 添加处理中状态的统计
        processing_count = stats.get('status', {}).get('processing', 0)
        if processing_count > 0:
            stats_text += f"\n处理中: {processing_count}"
        
        # 添加人脸和人物统计
        faces_count = stats.get('faces', 0)
        persons_count = stats.get('persons', 0)
        stats_text += f"\n人脸数: {faces_count}"
        stats_text += f"\n人物数: {persons_count}"
        
        self.stats_label.setText(stats_text)
    
    def filter_photos(self, category):
        """根据分类筛选照片"""
        self.current_filter = category
        self.load_photos(category)
        if category:
            self.status_bar.showMessage(f"筛选: {category}", 3000)
        else:
            self.status_bar.showMessage("显示全部照片", 3000)

    def apply_filter_from_combo(self):
        """从筛选下拉框应用分类过滤。"""
        category = self.filter_combo.currentData()
        self.filter_photos(category)
    
    def refresh_photos(self):
        """刷新照片显示"""
        self.load_photos(self.current_filter)
        self.status_bar.showMessage("照片列表已刷新", 3000)

    def _selected_photo_items(self):
        return self.photo_list.selectedItems() or []

    def _selected_photo_ids(self):
        ids = []
        for item in self._selected_photo_items():
            photo = item.data(Qt.ItemDataRole.UserRole)
            if photo and "id" in photo:
                ids.append(photo["id"])
        return ids

    def update_selected_photo_category(self):
        """将选中照片的分类更新为下拉框选择的值。"""
        category = self.set_category_combo.currentData()
        photo_ids = self._selected_photo_ids()
        if not photo_ids:
            QMessageBox.information(self, "提示", "请先选中照片，再修改分类。")
            return

        updated = 0
        for pid in photo_ids:
            if set_photo_category(pid, category):
                updated += 1
        self.status_bar.showMessage(f"已更新 {updated} 张照片的分类为 {category}", 5000)
        self.refresh_photos()

    def tag_faces_for_selection(self):
        """将当前选中照片中的人脸关联到指定人物。"""
        name = self.person_input.text().strip()
        if not name:
            QMessageBox.information(self, "提示", "请先输入人物名称。")
            return

        photo_ids = self._selected_photo_ids()
        if not photo_ids:
            QMessageBox.information(self, "提示", "请先选中包含人脸的照片。")
            return

        person_id = get_or_create_person(name)
        if person_id is None:
            QMessageBox.warning(self, "错误", "无法创建或获取人物条目。")
            return

        tagged = 0
        for pid in photo_ids:
            faces = get_faces_by_photo_id(pid)
            face_ids = [f["id"] for f in faces]
            tagged += assign_faces_to_person(face_ids, person_id)

        self.status_bar.showMessage(f"已为 {len(photo_ids)} 张照片的 {tagged} 张人脸标记为 {name}", 5000)
        self.person_input.clear()
    
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
            self.current_library_path = directory
            self.library_path_label.setText(directory)
            self.status_bar.showMessage(f"开始扫描目录: {directory}", 3000)
            self.start_scan(directory)
    
    def start_scan(self, directory):
        """开始扫描指定目录"""
        # 更新当前库路径
        self.current_library_path = directory
        self.library_path_label.setText(directory)
        
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
        # 重新加载照片
        self.load_photos(self.current_filter)
    
    def on_scan_error(self, error_msg):
        """处理扫描错误"""
        self.status_bar.showMessage(f"扫描错误: {error_msg}", 5000)
        self.scan_worker = None
    
    def switch_to_gallery(self):
        """切换到图库视图"""
        self.status_bar.showMessage("已切换到图库视图", 3000)
    
    def switch_to_people(self):
        """切换到人物视图(占位函数)"""
        self.status_bar.showMessage("人物视图功能将在阶段5实现", 3000)
    
    def show_db_info(self):
        """显示数据库信息"""
        # 更新统计信息显示
        self.update_stats()
        self.status_bar.showMessage("数据库信息已更新", 3000)
    
    def show_about(self):
        """显示关于信息"""
        from PyQt6.QtWidgets import QMessageBox
        
        QMessageBox.about(self, "关于 Photoye", 
            """
            <h3>Photoye - 本地智能照片管理助手</h3>
            <p><b>版本:</b> 1.0 (阶段4)</p>
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
