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
                             QMessageBox, QDialog, QVBoxLayout as QVBox, QHBoxLayout as QHBox, 
                             QScrollArea, QCheckBox)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QPixmap, QImage
from database import (
    init_db,
    get_all_photos,
    get_faces_by_photo_id,
    get_or_create_person,
    assign_faces_to_person,
    set_photo_category,
    list_persons,
    get_unlabeled_faces,
    cleanup_on_exit,
    clear_temp_photos,
)
from worker import ScanWorker


class PhotoyeMainWindow(QMainWindow):
    """Photoye主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.scan_worker = None
        self.current_filter = None
        self.current_library_path = None
        self.pending_face_naming = False
        self.selected_model_profile = None
        # 在启动时清空上次的临时照片数据
        clear_temp_photos()
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
        """创建左侧导航面板（带滚动）"""
        # 外层容器
        nav_container = QWidget()
        nav_container.setFixedWidth(300)
        nav_container.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-right: 1px solid #ddd;
            }
        """)
        
        container_layout = QVBoxLayout(nav_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        # 导航标题（固定不滚动）
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
        container_layout.addWidget(nav_title)
        
        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f5f5f5;
            }
        """)
        
        # 滚动内容
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: #f5f5f5;")
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(5, 5, 5, 5)
        
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

        # 统一的筛选与标记面板
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
        
        scroll.setWidget(scroll_content)
        container_layout.addWidget(scroll)
        
        return nav_container

    def create_filter_tag_panel(self):
        """筛选和标记区域，支持分类/人脸两种模式，并提供命名入口。"""
        panel = QGroupBox("筛选与标记")
        box = QVBoxLayout(panel)

        # 筛选模型/模式
        self.filter_mode_combo = QComboBox()
        self.filter_mode_combo.addItem("全部", userData={"mode": "all"})
        self.filter_mode_combo.addItem("按分类", userData={"mode": "category"})
        self.filter_mode_combo.addItem("按人脸", userData={"mode": "person"})
        self.filter_mode_combo.currentIndexChanged.connect(self._on_filter_mode_changed)

        box.addWidget(QLabel("筛选类型"))
        box.addWidget(self.filter_mode_combo)

        # 分类筛选控件（改为多选）
        self.filter_category_checks = {}
        category_list = ["单人照", "合照", "风景", "建筑", "动物", "室内", "美食", "文档"]
        for cat in category_list:
            cb = QCheckBox(cat)
            self.filter_category_checks[cat] = cb
            box.addWidget(cb)
        
        box.addWidget(QLabel("按分类"))

        # 人脸筛选控件
        self.filter_person_combo = QComboBox()
        self.refresh_person_filter_options()
        refresh_person_btn = QPushButton("刷新人物列表")
        refresh_person_btn.clicked.connect(self.refresh_person_filter_options)

        self.ensure_faces_btn = QPushButton("扫描并更新人脸索引")
        self.ensure_faces_btn.clicked.connect(self.ensure_faces_indexed)

        self.person_category_filter = QComboBox()
        self.person_category_filter.addItem("不限分类", userData=None)
        self.person_category_filter.addItem("仅合照", userData="合照")
        self.person_category_filter.addItem("仅单人照", userData="单人照")

        box.addWidget(QLabel("按人脸"))
        box.addWidget(self.filter_person_combo)
        box.addWidget(refresh_person_btn)
        box.addWidget(self.ensure_faces_btn)
        box.addWidget(QLabel("附加分类"))
        box.addWidget(self.person_category_filter)

        # 模型选择与应用
        self.model_profile_combo = QComboBox()
        self.model_profile_combo.addItem("平衡 (balanced)", userData="balanced")
        self.model_profile_combo.addItem("快速 (speed)", userData="speed")
        self.model_profile_combo.addItem("高精度 (accuracy)", userData="accuracy")
        self.model_profile_combo.addItem("零样本 (zeroshot)", userData="zeroshot")
        apply_model_btn = QPushButton("应用模型并重新分析")
        apply_model_btn.clicked.connect(self.apply_model_and_rescan)

        box.addWidget(QLabel("选择分析模型"))
        box.addWidget(self.model_profile_combo)
        box.addWidget(apply_model_btn)

        # 应用筛选
        apply_filter_btn = QPushButton("应用筛选")
        apply_filter_btn.clicked.connect(self.apply_filter)
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

        box.addWidget(QLabel("人脸命名/标记"))
        box.addWidget(self.person_input)
        box.addWidget(tag_btn)

        # 初始化控件可用性
        self._on_filter_mode_changed()

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
        
        self.photo_list.itemClicked.connect(self.on_photo_item_clicked)
        
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
    
    def load_photos(self, filter_spec=None):
        """加载照片到界面，支持分类/人脸等筛选。"""
        filter_spec = filter_spec or self.current_filter

        if isinstance(filter_spec, dict):
            mode = filter_spec.get("mode")
        else:
            mode = None

        categories = None  # 改为 categories（列表）
        person_id = None
        has_faces = None
        unlabeled_faces = False

        if mode == "category":
            categories = filter_spec.get("categories")  # 获取列表
        elif mode == "person":
            person_id = filter_spec.get("person_id")
            unlabeled_faces = filter_spec.get("unlabeled", False)
            category = filter_spec.get("category")
            if filter_spec.get("any_face"):
                has_faces = True
            elif person_id is None and not unlabeled_faces:
                has_faces = True

        # 获取照片数据
        if self.current_library_path:
            photos = get_all_photos(
                categories=categories,
                library_path=self.current_library_path,
                person_id=person_id,
                has_faces=has_faces,
                unlabeled_faces=unlabeled_faces,
            )
        else:
            photos = get_all_photos(
                categories=categories,
                person_id=person_id,
                has_faces=has_faces,
                unlabeled_faces=unlabeled_faces,
            )
        
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
    
    def _on_filter_mode_changed(self):
        """切换筛选模式时同步控件状态。"""
        mode_data = self.filter_mode_combo.currentData() or {"mode": "all"}
        mode = mode_data.get("mode")
        # 启用/禁用分类检查框
        for cb in self.filter_category_checks.values():
            cb.setEnabled(mode == "category")
        self.filter_person_combo.setEnabled(mode == "person")
        self.ensure_faces_btn.setEnabled(mode == "person")
        self.person_category_filter.setEnabled(mode == "person")

    def refresh_person_filter_options(self):
        """刷新人物下拉框，含未命名项。"""
        current_value = None
        if hasattr(self, "filter_person_combo"):
            current_value = self.filter_person_combo.currentData()

        self.filter_person_combo.clear()
        self.filter_person_combo.addItem("全部含人脸", userData="__any_face__")
        self.filter_person_combo.addItem("未命名人脸", userData="__unlabeled__")

        persons = list_persons()
        for p in persons:
            label = f"{p.get('name', '未命名')} ({p.get('photo_count', 0)} 张)"
            self.filter_person_combo.addItem(label, userData=p.get("id"))

        # 尝试恢复先前选择
        if current_value is not None:
            index = self.filter_person_combo.findData(current_value)
            if index >= 0:
                self.filter_person_combo.setCurrentIndex(index)

    def apply_filter(self):
        """根据当前模式应用筛选。"""
        mode_data = self.filter_mode_combo.currentData() or {"mode": "all"}
        mode = mode_data.get("mode")

        if mode == "category":
            # 获取所有被选中的分类
            selected_categories = [cat for cat, cb in self.filter_category_checks.items() if cb.isChecked()]
            self.current_filter = {"mode": "category", "categories": selected_categories if selected_categories else None}
            self.load_photos(self.current_filter)
            msg = f"筛选分类: {',  '.join(selected_categories) if selected_categories else '全部'}"
        elif mode == "person":
            person_value = self.filter_person_combo.currentData()
            person_category = self.person_category_filter.currentData()
            if person_value == "__unlabeled__":
                self.current_filter = {"mode": "person", "unlabeled": True, "category": person_category}
                msg = "筛选: 未命名人脸"
            elif person_value == "__any_face__":
                self.current_filter = {"mode": "person", "any_face": True, "category": person_category}
                msg = "筛选: 任意含人脸照片"
            else:
                self.current_filter = {"mode": "person", "person_id": person_value, "category": person_category}
                msg = f"筛选人物: {self.filter_person_combo.currentText()}"
            self.load_photos(self.current_filter)
        else:
            self.current_filter = None
            self.load_photos(None)
            msg = "显示全部照片"

        self.status_bar.showMessage(msg, 3000)

    def apply_model_and_rescan(self):
        """按照当前筛选模式选择模型后重新分析，并在按人脸时触发命名流程。"""
        mode_data = self.filter_mode_combo.currentData() or {"mode": "all"}
        mode = mode_data.get("mode")
        self.selected_model_profile = self.model_profile_combo.currentData()

        if not self.current_library_path:
            QMessageBox.information(self, "提示", "请先选择照片库，再应用模型重新分析。")
            return

        # 按人脸模式时，扫描完成后需要提示命名
        self.pending_face_naming = mode == "person"
        
        # 传递 analyze=True 让扫描线程执行分析
        self.status_bar.showMessage(f"使用模型 {self.selected_model_profile or '默认'} 进行分析...", 4000)
        self.start_scan(self.current_library_path, analyze=True)

    def ensure_faces_indexed(self):
        """在按人脸筛选时触发扫描，确保所有照片有人脸索引。"""
        if not self.current_library_path:
            QMessageBox.information(self, "提示", "请先选择照片库，然后再扫描人脸索引。")
            return

        if self.scan_worker and self.scan_worker.isRunning():
            QMessageBox.information(self, "提示", "正在扫描中，请稍候完成后再试。")
            return

        self.status_bar.showMessage("重新扫描以更新人脸索引...", 3000)
        self.start_scan(self.current_library_path)

    def prompt_name_unlabeled_faces(self):
        """弹出未命名人脸命名对话框，将识别出的不同人脸命名。"""
        faces = get_unlabeled_faces(limit=30)
        if not faces:
            QMessageBox.information(self, "提示", "没有未命名的人脸需要标记。")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("未命名人脸标记")
        dialog.resize(520, 640)

        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vbox = QVBoxLayout(container)

        entries = []

        for idx, face in enumerate(faces, start=1):
            row = QWidget()
            row_layout = QHBox()
            row.setLayout(row_layout)

            thumb = self._face_thumbnail(face)
            thumb_label = QLabel()
            thumb_label.setPixmap(thumb)
            thumb_label.setFixedSize(120, 120)
            thumb_label.setScaledContents(True)

            name_edit = QLineEdit()
            name_edit.setPlaceholderText(f"人物{idx}")

            row_layout.addWidget(thumb_label)
            row_layout.addWidget(name_edit)
            vbox.addWidget(row)

            entries.append((face, name_edit))

        container.setLayout(vbox)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        buttons = QWidget()
        buttons_layout = QHBox()
        buttons.setLayout(buttons_layout)
        ok_btn = QPushButton("提交并标记")
        cancel_btn = QPushButton("取消")
        buttons_layout.addWidget(ok_btn)
        buttons_layout.addWidget(cancel_btn)
        layout.addWidget(buttons)

        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            assigned = 0
            for face, edit in entries:
                name = edit.text().strip()
                if not name:
                    continue
                person_id = get_or_create_person(name)
                if person_id:
                    assigned += assign_faces_to_person([face["id"]], person_id)
            self.refresh_person_filter_options()
            self.status_bar.showMessage(f"已标记 {assigned} 张人脸", 4000)
            self.load_photos(self.current_filter)

    def _face_thumbnail(self, face_record):
        """根据照片和bbox裁剪生成人脸缩略图。"""
        filepath = face_record.get("filepath")
        bbox = face_record.get("bbox", [0, 0, 120, 120])
        if not filepath or not os.path.exists(filepath):
            placeholder = QPixmap(120, 120)
            placeholder.fill(Qt.GlobalColor.lightGray)
            return placeholder

        image = QImage(filepath)
        if image.isNull():
            placeholder = QPixmap(120, 120)
            placeholder.fill(Qt.GlobalColor.lightGray)
            return placeholder

        x1, y1, x2, y2 = bbox
        w = max(20, x2 - x1)
        h = max(20, y2 - y1)
        cropped = image.copy(x1, y1, w, h)
        thumb = cropped.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        return QPixmap.fromImage(thumb)
    
    def on_photo_item_clicked(self, item: QListWidgetItem):
        """点击照片项时，在状态栏显示详细分类信息。"""
        photo_data = item.data(Qt.ItemDataRole.UserRole)
        if not photo_data:
            return

        photo_id = photo_data.get("id")
        from database import get_raw_classification_for_photo
        
        # 这是一个假设的函数，需要你在database.py中实现
        # 它应该从一个新表或字段中获取存储的原始分类结果
        # 这里我们暂时用一个模拟数据
        # raw_scores = get_raw_classification_for_photo(photo_id) 
        
        # 暂时无法获取原始分类，先显示已知信息
        category = photo_data.get('category', 'N/A')
        filepath = photo_data.get('filepath', 'N/A')
        
        # 模拟一个原始分类信息
        # 在真实场景中，你需要从数据库读取分析时保存的原始分类字典
        import random
        categories = ["风景", "建筑", "动物", "文档", "室内", "美食", "单人照", "合照"]
        mock_scores = {cat: round(random.random(), 2) for cat in categories}
        
        # 找到分数最高的几个
        top_3 = sorted(mock_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # 格式化显示信息
        details = ", ".join([f"{cat}: {score:.2f}" for cat, score in top_3])
        
        self.status_bar.showMessage(f"'{os.path.basename(filepath)}' - 最终分类: {category} | 模型原始判断(模拟): {details}", 10000)


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
    
    def start_scan(self, directory, analyze: bool = False):
        """开始扫描指定目录"""
        # 更新当前库路径
        self.current_library_path = directory
        self.library_path_label.setText(directory)

        # 创建并启动扫描工作线程，带上所选模型档位和分析标志
        self.scan_worker = ScanWorker(directory, model_profile=self.selected_model_profile, analyze=analyze)
        
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
        # 若按人脸模式请求过模型分析，则提示命名未命名人脸
        if self.pending_face_naming:
            self.pending_face_naming = False
            self.prompt_name_unlabeled_faces()
    
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

    def closeEvent(self, event):
        """窗口关闭事件，清理临时数据。"""
        cleanup_on_exit()
        event.accept()


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
