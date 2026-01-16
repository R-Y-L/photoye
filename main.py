#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - 本地智能照片管理助手
主程序入口和用户界面

版本: 2.0 (重构版)
"""

import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QMenuBar, QStatusBar, QLabel, QSplitter, 
                             QFileDialog, QListWidget, QListWidgetItem, QPushButton,
                             QButtonGroup, QGroupBox, QGridLayout, QLineEdit, QComboBox,
                             QMessageBox, QDialog, QVBoxLayout as QVBox, QHBoxLayout as QHBox, 
                             QScrollArea, QCheckBox, QStackedWidget, QFrame, QInputDialog)
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
    clear_all_ai_data,
    get_all_persons_with_sample_faces,
    get_person_with_faces,
    get_photos_by_person,
    delete_person,
    rename_person,
)
from worker import ScanWorker, ThumbnailWorker, FaceAnalysisWorker


class PhotoyeMainWindow(QMainWindow):
    """Photoye主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.scan_worker = None
        self.face_worker = None
        self.thumbnail_worker = None
        self.current_filter = None
        self.current_library_path = None
        self.pending_face_naming = False
        self.selected_model_profile = None
        self.current_view_mode = "gallery"  # "gallery" or "people"
        
        # 缩略图缓存 {filepath: QPixmap}
        self.thumbnail_cache = {}
        
        # 在启动时清空上次的临时照片数据
        clear_temp_photos()
        self.init_ui()
        self.init_database()
        self.init_thumbnail_worker()
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
        
        # 右侧内容区 - 使用 StackedWidget 切换视图
        self.content_stack = QStackedWidget()
        
        # 图库视图
        self.photo_display = self.create_photo_display()
        self.content_stack.addWidget(self.photo_display)
        
        # 人物视图
        self.people_display = self.create_people_display()
        self.content_stack.addWidget(self.people_display)
        
        main_splitter.addWidget(self.content_stack)
        
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
        """筛选和标记区域，筛选即时响应，AI分析专门用于人脸检测。"""
        panel = QGroupBox("筛选与操作")
        box = QVBoxLayout(panel)

        # ========== 第〇部分：语义搜索（CLIP） ==========
        box.addWidget(QLabel("🔎 语义搜索"))
        
        search_layout = QHBoxLayout()
        self.semantic_search_input = QLineEdit()
        self.semantic_search_input.setPlaceholderText("输入描述，如：海边的合照、生日派对...")
        self.semantic_search_input.returnPressed.connect(self._on_semantic_search)
        search_layout.addWidget(self.semantic_search_input)
        
        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self._on_semantic_search)
        search_layout.addWidget(search_btn)
        box.addLayout(search_layout)
        
        # 搜索状态标签
        self.semantic_search_label = QLabel("")
        self.semantic_search_label.setStyleSheet("color: #666; font-size: 11px;")
        box.addWidget(self.semantic_search_label)
        
        # 清除搜索按钮
        clear_search_btn = QPushButton("清除搜索结果")
        clear_search_btn.clicked.connect(self._clear_semantic_search)
        box.addWidget(clear_search_btn)
        
        # ========== 分隔线 ==========
        separator0 = QFrame()
        separator0.setFrameShape(QFrame.Shape.HLine)
        separator0.setStyleSheet("color: #ccc;")
        box.addWidget(separator0)

        # ========== 第一部分：筛选模式（即时响应） ==========
        box.addWidget(QLabel("📂 筛选模式"))
        self.filter_mode_combo = QComboBox()
        self.filter_mode_combo.addItem("全部照片", userData="all")
        self.filter_mode_combo.addItem("按分类筛选", userData="category")
        self.filter_mode_combo.addItem("按人物筛选", userData="person")
        self.filter_mode_combo.currentIndexChanged.connect(self._on_filter_mode_changed)
        box.addWidget(self.filter_mode_combo)

        # ========== 第二部分：分类多选区域（即时响应） ==========
        self.category_group = QGroupBox("选择分类（可多选）")
        category_layout = QGridLayout(self.category_group)
        category_layout.setSpacing(6)
        
        checkbox_style = """
            QCheckBox {
                spacing: 8px;
                padding: 4px;
                font-size: 13px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #999;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #4a90d9;
                border-radius: 3px;
                background-color: #4a90d9;
            }
        """
        
        self.filter_category_checks = {}
        category_list = ["单人照", "合照", "风景", "建筑", "动物", "室内", "美食", "文档"]
        for i, cat in enumerate(category_list):
            cb = QCheckBox(cat)
            cb.setStyleSheet(checkbox_style)
            # 即时响应：勾选后立即应用筛选
            cb.stateChanged.connect(self._on_category_changed)
            self.filter_category_checks[cat] = cb
            category_layout.addWidget(cb, i // 2, i % 2)
        
        # 选中数量反馈标签
        self.category_selection_label = QLabel("已选: 0 项")
        self.category_selection_label.setStyleSheet("color: #666; font-size: 12px;")
        category_layout.addWidget(self.category_selection_label, len(category_list) // 2, 0, 1, 2)
        
        # 全选/取消全选按钮
        select_btns = QWidget()
        select_btns_layout = QHBoxLayout(select_btns)
        select_btns_layout.setContentsMargins(0, 0, 0, 0)
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(lambda: self._set_all_categories(True))
        deselect_all_btn = QPushButton("取消")
        deselect_all_btn.clicked.connect(lambda: self._set_all_categories(False))
        select_btns_layout.addWidget(select_all_btn)
        select_btns_layout.addWidget(deselect_all_btn)
        category_layout.addWidget(select_btns, len(category_list) // 2 + 1, 0, 1, 2)
        
        box.addWidget(self.category_group)

        # ========== 第三部分：人物筛选区域（即时响应） ==========
        self.person_group = QGroupBox("选择人物")
        person_layout = QVBoxLayout(self.person_group)
        
        self.filter_person_combo = QComboBox()
        self.filter_person_combo.currentIndexChanged.connect(self._on_person_changed)
        self.refresh_person_filter_options()
        person_layout.addWidget(self.filter_person_combo)
        
        refresh_person_btn = QPushButton("🔄 刷新人物列表")
        refresh_person_btn.clicked.connect(self.refresh_person_filter_options)
        person_layout.addWidget(refresh_person_btn)
        
        box.addWidget(self.person_group)

        # ========== 分隔线 ==========
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setStyleSheet("color: #ccc;")
        box.addWidget(separator1)

        # ========== 第四部分：AI 分析状态 (V2.2 自动化) ==========
        box.addWidget(QLabel("🤖 AI 分析状态"))
        
        # AI 分析说明
        auto_info = QLabel("导入照片后自动进行:\n场景分类 → 人脸检测 → 人物聚类")
        auto_info.setStyleSheet("color: #888; font-size: 10px; padding: 4px;")
        auto_info.setWordWrap(True)
        box.addWidget(auto_info)
        
        # 人脸分析状态标签
        self.face_analysis_label = QLabel("等待导入照片...")
        self.face_analysis_label.setStyleSheet("color: #666; font-size: 11px;")
        box.addWidget(self.face_analysis_label)
        
        # 聚类状态标签
        self.cluster_label = QLabel("")
        self.cluster_label.setStyleSheet("color: #666; font-size: 11px;")
        box.addWidget(self.cluster_label)
        
        # 手动重新聚类按钮（可选操作）
        recluster_btn = QPushButton("🔄 重新聚类")
        recluster_btn.setToolTip("手动触发重新聚类（用于新增人脸后）")
        recluster_btn.setStyleSheet("QPushButton { padding: 4px; font-size: 11px; }")
        recluster_btn.clicked.connect(self.run_face_clustering)
        box.addWidget(recluster_btn)

        # ========== 分隔线 ==========
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet("color: #ccc;")
        box.addWidget(separator2)

        # ========== 第五部分：手动操作 ==========
        box.addWidget(QLabel("✏️ 手动操作"))
        
        # 修改分类
        self.set_category_combo = QComboBox()
        for cat in ["单人照", "合照", "风景", "建筑", "动物", "室内", "美食", "文档"]:
            self.set_category_combo.addItem(cat, userData=cat)
        box.addWidget(self.set_category_combo)
        
        set_cat_btn = QPushButton("设为此分类")
        set_cat_btn.clicked.connect(self.update_selected_photo_category)
        box.addWidget(set_cat_btn)

        # 人脸标记
        self.person_input = QLineEdit()
        self.person_input.setPlaceholderText("输入人物名称")
        box.addWidget(self.person_input)
        
        tag_btn = QPushButton("标记人脸")
        tag_btn.clicked.connect(self.tag_faces_for_selection)
        box.addWidget(tag_btn)

        # 初始化控件可用性
        self._on_filter_mode_changed()

        return panel
    
    def _set_all_categories(self, checked: bool):
        """全选或取消全选所有分类"""
        # 暂时阻止信号，避免多次触发筛选
        for cb in self.filter_category_checks.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self._update_category_selection_label()
        self._apply_filter_immediately()
    
    def _on_category_changed(self):
        """分类勾选变化时即时应用筛选"""
        self._update_category_selection_label()
        self._apply_filter_immediately()
    
    def _on_person_changed(self):
        """人物选择变化时即时应用筛选"""
        mode = self.filter_mode_combo.currentData() or "all"
        if mode == "person":
            self._apply_filter_immediately()
    
    def _on_semantic_search(self):
        """执行语义搜索"""
        query = self.semantic_search_input.text().strip()
        if not query:
            return
        
        self.semantic_search_label.setText("搜索中...")
        self.semantic_search_label.setStyleSheet("color: #4a90d9; font-size: 11px;")
        
        # 启动语义搜索线程
        from worker import SemanticSearchWorker
        self.semantic_search_worker = SemanticSearchWorker(query, top_k=100)
        self.semantic_search_worker.search_completed.connect(self._on_semantic_search_completed)
        self.semantic_search_worker.error_occurred.connect(self._on_semantic_search_error)
        self.semantic_search_worker.start()
    
    def _on_semantic_search_completed(self, results):
        """语义搜索完成"""
        if not results:
            self.semantic_search_label.setText("未找到匹配的照片")
            self.semantic_search_label.setStyleSheet("color: #999; font-size: 11px;")
            return
        
        # 显示结果数量和相似度范围
        top_sim = results[0][2] if results else 0
        self.semantic_search_label.setText(f"找到 {len(results)} 张相关照片 (最高相似度: {top_sim:.2f})")
        self.semantic_search_label.setStyleSheet("color: #4a90d9; font-size: 11px;")
        
        # 更新照片显示
        self.current_filter = {"mode": "semantic", "results": results}
        self._display_semantic_search_results(results)
    
    def _on_semantic_search_error(self, error_msg):
        """语义搜索错误"""
        self.semantic_search_label.setText(f"搜索失败: {error_msg}")
        self.semantic_search_label.setStyleSheet("color: #d94a4a; font-size: 11px;")
    
    def _display_semantic_search_results(self, results):
        """显示语义搜索结果"""
        # 清空当前显示
        self.photo_list.clear()
        
        # 创建默认占位符
        default_placeholder = QPixmap(150, 150)
        default_placeholder.fill(Qt.GlobalColor.lightGray)
        
        # 收集需要异步加载的路径
        paths_to_load = []
        
        # 按相似度排序的照片
        for photo_id, filepath, similarity in results:
            if not os.path.exists(filepath):
                continue
            
            item = QListWidgetItem()
            # 在文件名前显示相似度
            item.setText(f"[{similarity:.2f}] {os.path.basename(filepath)}")
            
            # 检查缓存中是否有缩略图
            if filepath in self.thumbnail_cache:
                item.setIcon(QIcon(self.thumbnail_cache[filepath]))
            else:
                item.setIcon(QIcon(default_placeholder))
                paths_to_load.append(filepath)
            
            # 存储照片数据
            item.setData(Qt.ItemDataRole.UserRole, {
                'id': photo_id,
                'filepath': filepath,
                'similarity': similarity
            })
            self.photo_list.addItem(item)
        
        # 异步加载缩略图
        if paths_to_load and self.thumbnail_worker:
            self.thumbnail_worker.add_paths(paths_to_load)
        
        self.status_bar.showMessage(f"语义搜索: 显示 {len(results)} 张照片")
    
    def _clear_semantic_search(self):
        """清除语义搜索结果，恢复正常显示"""
        self.semantic_search_input.clear()
        self.semantic_search_label.setText("")
        self.current_filter = None
        self.load_photos(None)
    
    def _apply_filter_immediately(self):
        """即时应用当前筛选条件"""
        mode = self.filter_mode_combo.currentData() or "all"
        
        if mode == "category":
            selected_categories = [cat for cat, cb in self.filter_category_checks.items() if cb.isChecked()]
            if selected_categories:
                self.current_filter = {"mode": "category", "categories": selected_categories}
                self.load_photos(self.current_filter)
            else:
                # 没有选择任何分类时显示全部
                self.current_filter = None
                self.load_photos(None)
        elif mode == "person":
            person_value = self.filter_person_combo.currentData()
            if person_value == "__unlabeled__":
                self.current_filter = {"mode": "person", "unlabeled": True}
            elif person_value == "__any_face__":
                self.current_filter = {"mode": "person", "any_face": True}
            else:
                self.current_filter = {"mode": "person", "person_id": person_value}
            self.load_photos(self.current_filter)
        else:
            self.current_filter = None
            self.load_photos(None)
    
    def _update_category_selection_label(self):
        """更新分类选中数量的反馈标签"""
        count = sum(1 for cb in self.filter_category_checks.values() if cb.isChecked())
        selected_names = [cat for cat, cb in self.filter_category_checks.items() if cb.isChecked()]
        if count == 0:
            self.category_selection_label.setText("已选: 0 项")
            self.category_selection_label.setStyleSheet("color: #999; font-size: 12px;")
        elif count <= 3:
            self.category_selection_label.setText(f"已选: {', '.join(selected_names)}")
            self.category_selection_label.setStyleSheet("color: #4a90d9; font-size: 12px; font-weight: bold;")
        else:
            self.category_selection_label.setText(f"已选: {count} 项")
            self.category_selection_label.setStyleSheet("color: #4a90d9; font-size: 12px; font-weight: bold;")
    
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
        
        # 视图切换按钮
        switch_to_people_btn = QPushButton("切换到人物视图")
        switch_to_people_btn.clicked.connect(self.switch_to_people)
        header_layout.addWidget(switch_to_people_btn)
        
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
    
    def create_people_display(self):
        """创建人物视图展示区"""
        display_widget = QWidget()
        display_widget.setStyleSheet("""
            QWidget {
                background-color: white;
            }
        """)
        
        layout = QVBoxLayout(display_widget)
        
        # 展示区标题和工具栏
        header_layout = QHBoxLayout()
        
        display_title = QLabel("人物管理")
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
        
        # 视图切换按钮
        switch_to_gallery_btn = QPushButton("切换到图库视图")
        switch_to_gallery_btn.clicked.connect(self.switch_to_gallery)
        header_layout.addWidget(switch_to_gallery_btn)
        
        # 刷新按钮
        refresh_people_btn = QPushButton("刷新")
        refresh_people_btn.clicked.connect(self.load_people_view)
        header_layout.addWidget(refresh_people_btn)
        
        layout.addLayout(header_layout)
        
        # 人物列表滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        self.people_container = QWidget()
        self.people_layout = QVBoxLayout(self.people_container)
        self.people_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll.setWidget(self.people_container)
        layout.addWidget(scroll)
        
        return display_widget
    
    def load_people_view(self):
        """加载人物视图数据"""
        # 清空现有内容
        while self.people_layout.count():
            child = self.people_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # 获取所有人物及其样本人脸
        persons = get_all_persons_with_sample_faces(limit_faces=4)
        
        if not persons:
            empty_label = QLabel("暂无人物数据\n\n请先导入照片并使用人脸识别模型分析")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setStyleSheet("""
                QLabel {
                    color: #888;
                    font-size: 14px;
                    padding: 40px;
                }
            """)
            self.people_layout.addWidget(empty_label)
            return
        
        for person in persons:
            person_card = self._create_person_card(person)
            self.people_layout.addWidget(person_card)
        
        self.people_layout.addStretch()
        self.status_bar.showMessage(f"已加载 {len(persons)} 个人物", 3000)
    
    def _create_person_card(self, person: dict):
        """创建单个人物卡片"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 5px;
            }
            QFrame:hover {
                background-color: #f0f7ff;
                border-color: #4a90d9;
            }
        """)
        
        card_layout = QHBoxLayout(card)
        
        # 人脸样本缩略图区域
        faces_widget = QWidget()
        faces_layout = QHBoxLayout(faces_widget)
        faces_layout.setContentsMargins(5, 5, 5, 5)
        faces_layout.setSpacing(5)
        
        sample_faces = person.get('sample_faces', [])
        for face in sample_faces[:4]:  # 最多显示4张
            thumb = self._face_thumbnail_from_data(face)
            thumb_label = QLabel()
            thumb_label.setPixmap(thumb)
            thumb_label.setFixedSize(60, 60)
            thumb_label.setScaledContents(True)
            thumb_label.setStyleSheet("border: 1px solid #ccc; border-radius: 4px;")
            faces_layout.addWidget(thumb_label)
        
        # 如果没有人脸样本，显示占位符
        if not sample_faces:
            placeholder = QLabel("无照片")
            placeholder.setFixedSize(60, 60)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("background-color: #eee; border-radius: 4px; color: #888;")
            faces_layout.addWidget(placeholder)
        
        card_layout.addWidget(faces_widget)
        
        # 人物信息
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        name_label = QLabel(person.get('name', '未命名'))
        name_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        info_layout.addWidget(name_label)
        
        stats_label = QLabel(f"人脸数: {person.get('face_count', 0)} | 照片数: {person.get('photo_count', 0)}")
        stats_label.setStyleSheet("color: #666; font-size: 12px;")
        info_layout.addWidget(stats_label)
        
        card_layout.addWidget(info_widget, stretch=1)
        
        # 操作按钮
        buttons_widget = QWidget()
        buttons_layout = QVBoxLayout(buttons_widget)
        buttons_layout.setSpacing(4)
        
        view_btn = QPushButton("查看照片")
        view_btn.setFixedWidth(80)
        view_btn.clicked.connect(lambda checked, pid=person['id']: self.view_person_photos(pid))
        buttons_layout.addWidget(view_btn)
        
        rename_btn = QPushButton("重命名")
        rename_btn.setFixedWidth(80)
        rename_btn.clicked.connect(lambda checked, pid=person['id'], name=person.get('name', ''): self.rename_person_dialog(pid, name))
        buttons_layout.addWidget(rename_btn)
        
        delete_btn = QPushButton("删除")
        delete_btn.setFixedWidth(80)
        delete_btn.setStyleSheet("color: #c00;")
        delete_btn.clicked.connect(lambda checked, pid=person['id'], name=person.get('name', ''): self.delete_person_confirm(pid, name))
        buttons_layout.addWidget(delete_btn)
        
        card_layout.addWidget(buttons_widget)
        
        return card
    
    def _face_thumbnail_from_data(self, face_data: dict):
        """从人脸数据生成缩略图"""
        filepath = face_data.get("photo_filepath")
        bbox = face_data.get("bbox", [0, 0, 60, 60])
        
        if not filepath or not os.path.exists(filepath):
            placeholder = QPixmap(60, 60)
            placeholder.fill(Qt.GlobalColor.lightGray)
            return placeholder
        
        image = QImage(filepath)
        if image.isNull():
            placeholder = QPixmap(60, 60)
            placeholder.fill(Qt.GlobalColor.lightGray)
            return placeholder
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w = max(20, x2 - x1)
        h = max(20, y2 - y1)
        cropped = image.copy(x1, y1, w, h)
        thumb = cropped.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        return QPixmap.fromImage(thumb)
    
    def view_person_photos(self, person_id: int):
        """查看指定人物的所有照片"""
        # 切换到图库视图并筛选该人物
        self.current_filter = {"mode": "person", "person_id": person_id}
        self.switch_to_gallery()
        self.load_photos(self.current_filter)
        
        # 更新人物下拉框选择
        index = self.filter_person_combo.findData(person_id)
        if index >= 0:
            self.filter_person_combo.setCurrentIndex(index)
        
        self.status_bar.showMessage(f"已筛选人物照片", 3000)
    
    def rename_person_dialog(self, person_id: int, current_name: str):
        """重命名人物对话框"""
        new_name, ok = QInputDialog.getText(
            self, "重命名人物", 
            f"当前名称: {current_name}\n请输入新名称:",
            text=current_name
        )
        
        if ok and new_name.strip():
            if rename_person(person_id, new_name.strip()):
                self.status_bar.showMessage(f"已将 '{current_name}' 重命名为 '{new_name.strip()}'", 3000)
                self.load_people_view()
                self.refresh_person_filter_options()
            else:
                QMessageBox.warning(self, "错误", "重命名失败，可能名称已存在")
    
    def delete_person_confirm(self, person_id: int, name: str):
        """确认删除人物"""
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除人物 '{name}' 吗？\n\n注意：关联的人脸记录将变为未命名状态，不会删除照片。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if delete_person(person_id):
                self.status_bar.showMessage(f"已删除人物 '{name}'", 3000)
                self.load_people_view()
                self.refresh_person_filter_options()
            else:
                QMessageBox.warning(self, "错误", "删除失败")
    
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
    
    def init_thumbnail_worker(self):
        """初始化缩略图工作线程"""
        self.thumbnail_worker = ThumbnailWorker(thumbnail_size=150)
        self.thumbnail_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        self.thumbnail_worker.start()
    
    def _on_thumbnail_ready(self, filepath: str, pixmap):
        """处理缩略图生成完成"""
        # 缓存缩略图
        self.thumbnail_cache[filepath] = pixmap
        
        # 更新列表中对应的项
        for i in range(self.photo_list.count()):
            item = self.photo_list.item(i)
            photo_data = item.data(Qt.ItemDataRole.UserRole)
            if photo_data and photo_data.get('filepath') == filepath:
                item.setIcon(QIcon(pixmap))
                break
    
    def load_photos(self, filter_spec=None):
        """加载照片到界面，支持分类/人脸等筛选。使用异步缩略图加载避免卡顿。"""
        filter_spec = filter_spec or self.current_filter

        if isinstance(filter_spec, dict):
            mode = filter_spec.get("mode")
        else:
            mode = None

        categories = None
        person_id = None
        has_faces = None
        unlabeled_faces = False

        if mode == "category":
            categories = filter_spec.get("categories")
        elif mode == "person":
            person_id = filter_spec.get("person_id")
            unlabeled_faces = filter_spec.get("unlabeled", False)
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
        
        # 收集需要异步加载的路径
        paths_to_load = []
        
        # 创建默认占位符
        default_placeholder = QPixmap(150, 150)
        default_placeholder.fill(Qt.GlobalColor.lightGray)
        
        # 添加照片到列表（先用占位符，缩略图异步加载）
        for photo in photos:
            item = QListWidgetItem()
            item.setText(os.path.basename(photo['filepath']))
            filepath = photo['filepath']
            
            # 检查缓存中是否有缩略图
            if filepath in self.thumbnail_cache:
                item.setIcon(QIcon(self.thumbnail_cache[filepath]))
            else:
                # 使用占位符，稍后异步加载
                item.setIcon(QIcon(default_placeholder))
                paths_to_load.append(filepath)
                
            item.setData(Qt.ItemDataRole.UserRole, photo)
            self.photo_list.addItem(item)
        
        # 异步加载缩略图
        if paths_to_load and self.thumbnail_worker:
            self.thumbnail_worker.add_paths(paths_to_load)
        
        # 更新统计信息
        self.update_stats()
        self.status_bar.showMessage(f"加载了 {len(photos)} 张照片", 3000)
    
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
        mode = self.filter_mode_combo.currentData() or "all"
        
        # 分类区域：仅在"按分类筛选"时显示
        self.category_group.setVisible(mode == "category")
        
        # 人物区域：仅在"按人物筛选"时显示
        self.person_group.setVisible(mode == "person")

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

    def run_face_analysis(self):
        """运行人脸检测与识别（独立于照片导入）"""
        if not self.current_library_path:
            QMessageBox.information(self, "提示", "请先选择照片库，再运行人脸分析。")
            return
        
        if self.face_worker and self.face_worker.isRunning():
            QMessageBox.information(self, "提示", "人脸分析正在进行中，请稍候...")
            return
        
        # 创建并启动人脸分析线程
        self.face_worker = FaceAnalysisWorker(
            library_path=self.current_library_path,
            model_profile=self.selected_model_profile
        )
        
        self.face_worker.progress_updated.connect(self._on_face_analysis_progress)
        self.face_worker.face_detected.connect(self._on_face_detected)
        self.face_worker.analysis_completed.connect(self._on_face_analysis_completed)
        self.face_worker.error_occurred.connect(self._on_face_analysis_error)
        
        self.face_worker.start()
        
        self.face_analysis_label.setText("正在分析人脸...")
        self.status_bar.showMessage("开始人脸检测与识别...", 3000)
    
    def _on_face_analysis_progress(self, current, total):
        """处理人脸分析进度"""
        self.face_analysis_label.setText(f"分析中: {current}/{total}")
        self.status_bar.showMessage(f"人脸分析进度: {current}/{total}")
    
    def _on_face_detected(self, filepath, face_count):
        """处理检测到人脸"""
        print(f"检测到 {face_count} 个人脸: {os.path.basename(filepath)}")
    
    def _on_face_analysis_completed(self, total_photos, total_faces):
        """处理人脸分析完成"""
        self.face_analysis_label.setText(f"完成: {total_photos} 张照片, {total_faces} 个人脸")
        self.status_bar.showMessage(f"人脸分析完成: 处理 {total_photos} 张照片，检测到 {total_faces} 个人脸", 5000)
        
        # 刷新人物列表
        self.refresh_person_filter_options()
        
        # 如果检测到人脸，提示命名
        if total_faces > 0:
            self.prompt_name_unlabeled_faces()
        
        # 重新加载照片
        self.load_photos(self.current_filter)
    
    def _on_face_analysis_error(self, error_msg):
        """处理人脸分析错误"""
        self.face_analysis_label.setText("分析失败")
        self.status_bar.showMessage(f"人脸分析错误: {error_msg}", 5000)

    def run_face_clustering(self):
        """运行人脸聚类，将相似人脸分组"""
        from worker import ClusteringWorker
        
        if hasattr(self, 'cluster_worker') and self.cluster_worker and self.cluster_worker.isRunning():
            QMessageBox.information(self, "提示", "聚类正在进行中，请稍候...")
            return
        
        self.cluster_label.setText("正在聚类...")
        self.cluster_label.setStyleSheet("color: #4a90d9; font-size: 11px;")
        
        # 创建聚类工作线程
        self.cluster_worker = ClusteringWorker(eps=0.7, min_samples=2)
        self.cluster_worker.progress_updated.connect(self._on_clustering_progress)
        self.cluster_worker.clustering_completed.connect(self._on_clustering_completed)
        self.cluster_worker.error_occurred.connect(self._on_clustering_error)
        self.cluster_worker.start()
    
    def _on_clustering_progress(self, current, total):
        """聚类进度更新"""
        self.cluster_label.setText(f"聚类中: {current}%")
    
    def _on_clustering_completed(self, result):
        """聚类完成"""
        n_clusters = result.get('n_clusters', 0)
        n_noise = result.get('n_noise', 0)
        n_faces = result.get('n_faces', 0)
        
        self.cluster_label.setText(f"完成: {n_clusters} 个人物, {n_noise} 个噪声")
        self.cluster_label.setStyleSheet("color: #4a4; font-size: 11px;")
        self.status_bar.showMessage(f"聚类完成: {n_faces} 个人脸分为 {n_clusters} 组，{n_noise} 个无法归类", 5000)
        
        # 刷新人物视图
        self.refresh_person_filter_options()
        if hasattr(self, 'people_view') and self.stacked_widget.currentWidget() == self.people_view:
            self.load_people_view()
    
    def _on_clustering_error(self, error_msg):
        """聚类错误"""
        self.cluster_label.setText("聚类失败")
        self.cluster_label.setStyleSheet("color: #d94a4a; font-size: 11px;")
        self.status_bar.showMessage(f"聚类错误: {error_msg}", 5000)

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
        
        # 显示已知信息
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
    
    def start_scan(self, directory):
        """开始扫描指定目录（V2.2 完整自动化流水线）"""
        # 更新当前库路径
        self.current_library_path = directory
        self.library_path_label.setText(directory)

        # 创建并启动扫描工作线程（自动化流水线）
        self.scan_worker = ScanWorker(directory, model_profile=self.selected_model_profile)
        
        # 连接信号
        self.scan_worker.progress_updated.connect(self.on_scan_progress)
        self.scan_worker.stage_changed.connect(self.on_stage_changed)
        self.scan_worker.file_found.connect(self.on_file_found)
        self.scan_worker.scan_completed.connect(self.on_scan_completed)
        self.scan_worker.pipeline_completed.connect(self.on_pipeline_completed)
        self.scan_worker.error_occurred.connect(self.on_scan_error)
        
        # 启动线程
        self.scan_worker.start()
        
        self.status_bar.showMessage("正在扫描并分析照片...")
        self.face_analysis_label.setText("🔄 自动分析中...")
        self.cluster_label.setText("等待人脸检测完成...")
    
    def on_stage_changed(self, stage_desc):
        """处理阶段变化"""
        self.face_analysis_label.setText(stage_desc)
        self.status_bar.showMessage(stage_desc)
    
    def on_scan_progress(self, current, total):
        """处理扫描进度更新"""
        self.status_bar.showMessage(f"已处理 {current}/{total} 个文件")
    
    def on_file_found(self, filepath):
        """处理发现新文件"""
        filename = os.path.basename(filepath)
        print(f"发现新文件: {filename}")
    
    def on_scan_completed(self, total_files):
        """处理文件扫描完成（流水线继续进行）"""
        self.status_bar.showMessage(f"扫描完成: {total_files} 个文件，正在分析...", 3000)
    
    def on_pipeline_completed(self, stats):
        """处理整个流水线完成"""
        total = stats.get('total_files', 0)
        faces = stats.get('faces_detected', 0)
        clusters = stats.get('clusters_created', 0)
        noise = stats.get('noise_faces', 0)
        corrected = stats.get('categories_corrected', 0)
        
        # 更新UI标签
        self.face_analysis_label.setText(f"✅ 检测到 {faces} 个人脸")
        self.face_analysis_label.setStyleSheet("color: #4a4; font-size: 11px;")
        
        if clusters > 0:
            self.cluster_label.setText(f"✅ {clusters} 个人物, {noise} 个噪声")
            self.cluster_label.setStyleSheet("color: #4a4; font-size: 11px;")
        else:
            self.cluster_label.setText("无需聚类")
            self.cluster_label.setStyleSheet("color: #666; font-size: 11px;")
        
        # 显示完成消息
        msg = f"分析完成: {total} 张照片, {faces} 个人脸, {clusters} 个人物"
        if corrected > 0:
            msg += f", {corrected} 个分类修正"
        self.status_bar.showMessage(msg, 8000)
        
        self.scan_worker = None
        
        # 重新加载照片和人物视图
        self.load_photos(self.current_filter)
        self.update_stats()
        self.refresh_person_filter_options()
        
        # 如果当前在人物视图，刷新它
        if self.current_view_mode == "people":
            self.load_people_view()
    
    def on_scan_error(self, error_msg):
        """处理扫描错误"""
        self.status_bar.showMessage(f"扫描错误: {error_msg}", 5000)
        self.scan_worker = None
    
    def switch_to_gallery(self):
        """切换到图库视图"""
        self.current_view_mode = "gallery"
        self.content_stack.setCurrentIndex(0)
        self.load_photos(self.current_filter)
        self.status_bar.showMessage("已切换到图库视图", 3000)
    
    def switch_to_people(self):
        """切换到人物视图"""
        self.current_view_mode = "people"
        self.content_stack.setCurrentIndex(1)
        self.load_people_view()
        self.status_bar.showMessage("已切换到人物视图", 3000)
    
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
        """窗口关闭事件，清空所有 AI 数据。"""
        clear_all_ai_data()
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
