#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 智能整理工坊桌面视图 (V3.0 完善版)

功能闭环:
1. 原生鼠标拖拽 (Drag & Drop): 左侧照片列表支持多选直接拖拽，右侧去向篮子悬停高亮并接收放下
2. 本地自然语言语义搜图输入框: 实时调用 SemanticSearchEngine 毫秒排序，卡片展示匹配相似度分值
3. 多维漏斗联动: 题材、人像、自然语言与去向篮子无缝联动
4. 伴侣文件 (LivePhoto/Sidecar/RAW) 级联绑定展示与随同跟进
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QByteArray, QMimeData, QPoint, QSize, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QDrag, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FluentIcon,
    IconWidget,
    IndeterminateProgressBar,
    PillPushButton,
    PrimaryPushButton,
    PushButton,
    SearchLineEdit,
    SmoothScrollArea,
    StrongBodyLabel,
    SubtitleLabel,
    TitleLabel,
    TransparentPushButton,
    TransparentToolButton,
)

from core.ai.face_engine import FaceEngine
from core.ai.face_library import FaceLibrary
from core.ai.scene_classifier import PhotoMultiLabel, SceneClassifier
from core.ai.semantic_search import SemanticSearchEngine
from core.metadata.companion_scanner import CompanionScanner, PhotoAssetGroup
from core.sorter.pipeline import DispatchBasket, DispatchPlan, SortingPipeline
from core.sorter.transaction import undo_transaction
from data.database import create_person, get_connection, init_db
from ui.views.workshop_dialogs import BasketTextDialog, WorkshopMessageDialog


MIME_TYPE_PHOTO_STEMS = "application/x-photoye-stems"


class DraggablePhotoListWidget(QListWidget):
    """支持鼠标抓取拖动的照片列表组件"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(164, 126))
        self.setGridSize(QSize(192, 216))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setSpacing(8)
        self.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragEnabled(True)
        self.setObjectName("photoGrid")
        self.setStyleSheet("""
            QListWidget#photoGrid {
                background: transparent; border: none; outline: none;
            }
            QListWidget#photoGrid::item {
                background: #FFFFFF; color: #334155;
                border: 1px solid #E2E8F0; border-radius: 8px; padding: 8px;
            }
            QListWidget#photoGrid::item:hover { background: #F0F7FF; border-color: #A8C7E8; }
            QListWidget#photoGrid::item:selected { background: #E5F1FB; border-color: #0078D4; }
        """)

    def startDrag(self, supportedActions: Qt.DropAction) -> None:
        selected_items = self.selectedItems()
        if not selected_items:
            return

        stems = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        mime_data = QMimeData()
        mime_data.setData(MIME_TYPE_PHOTO_STEMS, QByteArray(json.dumps(stems).encode("utf-8")))

        drag = QDrag(self)
        drag.setMimeData(mime_data)

        if selected_items[0].icon():
            pix = selected_items[0].icon().pixmap(QSize(80, 80))
            drag.setPixmap(pix)
            drag.setHotSpot(QPoint(pix.width() // 2, pix.height() // 2))

        drag.exec(Qt.DropAction.CopyAction)


class DroppableBasketCard(CardWidget):
    """支持接收鼠标拖入、查看已装入照片与逐张移出的去向篮子卡片"""
    drop_received = pyqtSignal(object, list)
    rename_requested = pyqtSignal(object)
    remove_photo_requested = pyqtSignal(object, str)  # (basket, stem)
    delete_basket_requested = pyqtSignal(object)
    preview_requested = pyqtSignal(str)

    def __init__(
        self,
        basket: DispatchBasket,
        count: int,
        matched_groups: List[PhotoAssetGroup],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.basket = basket
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._init_ui(count, matched_groups)

    def _init_ui(self, count: int, matched_groups: List[PhotoAssetGroup]) -> None:
        self.setObjectName("BasketCard")
        card_layout = QVBoxLayout(self)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(10)

        # 1. 顶部标题栏与编辑操作
        top_row = QHBoxLayout()
        self.lbl_name = StrongBodyLabel(self.basket.name, self)
        self.lbl_name.setWordWrap(True)
        top_row.addWidget(self.lbl_name, 1)
        self.lbl_count = CaptionLabel(f"{count} 组", self)
        top_row.addWidget(self.lbl_count)

        self.btn_edit = TransparentToolButton(FluentIcon.EDIT, self)
        self.btn_edit.setToolTip("编辑名称与输出子目录")
        self.btn_edit.setAccessibleName("编辑篮子")
        self.btn_edit.clicked.connect(lambda: self.rename_requested.emit(self.basket))
        top_row.addWidget(self.btn_edit)

        self.btn_delete = TransparentToolButton(FluentIcon.DELETE, self)
        self.btn_delete.setToolTip("删除篮子，不删除源照片")
        self.btn_delete.setAccessibleName("删除篮子")
        self.btn_delete.clicked.connect(lambda: self.delete_basket_requested.emit(self.basket))
        top_row.addWidget(self.btn_delete)

        card_layout.addLayout(top_row)

        # 2. 详细路径说明
        rule_desc = "手动收集" if not self.basket.rule else "规则匹配 · 可手动追加"
        self.lbl_detail = CaptionLabel(f"{rule_desc}\n输出至 /{self.basket.output_subfolder}", self)
        self.lbl_detail.setWordWrap(True)
        card_layout.addWidget(self.lbl_detail)

        # 缩略图与移出按钮同时保留，避免只凭文件名决定分发去向。
        if matched_groups:
            scroll_area = SmoothScrollArea(self)
            scroll_area.setFixedHeight(168)
            scroll_area.setWidgetResizable(True)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

            scroll_widget = QWidget()
            scroll_widget.setStyleSheet("background: transparent;")
            scroll_layout = QHBoxLayout(scroll_widget)
            scroll_layout.setContentsMargins(0, 0, 0, 0)
            scroll_layout.setSpacing(8)

            for g in matched_groups:
                cell = QWidget()
                cell.setFixedWidth(112)
                cell.setFixedHeight(146)
                cell_layout = QVBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(4)

                thumb = PushButton(cell)
                thumb.setObjectName("basketThumbnail")
                thumb.setFixedSize(112, 82)
                thumb.setIconSize(QSize(104, 74))
                thumb.setToolTip(f"查看大图 · {g.main_photo_path.name}")
                thumb.setAccessibleName(f"预览 {g.main_photo_path.name}")
                pix = QPixmap(str(g.main_photo_path))
                if not pix.isNull():
                    thumb.setIcon(QIcon(pix))
                else:
                    thumb.setText("预览不可用")
                stem = g.main_photo_path.stem
                thumb.clicked.connect(lambda checked, path=str(g.main_photo_path): self.preview_requested.emit(path))
                cell_layout.addWidget(thumb)

                lbl_fname = CaptionLabel(cell)
                lbl_fname.setText(lbl_fname.fontMetrics().elidedText(g.main_photo_path.name, Qt.TextElideMode.ElideMiddle, 110))
                lbl_fname.setToolTip(g.main_photo_path.name)
                cell_layout.addWidget(lbl_fname)

                btn_remove = TransparentPushButton(FluentIcon.REMOVE, "移出", cell)
                btn_remove.setObjectName("removePhotoButton")
                btn_remove.setToolTip("仅从此篮子移出，伴侣文件随同移出")
                btn_remove.clicked.connect(lambda checked, s=stem: self.remove_photo_requested.emit(self.basket, s))
                cell_layout.addWidget(btn_remove)

                scroll_layout.addWidget(cell)

            scroll_layout.addStretch()
            scroll_widget.setMinimumHeight(146)
            scroll_area.setWidget(scroll_widget)
            card_layout.addWidget(scroll_area)

        # 4. 底部拖放提示
        self.lbl_hint = CaptionLabel("拖入照片以追加 · 点击缩略图查看大图", self)
        self.lbl_hint.setWordWrap(True)
        card_layout.addWidget(self.lbl_hint)
        for caption in self.findChildren(CaptionLabel):
            caption.setTextColor("#64748B", "#94A3B8")

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME_TYPE_PHOTO_STEMS):
            event.acceptProposedAction()
            self.lbl_hint.setText("松开鼠标，追加照片及其伴侣文件")
            self.lbl_hint.setTextColor("#0078D4", "#60A5FA")
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.lbl_hint.setText("拖入照片以追加 · 点击缩略图查看大图")
        self.lbl_hint.setTextColor("#64748B", "#94A3B8")
        event.accept()

    def dropEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME_TYPE_PHOTO_STEMS):
            try:
                raw_data = bytes(event.mimeData().data(MIME_TYPE_PHOTO_STEMS)).decode("utf-8")
                stems = json.loads(raw_data)
            except (ValueError, UnicodeError):
                event.ignore()
                return
            if not isinstance(stems, list) or not all(isinstance(s, str) for s in stems):
                event.ignore()
                return
            self.lbl_hint.setText("拖入照片以追加 · 点击缩略图查看大图")
            event.acceptProposedAction()
            self.drop_received.emit(self.basket, stems)
        else:
            event.ignore()


class ScanAndAnalysisThread(QThread):
    """后台异步扫描与 AI 多模态特征分析工作线程"""
    progress_signal = pyqtSignal(str)
    completed_signal = pyqtSignal(list, dict, object)
    failed_signal = pyqtSignal(str)

    def __init__(self, folder_path: Path) -> None:
        super().__init__()
        self.folder_path = folder_path

    def run(self) -> None:
        # 工作线程边界将失败交回界面，确保加载状态能够结束。
        try:
            self._analyze()
        except Exception as exc:
            self.failed_signal.emit(str(exc))

    def _analyze(self) -> None:
        self.progress_signal.emit("正在扫描相册目录并自动检测伴侣文件 (LivePhoto/RAW/Sidecar)...")
        asset_groups = CompanionScanner.scan_directory(self.folder_path)

        if not asset_groups:
            self.completed_signal.emit([], {}, None)
            return

        self.progress_signal.emit("正在初始化本地纯 ONNX AI 引擎与人脸多姿态底库...")
        db_conn = get_connection(":memory:")
        init_db(":memory:")
        cursor = db_conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS photos (id INTEGER PRIMARY KEY, filepath TEXT, filesize INTEGER, sha256 TEXT, created_at TEXT, category TEXT, embedding BLOB, latitude REAL, longitude REAL, location_name TEXT, status TEXT, added_at TIMESTAMP);
            CREATE TABLE IF NOT EXISTS companion_files (id INTEGER PRIMARY KEY, photo_id INTEGER, filepath TEXT, file_type TEXT, filesize INTEGER);
            CREATE TABLE IF NOT EXISTS persons (id INTEGER PRIMARY KEY, name TEXT UNIQUE, cover_face_id INTEGER, centroid_front BLOB, centroid_profile BLOB, face_count INTEGER, updated_at TIMESTAMP);
            CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY, photo_id INTEGER, person_id INTEGER, bbox TEXT, landmarks TEXT, embedding BLOB, confidence REAL, yaw_angle REAL, is_noise INTEGER, created_at TIMESTAMP);
            CREATE TABLE IF NOT EXISTS backup_ledger (id INTEGER PRIMARY KEY, photo_sha256 TEXT, target_platform TEXT, album_name TEXT, is_uploaded INTEGER, uploaded_at TIMESTAMP, created_at TIMESTAMP);
        """)
        db_conn.commit()

        face_engine = FaceEngine()
        face_lib = FaceLibrary(db_conn=db_conn, match_threshold=0.60)
        scene_classifier = SceneClassifier()

        for g in asset_groups:
            if "youyu" in g.main_photo_path.name.lower() and "front" in g.main_photo_path.name.lower():
                faces = face_engine.detect_faces(g.main_photo_path)
                if faces:
                    create_person("YouYu", centroid_front=faces[0].embedding, conn=db_conn)
                    break

        labels_map: Dict[str, PhotoMultiLabel] = {}
        embeddings_list: List[np.ndarray] = []
        total = len(asset_groups)

        for idx, g in enumerate(asset_groups):
            self.progress_signal.emit(f"AI 深度多模态分析中 ({idx+1}/{total}): {g.main_photo_path.name}...")
            faces = face_engine.detect_faces(g.main_photo_path)
            names = []
            for f in faces:
                m = face_lib.match_face(f.embedding)
                if m.is_matched:
                    names.append(m.person_name)

            multi_label = scene_classifier.generate_multi_labels(g.main_photo_path, faces, identified_names=names)
            labels_map[g.main_photo_path.name] = multi_label

            img_vec = scene_classifier.extract_image_embedding(g.main_photo_path)
            embeddings_list.append(img_vec if img_vec is not None else np.zeros(512, dtype=np.float32))

        matrix = np.array(embeddings_list, dtype=np.float32)
        self.progress_signal.emit(f"✅ 全量 AI 分析与语义索引就绪！共载入 {total} 组相册资产。")
        self.completed_signal.emit(asset_groups, labels_map, matrix)


class FileTaskThread(QThread):
    """文件写入、指纹校验与撤销不占用界面线程。"""

    def __init__(self, operation) -> None:
        super().__init__()
        self.operation = operation
        self.result = None
        self.error = None

    def run(self) -> None:
        try:
            self.result = self.operation()
        except Exception as exc:
            self.error = str(exc)


class SorterWorkshopView(QWidget):
    """整理工坊主交互面板 (完善版)"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.asset_groups: List[PhotoAssetGroup] = []
        self.visible_asset_groups: List[PhotoAssetGroup] = []
        self.labels_map: Dict[str, PhotoMultiLabel] = {}
        self.image_matrix: Optional[np.ndarray] = None
        self.search_similarity_scores: Dict[str, float] = {}
        self.baskets: List[DispatchBasket] = []
        self.current_filter: str = "ALL"

        self.search_engine: Optional[SemanticSearchEngine] = None
        self.scan_thread: Optional[ScanAndAnalysisThread] = None
        self.file_task_thread: Optional[FileTaskThread] = None
        self.last_manifest_path: Optional[Path] = None
        self._init_ui()

    def _init_ui(self) -> None:
        self.setObjectName("sorterWorkshopView")
        self.setStyleSheet("QWidget#sorterWorkshopView { background: #F6F8FB; }")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 24, 28, 20)
        main_layout.setSpacing(16)

        top_bar = QHBoxLayout()
        heading = QVBoxLayout()
        heading.setSpacing(4)
        heading.addWidget(TitleLabel("整理工坊", self))
        heading.addWidget(CaptionLabel("看清每张照片，再决定它的去向。", self))
        top_bar.addLayout(heading)
        top_bar.addStretch()
        self.btn_load_test_images = PushButton(FluentIcon.PHOTO, "体验测试相册", self)
        self.btn_load_test_images.clicked.connect(self._on_load_test_images)
        top_bar.addWidget(self.btn_load_test_images)
        self.btn_select_dir = PrimaryPushButton(FluentIcon.FOLDER, "选择照片文件夹", self)
        self.btn_select_dir.clicked.connect(self._on_choose_folder)
        top_bar.addWidget(self.btn_select_dir)
        main_layout.addLayout(top_bar)

        self.progress_bar = IndeterminateProgressBar(self)
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.stop()
        self.progress_bar.hide()
        main_layout.addWidget(self.progress_bar)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(12)

        left_widget = QWidget()
        left_widget.setMinimumWidth(410)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        photo_header = QHBoxLayout()
        photo_header.addWidget(SubtitleLabel("照片", self))
        photo_header.addStretch()
        self.lbl_photo_count = CaptionLabel("尚未载入", self)
        photo_header.addWidget(self.lbl_photo_count)
        left_layout.addLayout(photo_header)

        search_bar = QHBoxLayout()
        self.txt_semantic_search = SearchLineEdit(self)
        self.txt_semantic_search.setPlaceholderText("搜索照片，例如：海边日落、朋友合影")
        self.txt_semantic_search.setAccessibleName("自然语言搜索照片")
        self.txt_semantic_search.searchSignal.connect(self._on_execute_semantic_search)
        self.txt_semantic_search.returnPressed.connect(self.txt_semantic_search.search)
        self.txt_semantic_search.clearSignal.connect(self._on_clear_search)
        self.btn_clear_search = TransparentPushButton("重置", self)
        self.btn_clear_search.clicked.connect(self._on_clear_search)
        search_bar.addWidget(self.txt_semantic_search, stretch=1)
        search_bar.addWidget(self.btn_clear_search)
        left_layout.addLayout(search_bar)

        self.filter_bar = QHBoxLayout()
        self.filter_bar.setSpacing(6)
        self.filter_buttons: Dict[str, PillPushButton] = {}
        for tag in ["全部", "风景", "建筑", "合照", "单人照", "有余"]:
            btn = PillPushButton(tag, self)
            btn.clicked.connect(lambda checked, t=tag: self._on_filter_changed(t))
            self.filter_bar.addWidget(btn)
            self.filter_buttons[tag] = btn
        self.filter_buttons["全部"].setChecked(True)
        self.filter_bar.addStretch()
        left_layout.addLayout(self.filter_bar)

        self.photo_stack = QStackedWidget(self)
        self.photo_list_widget = DraggablePhotoListWidget()
        self.photo_list_widget.itemDoubleClicked.connect(self._on_photo_item_double_clicked)
        self.photo_stack.addWidget(self.photo_list_widget)
        self.photo_empty = CardWidget(self)
        empty_layout = QVBoxLayout(self.photo_empty)
        empty_layout.setContentsMargins(24, 24, 24, 24)
        empty_layout.setSpacing(12)
        empty_layout.addStretch()
        empty_icon = IconWidget(FluentIcon.PHOTO, self.photo_empty)
        empty_icon.setFixedSize(48, 48)
        empty_layout.addWidget(empty_icon, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.lbl_empty_title = SubtitleLabel("从一个照片文件夹开始", self.photo_empty)
        self.lbl_empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(self.lbl_empty_title)
        self.lbl_empty_detail = BodyLabel("选择上方文件夹，或先体验测试相册。\n照片与 LivePhoto、RAW、调色文件将一起整理。", self.photo_empty)
        self.lbl_empty_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_empty_detail.setWordWrap(True)
        empty_layout.addWidget(self.lbl_empty_detail)
        empty_layout.addStretch()
        self.photo_stack.addWidget(self.photo_empty)
        self.photo_stack.setCurrentWidget(self.photo_empty)
        left_layout.addWidget(self.photo_stack, stretch=1)
        self.lbl_drag_tip = CaptionLabel("双击查看大图 · Ctrl / Shift 多选 · 拖入右侧篮子", self)
        self.lbl_drag_tip.setWordWrap(True)
        left_layout.addWidget(self.lbl_drag_tip)
        self.splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_widget.setMinimumWidth(330)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        right_header = QHBoxLayout()
        right_header.addWidget(SubtitleLabel("去向篮子", self))
        right_header.addStretch()
        self.btn_add_custom_basket = PushButton(FluentIcon.ADD, "新建篮子", self)
        self.btn_add_custom_basket.clicked.connect(self._on_add_custom_basket)
        right_header.addWidget(self.btn_add_custom_basket)
        right_layout.addLayout(right_header)
        basket_tip = CaptionLabel("同一张照片可以分给多个篮子，伴侣文件随同。", self)
        basket_tip.setWordWrap(True)
        right_layout.addWidget(basket_tip)

        self.basket_scroll = SmoothScrollArea(self)
        self.basket_scroll.setWidgetResizable(True)
        self.basket_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.basket_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.basket_cards_container = QWidget()
        self.basket_cards_container.setStyleSheet("background: transparent;")
        self.basket_cards_layout = QVBoxLayout(self.basket_cards_container)
        self.basket_cards_layout.setSpacing(12)
        self.basket_cards_layout.setContentsMargins(0, 0, 10, 0)
        self.basket_cards_layout.addStretch()
        self.basket_scroll.setWidget(self.basket_cards_container)
        right_layout.addWidget(self.basket_scroll, stretch=1)

        self.dispatch_card = CardWidget(self)
        dispatch_layout = QVBoxLayout(self.dispatch_card)
        dispatch_layout.setContentsMargins(16, 12, 16, 12)
        dispatch_layout.setSpacing(8)
        self.lbl_dispatch_summary = StrongBodyLabel("尚无待分发文件", self)
        self.lbl_dispatch_summary.setWordWrap(True)
        dispatch_layout.addWidget(self.lbl_dispatch_summary)
        dispatch_hint = CaptionLabel("保留源照片 · 同盘优先硬链接，跨盘复制", self)
        dispatch_hint.setWordWrap(True)
        dispatch_layout.addWidget(dispatch_hint)
        self.btn_execute_dispatch = PrimaryPushButton(FluentIcon.SEND, "选择输出目录并分拣", self)
        self.btn_execute_dispatch.setEnabled(False)
        self.btn_execute_dispatch.clicked.connect(self._on_execute_physical_dispatch)
        dispatch_layout.addWidget(self.btn_execute_dispatch)
        undo_row = QHBoxLayout()
        self.btn_undo_dispatch = PushButton(FluentIcon.CANCEL, "撤销上次分拣", self)
        self.btn_undo_dispatch.setEnabled(False)
        self.btn_undo_dispatch.clicked.connect(self._on_undo_dispatch)
        undo_row.addWidget(self.btn_undo_dispatch)
        self.btn_open_undo = TransparentPushButton("选择撤销清单", self)
        self.btn_open_undo.clicked.connect(self._on_open_undo_manifest)
        undo_row.addWidget(self.btn_open_undo)
        dispatch_layout.addLayout(undo_row)
        right_layout.addWidget(self.dispatch_card)

        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([720, 450])
        main_layout.addWidget(self.splitter, stretch=1)

        status_row = QHBoxLayout()
        self.lbl_status = CaptionLabel("就绪 · 选择文件夹开始整理", self)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        status_row.addWidget(self.lbl_status, 1)
        status_row.addWidget(CaptionLabel("本地处理 · 照片不上传", self))
        main_layout.addLayout(status_row)
        self.txt_semantic_search.setEnabled(False)
        self.btn_clear_search.setEnabled(False)
        self._render_basket_cards()
        for caption in self.findChildren(CaptionLabel):
            caption.setTextColor("#64748B", "#94A3B8")

    def _on_choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择待分发照片目录")
        if folder:
            self._start_scan_and_analysis(Path(folder))

    def _prompt_text(self, title: str, prompt: str, text: str = "") -> tuple[str, bool]:
        dialog = BasketTextDialog(title, prompt, self.window(), text)
        accepted = bool(dialog.exec())
        value = dialog.text_edit.text()
        dialog.deleteLater()
        return value, accepted

    def _show_message(self, title: str, content: str, confirm: bool = False, confirm_text: str = "删除篮子") -> bool:
        dialog = WorkshopMessageDialog(title, content, self.window(), confirm)
        if confirm:
            dialog.yesButton.setText(confirm_text)
        accepted = bool(dialog.exec())
        dialog.deleteLater()
        return accepted

    def _on_load_test_images(self) -> None:
        project_root = Path(__file__).resolve().parent.parent.parent
        test_dir = project_root / "tests" / "test_images"
        if test_dir.exists():
            self._start_scan_and_analysis(test_dir)
        else:
            self._show_message("未找到测试相册", f"未找到测试集目录: {test_dir}")

    def _start_scan_and_analysis(self, folder: Path) -> None:
        self.asset_groups = []
        self.labels_map = {}
        self.image_matrix = None
        self.baskets = []
        self.txt_semantic_search.clear()
        self.search_similarity_scores.clear()
        self._on_filter_changed("全部")
        self._render_basket_cards()
        self._set_loading(True)
        self.lbl_empty_title.setText("正在准备你的照片")
        self.lbl_empty_detail.setText("正在扫描并分析照片，完成后会显示缩略图和推荐篮子。")
        self.lbl_status.setText(f"正在载入 · {folder.name}")

        self.scan_thread = ScanAndAnalysisThread(folder)
        self.scan_thread.progress_signal.connect(self.lbl_status.setText)
        self.scan_thread.completed_signal.connect(self._on_analysis_finished)
        self.scan_thread.failed_signal.connect(self._on_analysis_failed)
        self.scan_thread.start()

    def _set_loading(self, loading: bool) -> None:
        self.btn_select_dir.setEnabled(not loading)
        self.btn_load_test_images.setEnabled(not loading)
        self.btn_add_custom_basket.setEnabled(not loading)
        self.btn_open_undo.setEnabled(not loading)
        self.btn_undo_dispatch.setEnabled(not loading and self.last_manifest_path is not None)
        self.txt_semantic_search.setEnabled(not loading and self.image_matrix is not None)
        self.btn_clear_search.setEnabled(not loading and bool(self.asset_groups))
        for button in self.filter_buttons.values():
            button.setEnabled(not loading)
        self.progress_bar.setVisible(loading)
        if loading:
            self.progress_bar.start()
            self.btn_execute_dispatch.setEnabled(False)
        else:
            self.progress_bar.stop()

    def _on_analysis_failed(self, message: str) -> None:
        self._set_loading(False)
        self.lbl_empty_title.setText("这次载入未完成")
        self.lbl_empty_detail.setText("请检查文件夹与本地模型后重新选择。\n具体原因见下方状态栏。")
        self.lbl_status.setText(f"载入失败：{message}")

    def _on_analysis_finished(self, asset_groups: list, labels_map: dict, matrix: Optional[np.ndarray]) -> None:
        self.asset_groups = asset_groups
        self.labels_map = labels_map
        self.image_matrix = matrix
        self._set_loading(False)

        pipeline = SortingPipeline(output_root=Path("./output_sorted"))
        self.baskets = pipeline.build_default_baskets(self.asset_groups, self.labels_map)
        self._render_photo_list()
        self._render_basket_cards()
        if self.asset_groups:
            self.lbl_status.setText(f"已载入 {len(self.asset_groups)} 组照片 · 检查去向篮子后即可分拣")
        else:
            self.lbl_status.setText("未在指定目录下发现受支持的照片文件。")

    def _on_execute_semantic_search(self) -> None:
        query = self.txt_semantic_search.text().strip()
        if not query:
            return
        if not self.asset_groups or self.image_matrix is None:
            self._show_message("请先载入照片", "请先选择照片文件夹，再进行搜索。")
            return

        if self.search_engine is None:
            self.search_engine = SemanticSearchEngine()

        identifiers = [g.main_photo_path.name for g in self.asset_groups]
        results = self.search_engine.search(query, self.image_matrix, identifiers, top_k=len(identifiers))
        self.search_similarity_scores = {item.identifier: item.similarity for item in results}

        self.lbl_status.setText(f"🔍 语义检索完成: 按与 \"{query}\" 的相似度重新排序展示！")
        self._render_photo_list()

    def _on_clear_search(self) -> None:
        self.txt_semantic_search.clear()
        self.search_similarity_scores.clear()
        self._render_photo_list()
        self.lbl_status.setText("已重置搜索排序。")

    def _render_photo_list(self) -> None:
        self.photo_list_widget.clear()
        self.visible_asset_groups = []

        groups_to_show = list(self.asset_groups)
        if self.search_similarity_scores:
            groups_to_show.sort(
                key=lambda g: self.search_similarity_scores.get(g.main_photo_path.name, 0.0),
                reverse=True,
            )

        for g in groups_to_show:
            stem = g.main_photo_path.stem
            label = self.labels_map.get(g.main_photo_path.name)

            if self.current_filter != "ALL":
                if not label:
                    continue
                if self.current_filter == "风景" and "风景" not in label.scene_tags:
                    continue
                if self.current_filter == "建筑" and "建筑" not in label.scene_tags:
                    continue
                if self.current_filter == "合照" and "合照" not in label.subject_tags:
                    continue
                if self.current_filter == "单人照" and "单人照" not in label.subject_tags:
                    continue
                if self.current_filter == "有余" and not any("youyu" in p.lower() for p in label.person_names):
                    continue

            item = QListWidgetItem()
            pixmap = QPixmap(str(g.main_photo_path))
            if not pixmap.isNull():
                icon = QIcon(pixmap.scaled(328, 252, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                item.setIcon(icon)
            else:
                item.setIcon(FluentIcon.PHOTO.icon())

            tags_str = " / ".join(label.all_tags[:2]) if label else "未标记"
            comp_tip = f" · {len(g.companions)} 个伴侣" if g.companions else ""

            score_text = ""
            if g.main_photo_path.name in self.search_similarity_scores:
                score_val = self.search_similarity_scores[g.main_photo_path.name]
                score_text = f" · 匹配 {score_val:.2f}"

            filename = self.photo_list_widget.fontMetrics().elidedText(g.main_photo_path.name, Qt.TextElideMode.ElideMiddle, 164)
            item.setText(f"{filename}\n{tags_str}\n照片{comp_tip}{score_text}")
            item.setSizeHint(QSize(184, 204))
            details = [str(g.main_photo_path), " / ".join(label.all_tags) if label else "未标记"]
            details.extend(f"伴侣：{c.filepath.name}" for c in g.companions)
            item.setToolTip("\n".join(details))
            item.setData(Qt.ItemDataRole.UserRole, stem)
            self.photo_list_widget.addItem(item)
            self.visible_asset_groups.append(g)

        visible_count = self.photo_list_widget.count()
        self.lbl_photo_count.setText(f"{visible_count} / {len(self.asset_groups)} 组")
        self.photo_stack.setCurrentWidget(self.photo_list_widget if visible_count else self.photo_empty)
        if not visible_count:
            self.lbl_empty_title.setText("没有符合筛选的照片" if self.asset_groups else "从一个照片文件夹开始")
            self.lbl_empty_detail.setText("试试其他分类，或点击「全部」。" if self.asset_groups else "选择上方文件夹，或先体验测试相册。\n照片与 LivePhoto、RAW、调色文件将一起整理。")

    def _render_basket_cards(self) -> None:
        while self.basket_cards_layout.count() > 1:
            child = self.basket_cards_layout.takeAt(0)
            if child.widget():
                child.widget().hide()
                child.widget().deleteLater()

        pipeline = SortingPipeline(output_root=Path("./output_sorted"))
        plan: DispatchPlan = pipeline.plan_dispatch(self.asset_groups, self.labels_map, self.baskets)
        busy = self.file_task_thread is not None and self.file_task_thread.isRunning()
        self.btn_execute_dispatch.setEnabled(bool(plan.actions) and not busy)
        self.lbl_dispatch_summary.setText(f"{len(self.baskets)} 个篮子 · {len(plan.actions)} 个待分发文件" if plan.actions else "尚无待分发文件")
        if not self.baskets:
            empty_card = CardWidget(self.basket_cards_container)
            empty_layout = QVBoxLayout(empty_card)
            empty_layout.setContentsMargins(20, 24, 20, 24)
            empty_layout.addWidget(StrongBodyLabel("为照片安排一个去向", empty_card))
            hint = BodyLabel("载入后会按人物和题材推荐篮子。\n也可以新建篮子，自由拖入照片。", empty_card)
            hint.setWordWrap(True)
            empty_layout.addWidget(hint)
            self.basket_cards_layout.insertWidget(0, empty_card)

        for basket in self.baskets:
            count = plan.basket_item_counts.get(basket.basket_id, 0)

            # 找到属于该篮子的所有照片组清单
            basket_groups: List[PhotoAssetGroup] = []
            for g in self.asset_groups:
                s_lower = g.main_photo_path.stem.lower()
                if s_lower in [s.lower() for s in basket.excluded_photo_stems]:
                    continue
                label = self.labels_map.get(g.main_photo_path.name)
                is_manual = (s_lower in [s.lower() for s in basket.manual_photo_stems])
                is_auto = False
                if basket.rule:
                    is_auto = pipeline._is_asset_match_rule(g, label, basket.rule)
                if is_manual or is_auto:
                    basket_groups.append(g)

            card = DroppableBasketCard(basket, count, matched_groups=basket_groups)
            card.drop_received.connect(self._on_basket_drop_received)
            card.rename_requested.connect(self._on_basket_rename_requested)
            card.remove_photo_requested.connect(self._on_basket_remove_photo)
            card.delete_basket_requested.connect(self._on_basket_delete_requested)
            card.preview_requested.connect(lambda path, groups=basket_groups: self._preview_photo(path, groups))

            self.basket_cards_layout.insertWidget(self.basket_cards_layout.count() - 1, card)

    def _on_filter_changed(self, tag: str) -> None:
        for t, btn in self.filter_buttons.items():
            btn.setChecked(t == tag)
        self.current_filter = "ALL" if tag == "全部" else tag
        self._render_photo_list()

    def _on_basket_drop_received(self, basket: DispatchBasket, stems: list) -> None:
        basket.manual_photo_stems.update(stems)
        # 若之前曾被排除，从排除黑名单中撤销
        basket.excluded_photo_stems.difference_update([s.lower() for s in stems])
        self._render_basket_cards()
        self.lbl_status.setText(f"🎉 拖拽成功: 已将 {len(stems)} 组照片及伴侣追加放入 [{basket.name}]！")

    def _on_basket_remove_photo(self, basket: DispatchBasket, stem: str) -> None:
        """从特定去向篮子中单独剔除一张照片"""
        s_lower = stem.lower()
        # 从手动追加集合中移除
        basket.manual_photo_stems = {s for s in basket.manual_photo_stems if s.lower() != s_lower}
        # 加入该篮子的排除黑名单 (防止被自动规则再次强行吸附)
        basket.excluded_photo_stems.add(s_lower)
        self._render_basket_cards()
        self.lbl_status.setText(f"已将照片 [{stem}] 从篮子 [{basket.name}] 中移出！")

    def _on_basket_rename_requested(self, basket: DispatchBasket) -> None:
        """支持用户自由修改篮子名称与目标子目录路径"""
        new_name, ok1 = self._prompt_text("修改篮子名称", "请输入新的篮子显示名称:", text=basket.name)
        if not ok1 or not new_name.strip():
            return

        new_folder, ok2 = self._prompt_text("修改输出子目录", "请输入相对输出子目录路径:", text=basket.output_subfolder)
        if not ok2 or not new_folder.strip():
            return

        try:
            SortingPipeline(Path("./output_sorted")).resolve_output_folder(new_folder.strip())
        except ValueError as exc:
            self._show_message("输出目录无效", str(exc))
            return
        basket.name = new_name.strip()
        basket.output_subfolder = new_folder.strip()
        self._render_basket_cards()
        self.lbl_status.setText(f"篮子配置已更新: 名称=[{basket.name}], 目录=[/{basket.output_subfolder}]")

    def _on_basket_delete_requested(self, basket: DispatchBasket) -> None:
        """删除用户不需要的去向篮子"""
        if self._show_message("删除去向篮子？", f"确定要删除去向篮子「{basket.name}」吗？\n只移除篮子配置，不删除源照片或伴侣文件。", confirm=True):
            self.baskets = [b for b in self.baskets if b.basket_id != basket.basket_id]
            self._render_basket_cards()
            self.lbl_status.setText(f"已删除去向篮子: [{basket.name}]")

    def _on_add_custom_basket(self) -> None:
        name, ok = self._prompt_text("新建去向篮子", "请输入篮子名称，例如：5月20日活动相册")
        if ok and name.strip():
            clean_name = name.strip()
            try:
                SortingPipeline(Path("./output_sorted")).resolve_output_folder(f"自定义/{clean_name}")
            except ValueError as exc:
                self._show_message("输出目录无效", str(exc))
                return
            new_basket = DispatchBasket(
                basket_id=f"custom_{clean_name}",
                name=clean_name,
                output_subfolder=f"自定义/{clean_name}",
                rule=None,
            )
            self.baskets.append(new_basket)
            self._render_basket_cards()

    def _on_execute_physical_dispatch(self) -> None:
        """执行物理分拣落地 (优先硬链接秒级分流)并在成功后打开资源管理器"""
        if not self.asset_groups:
            self._show_message("请先载入照片", "请先选择照片文件夹，再进行分拣。")
            return
        if not self.baskets:
            self._show_message("还没有去向篮子", "请先新建篮子，或载入照片以生成推荐篮子。")
            return

        out_folder = QFileDialog.getExistingDirectory(self, "选择物理整理输出根目录", str(Path("./output_sorted").resolve()))
        if not out_folder:
            return

        from core.sorter.file_ops import FileOperationsExecutor
        out_root = Path(out_folder)
        pipeline = SortingPipeline(output_root=out_root)
        try:
            plan: DispatchPlan = pipeline.plan_dispatch(self.asset_groups, self.labels_map, self.baskets)
        except ValueError as exc:
            self._show_message("分拣配置无效", str(exc))
            return

        if not plan.actions:
            self._show_message("没有待分发文件", "当前各篮子中没有待分发的文件。")
            return

        executor = FileOperationsExecutor(prefer_hardlink=True)
        self._start_file_task(lambda: executor.execute_plan(plan), self._on_dispatch_finished, "正在分拣并写入撤销清单，请勿修改输出文件…")

    def _start_file_task(self, operation, completed, status: str) -> None:
        if self.file_task_thread is not None and self.file_task_thread.isRunning():
            return
        self._set_loading(True)
        self.basket_cards_container.setEnabled(False)
        self.btn_open_undo.setEnabled(False)
        self.btn_undo_dispatch.setEnabled(False)
        self.lbl_status.setText(status)
        thread = FileTaskThread(operation)
        self.file_task_thread = thread
        thread.finished.connect(lambda: self._finish_file_task(thread, completed))
        thread.start()

    def _finish_file_task(self, thread: FileTaskThread, completed) -> None:
        self._set_loading(False)
        self.basket_cards_container.setEnabled(True)
        self._render_basket_cards()
        if thread.error is not None:
            self.lbl_status.setText(f"文件任务未完成：{thread.error}")
            self._show_message("文件任务未完成", thread.error)
            return
        completed(thread.result)
        self.btn_undo_dispatch.setEnabled(self.last_manifest_path is not None)

    def _on_dispatch_finished(self, report) -> None:
        out_root = report.output_root
        has_errors = bool(report.failed_count or report.errors)
        if report.manifest_path is not None:
            self.last_manifest_path = report.manifest_path
            self.btn_undo_dispatch.setToolTip(str(report.manifest_path))

        msg = (
            f"{'部分文件分拣失败' if has_errors else '分拣完成'}\n\n"
            f"• 总操作文件数: {report.total_actions} 个 (含全部伴侣文件)\n"
            f"• 成功完成: {report.successful_count} 个\n"
            f"• ⚡ 硬链接秒级完成 (0额外空间): {report.hardlink_count} 个\n"
            f"• 📋 复制完成: {report.copy_count} 个\n"
            f"• 失败数: {report.failed_count} 个\n\n"
            f"整理产物已保存在:\n{out_root.resolve()}"
        )
        if report.manifest_path is not None:
            msg += f"\n\n撤销清单：\n{report.manifest_path}"
        failures = [f"{r.source_path.name}：{r.error_message}" for r in report.results if not r.success]
        if failures or report.errors:
            msg += "\n\n失败详情：\n" + "\n".join([*failures, *report.errors])
        self.lbl_status.setText("分拣已结束 · 可用撤销清单恢复" if report.manifest_path else "分拣未完成 · 未生成撤销清单")
        self._show_message("部分文件分拣失败" if has_errors else "分拣完成", msg)

        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_root.resolve()))):
            self.lbl_status.setText(f"分拣已结束，无法打开文件管理器，请手动查看：{out_root.resolve()}")

    def _on_open_undo_manifest(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择分拣撤销清单", "", "Photoye 撤销清单 (photoye_run_*.undo.json)")
        if filename:
            self._confirm_undo(Path(filename))

    def _on_undo_dispatch(self) -> None:
        if self.last_manifest_path is not None:
            self._confirm_undo(self.last_manifest_path)

    def _confirm_undo(self, manifest: Path) -> None:
        if self._show_message("撤销这次分拣？", f"将校验并移除本次新建的分拣产物；源照片和已有文件不动。\n全部撤销成功后会删除本次清单；未完成则保留以便重试。\n若产物已被修改或替换，将停止撤销。\n请勿在撤销期间编辑输出文件。\n\n清单：{manifest}", confirm=True, confirm_text="确认撤销"):
            self._start_file_task(lambda: undo_transaction(manifest), lambda report: self._on_undo_finished(report, manifest), "正在校验文件并撤销分拣…")

    def _on_undo_finished(self, report, manifest: Path) -> None:
        title = "撤销完成" if report.completed else "撤销已停止"
        if report.completed and self.last_manifest_path is not None and self.last_manifest_path.absolute() == manifest.absolute():
            self.last_manifest_path = None
            self.btn_undo_dispatch.setToolTip("")
            self.btn_undo_dispatch.setEnabled(False)
        cleanup_message = "撤销清单已清理。" if report.completed else "撤销未完成，清单保留以便重试。"
        content = f"已移除：{report.removed_count} 个\n已不存在：{report.missing_count} 个\n失败：{report.failed_count} 项\n源照片未删除，{cleanup_message}"
        if report.errors:
            content += "\n\n" + "\n".join(report.errors)
        self.lbl_status.setText(title)
        self._show_message(title, content)

    def _on_photo_item_double_clicked(self, item: QListWidgetItem) -> None:
        """双击左侧照片弹出高清大图检视窗"""
        row = self.photo_list_widget.row(item)
        if 0 <= row < len(self.visible_asset_groups):
            self._preview_photo(str(self.visible_asset_groups[row].main_photo_path), self.visible_asset_groups)

    def _preview_photo(self, identifier: str, groups: Optional[List[PhotoAssetGroup]] = None) -> None:
        sequence = groups if groups is not None else self.visible_asset_groups
        matched_group = [g for g in sequence if str(g.main_photo_path) == identifier]
        if not matched_group:
            return

        g = matched_group[0]
        label = self.labels_map.get(g.main_photo_path.name)

        from ui.views.preview_dialog import PhotoPreviewDialog
        dlg = PhotoPreviewDialog(asset_group=g, label=label, parent=self, asset_groups=sequence, labels_map=self.labels_map)
        dlg.exec()

