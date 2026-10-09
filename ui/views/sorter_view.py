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
from typing import Dict, List, Optional, Set

import numpy as np
from PyQt6.QtCore import QByteArray, QMimeData, QPoint, QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDrag, QFont, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.ai.face_engine import FaceEngine
from core.ai.face_library import FaceLibrary
from core.ai.scene_classifier import PhotoMultiLabel, SceneClassifier
from core.ai.semantic_search import SemanticSearchEngine
from core.metadata.companion_scanner import CompanionScanner, PhotoAssetGroup
from core.sorter.pipeline import BasketFilterRule, DispatchBasket, DispatchPlan, SortingPipeline
from data.database import create_person, get_connection, init_db


MIME_TYPE_PHOTO_STEMS = "application/x-photoye-stems"


class DraggablePhotoListWidget(QListWidget):
    """支持鼠标抓取拖动的照片列表组件"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(120, 120))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setSpacing(10)
        self.setDragEnabled(True)

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


class DroppableBasketCard(QFrame):
    """支持接收鼠标拖入放下的去向篮子卡片"""
    drop_received = pyqtSignal(object, list)

    def __init__(self, basket: DispatchBasket, count: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.basket = basket
        self.setAcceptDrops(True)
        self._init_ui(count)

    def _init_ui(self, count: int) -> None:
        self.setObjectName("BasketCard")
        self.setStyleSheet("""
            QFrame#BasketCard {
                background: #F8F9FA;
                border: 2px dashed #B0BEC5;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        card_layout = QHBoxLayout(self)
        card_layout.setContentsMargins(10, 10, 10, 10)

        info_layout = QVBoxLayout()
        self.lbl_name = QLabel(f"🧺 {self.basket.name}")
        self.lbl_name.setStyleSheet("font-weight: bold; font-size: 14px; color: #0288D1;")
        
        rule_desc = "手动拖拽库" if not self.basket.rule else "自动规则 + 拖拽"
        self.lbl_detail = QLabel(f"子目录: /{self.basket.output_subfolder} | 已汇集: {count} 组相册资产 ({rule_desc})")
        self.lbl_detail.setStyleSheet("color: #666; font-size: 11px;")
        
        info_layout.addWidget(self.lbl_name)
        info_layout.addWidget(self.lbl_detail)
        card_layout.addLayout(info_layout, stretch=1)

        self.lbl_hint = QLabel("👈 拖拽照片到此处放入")
        self.lbl_hint.setStyleSheet("color: #90A4AE; font-size: 11px; font-style: italic;")
        card_layout.addWidget(self.lbl_hint)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME_TYPE_PHOTO_STEMS):
            event.acceptProposedAction()
            self.setStyleSheet("""
                QFrame#BasketCard {
                    background: #E1F5FE;
                    border: 2px solid #0288D1;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet("""
            QFrame#BasketCard {
                background: #F8F9FA;
                border: 2px dashed #B0BEC5;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        event.accept()

    def dropEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME_TYPE_PHOTO_STEMS):
            raw_data = event.mimeData().data(MIME_TYPE_PHOTO_STEMS).data().decode("utf-8")
            stems = json.loads(raw_data)
            self.drop_received.emit(self.basket, stems)
            self.dragLeaveEvent(event)
            event.acceptProposedAction()
        else:
            event.ignore()


class ScanAndAnalysisThread(QThread):
    """后台异步扫描与 AI 多模态特征分析工作线程"""
    progress_signal = pyqtSignal(str)
    completed_signal = pyqtSignal(list, dict, object)

    def __init__(self, folder_path: Path) -> None:
        super().__init__()
        self.folder_path = folder_path

    def run(self) -> None:
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


class SorterWorkshopView(QWidget):
    """整理工坊主交互面板 (完善版)"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.asset_groups: List[PhotoAssetGroup] = []
        self.labels_map: Dict[str, PhotoMultiLabel] = {}
        self.image_matrix: Optional[np.ndarray] = None
        self.search_similarity_scores: Dict[str, float] = {}
        self.baskets: List[DispatchBasket] = []
        self.current_filter: str = "ALL"

        self.search_engine: Optional[SemanticSearchEngine] = None
        self.scan_thread: Optional[ScanAndAnalysisThread] = None
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. 顶部操作栏
        top_bar = QHBoxLayout()
        self.btn_select_dir = QPushButton("📂 选择相册文件夹")
        self.btn_select_dir.setFixedHeight(38)
        self.btn_select_dir.setStyleSheet("QPushButton { font-weight: bold; font-size: 13px; background-color: #0288D1; color: white; border-radius: 4px; padding: 0 16px; } QPushButton:hover { background-color: #0277BD; }")
        self.btn_select_dir.clicked.connect(self._on_choose_folder)

        self.btn_load_test_images = QPushButton("🚀 载入内置测试集 (23张)")
        self.btn_load_test_images.setFixedHeight(38)
        self.btn_load_test_images.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 4px; padding: 0 14px; font-weight: bold; } QPushButton:hover { background-color: #388E3C; }")
        self.btn_load_test_images.clicked.connect(self._on_load_test_images)

        self.lbl_status = QLabel("就绪。请选择待整理的照片文件夹，或直接载入默认测试集。")
        self.lbl_status.setStyleSheet("color: #555; font-size: 13px;")

        top_bar.addWidget(self.btn_select_dir)
        top_bar.addWidget(self.btn_load_test_images)
        top_bar.addWidget(self.lbl_status, stretch=1)
        main_layout.addLayout(top_bar)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #E0E0E0;")
        main_layout.addWidget(line)

        # 2. 中间左右双分区
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---------- 左侧面板: 待分发候选池 ----------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)

        # 语义搜索输入条
        search_bar = QHBoxLayout()
        self.txt_semantic_search = QLineEdit()
        self.txt_semantic_search.setFixedHeight(34)
        self.txt_semantic_search.setPlaceholderText("🔍 输入自然语言查图 (如: 海边日落、树林风景、朋友合影、个人肖像)... 回车搜图")
        self.txt_semantic_search.setStyleSheet("QLineEdit { border: 1px solid #B0BEC5; border-radius: 4px; padding: 0 10px; font-size: 13px; background: white; } QLineEdit:focus { border: 2px solid #0288D1; }")
        self.txt_semantic_search.returnPressed.connect(self._on_execute_semantic_search)

        self.btn_search = QPushButton("搜索")
        self.btn_search.setFixedHeight(34)
        self.btn_search.setStyleSheet("QPushButton { background: #0288D1; color: white; border-radius: 4px; padding: 0 14px; font-weight: bold; }")
        self.btn_search.clicked.connect(self._on_execute_semantic_search)

        self.btn_clear_search = QPushButton("重置")
        self.btn_clear_search.setFixedHeight(34)
        self.btn_clear_search.setStyleSheet("QPushButton { background: #ECEFF1; color: #455A64; border-radius: 4px; padding: 0 10px; }")
        self.btn_clear_search.clicked.connect(self._on_clear_search)

        search_bar.addWidget(self.txt_semantic_search, stretch=1)
        search_bar.addWidget(self.btn_search)
        search_bar.addWidget(self.btn_clear_search)
        left_layout.addLayout(search_bar)

        # 快捷漏斗栏
        self.filter_bar = QHBoxLayout()
        self.lbl_filter_title = QLabel("快捷分类:")
        self.lbl_filter_title.setStyleSheet("font-weight: bold; color: #333;")
        self.filter_bar.addWidget(self.lbl_filter_title)

        self.filter_buttons: Dict[str, QPushButton] = {}
        for tag in ["全部", "风景", "建筑", "合照", "单人照", "有余"]:
            btn = QPushButton(tag)
            btn.setCheckable(True)
            btn.setStyleSheet("QPushButton { padding: 4px 10px; border-radius: 12px; border: 1px solid #CCC; background: #FFF; } QPushButton:checked { background: #0288D1; color: white; border: 1px solid #0288D1; font-weight: bold; }")
            btn.clicked.connect(lambda checked, t=tag: self._on_filter_changed(t))
            self.filter_bar.addWidget(btn)
            self.filter_buttons[tag] = btn

        self.filter_buttons["全部"].setChecked(True)
        self.filter_bar.addStretch()
        left_layout.addLayout(self.filter_bar)

        # 原生支持拖拽的照片列表
        self.photo_list_widget = DraggablePhotoListWidget()
        self.photo_list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #DDD;
                border-radius: 6px;
                background-color: #FAFAFA;
                padding: 8px;
            }
            QListWidget::item {
                background: white;
                border: 1px solid #EEE;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item:selected {
                background: #E1F5FE;
                border: 2px solid #0288D1;
            }
        """)
        left_layout.addWidget(self.photo_list_widget, stretch=1)

        # 底部提示
        left_bottom = QHBoxLayout()
        self.lbl_drag_tip = QLabel("💡 提示：按住鼠标可直接将选中的照片拖入右侧篮子中放入")
        self.lbl_drag_tip.setStyleSheet("color: #78909C; font-size: 12px;")
        left_bottom.addWidget(self.lbl_drag_tip)
        left_bottom.addStretch()
        left_layout.addLayout(left_bottom)

        splitter.addWidget(left_widget)

        # ---------- 右侧面板: 多路去向篮子栏 ----------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)

        right_header = QHBoxLayout()
        lbl_baskets_title = QLabel("🧺 多路分发去向篮子 (直接拖拽照片进入)")
        lbl_baskets_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")
        right_header.addWidget(lbl_baskets_title)
        right_header.addStretch()

        self.btn_add_custom_basket = QPushButton("+ 新建篮子")
        self.btn_add_custom_basket.setStyleSheet("background: #EDE7F6; color: #512DA8; border: 1px solid #D1C4E9; padding: 4px 10px; border-radius: 4px; font-weight: bold;")
        self.btn_add_custom_basket.clicked.connect(self._on_add_custom_basket)
        right_header.addWidget(self.btn_add_custom_basket)
        right_layout.addLayout(right_header)

        self.basket_scroll = QScrollArea()
        self.basket_scroll.setWidgetResizable(True)
        self.basket_scroll.setStyleSheet("QScrollArea { border: 1px solid #DDD; border-radius: 6px; background: white; }")

        self.basket_cards_container = QWidget()
        self.basket_cards_layout = QVBoxLayout(self.basket_cards_container)
        self.basket_cards_layout.setSpacing(10)
        self.basket_cards_layout.setContentsMargins(10, 10, 10, 10)
        self.basket_cards_layout.addStretch()

        self.basket_scroll.setWidget(self.basket_cards_container)
        right_layout.addWidget(self.basket_scroll, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setSizes([700, 480])
        main_layout.addWidget(splitter, stretch=1)

    def _on_choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择待分发照片目录")
        if folder:
            self._start_scan_and_analysis(Path(folder))

    def _on_load_test_images(self) -> None:
        project_root = Path(__file__).resolve().parent.parent.parent
        test_dir = project_root / "tests" / "test_images"
        if test_dir.exists():
            self._start_scan_and_analysis(test_dir)
        else:
            QMessageBox.warning(self, "提示", f"未找到测试集目录: {test_dir}")

    def _start_scan_and_analysis(self, folder: Path) -> None:
        self.btn_select_dir.setEnabled(False)
        self.btn_load_test_images.setEnabled(False)
        self.photo_list_widget.clear()
        self.search_similarity_scores.clear()

        self.scan_thread = ScanAndAnalysisThread(folder)
        self.scan_thread.progress_signal.connect(self.lbl_status.setText)
        self.scan_thread.completed_signal.connect(self._on_analysis_finished)
        self.scan_thread.start()

    def _on_analysis_finished(self, asset_groups: list, labels_map: dict, matrix: Optional[np.ndarray]) -> None:
        self.btn_select_dir.setEnabled(True)
        self.btn_load_test_images.setEnabled(True)
        self.asset_groups = asset_groups
        self.labels_map = labels_map
        self.image_matrix = matrix

        if not self.asset_groups:
            self.lbl_status.setText("⚠️ 未在指定目录下发现受支持的照片文件。")
            return

        pipeline = SortingPipeline(output_root=Path("./output_sorted"))
        self.baskets = pipeline.build_default_baskets(self.asset_groups, self.labels_map)
        self._render_photo_list()
        self._render_basket_cards()

    def _on_execute_semantic_search(self) -> None:
        query = self.txt_semantic_search.text().strip()
        if not query:
            return
        if not self.asset_groups or self.image_matrix is None:
            QMessageBox.information(self, "提示", "请先载入相册照片！")
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
                icon = QIcon(pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                item.setIcon(icon)

            tags_str = "/".join(label.all_tags[:2]) if label else "未标记"
            comp_tip = f" (+{len(g.companions)}伴侣)" if g.companions else ""

            score_text = ""
            if g.main_photo_path.name in self.search_similarity_scores:
                score_val = self.search_similarity_scores[g.main_photo_path.name]
                score_text = f" [匹配:{score_val:.2f}]"

            item.setText(f"{g.main_photo_path.name[:13]}..\n[{tags_str}]{score_text}{comp_tip}")
            item.setData(Qt.ItemDataRole.UserRole, stem)
            self.photo_list_widget.addItem(item)

    def _render_basket_cards(self) -> None:
        while self.basket_cards_layout.count() > 1:
            child = self.basket_cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        pipeline = SortingPipeline(output_root=Path("./output_sorted"))
        plan: DispatchPlan = pipeline.plan_dispatch(self.asset_groups, self.labels_map, self.baskets)

        for basket in self.baskets:
            count = plan.basket_item_counts.get(basket.basket_id, 0)
            card = DroppableBasketCard(basket, count)
            card.drop_received.connect(self._on_basket_drop_received)
            self.basket_cards_layout.insertWidget(self.basket_cards_layout.count() - 1, card)

    def _on_filter_changed(self, tag: str) -> None:
        for t, btn in self.filter_buttons.items():
            btn.setChecked(t == tag)
        self.current_filter = "ALL" if tag == "全部" else tag
        self._render_photo_list()

    def _on_basket_drop_received(self, basket: DispatchBasket, stems: list) -> None:
        basket.manual_photo_stems.update(stems)
        self._render_basket_cards()
        self.lbl_status.setText(f"🎉 拖拽成功: 已将 {len(stems)} 组照片及伴侣追加放入 [{basket.name}]！")

    def _on_add_custom_basket(self) -> None:
        name, ok = QInputDialog.getText(self, "新建自定义去向篮子", "请输入篮子名称 (如: 5月20日活动相册):")
        if ok and name.strip():
            clean_name = name.strip()
            new_basket = DispatchBasket(
                basket_id=f"custom_{clean_name}",
                name=clean_name,
                output_subfolder=f"自定义/{clean_name}",
                rule=None,
            )
