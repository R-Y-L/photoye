#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 照片高保真大图检视器组件 (V3.0)

支持双击/点击任意照片弹出高清大图弹窗，查看人像识别标签、伴侣文件详情与 EXIF 元数据。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QPainter, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FluentIcon,
    PrimaryPushButton,
    PushButton,
    SmoothScrollArea,
    StrongBodyLabel,
    SubtitleLabel,
    TransparentToolButton,
)

from core.ai.scene_classifier import PhotoMultiLabel
from core.metadata.companion_scanner import PhotoAssetGroup


class PhotoImageView(QGraphicsView):
    """通过视图变换缩放原图，避免为高倍率生成巨型位图。"""

    zoom_changed = pyqtSignal(float)
    navigation_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.image_item = QGraphicsPixmapItem()
        self.image_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.image_item)
        self.error_item = self.scene().addText("无法加载照片预览")
        self.error_item.setDefaultTextColor(QColor("#FFFFFF"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setStyleSheet("QGraphicsView { background: #18212F; border: none; border-radius: 8px; }")
        self.setBackgroundBrush(QColor("#18212F"))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.fit_mode = True

    @property
    def zoom(self) -> float:
        return self.transform().m11()

    def set_photo(self, pixmap: QPixmap) -> None:
        self.image_item.setPixmap(pixmap)
        self.error_item.setVisible(pixmap.isNull())
        self.setSceneRect(self.error_item.boundingRect() if pixmap.isNull() else self.image_item.boundingRect())
        self.fit_photo()

    def fit_photo(self) -> None:
        self.fit_mode = True
        self.resetTransform()
        if not self.image_item.pixmap().isNull():
            self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self.sceneRect().center())
        self.zoom_changed.emit(self.zoom)

    def set_zoom(self, zoom: float, anchor: Optional[QPoint] = None) -> None:
        if self.image_item.pixmap().isNull():
            return
        # 极大图片的适应窗口倍率也应能作为缩小边界。
        fit_scale = min(self.viewport().width() / self.sceneRect().width(), self.viewport().height() / self.sceneRect().height())
        zoom = max(min(0.05, fit_scale), min(8.0, zoom))
        anchor = anchor if anchor is not None else self.viewport().rect().center()
        before = self.mapToScene(anchor)
        self.scale(zoom / self.zoom, zoom / self.zoom)
        after = self.mapToScene(anchor)
        self.translate(after.x() - before.x(), after.y() - before.y())
        self.fit_mode = False
        self.zoom_changed.emit(self.zoom)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                if not self.image_item.pixmap().isNull():
                    steps = max(-4.0, min(4.0, delta / 120))
                    self.set_zoom(self.zoom * (1.2 ** steps), event.position().toPoint())
            else:
                self.navigation_requested.emit(-1 if delta > 0 else 1)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self.fit_mode:
                self.set_zoom(1.0, event.position().toPoint())
            else:
                self.fit_photo()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit_photo()


class PhotoPreviewDialog(QDialog):
    """照片大图检视窗口"""

    def __init__(
        self,
        asset_group: PhotoAssetGroup,
        label: Optional[PhotoMultiLabel] = None,
        parent: Optional[QWidget] = None,
        *,
        asset_groups: Optional[List[PhotoAssetGroup]] = None,
        labels_map: Optional[Dict[str, PhotoMultiLabel]] = None,
    ) -> None:
        super().__init__(parent)
        self.asset_group = asset_group
        self.label = label
        self.asset_groups = list(asset_groups) if asset_groups else [asset_group]
        self.current_index = next((i for i, group in enumerate(self.asset_groups) if group.main_photo_path == asset_group.main_photo_path), -1)
        if self.current_index < 0:
            self.asset_groups = [asset_group]
            self.current_index = 0
        self.labels_map = dict(labels_map or {})
        if label is not None:
            self.labels_map[str(asset_group.main_photo_path)] = label
        self._source_pixmap = QPixmap(str(asset_group.main_photo_path))
        self._init_ui()
        self._show_photo()

    def _init_ui(self) -> None:
        self.setWindowTitle(f"照片检视 - {self.asset_group.main_photo_path.name}")
        self.resize(1040, 720)
        self.setMinimumSize(700, 500)
        self.setStyleSheet("PhotoPreviewDialog { background: #F3F6FA; }")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        image_column = QVBoxLayout()
        image_column.setSpacing(10)
        nav_bar = QHBoxLayout()
        self.btn_previous = TransparentToolButton(FluentIcon.LEFT_ARROW, self)
        self.btn_previous.setToolTip("上一张（←）")
        self.btn_previous.setAccessibleName("上一张")
        self.btn_previous.clicked.connect(lambda: self._navigate(-1))
        self.btn_next = TransparentToolButton(FluentIcon.RIGHT_ARROW, self)
        self.btn_next.setToolTip("下一张（→）")
        self.btn_next.setAccessibleName("下一张")
        self.btn_next.clicked.connect(lambda: self._navigate(1))
        self.lbl_position = CaptionLabel(self)
        nav_bar.addWidget(self.btn_previous)
        nav_bar.addWidget(self.lbl_position)
        nav_bar.addWidget(self.btn_next)
        nav_bar.addStretch()
        image_column.addLayout(nav_bar)
        self.image_area = PhotoImageView(self)
        self.image_area.navigation_requested.connect(self._navigate)
        image_column.addWidget(self.image_area, 1)

        zoom_bar = QHBoxLayout()
        self.btn_zoom_out = TransparentToolButton(FluentIcon.REMOVE, self)
        self.btn_zoom_out.setToolTip("缩小（−）")
        self.btn_zoom_out.setAccessibleName("缩小")
        self.btn_zoom_out.clicked.connect(lambda: self.image_area.set_zoom(self.image_area.zoom / 1.2))
        self.btn_zoom_in = TransparentToolButton(FluentIcon.ADD, self)
        self.btn_zoom_in.setToolTip("放大（+）")
        self.btn_zoom_in.setAccessibleName("放大")
        self.btn_zoom_in.clicked.connect(lambda: self.image_area.set_zoom(self.image_area.zoom * 1.2))
        self.lbl_zoom = CaptionLabel(self)
        self.lbl_zoom.setMinimumWidth(46)
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_area.zoom_changed.connect(lambda zoom: self.lbl_zoom.setText(f"{zoom:.0%}"))
        self.btn_fit = PushButton("适应窗口", self)
        self.btn_fit.clicked.connect(self.image_area.fit_photo)
        self.btn_actual = PushButton("1:1", self)
        self.btn_actual.setToolTip("原始尺寸（1）")
        self.btn_actual.clicked.connect(lambda: self.image_area.set_zoom(1.0))
        for widget in (self.btn_zoom_out, self.lbl_zoom, self.btn_zoom_in):
            zoom_bar.addWidget(widget)
        zoom_bar.addStretch()
        zoom_bar.addWidget(self.btn_fit)
        zoom_bar.addWidget(self.btn_actual)
        image_column.addLayout(zoom_bar)
        hint = CaptionLabel("滚轮 / ← → 切图 · Ctrl+滚轮缩放 · 拖动查看 · 双击原图/适应", self)
        hint.setWordWrap(True)
        image_column.addWidget(hint)
        main_layout.addLayout(image_column, stretch=1)

        for key, action in (
            ("Left", lambda: self._navigate(-1)), ("Right", lambda: self._navigate(1)),
            ("+", self.btn_zoom_in.click), ("=", self.btn_zoom_in.click),
            ("-", self.btn_zoom_out.click), ("0", self.image_area.fit_photo),
            ("1", self.btn_actual.click),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)

        meta_panel = CardWidget(self)
        meta_panel.setMinimumWidth(240)
        meta_panel.setMaximumWidth(300)
        meta_layout = QVBoxLayout(meta_panel)
        meta_layout.setContentsMargins(16, 20, 16, 16)
        meta_layout.setSpacing(16)
        meta_layout.addWidget(SubtitleLabel("照片详情", meta_panel))

        self.detail_scroll = SmoothScrollArea(meta_panel)
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.detail_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        meta_layout.addWidget(self.detail_scroll, 1)
        self.btn_close = PrimaryPushButton("完成", meta_panel)
        self.btn_close.clicked.connect(self.accept)
        meta_layout.addWidget(self.btn_close)
        main_layout.addWidget(meta_panel)

    def _show_photo(self) -> None:
        self.asset_group = self.asset_groups[self.current_index]
        path = self.asset_group.main_photo_path
        self.label = self.labels_map.get(str(path)) or self.labels_map.get(path.name)
        self.setWindowTitle(f"照片检视 - {path.name}")
        self._source_pixmap = QPixmap(str(path))
        self.image_area.set_photo(self._source_pixmap)
        self.lbl_position.setText(f"{self.current_index + 1} / {len(self.asset_groups)}")
        self.btn_previous.setEnabled(self.current_index > 0)
        self.btn_next.setEnabled(self.current_index < len(self.asset_groups) - 1)
        for button in (self.btn_zoom_out, self.btn_zoom_in, self.btn_actual, self.btn_fit):
            button.setEnabled(not self._source_pixmap.isNull())

        old_details = self.detail_scroll.takeWidget()
        if old_details is not None:
            old_details.deleteLater()
        details = QWidget()
        details.setStyleSheet("background: transparent;")
        detail_layout = QVBoxLayout(details)
        detail_layout.setContentsMargins(0, 0, 10, 0)
        detail_layout.setSpacing(10)
        lbl_title = StrongBodyLabel(self.asset_group.main_photo_path.name, details)
        lbl_title.setWordWrap(True)
        lbl_title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        detail_layout.addWidget(lbl_title)
        info = CaptionLabel(f"{self.asset_group.filesize / 1024:.0f} KB · {self._source_pixmap.width()} × {self._source_pixmap.height()}", details)
        detail_layout.addWidget(info)
        if self.asset_group.created_at:
            date_label = CaptionLabel(f"拍摄时间 {self.asset_group.created_at}", details)
            date_label.setWordWrap(True)
            detail_layout.addWidget(date_label)

        num_comp = len(self.asset_group.companions)
        detail_layout.addSpacing(8)
        detail_layout.addWidget(StrongBodyLabel(f"伴侣文件 · {num_comp}", details))
        if not num_comp:
            detail_layout.addWidget(CaptionLabel("这张照片没有绑定伴侣文件", details))
        for c in self.asset_group.companions:
            lbl_c = BodyLabel(f"{c.filepath.name}\n{c.file_type} · {c.filesize // 1024} KB", details)
            lbl_c.setWordWrap(True)
            detail_layout.addWidget(lbl_c)

        detail_layout.addSpacing(8)
        detail_layout.addWidget(StrongBodyLabel("AI 识别标签", details))
        if self.label and self.label.all_tags:
            for tag in self.label.all_tags:
                tag_badge = CaptionLabel(tag, details)
                tag_badge.setObjectName("tagBadge")
                tag_badge.setWordWrap(True)
                tag_badge.setStyleSheet("background: #EDF5FF; color: #1D5D96; border-radius: 6px; padding: 6px 10px;")
                detail_layout.addWidget(tag_badge)
        else:
            detail_layout.addWidget(CaptionLabel("暂未分析标签", details))
        detail_layout.addStretch()
        self.detail_scroll.setWidget(details)
        for caption in self.findChildren(CaptionLabel):
            if caption.objectName() != "tagBadge":
                caption.setTextColor("#64748B", "#94A3B8")

    def _navigate(self, step: int) -> None:
        index = self.current_index + step
        if 0 <= index < len(self.asset_groups):
            self.current_index = index
            self._show_photo()
