#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 照片高保真大图检视器组件 (V3.0)

支持双击/点击任意照片弹出高清大图弹窗，查看人像识别标签、伴侣文件详情与 EXIF 元数据。
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    PrimaryPushButton,
    SmoothScrollArea,
    StrongBodyLabel,
    SubtitleLabel,
)

from core.ai.scene_classifier import PhotoMultiLabel
from core.metadata.companion_scanner import PhotoAssetGroup


class PhotoPreviewDialog(QDialog):
    """照片大图检视窗口"""

    def __init__(
        self,
        asset_group: PhotoAssetGroup,
        label: Optional[PhotoMultiLabel] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.asset_group = asset_group
        self.label = label
        self._source_pixmap = QPixmap(str(asset_group.main_photo_path))
        self._init_ui()

    def _init_ui(self) -> None:
        self.setWindowTitle(f"照片检视 - {self.asset_group.main_photo_path.name}")
        self.resize(1040, 720)
        self.setMinimumSize(700, 500)
        self.setStyleSheet("PhotoPreviewDialog { background: #F3F6FA; }")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        self.image_area = QScrollArea(self)
        self.image_area.setWidgetResizable(True)
        self.image_area.setStyleSheet("QScrollArea { background: #18212F; border: none; border-radius: 8px; }")

        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumSize(1, 1)
        self.lbl_image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.lbl_image.setStyleSheet("background: #18212F; color: #FFFFFF;")

        if self._source_pixmap.isNull():
            self.lbl_image.setText("无法加载照片预览")
        self.image_area.setWidget(self.lbl_image)
        self.image_area.viewport().installEventFilter(self)
        main_layout.addWidget(self.image_area, stretch=1)

        meta_panel = CardWidget(self)
        meta_panel.setMinimumWidth(240)
        meta_panel.setMaximumWidth(300)
        meta_layout = QVBoxLayout(meta_panel)
        meta_layout.setContentsMargins(16, 20, 16, 16)
        meta_layout.setSpacing(16)
        meta_layout.addWidget(SubtitleLabel("照片详情", meta_panel))

        detail_scroll = SmoothScrollArea(meta_panel)
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        detail_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
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
        detail_scroll.setWidget(details)
        meta_layout.addWidget(detail_scroll, 1)
        self.btn_close = PrimaryPushButton("完成", meta_panel)
        self.btn_close.clicked.connect(self.accept)
        meta_layout.addWidget(self.btn_close)
        main_layout.addWidget(meta_panel)
        for caption in self.findChildren(CaptionLabel):
            if caption.objectName() != "tagBadge":
                caption.setTextColor("#64748B", "#94A3B8")

    def eventFilter(self, watched, event) -> bool:
        if watched is self.image_area.viewport() and event.type() == QEvent.Type.Resize:
            if not self._source_pixmap.isNull():
                self.lbl_image.setPixmap(self._source_pixmap.scaled(
                    event.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        return super().eventFilter(watched, event)
