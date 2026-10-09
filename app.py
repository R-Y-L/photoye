#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye V3.0 桌面应用统一启动入口

启动现代化 PyQt6 桌面应用外壳，默认载入智能整理工坊视图。
支持通过 start_photoye.bat、python app.py 或 python run.py 一键拉起。
"""

import os
import sys
from pathlib import Path

# 保证当前工作目录位于 Python 检索根路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QStatusBar

from ui.views.sorter_view import SorterWorkshopView


class PhotoyeApplication(QMainWindow):
    """Photoye V3.0 桌面顶层主窗口"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Photoye - 本地智能照片整理与回忆展厅 (V3.0)")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)

        # 核心主工作区
        self.sorter_view = SorterWorkshopView(self)
        self.setCentralWidget(self.sorter_view)

        # 底部状态栏
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("QStatusBar { color: #666; font-size: 12px; background: #F5F5F5; }")
        self.status_bar.showMessage("就绪 · 隐私至上 · 本地优先 · 纯 ONNX 运行时就绪")
        self.setStatusBar(self.status_bar)


def main() -> None:
    """应用主入口函数"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 全局字体配置
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    window = PhotoyeApplication()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
