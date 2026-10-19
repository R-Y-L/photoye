#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye V3.0 桌面应用统一启动入口 (Fluent 风格)

基于 PyQt6-Fluent-Widgets 构建浅色整理工作台，支持既有启动脚本。
"""

import os
import sys
from pathlib import Path

# 保证当前工作目录位于 Python 检索根路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import (
    FluentIcon,
    FluentWindow,
    NavigationItemPosition,
    Theme,
    setTheme,
    setThemeColor,
)

from ui.views.sorter_view import SorterWorkshopView


class PhotoyeApplication(FluentWindow):
    """Photoye V3.0 现代 Fluent 风格桌面主窗口"""

    def __init__(self) -> None:
        setThemeColor("#0078D4")
        setTheme(Theme.LIGHT)
        super().__init__()
        self.setMicaEffectEnabled(False)
        self.setWindowTitle("Photoye · 本地照片整理")
        self.resize(1360, 860)
        self.setMinimumSize(1000, 680)

        # 核心子界面装配 (必须设置 objectName 以供 FluentWindow 索引)
        self.sorter_view = SorterWorkshopView(self)
        self.sorter_view.setObjectName("sorterWorkshopView")
        self.addSubInterface(
            self.sorter_view,
            FluentIcon.FOLDER,
            "智能整理工坊",
            position=NavigationItemPosition.TOP,
        )

        # 导航栏宽度与显示设置
        self.navigationInterface.setExpandWidth(180)

        # 窗口居中
        self._center_window()

    def _center_window(self) -> None:
        screen = QApplication.primaryScreen()
        if screen:
            rect = screen.availableGeometry()
            x = rect.x() + (rect.width() - self.width()) // 2
            y = rect.y() + (rect.height() - self.height()) // 2
            self.move(x, y)

    def closeEvent(self, event) -> None:
        thread = self.sorter_view.scan_thread
        if thread is not None and thread.isRunning():
            self.sorter_view.lbl_status.setText("正在分析照片，请等待当前任务结束后关闭窗口。")
            event.ignore()
            return
        super().closeEvent(event)


def main() -> None:
    """应用主入口函数"""
    app = QApplication(sys.argv)

    # 全局字体配置
    font = QFont("Microsoft YaHei UI", 9)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    window = PhotoyeApplication()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
