"""整理工坊的 Fluent 输入、确认与结果弹窗。"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget
from qfluentwidgets import BodyLabel, LineEdit, MessageBoxBase, SmoothScrollArea, SubtitleLabel


class BasketTextDialog(MessageBoxBase):
    """保留单次输入及取消语义，空白输入不允许确认。"""

    def __init__(self, title: str, prompt: str, parent: QWidget, text: str = "") -> None:
        super().__init__(parent)
        self.title_label = SubtitleLabel(title, self)
        self.prompt_label = BodyLabel(prompt, self)
        self.prompt_label.setWordWrap(True)
        self.text_edit = LineEdit(self)
        self.text_edit.setAccessibleName(prompt)
        self.text_edit.setText(text)
        self.text_edit.setClearButtonEnabled(True)
        self.viewLayout.addWidget(self.title_label)
        self.viewLayout.addWidget(self.prompt_label)
        self.viewLayout.addWidget(self.text_edit)
        self.widget.setFixedWidth(min(460, parent.width() - 64))
        self.yesButton.setText("确定")
        self.cancelButton.setText("取消")
        self.yesButton.setEnabled(bool(text.strip()))
        self.text_edit.textChanged.connect(lambda value: self.yesButton.setEnabled(bool(value.strip())))
        self.text_edit.returnPressed.connect(self.yesButton.click)
        self.text_edit.setFocus()
        self.text_edit.selectAll()

    def validate(self) -> bool:
        return bool(self.text_edit.text().strip())


class WorkshopMessageDialog(MessageBoxBase):
    """长结果可滚动，文件名及目录只按纯文本显示。"""

    def __init__(self, title: str, content: str, parent: QWidget, confirm: bool = False) -> None:
        super().__init__(parent)
        self.title_label = SubtitleLabel(title, self)
        self.viewLayout.addWidget(self.title_label)
        scroll = SmoothScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.content_label = BodyLabel(content, self)
        self.content_label.setTextFormat(Qt.TextFormat.PlainText)
        self.content_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.content_label.setWordWrap(True)
        self.content_label.setStyleSheet("background: transparent;")
        scroll.setWidget(self.content_label)
        width = min(520, parent.width() - 64)
        self.widget.setFixedWidth(width)
        text_width = width - 64
        text_height = self.content_label.fontMetrics().boundingRect(
            0, 0, text_width, 100000, Qt.TextFlag.TextWordWrap, content,
        ).height()
        scroll.setFixedHeight(min(max(48, text_height + 12), max(80, parent.height() - 260)))
        self.viewLayout.addWidget(scroll)
        self.yesButton.setText("删除篮子" if confirm else "知道了")
        self.cancelButton.setText("取消")
        if confirm:
            self.cancelButton.setFocus()
        else:
            self.hideCancelButton()