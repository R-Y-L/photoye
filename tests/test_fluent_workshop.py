"""Fluent 整理工坊行为回归；用临时照片隔离模型推理与业务数据。"""

import ast
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtCore import QByteArray, QEvent, QMimeData, QPoint, QPointF, Qt, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent, QWheelEvent
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import BodyLabel, CaptionLabel, LineEdit, MessageBoxBase, PrimaryPushButton, PushButton, SearchLineEdit, SmoothScrollArea, StrongBodyLabel

from app import PhotoyeApplication
from core.ai.scene_classifier import PhotoMultiLabel
from core.ai.semantic_search import SearchResultItem
from core.metadata.companion_scanner import CompanionAsset, PhotoAssetGroup
from core.sorter.pipeline import DispatchBasket, SortingPipeline
from ui.views import sorter_view
from ui.views.preview_dialog import PhotoPreviewDialog
from ui.views.workshop_dialogs import BasketTextDialog, WorkshopMessageDialog
from ui.views.sorter_view import (
    MIME_TYPE_PHOTO_STEMS,
    DroppableBasketCard,
    ScanAndAnalysisThread,
    SorterWorkshopView,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


@pytest.fixture
def album(tmp_path):
    groups = []
    labels = {}
    for name, color, scene in (("lake", "steelblue", "风景"), ("city", "tan", "建筑")):
        photo = tmp_path / f"{name}.jpg"
        Image.new("RGB", (640, 400), color).save(photo)
        groups.append(PhotoAssetGroup(main_photo_path=photo, filesize=photo.stat().st_size))
        labels[photo.name] = PhotoMultiLabel(
            subject_tags=["无人物"], person_names=[], scene_tags=[scene],
            scene_scores={scene: 0.8}, all_tags=[scene, "无人物"],
        )
    companion = tmp_path / "lake.xmp"
    companion.write_bytes(b"sidecar")
    groups[0].companions.append(CompanionAsset(
        filepath=companion, file_type="sidecar", filesize=companion.stat().st_size,
    ))
    return groups, labels, np.zeros((2, 512), dtype=np.float32)


@pytest.fixture
def view(qapp):
    widget = SorterWorkshopView()
    widget.resize(1220, 760)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def active_cards(view):
    return [
        view.basket_cards_layout.itemAt(i).widget()
        for i in range(view.basket_cards_layout.count() - 1)
        if isinstance(view.basket_cards_layout.itemAt(i).widget(), DroppableBasketCard)
    ]


@pytest.fixture
def modal_driver(monkeypatch):
    """定时操作真实模态窗口，保留 exec、验证和淡出关闭路径。"""
    actions = []
    original_exec = MessageBoxBase.exec

    def execute(dialog):
        assert actions, f"Unexpected dialog: {type(dialog).__name__}"
        action = actions.pop(0)
        errors = []

        def interact():
            try:
                assert dialog.isVisible()
                assert isinstance(dialog.yesButton, PrimaryPushButton)
                action(dialog)
            except Exception as exc:
                errors.append(exc)
                dialog.reject()

        timeout = QTimer(dialog)
        timeout.setSingleShot(True)
        timeout.timeout.connect(dialog.reject)
        timeout.start(3000)
        QTimer.singleShot(0, interact)
        result = original_exec(dialog)
        timeout.stop()
        if errors:
            raise errors[0]
        return result

    monkeypatch.setattr(BasketTextDialog, "exec", execute)
    monkeypatch.setattr(WorkshopMessageDialog, "exec", execute)
    yield actions
    assert not actions, "Expected dialog was not opened"


def test_window_starts_with_fluent_controls(qapp):
    window = PhotoyeApplication()
    try:
        window.show()
        qapp.processEvents()
        assert window.isVisible()
        assert isinstance(window.sorter_view.btn_select_dir, PrimaryPushButton)
        assert isinstance(window.sorter_view.txt_semantic_search, SearchLineEdit)
        assert window.sorter_view.photo_stack.currentWidget() is window.sorter_view.photo_empty
        assert not window.sorter_view.btn_execute_dispatch.isEnabled()
    finally:
        window.close()
        window.deleteLater()


def test_load_filter_and_search_reset(view, album, qapp):
    view._on_analysis_finished(*album)
    qapp.processEvents()
    assert view.photo_list_widget.count() == 2
    assert not view.photo_list_widget.item(0).icon().isNull()
    assert "lake.xmp" in view.photo_list_widget.item(0).toolTip()
    assert len(active_cards(view)) == 2
    assert view.btn_execute_dispatch.isEnabled()
    assert "3 个待分发文件" in view.lbl_dispatch_summary.text()
    QTest.mouseClick(view.filter_buttons["风景"], Qt.MouseButton.LeftButton)
    assert view.current_filter == "风景"
    assert view.photo_list_widget.count() == 1
    assert view.photo_list_widget.item(0).data(Qt.ItemDataRole.UserRole) == "lake"
    QTest.mouseClick(view.filter_buttons["合照"], Qt.MouseButton.LeftButton)
    assert view.photo_list_widget.count() == 0
    assert view.photo_stack.currentWidget() is view.photo_empty
    QTest.mouseClick(view.filter_buttons["全部"], Qt.MouseButton.LeftButton)
    assert view.photo_list_widget.count() == 2

    # 搜索服务在边界处替身化，真实搜索信号、排序和清除路径仍由界面执行。
    engine = Mock()
    engine.search.return_value = [SearchResultItem("city.jpg", 0.9), SearchResultItem("lake.jpg", 0.2)]
    view.search_engine = engine
    view.txt_semantic_search.setText("city")
    QTest.keyClick(view.txt_semantic_search, Qt.Key.Key_Return)
    engine.search.assert_called_once()
    assert view.photo_list_widget.item(0).data(Qt.ItemDataRole.UserRole) == "city"
    QTest.mouseClick(view.btn_clear_search, Qt.MouseButton.LeftButton)
    assert view.txt_semantic_search.text() == ""
    assert view.search_similarity_scores == {}
    assert view.photo_list_widget.item(0).data(Qt.ItemDataRole.UserRole) == "lake"
    view.txt_semantic_search.setText("city")
    QTest.mouseClick(view.txt_semantic_search.searchButton, Qt.MouseButton.LeftButton)
    assert engine.search.call_count == 2
    QTest.mouseClick(view.txt_semantic_search.clearButton, Qt.MouseButton.LeftButton)
    assert view.search_similarity_scores == {}


def test_real_drop_event_and_remove_preserve_companions(view, album, qapp):
    view._on_analysis_finished(*album)
    basket = DispatchBasket("manual", "精选", "selected")
    view.baskets = [basket]
    view._render_basket_cards()
    qapp.processEvents()
    card = active_cards(view)[0]
    mime = QMimeData()
    mime.setData(MIME_TYPE_PHOTO_STEMS, QByteArray(json.dumps(["lake"]).encode()))
    enter = QDragEnterEvent(QPoint(10, 10), Qt.DropAction.CopyAction, mime,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(card, enter)
    assert enter.isAccepted()
    drop = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime,
                      Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(card, drop)
    assert drop.isAccepted()
    assert basket.manual_photo_stems == {"lake"}
    plan = SortingPipeline("unused").plan_dispatch(view.asset_groups, view.labels_map, view.baskets)
    assert [a.source_path.name for a in plan.actions] == ["lake.jpg", "lake.xmp"]
    qapp.processEvents()
    remove = active_cards(view)[0].findChild(PushButton, "removePhotoButton")
    QTest.mouseClick(remove, Qt.MouseButton.LeftButton)
    assert "lake" in basket.excluded_photo_stems
    assert not view.btn_execute_dispatch.isEnabled()
    assert not SortingPipeline("unused").plan_dispatch(view.asset_groups, view.labels_map, view.baskets).actions


def test_drag_source_encodes_all_selected_photos(view, album, monkeypatch):
    view._on_analysis_finished(*album)
    for index in range(view.photo_list_widget.count()):
        view.photo_list_widget.item(index).setSelected(True)
    captured = []

    def capture_drag(drag, actions):
        captured.extend(json.loads(bytes(drag.mimeData().data(MIME_TYPE_PHOTO_STEMS))))
        return Qt.DropAction.CopyAction

    monkeypatch.setattr(sorter_view.QDrag, "exec", capture_drag)
    view.photo_list_widget.startDrag(Qt.DropAction.CopyAction)
    assert captured == ["lake", "city"]


def test_basket_preview_opens_real_dialog(view, album, monkeypatch, qapp):
    view._on_analysis_finished(*album)
    qapp.processEvents()
    opened = []

    def inspect_dialog(dialog):
        dialog.show()
        qapp.processEvents()
        assert not dialog.image_area.image_item.pixmap().isNull()
        assert dialog.asset_group.main_photo_path == album[0][0].main_photo_path
        opened.append(dialog)
        dialog.close()
        return 0

    monkeypatch.setattr(PhotoPreviewDialog, "exec", inspect_dialog)
    card = next(c for c in active_cards(view) if c.basket.rule.scene_tags == ["风景"])
    QTest.mouseClick(card.findChild(PushButton, "basketThumbnail"), Qt.MouseButton.LeftButton)
    view.photo_list_widget.itemDoubleClicked.emit(view.photo_list_widget.item(0))
    assert len(opened) == 2


def test_preview_resizes_without_distortion(qapp, album):
    dialog = PhotoPreviewDialog(album[0][0], album[1]["lake.jpg"])
    try:
        dialog.show()
        qapp.processEvents()
        first = dialog.image_area.transform().mapRect(dialog.image_area.image_item.boundingRect()).size()
        assert first.width() / first.height() == pytest.approx(1.6, abs=0.01)
        assert any(label.text() == "风景" for label in dialog.findChildren(CaptionLabel))
        dialog.resize(800, 550)
        qapp.processEvents()
        smaller = dialog.image_area.transform().mapRect(dialog.image_area.image_item.boundingRect()).size()
        assert smaller.width() < first.width()
        assert smaller.width() / smaller.height() == pytest.approx(1.6, abs=0.01)
        QTest.mouseClick(dialog.btn_close, Qt.MouseButton.LeftButton)
        assert not dialog.isVisible()
    finally:
        dialog.close()


@pytest.fixture
def preview(qapp, album):
    dialog = PhotoPreviewDialog(album[0][0], album[1]["lake.jpg"], asset_groups=album[0], labels_map=album[1])
    dialog.resize(900, 620)
    dialog.show()
    dialog.activateWindow()
    qapp.processEvents()
    yield dialog
    dialog.close()
    dialog.deleteLater()
    qapp.processEvents()


def send_wheel(image_view, delta, position=None, modifiers=Qt.KeyboardModifier.NoModifier):
    viewport = image_view.viewport()
    position = position or viewport.rect().center()
    event = QWheelEvent(
        QPointF(position), QPointF(viewport.mapToGlobal(position)), QPoint(), QPoint(0, delta),
        Qt.MouseButton.NoButton, modifiers, Qt.ScrollPhase.NoScrollPhase, False,
    )
    QApplication.sendEvent(viewport, event)
    assert event.isAccepted()


def test_preview_wheel_zoom_buttons_and_limits(preview, qapp):
    image = preview.image_area
    fitted = image.zoom
    send_wheel(image, 120, modifiers=Qt.KeyboardModifier.ControlModifier)
    assert image.zoom == pytest.approx(fitted * 1.2)
    assert not image.fit_mode
    assert preview.current_index == 0
    send_wheel(image, -120, modifiers=Qt.KeyboardModifier.ControlModifier)
    assert image.zoom == pytest.approx(fitted)
    QTest.mouseClick(preview.btn_zoom_in, Qt.MouseButton.LeftButton)
    assert image.zoom > fitted
    QTest.mouseClick(preview.btn_zoom_out, Qt.MouseButton.LeftButton)
    assert image.zoom == pytest.approx(fitted)
    QTest.mouseClick(preview.btn_actual, Qt.MouseButton.LeftButton)
    assert image.zoom == pytest.approx(1.0)
    assert preview.lbl_zoom.text() == "100%"
    for _ in range(20):
        send_wheel(image, 480, modifiers=Qt.KeyboardModifier.ControlModifier)
    assert image.zoom == pytest.approx(8.0)
    for _ in range(20):
        send_wheel(image, -480, modifiers=Qt.KeyboardModifier.ControlModifier)
    assert image.zoom == pytest.approx(0.05)
    assert preview.current_index == 0
    QTest.mouseClick(preview.btn_fit, Qt.MouseButton.LeftButton)
    assert image.fit_mode
    assert image.zoom == pytest.approx(fitted)
    qapp.processEvents()


def test_preview_wheel_anchor_drag_and_double_click(preview, qapp):
    image = preview.image_area
    QTest.mouseClick(preview.btn_actual, Qt.MouseButton.LeftButton)
    for _ in range(5):
        QTest.mouseClick(preview.btn_zoom_in, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    viewport = image.viewport()
    anchor = viewport.rect().center() + QPoint(20, 15)
    before = image.mapToScene(anchor)
    send_wheel(image, 120, anchor, Qt.KeyboardModifier.ControlModifier)
    after = image.mapToScene(anchor)
    assert abs(before.x() - after.x()) < 1.0
    assert abs(before.y() - after.y()) < 1.0
    start = viewport.rect().center()
    end = start - QPoint(40, 30)
    h_before = image.horizontalScrollBar().value()
    v_before = image.verticalScrollBar().value()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    move = QMouseEvent(QEvent.Type.MouseMove, QPointF(end), QPointF(viewport.mapToGlobal(end)),
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(viewport, move)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
    assert image.horizontalScrollBar().value() > h_before
    assert image.verticalScrollBar().value() > v_before
    QTest.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=start)
    assert image.fit_mode
    QTest.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=start)
    assert not image.fit_mode
    assert image.zoom == pytest.approx(1.0)


def test_preview_keyboard_navigation_syncs_details_and_boundaries(preview, album, qapp):
    assert not preview.btn_previous.isEnabled()
    assert preview.lbl_position.text() == "1 / 2"
    QTest.keyClick(preview.image_area.viewport(), Qt.Key.Key_Left)
    assert preview.current_index == 0
    QTest.mouseClick(preview.btn_zoom_in, Qt.MouseButton.LeftButton)
    assert not preview.image_area.fit_mode
    QTest.keyClick(preview.image_area.viewport(), Qt.Key.Key_Right)
    qapp.processEvents()
    assert preview.asset_group is album[0][1]
    assert preview.current_index == 1
    assert preview.label is album[1]["city.jpg"]
    assert "city.jpg" in preview.windowTitle()
    assert preview.lbl_position.text() == "2 / 2"
    assert not preview.btn_next.isEnabled()
    assert preview.image_area.fit_mode
    details = preview.detail_scroll.widget()
    assert [label.text() for label in details.findChildren(CaptionLabel, "tagBadge")] == ["建筑", "无人物"]
    assert any(label.text() == "伴侣文件 · 0" for label in details.findChildren(StrongBodyLabel))
    QTest.keyClick(preview.btn_close, Qt.Key.Key_Right)
    assert preview.current_index == 1
    QTest.keyClick(preview.btn_close, Qt.Key.Key_Left)
    assert preview.asset_group is album[0][0]
    assert "lake.xmp" in "\n".join(label.text() for label in preview.detail_scroll.widget().findChildren(BodyLabel))
    QTest.mouseClick(preview.btn_next, Qt.MouseButton.LeftButton)
    assert preview.current_index == 1
    QTest.mouseClick(preview.btn_previous, Qt.MouseButton.LeftButton)
    assert preview.current_index == 0
    QTest.keyClick(preview, Qt.Key.Key_Escape)
    assert not preview.isVisible()


def test_preview_keyboard_zoom_and_resize_modes(preview, qapp):
    QTest.keyClick(preview.image_area.viewport(), Qt.Key.Key_1)
    assert preview.image_area.zoom == pytest.approx(1.0)
    QTest.keyClick(preview.image_area.viewport(), Qt.Key.Key_Equal)
    assert preview.image_area.zoom == pytest.approx(1.2)
    QTest.keyClick(preview.image_area.viewport(), Qt.Key.Key_Minus)
    assert preview.image_area.zoom == pytest.approx(1.0)
    preview.resize(800, 550)
    qapp.processEvents()
    assert preview.image_area.zoom == pytest.approx(1.0)
    QTest.keyClick(preview.image_area.viewport(), Qt.Key.Key_0)
    assert preview.image_area.fit_mode
    fitted = preview.image_area.zoom
    preview.resize(1040, 720)
    qapp.processEvents()
    assert preview.image_area.zoom > fitted
    for button in (preview.btn_previous, preview.btn_next, preview.btn_fit, preview.btn_actual, preview.btn_zoom_in):
        assert preview.rect().contains(button.mapTo(preview, button.rect().bottomRight()))


def test_preview_missing_image_can_navigate_to_valid(qapp, album, tmp_path):
    missing = PhotoAssetGroup(main_photo_path=tmp_path / "missing.jpg", filesize=0)
    dialog = PhotoPreviewDialog(missing, asset_groups=[missing, album[0][0]], labels_map=album[1])
    try:
        dialog.show()
        dialog.activateWindow()
        qapp.processEvents()
        assert dialog.image_area.error_item.isVisible()
        assert not dialog.btn_zoom_in.isEnabled()
        send_wheel(dialog.image_area, -120, modifiers=Qt.KeyboardModifier.ControlModifier)
        assert dialog.image_area.image_item.pixmap().isNull()
        assert dialog.current_index == 0
        send_wheel(dialog.image_area, -120)
        assert not dialog.image_area.error_item.isVisible()
        assert dialog.btn_zoom_in.isEnabled()
        assert not dialog.image_area.image_item.pixmap().isNull()
    finally:
        dialog.close()


def test_preview_plain_wheel_navigates_without_zooming(preview, album):
    image = preview.image_area
    QTest.mouseClick(preview.btn_actual, Qt.MouseButton.LeftButton)
    send_wheel(image, 120)
    assert preview.current_index == 0
    assert image.zoom == pytest.approx(1.0)
    send_wheel(image, -120)
    assert preview.asset_group is album[0][1]
    assert preview.lbl_position.text() == "2 / 2"
    assert preview.label is album[1]["city.jpg"]
    assert image.fit_mode
    fitted = image.zoom
    send_wheel(image, -120)
    assert preview.current_index == 1
    assert image.zoom == pytest.approx(fitted)
    send_wheel(image, 120, modifiers=Qt.KeyboardModifier.ControlModifier)
    assert image.zoom == pytest.approx(fitted * 1.2)
    assert preview.current_index == 1
    send_wheel(image, 120)
    assert preview.asset_group is album[0][0]
    assert image.fit_mode


def test_preview_details_wheel_scrolls_without_switching(preview, album, qapp):
    label = album[1]["lake.jpg"]
    label.all_tags = [f"测试标签 {index}" for index in range(40)]
    preview._show_photo()
    qapp.processEvents()
    scroll = preview.detail_scroll
    assert scroll.verticalScrollBar().maximum() > 0
    navigation = QSignalSpy(preview.image_area.navigation_requested)
    changed = QSignalSpy(scroll.verticalScrollBar().valueChanged)
    send_wheel(scroll, -120)
    if scroll.verticalScrollBar().value() == 0:
        assert changed.wait(1000)
    assert scroll.verticalScrollBar().value() > 0
    assert preview.current_index == 0
    assert len(navigation) == 0


def test_preview_sequence_respects_grid_order_and_basket_scope(view, album, monkeypatch, qapp):
    view._on_analysis_finished(*album)
    seen = []

    def inspect_dialog(dialog):
        dialog.show()
        qapp.processEvents()
        seen.append(([g.main_photo_path.name for g in dialog.asset_groups], dialog.current_index))
        dialog.close()
        return 0

    monkeypatch.setattr(PhotoPreviewDialog, "exec", inspect_dialog)
    view.search_similarity_scores = {"city.jpg": 0.9, "lake.jpg": 0.2}
    view._render_photo_list()
    view.photo_list_widget.itemDoubleClicked.emit(view.photo_list_widget.item(1))
    assert seen[-1] == (["city.jpg", "lake.jpg"], 1)
    view._on_filter_changed("建筑")
    view.photo_list_widget.itemDoubleClicked.emit(view.photo_list_widget.item(0))
    assert seen[-1] == (["city.jpg"], 0)
    qapp.processEvents()
    card = next(c for c in active_cards(view) if c.basket.rule.scene_tags == ["风景"])
    QTest.mouseClick(card.findChild(PushButton, "basketThumbnail"), Qt.MouseButton.LeftButton)
    assert seen[-1] == (["lake.jpg"], 0)


def test_single_photo_preview_keeps_wheel_and_key_boundaries(qapp, album):
    dialog = PhotoPreviewDialog(album[0][0], album[1]["lake.jpg"])
    try:
        dialog.show()
        dialog.activateWindow()
        qapp.processEvents()
        assert not dialog.btn_previous.isEnabled()
        assert not dialog.btn_next.isEnabled()
        assert dialog.lbl_position.text() == "1 / 1"
        for delta in (-120, 120, 0):
            send_wheel(dialog.image_area, delta)
        for key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            QTest.keyClick(dialog.image_area.viewport(), key)
        assert dialog.current_index == 0
        fitted = dialog.image_area.zoom
        send_wheel(dialog.image_area, 120, modifiers=Qt.KeyboardModifier.ControlModifier)
        assert dialog.image_area.zoom == pytest.approx(fitted * 1.2)
    finally:
        dialog.close()


def test_reload_empty_and_failure_end_loading(view, album, monkeypatch):
    view._on_analysis_finished(*album)
    view._on_filter_changed("风景")
    thread = Mock()
    monkeypatch.setattr(sorter_view, "ScanAndAnalysisThread", Mock(return_value=thread))
    view._start_scan_and_analysis(album[0][0].main_photo_path.parent)
    thread.start.assert_called_once()
    assert view.current_filter == "ALL"
    assert view.photo_list_widget.count() == 0
    assert not view.baskets
    assert view.progress_bar.isVisible()
    assert not view.btn_select_dir.isEnabled()
    assert not view.btn_execute_dispatch.isEnabled()
    view._on_analysis_failed("missing model")
    assert view.btn_select_dir.isEnabled()
    assert not view.progress_bar.isVisible()
    assert "missing model" in view.lbl_status.text()
    view._on_analysis_finished([], {}, None)
    assert not view.btn_execute_dispatch.isEnabled()
    assert not view.txt_semantic_search.isEnabled()
    assert view.photo_stack.currentWidget() is view.photo_empty


def test_worker_failure_is_reported(qapp, tmp_path, monkeypatch):
    thread = ScanAndAnalysisThread(tmp_path)
    spy = QSignalSpy(thread.failed_signal)
    monkeypatch.setattr(thread, "_analyze", Mock(side_effect=RuntimeError("model unavailable")))
    thread.run()
    assert list(spy) == [["model unavailable"]]


@pytest.mark.parametrize("fails", [False, True])
def test_actual_thread_finishes_empty_or_failed_scan(view, qapp, tmp_path, monkeypatch, fails):
    if fails:
        monkeypatch.setattr(sorter_view.CompanionScanner, "scan_directory", Mock(side_effect=OSError("scan denied")))
    view._start_scan_and_analysis(tmp_path)
    thread = view.scan_thread
    try:
        finished = QSignalSpy(thread.finished)
        assert finished.wait(3000)
        qapp.processEvents()
        assert view.btn_select_dir.isEnabled()
        assert not view.progress_bar.isVisible()
        assert not view.btn_execute_dispatch.isEnabled()
        assert ("scan denied" if fails else "未在指定目录") in view.lbl_status.text()
    finally:
        assert thread.wait(3000)


def test_active_worker_prevents_unsafe_close(qapp):
    window = PhotoyeApplication()
    try:
        window.show()
        window.sorter_view.scan_thread = Mock()
        window.sorter_view.scan_thread.isRunning.return_value = True
        assert not window.close()
        assert window.isVisible()
        window.sorter_view.scan_thread.isRunning.return_value = False
        assert window.close()
    finally:
        window.sorter_view.scan_thread = None
        window.close()


def test_dispatch_button_writes_real_files(view, album, tmp_path, monkeypatch, modal_driver):
    view._on_analysis_finished(*album)
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(sorter_view.QFileDialog, "getExistingDirectory", lambda *args: str(output))
    messages = []

    def confirm_result(dialog):
        assert isinstance(dialog, WorkshopMessageDialog)
        assert dialog.title_label.text() == "分拣完成"
        assert not dialog.cancelButton.isVisible()
        messages.append(dialog.content_label.text())
        QTest.mouseClick(dialog.yesButton, Qt.MouseButton.LeftButton)

    modal_driver.append(confirm_result)
    opened = []
    monkeypatch.setattr(sorter_view.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    QTest.mouseClick(view.btn_execute_dispatch, Qt.MouseButton.LeftButton)
    for group in album[0]:
        scene = album[1][group.main_photo_path.name].scene_tags[0]
        target_dir = output / "题材" / scene
        assert (target_dir / group.main_photo_path.name).read_bytes() == group.main_photo_path.read_bytes()
        for companion in group.companions:
            assert (target_dir / companion.filepath.name).read_bytes() == companion.filepath.read_bytes()
    assert len(messages) == 1 and "失败数: 0" in messages[0]
    assert len(opened) == 1 and Path(opened[0].toLocalFile()) == output.resolve()


def test_edit_add_delete_baskets_leave_source_files(view, album, modal_driver, qapp):
    view._on_analysis_finished(*album)

    def enter_text(value, expected):
        def interact(dialog):
            assert isinstance(dialog, BasketTextDialog)
            assert isinstance(dialog.text_edit, LineEdit)
            assert dialog.text_edit.text() == expected
            dialog.text_edit.setText(value)
            QTest.mouseClick(dialog.yesButton, Qt.MouseButton.LeftButton)
        return interact

    modal_driver.extend([
        enter_text("旅行精选", ""), enter_text("新名称", "旅行精选"),
        enter_text("custom/new", "自定义/旅行精选"),
    ])
    QTest.mouseClick(view.btn_add_custom_basket, Qt.MouseButton.LeftButton)
    basket = view.baskets[-1]
    assert basket.name == "旅行精选"
    qapp.processEvents()
    card = next(c for c in active_cards(view) if c.basket is basket)
    QTest.mouseClick(card.btn_edit, Qt.MouseButton.LeftButton)
    assert basket.name == "新名称" and basket.output_subfolder == "custom/new"
    def confirm_delete(dialog):
        assert isinstance(dialog, WorkshopMessageDialog)
        assert dialog.yesButton.text() == "删除篮子"
        assert "不删除源照片" in dialog.content_label.text()
        QTest.mouseClick(dialog.yesButton, Qt.MouseButton.LeftButton)

    modal_driver.append(confirm_delete)
    qapp.processEvents()
    card = next(c for c in active_cards(view) if c.basket is basket)
    QTest.mouseClick(card.btn_delete, Qt.MouseButton.LeftButton)
    assert basket not in view.baskets
    assert all(g.main_photo_path.exists() for g in album[0])
    assert album[0][0].companions[0].filepath.exists()


def test_dispatch_cancel_and_partial_failure_are_truthful(view, album, tmp_path, monkeypatch, modal_driver):
    from core.sorter.file_ops import FileOperationsExecutor

    view._on_analysis_finished(*album)
    execute = Mock()
    monkeypatch.setattr(FileOperationsExecutor, "execute_plan", execute)
    monkeypatch.setattr(sorter_view.QFileDialog, "getExistingDirectory", lambda *args: "")
    QTest.mouseClick(view.btn_execute_dispatch, Qt.MouseButton.LeftButton)
    execute.assert_not_called()
    monkeypatch.setattr(sorter_view.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    execute.return_value = Mock(total_actions=3, successful_count=2, failed_count=1, hardlink_count=2, copy_count=0)
    messages = []

    def confirm_failure(dialog):
        assert dialog.title_label.text() == "部分文件分拣失败"
        messages.append(dialog.content_label.text())
        assert "部分文件分拣失败" in messages[-1]
        assert "失败数: 1" in messages[-1]
        QTest.mouseClick(dialog.yesButton, Qt.MouseButton.LeftButton)

    modal_driver.append(confirm_failure)
    monkeypatch.setattr(sorter_view.QDesktopServices, "openUrl", lambda *args: False)
    QTest.mouseClick(view.btn_execute_dispatch, Qt.MouseButton.LeftButton)
    assert len(messages) == 1
    assert "无法打开文件管理器" in view.lbl_status.text()


@pytest.mark.parametrize("cancel_at", [0, 1])
def test_edit_cancel_is_atomic(view, album, modal_driver, cancel_at):
    view._on_analysis_finished(*album)
    basket = view.baskets[0]
    original = (basket.name, basket.output_subfolder)
    if cancel_at:
        def rename(dialog):
            dialog.text_edit.setText("未保存的新名称")
            QTest.mouseClick(dialog.yesButton, Qt.MouseButton.LeftButton)
        modal_driver.append(rename)
    modal_driver.append(lambda dialog: QTest.mouseClick(dialog.cancelButton, Qt.MouseButton.LeftButton))
    view._on_basket_rename_requested(basket)
    assert (basket.name, basket.output_subfolder) == original


@pytest.mark.parametrize("use_escape", [False, True])
def test_create_cancel_and_delete_cancel_keep_state(view, album, modal_driver, use_escape):
    view._on_analysis_finished(*album)
    before = list(view.baskets)

    def cancel(dialog):
        if use_escape:
            QTest.keyClick(dialog, Qt.Key.Key_Escape)
        else:
            QTest.mouseClick(dialog.cancelButton, Qt.MouseButton.LeftButton)

    modal_driver.extend([cancel, cancel])
    view._on_add_custom_basket()
    assert view.baskets == before
    view._on_basket_delete_requested(view.baskets[0])
    assert view.baskets == before
    assert all(g.main_photo_path.exists() for g in album[0])


def test_input_rejects_whitespace_and_accepts_enter(view, modal_driver):
    def enter(dialog):
        assert not dialog.yesButton.isEnabled()
        dialog.text_edit.setText("   ")
        assert not dialog.validate()
        QTest.keyClick(dialog.text_edit, Qt.Key.Key_Return)
        assert dialog.isVisible()
        dialog.text_edit.setText("  周末精选  ")
        assert dialog.yesButton.isEnabled()
        QTest.keyClick(dialog.text_edit, Qt.Key.Key_Return)

    modal_driver.append(enter)
    QTest.mouseClick(view.btn_add_custom_basket, Qt.MouseButton.LeftButton)
    assert len(view.baskets) == 1
    assert view.baskets[0].name == "周末精选"


def test_message_long_paths_stay_readable_in_compact_window(view, modal_driver, qapp):
    view.resize(1000, 680)
    text = "整理结果\n" + "D:/照片/" + "很长的文件夹/" * 30 + "<相册>.jpg\n" + "成功 1 失败 0\n" * 30

    def inspect_message(dialog):
        qapp.processEvents()
        assert dialog.content_label.textFormat() == Qt.TextFormat.PlainText
        assert dialog.content_label.text() == text
        assert dialog.rect().contains(dialog.widget.geometry())
        scroll = dialog.findChild(SmoothScrollArea)
        assert scroll.verticalScrollBar().maximum() > 0
        assert not dialog.cancelButton.isVisible()
        QTest.mouseClick(dialog.yesButton, Qt.MouseButton.LeftButton)

    modal_driver.append(inspect_message)
    assert view._show_message("分拣结果", text)


@pytest.mark.parametrize("scenario", ["search", "empty", "no_baskets", "no_actions", "missing_test_album"])
def test_information_and_missing_album_use_fluent(view, album, monkeypatch, modal_driver, scenario):
    messages = []

    def inspect_message(dialog):
        assert isinstance(dialog, WorkshopMessageDialog)
        messages.append(dialog.content_label.text())
        QTest.mouseClick(dialog.yesButton, Qt.MouseButton.LeftButton)

    modal_driver.append(inspect_message)
    if scenario == "search":
        view.txt_semantic_search.setText("sunset")
        view._on_execute_semantic_search()
    elif scenario == "empty":
        view._on_execute_physical_dispatch()
    elif scenario == "missing_test_album":
        monkeypatch.setattr(Path, "exists", lambda path: False)
        view._on_load_test_images()
    else:
        view._on_analysis_finished(*album)
        view.baskets = [] if scenario == "no_baskets" else [DispatchBasket("empty", "空篮子", "empty")]
        monkeypatch.setattr(sorter_view.QFileDialog, "getExistingDirectory", lambda *args: "unused")
        view._on_execute_physical_dispatch()
    assert len(messages) == 1


def test_workshop_keeps_legacy_input_and_message_boxes_out():
    root = Path(__file__).resolve().parents[1]
    for filename in ("sorter_view.py", "workshop_dialogs.py", "preview_dialog.py"):
        tree = ast.parse((root / "ui" / "views" / filename).read_text(encoding="utf-8"))
        imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names]
        assert {"QInputDialog", "QMessageBox"}.isdisjoint(imports), filename


def test_minimum_window_keeps_actions_inside(qapp, album):
    window = PhotoyeApplication()
    try:
        window.resize(1000, 680)
        window.sorter_view._on_analysis_finished(*album)
        window.show()
        qapp.processEvents()
        assert window.size().width() == 1000
        for button in (window.sorter_view.btn_select_dir, window.sorter_view.btn_add_custom_basket,
                       window.sorter_view.btn_execute_dispatch):
            assert button.isVisible()
            top_left = button.mapTo(window, QPoint(0, 0))
            bottom_right = button.mapTo(window, button.rect().bottomRight())
            assert window.rect().contains(top_left)
            assert window.rect().contains(bottom_right)
        assert window.sorter_view.photo_list_widget.viewport().width() >= 184
        for card in active_cards(window.sorter_view):
            assert card.height() >= card.minimumSizeHint().height()
            scroll = card.findChild(SmoothScrollArea)
            assert scroll.geometry().bottom() < card.lbl_hint.geometry().top()
            assert card.lbl_hint.geometry().bottom() < card.height()
        grid = window.sorter_view.photo_list_widget
        assert grid.gridSize().width() > grid.item(0).sizeHint().width()
    finally:
        window.close()


def test_formal_dependency_check_includes_fluent(monkeypatch):
    import run

    monkeypatch.setattr(run.importlib.util, "find_spec", lambda name: None if name == "qfluentwidgets" else object())
    assert run.check_dependencies() == ["PyQt6-Fluent-Widgets"]


def test_formal_entry_opens_actual_window(qapp, monkeypatch):
    import run

    original_main = __import__("app").main
    # 复用本测试进程的 QApplication，入口本身仍实际构造并显示主窗口。
    monkeypatch.setattr("app.QApplication", Mock(side_effect=lambda *args: qapp))
    monkeypatch.setattr("app.QApplication.primaryScreen", qapp.primaryScreen)
    seen = []

    def inspect_window():
        windows = [w for w in qapp.topLevelWidgets() if isinstance(w, PhotoyeApplication) and w.isVisible()]
        seen.extend(windows)
        for window in windows:
            window.close()
        qapp.quit()

    monkeypatch.setattr(run, "check_dependencies", lambda: [])
    monkeypatch.setattr("app.main", original_main)
    QTimer.singleShot(50, inspect_window)
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 0
    assert seen