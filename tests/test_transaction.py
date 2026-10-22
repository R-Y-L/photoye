#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2-F4 真实临时文件事务、边界与受保护撤销回归测试。"""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.sorter import transaction
from core.sorter.transaction import TransactionJournal, UndoReport, undo_transaction


def read_manifest(journal):
    return json.loads(journal.path.read_text(encoding="utf-8"))


def write_manifest(journal, data):
    journal.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def batch(tmp_path):
    source_dir = tmp_path / "源相册"
    source_dir.mkdir()
    journal = TransactionJournal(tmp_path / "输出")
    sources = []
    targets = []
    for index in range(3):
        source = source_dir / f"照片_{index}.jpg"
        source.write_bytes(f"photo-{index}".encode())
        target = journal.output_root / source.name
        shutil.copy2(source, target)
        journal.record_created(source, target, "copy", index == 2)
        sources.append(source)
        targets.append(target)
    journal.finish()
    return journal, sources, targets


def assert_rejected(journal, targets):
    report = undo_transaction(journal.path)
    assert not report.completed
    assert report.failed_count > 0
    assert report.errors
    assert report.removed_count == 0
    assert all(target.exists() for target in targets)
    assert journal.path.is_file()
    return report


def test_mixed_hardlink_copy_and_companion_preserves_sources_and_old_files(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    journal = TransactionJournal(tmp_path / "output")
    old_file = journal.output_root / "IMG.JPG"
    old_file.write_bytes(b"old photo")
    sources = [source_dir / name for name in ("IMG.JPG", "IMG.MOV", "IMG.XMP")]
    targets = [journal.output_root / name for name in ("IMG_1.JPG", "IMG_1.MOV", "IMG_1.XMP")]
    for index, (source, target) in enumerate(zip(sources, targets)):
        source.write_bytes(f"original-{index}".encode())
        if index == 0:
            os.link(source, target)
            assert source.samefile(target)
        else:
            shutil.copy2(source, target)
        journal.record_created(source, target, "hardlink" if index == 0 else "copy", index > 0)
    before = [(source.read_bytes(), source.stat().st_ino, source.stat().st_mtime_ns) for source in sources]
    journal.finish()

    assert read_manifest(journal)["mode"] == "mixed"
    assert [item["is_companion"] for item in journal.operations] == [False, True, True]
    report = undo_transaction(journal.path)

    assert report == UndoReport(total_count=3, removed_count=3)
    assert report.completed
    assert all(not target.exists() for target in targets)
    assert before == [(source.read_bytes(), source.stat().st_ino, source.stat().st_mtime_ns) for source in sources]
    assert old_file.read_bytes() == b"old photo"
    assert not journal.path.exists()


@pytest.mark.parametrize("change", ["bytes", "size", "mtime", "same_size_and_mtime"])
def test_external_change_stops_entire_batch(batch, change):
    journal, sources, targets = batch
    target = targets[0]
    before = target.stat()
    if change == "bytes":
        target.write_bytes(b"changed")
    elif change == "size":
        target.write_bytes(b"larger external content")
    elif change == "mtime":
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    else:
        target.write_bytes(b"CHANGED")
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert target.stat().st_size == before.st_size
        assert target.stat().st_mtime_ns == before.st_mtime_ns
    report = assert_rejected(journal, targets)
    assert report.total_count == 3
    assert report.failed_count == 1
    assert report.missing_count == 0
    assert all(source.read_bytes() == f"photo-{index}".encode() for index, source in enumerate(sources))
    assert read_manifest(journal)["status"] == "completed"


def test_hardlink_external_change_is_also_protected(tmp_path):
    source = tmp_path / "original.jpg"
    source.write_bytes(b"original")
    journal = TransactionJournal(tmp_path / "output")
    target = journal.output_root / "link.jpg"
    os.link(source, target)
    journal.record_created(source, target, "hardlink", False)
    source.write_bytes(b"external edit via source")

    assert_rejected(journal, [target])
    assert source.read_bytes() == b"external edit via source"
    assert source.samefile(target)


def test_same_bytes_replacement_with_different_inode_is_rejected(batch):
    journal, _, targets = batch
    target = targets[0]
    before = target.stat()
    replacement = target.with_suffix(".replacement")
    shutil.copy2(target, replacement)
    assert replacement.stat().st_ino != before.st_ino
    os.replace(replacement, target)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert target.read_bytes() == b"photo-0"
    assert target.stat().st_mtime_ns == before.st_mtime_ns

    assert_rejected(journal, targets)


def test_missing_targets_are_skipped_and_manifest_is_cleaned(batch):
    journal, sources, targets = batch
    for target in targets:
        target.unlink()
    report = undo_transaction(journal.path)

    assert report == UndoReport(total_count=3, missing_count=3)
    assert report.completed
    assert all(source.exists() for source in sources)
    assert all(not target.exists() for target in targets)
    assert not journal.path.exists()


def test_previously_undone_manifest_is_also_cleaned(batch):
    journal, sources, targets = batch
    for target in targets:
        target.unlink()
    journal.finish("undone")
    assert undo_transaction(journal.path) == UndoReport(total_count=3, missing_count=3)
    assert not journal.path.exists() and all(source.exists() for source in sources)


def test_empty_transaction_cleans_only_its_own_manifest(tmp_path):
    journal = TransactionJournal(tmp_path / "output")
    other = TransactionJournal(journal.output_root)
    before = other.path.read_bytes()
    assert undo_transaction(journal.path) == UndoReport(total_count=0)
    assert not journal.path.exists() and other.path.read_bytes() == before


def test_missing_source_does_not_prevent_undo(batch):
    journal, sources, targets = batch
    for source in sources:
        source.unlink()

    assert undo_transaction(journal.path) == UndoReport(total_count=3, removed_count=3)
    assert all(not target.exists() for target in targets)


def test_group_rollback_missing_target_is_skipped(batch):
    journal, _, targets = batch
    targets[0].unlink()
    report = undo_transaction(journal.path)
    assert report == UndoReport(total_count=3, removed_count=2, missing_count=1)


@pytest.mark.parametrize("attack", [
    "relative", "traversal", "absolute_outside", "root", "source", "other_source",
    "duplicate", "manifest", "invalid_operation", "directory_target",
])
def test_invalid_operation_paths_never_delete_anything(batch, tmp_path, attack):
    journal, sources, targets = batch
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    data = read_manifest(journal)
    operation = data["operations"][0]
    if attack == "relative":
        operation["target"] = "relative.jpg"
    elif attack == "traversal":
        operation["target"] = str(journal.output_root / ".." / outside.name)
    elif attack == "absolute_outside":
        operation["target"] = str(outside)
    elif attack == "root":
        operation["target"] = str(journal.output_root)
    elif attack == "source":
        operation["source"] = operation["target"]
    elif attack == "other_source":
        data["operations"][1]["source"] = operation["target"]
    elif attack == "duplicate":
        data["operations"].append(dict(operation))
    elif attack == "manifest":
        operation["target"] = str(journal.path)
    elif attack == "invalid_operation":
        operation["type"] = "move"
    else:
        directory = journal.output_root / "pretend_file"
        directory.mkdir()
        operation["target"] = str(directory)
    write_manifest(journal, data)

    assert_rejected(journal, targets)
    assert outside.read_bytes() == b"outside"
    assert all(source.exists() for source in sources)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True), ("schema_version", "1"),
    ("run_id", 1), ("created_at", []), ("created_at", "invalid-date"),
    ("output_root", []), ("output_root", "relative"), ("mode", []),
    ("mode", "hardlink"), ("status", None), ("status", ""),
    ("created_directories", {}), ("operations", {}),
])
def test_non_operation_field_types_are_validated_before_delete(batch, field, value):
    journal, _, targets = batch
    data = read_manifest(journal)
    data[field] = value
    write_manifest(journal, data)
    assert_rejected(journal, targets)


@pytest.mark.parametrize("field,value", [
    ("source", []), ("source", "relative.jpg"), ("source", "bad\0path"),
    ("target", 42), ("is_companion", 1), ("type", []),
    ("fingerprint", {}), ("fingerprint", None),
])
def test_invalid_operation_field_types_are_rejected(batch, field, value):
    journal, _, targets = batch
    data = read_manifest(journal)
    data["operations"][0][field] = value
    write_manifest(journal, data)
    assert_rejected(journal, targets)


@pytest.mark.parametrize("field,value", [
    ("st_dev", True), ("st_ino", -1), ("size", "7"),
    ("mtime_ns", 1.5), ("sha256", "not-a-hash"), ("sha256", []),
])
def test_invalid_fingerprint_fields_are_rejected(batch, field, value):
    journal, _, targets = batch
    data = read_manifest(journal)
    data["operations"][0]["fingerprint"][field] = value
    write_manifest(journal, data)
    assert_rejected(journal, targets)


@pytest.mark.parametrize("content", [b"not json", b"[]", b"null", b"{", b"\xff\xfe"])
def test_invalid_json_returns_report_without_deletion(batch, content):
    journal, _, targets = batch
    journal.path.write_bytes(content)
    report = assert_rejected(journal, targets)
    assert report.total_count == 0


def test_duplicate_json_fields_are_rejected(batch):
    journal, _, targets = batch
    text = journal.path.read_text(encoding="utf-8")
    journal.path.write_text(text[:-2] + ', "status": "completed"\n}', encoding="utf-8")
    assert_rejected(journal, targets)


def test_manifest_outside_declared_root_is_rejected(batch, tmp_path):
    journal, _, targets = batch
    moved = tmp_path / "outside.undo.json"
    shutil.copy2(journal.path, moved)
    report = undo_transaction(moved)
    assert not report.completed
    assert report.removed_count == 0
    assert report.errors
    assert all(target.exists() for target in targets)


@pytest.mark.parametrize("attack", ["outside", "root", "wrong_identity", "file", "string", "bad_type"])
def test_forged_directories_are_rejected_before_file_deletion(batch, tmp_path, attack):
    journal, _, targets = batch
    directory = journal.output_root / "old_directory"
    directory.mkdir()
    outside = tmp_path / "outside_directory"
    outside.mkdir()
    data = read_manifest(journal)
    info = directory.stat()
    entry = {"path": str(directory), "st_dev": info.st_dev, "st_ino": info.st_ino}
    if attack == "outside":
        entry["path"] = str(outside)
    elif attack == "root":
        entry["path"] = str(journal.output_root)
    elif attack == "wrong_identity":
        entry["st_ino"] += 1
    elif attack == "file":
        entry["path"] = str(targets[0])
    elif attack == "string":
        entry = str(directory)
    else:
        entry["st_ino"] = True
    data["created_directories"] = [entry]
    write_manifest(journal, data)

    assert_rejected(journal, targets)
    assert directory.is_dir()
    assert outside.is_dir()


def test_only_recorded_empty_directories_are_cleaned(tmp_path):
    journal = TransactionJournal(tmp_path / "output")
    old = journal.output_root / "old"
    old.mkdir()
    unrecorded = journal.output_root / "unrecorded"
    unrecorded.mkdir()
    parent = journal.output_root / "new"
    child = parent / "child"
    child.mkdir(parents=True)
    nonempty = journal.output_root / "new_nonempty"
    nonempty.mkdir()
    unrelated = nonempty / "keep.txt"
    unrelated.write_bytes(b"unrelated")
    journal.record_directories([parent, child, nonempty])
    source = tmp_path / "source.jpg"
    source.write_bytes(b"original")
    for target in (old / source.name, child / source.name):
        shutil.copy2(source, target)
        journal.record_created(source, target, "copy", False)
    journal.finish()

    assert undo_transaction(journal.path) == UndoReport(total_count=2, removed_count=2)
    assert old.is_dir()
    assert unrecorded.is_dir()
    assert not child.exists()
    assert not parent.exists()
    assert nonempty.is_dir()
    assert unrelated.read_bytes() == b"unrelated"
    assert journal.output_root.is_dir()
    assert not journal.path.exists()


def test_recorded_directory_replacement_is_rejected(batch):
    journal, _, targets = batch
    directory = journal.output_root / "new"
    directory.mkdir()
    journal.record_directories([directory])
    directory.rename(journal.output_root / "original_directory")
    directory.mkdir()
    assert_rejected(journal, targets)
    assert directory.is_dir()


def make_symlink(link, target, is_directory=False):
    try:
        link.symlink_to(target, target_is_directory=is_directory)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows 当前账户无符号链接创建权限")
        raise


@pytest.mark.parametrize("kind", ["target", "parent", "dangling", "manifest", "root"])
def test_symlink_paths_are_rejected(batch, tmp_path, kind):
    journal, _, targets = batch
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "photo.jpg"
    external.write_bytes(b"external")
    data = read_manifest(journal)
    if kind in ("target", "dangling"):
        link = journal.output_root / "linked.jpg"
        make_symlink(link, external if kind == "target" else outside / "missing.jpg")
        data["operations"][0]["target"] = str(link)
        write_manifest(journal, data)
    elif kind == "parent":
        link = journal.output_root / "linked_directory"
        make_symlink(link, outside, True)
        data["operations"][0]["target"] = str(link / external.name)
        write_manifest(journal, data)
    elif kind == "manifest":
        link = journal.output_root / "linked.undo.json"
        make_symlink(link, journal.path)
        report = undo_transaction(link)
        assert not report.completed
        assert report.removed_count == 0
        assert all(target.exists() for target in targets)
        return
    else:
        link = tmp_path / "linked_root"
        make_symlink(link, journal.output_root, True)
        data["output_root"] = str(link)
        write_manifest(journal, data)
    assert_rejected(journal, targets)
    assert external.read_bytes() == b"external"


def test_unique_names_and_utf8_manifests_with_fixed_timestamp(tmp_path, monkeypatch):
    class FixedDatetime:
        @staticmethod
        def now(tz):
            return datetime(2026, 10, 19, 12, tzinfo=timezone.utc)

    monkeypatch.setattr(transaction, "datetime", FixedDatetime)
    first = TransactionJournal(tmp_path / "中文输出")
    second = TransactionJournal(tmp_path / "中文输出")
    assert first.path != second.path
    assert first.path.name.startswith("photoye_run_20261019_120000_")
    assert first.path.name.endswith(".undo.json")
    assert read_manifest(first)["run_id"] != read_manifest(second)["run_id"]
    assert "中文输出" in first.path.read_bytes().decode("utf-8")
    assert first.path.parent == first.output_root.resolve()


def test_streamed_fingerprint_records_all_required_fields(tmp_path):
    source = tmp_path / "large.jpg"
    content = b"photo" * (1024 * 1024)
    source.write_bytes(content)
    journal = TransactionJournal(tmp_path / "output")
    target = journal.output_root / source.name
    shutil.copy2(source, target)
    journal.record_created(source, target, "copy", False)
    info = target.stat()
    assert journal.operations[0]["fingerprint"] == {
        "st_dev": info.st_dev, "st_ino": info.st_ino, "size": len(content),
        "mtime_ns": info.st_mtime_ns, "sha256": hashlib.sha256(content).hexdigest(),
    }
    assert undo_transaction(journal.path).removed_count == 1


def test_record_failure_preserves_disk_and_in_memory_state(batch, monkeypatch):
    journal, sources, targets = batch
    target = journal.output_root / "new.jpg"
    shutil.copy2(sources[0], target)
    directory = journal.output_root / "new_directory"
    directory.mkdir()
    before_disk = journal.path.read_bytes()
    before_memory = journal.operations
    observed = []

    def fail_replace(source, destination):
        temporary = Path(source)
        assert temporary.parent == journal.path.parent
        assert Path(destination) == journal.path
        observed.append(json.loads(temporary.read_text(encoding="utf-8")))
        raise OSError("simulated atomic replace failure")

    with monkeypatch.context() as patch:
        patch.setattr(transaction.os, "replace", fail_replace)
        with pytest.raises(OSError, match="simulated atomic replace failure"):
            journal.record_created(sources[0], target, "copy", False)
        with pytest.raises(OSError, match="simulated atomic replace failure"):
            journal.record_directories([directory])
        with pytest.raises(OSError, match="simulated atomic replace failure"):
            journal.finish("failed")
    assert len(observed) == 3
    assert journal.operations == before_memory
    assert journal.path.read_bytes() == before_disk
    assert read_manifest(journal)["created_directories"] == []
    assert read_manifest(journal)["status"] == "completed"
    assert not list(journal.output_root.glob("*.tmp"))
    assert undo_transaction(journal.path).removed_count == 3
    assert target.exists()
    assert directory.exists()
    assert all(not item.exists() for item in targets)


def test_constructor_reports_real_write_failure(tmp_path, monkeypatch):
    def fail_replace(*args):
        raise OSError("cannot create journal")

    monkeypatch.setattr(transaction.os, "replace", fail_replace)
    root = tmp_path / "output"
    with pytest.raises(OSError, match="cannot create journal"):
        TransactionJournal(root)
    assert list(root.iterdir()) == []


def test_constructor_rejects_file_as_output_root(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"keep")
    with pytest.raises(OSError):
        TransactionJournal(path)
    assert path.read_bytes() == b"keep"


@pytest.mark.parametrize("attack", ["source", "outside", "root", "duplicate", "type", "flag", "missing"])
def test_record_created_rejects_invalid_calls_without_persisting(batch, tmp_path, attack):
    journal, sources, targets = batch
    before = journal.path.read_bytes()
    source, target, operation_type, companion = sources[0], targets[0], "copy", False
    if attack == "source":
        source = target
    elif attack == "outside":
        target = source
    elif attack == "root":
        target = journal.output_root
    elif attack == "type":
        operation_type = "delete"
    elif attack == "flag":
        companion = 1
    elif attack == "missing":
        target = journal.output_root / "missing.jpg"
    with pytest.raises((OSError, ValueError)):
        journal.record_created(source, target, operation_type, companion)
    assert journal.path.read_bytes() == before
    assert len(journal.operations) == 3
    assert all(target.exists() for target in targets)


@pytest.mark.parametrize("attack", ["root", "outside", "file", "missing", "not_list"])
def test_record_directories_rejects_invalid_calls(batch, tmp_path, attack):
    journal, _, targets = batch
    values = {
        "root": [journal.output_root], "outside": [tmp_path],
        "file": [targets[0]], "missing": [journal.output_root / "missing"],
        "not_list": "not a list",
    }
    before = journal.path.read_bytes()
    with pytest.raises((OSError, ValueError)):
        journal.record_directories(values[attack])
    assert journal.path.read_bytes() == before


def test_unlink_failure_stops_in_reverse_order_and_retry_continues(batch, monkeypatch):
    journal, sources, targets = batch
    original_unlink = Path.unlink
    attempted = []

    def fail_second(path, *args, **kwargs):
        attempted.append(path)
        if path == targets[1]:
            raise PermissionError("simulated file lock")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_second)
        report = undo_transaction(journal.path)
    assert report.total_count == 3
    assert report.removed_count == 1
    assert report.missing_count == 0
    assert report.failed_count == 1
    assert "simulated file lock" in report.errors[0]
    assert attempted == [targets[2], targets[1]]
    assert targets[0].exists() and targets[1].exists()
    assert not targets[2].exists()
    assert read_manifest(journal)["status"] == "completed"
    assert undo_transaction(journal.path) == UndoReport(total_count=3, removed_count=2, missing_count=1)
    assert all(source.exists() for source in sources)


def test_retry_after_partial_failure_still_rejects_changed_target(batch, monkeypatch):
    journal, _, targets = batch
    original_unlink = Path.unlink

    def fail_second(path, *args, **kwargs):
        if path == targets[1]:
            raise PermissionError("locked")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_second)
        first = undo_transaction(journal.path)
    assert first.removed_count == 1 and first.failed_count == 1
    targets[0].write_bytes(b"external change")
    retry = assert_rejected(journal, targets[:2])
    assert retry.missing_count == 1
    assert retry.failed_count == 1


def test_manifest_delete_failure_is_reported_and_retry_cleans_it(batch, monkeypatch):
    journal, _, targets = batch
    original_unlink = Path.unlink

    def fail_manifest(path, *args, **kwargs):
        if path == journal.path:
            raise PermissionError("manifest locked")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_manifest)
        report = undo_transaction(journal.path)
    assert report.removed_count == 3
    assert report.failed_count == 1
    assert "撤销清单清理失败" in report.errors[0] and "manifest locked" in report.errors[0]
    assert all(not target.exists() for target in targets)
    assert read_manifest(journal)["status"] == "completed"
    assert undo_transaction(journal.path) == UndoReport(total_count=3, missing_count=3)
    assert not journal.path.exists()


def test_manifest_replaced_during_undo_is_not_deleted(batch, monkeypatch):
    journal, sources, targets = batch
    original_unlink = Path.unlink

    def replace_manifest(path, *args, **kwargs):
        result = original_unlink(path, *args, **kwargs)
        if path == targets[0]:
            replacement = journal.output_root / "replacement.json"
            replacement.write_bytes(journal.path.read_bytes())
            os.replace(replacement, journal.path)
        return result

    monkeypatch.setattr(Path, "unlink", replace_manifest)
    report = undo_transaction(journal.path)
    assert report.removed_count == 3 and report.failed_count == 1
    assert "清单已发生变化" in report.errors[0]
    assert journal.path.exists() and all(source.exists() for source in sources)


def test_directory_cleanup_failure_keeps_manifest_for_retry(batch, monkeypatch):
    journal, _, targets = batch
    directory = journal.output_root / "new"
    directory.mkdir()
    journal.record_directories([directory])
    original_rmdir = Path.rmdir

    def deny(path):
        if path == directory:
            raise PermissionError("directory locked")
        return original_rmdir(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rmdir", deny)
        report = undo_transaction(journal.path)
    assert report.removed_count == 3 and report.failed_count == 1
    assert journal.path.exists() and directory.exists()
    assert all(not target.exists() for target in targets)
    assert undo_transaction(journal.path) == UndoReport(total_count=3, missing_count=3)
    assert not journal.path.exists() and not directory.exists()


def test_missing_manifest_returns_failed_report(tmp_path):
    report = undo_transaction(tmp_path / "missing.undo.json")
    assert report.total_count == 0
    assert report.failed_count == 1
    assert not report.completed
    assert report.errors