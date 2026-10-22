#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 Python 分拣事务清单与先整批校验、后删除的受保护撤销。"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


_SCHEMA_VERSION = 1
_OPERATION_TYPES = {"hardlink", "copy"}
_FINGERPRINT_FIELDS = {"st_dev", "st_ino", "size", "mtime_ns", "sha256"}
_MANIFEST_FIELDS = {
    "schema_version", "run_id", "created_at", "output_root", "mode",
    "operations", "status", "created_directories",
}


@dataclass
class UndoReport:
    """撤销结果；未尝试的剩余文件不冒充删除成功或删除失败。"""

    total_count: int
    removed_count: int = 0
    missing_count: int = 0
    failed_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.failed_count == 0


def _atomic_write(path: Path, data: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException as error:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise error from cleanup_error
        raise


def _absolute_path(value: str) -> Path:
    if type(value) is not str or not value or "\0" in value:
        raise ValueError("路径必须是非空字符串且不含 NUL")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"路径必须是无父级跳转的绝对路径: {value}")
    return path


def _is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _validate_root(root: Path) -> None:
    info = root.lstat()
    if _is_link(info) or not stat.S_ISDIR(info.st_mode) or root.resolve() != root:
        raise ValueError(f"输出根目录必须保持为真实目录: {root}")


def _safe_child(path: Path, root: Path) -> Path:
    _validate_root(root)
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"路径必须严格位于输出根目录内: {path}")
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"路径解析后逃逸输出根目录: {path}")
    current = path
    while current != root:
        try:
            info = current.lstat()
        except FileNotFoundError:
            pass
        else:
            if _is_link(info):
                raise ValueError(f"事务路径不能包含符号链接或重解析点: {current}")
        current = current.parent
    return resolved


def _identity(info: os.stat_result) -> dict:
    return {"st_dev": info.st_dev, "st_ino": info.st_ino}


def _file_metadata(info: os.stat_result) -> dict:
    return {**_identity(info), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def _fingerprint(path: Path) -> dict:
    before = path.lstat()
    if _is_link(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"目标必须为真实普通文件: {path}")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if _file_metadata(before) != _file_metadata(opened):
            raise ValueError(f"指纹读取前文件身份发生变化: {path}")
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    current = path.lstat()
    if (
        _is_link(current)
        or _file_metadata(before) != _file_metadata(after)
        or _file_metadata(before) != _file_metadata(current)
    ):
        raise ValueError(f"指纹读取期间文件发生变化: {path}")
    return {**_file_metadata(after), "sha256": digest.hexdigest()}


fingerprint = _fingerprint


def _mode(operations: list[dict]) -> str:
    types = {operation["type"] for operation in operations}
    return next(iter(types)) if len(types) == 1 else "mixed" if types else "pending"


class TransactionJournal:
    """仅登记调用方刚创建的文件和目录；持久化成功后才发布内存状态。"""

    def __init__(self, output_root: Path) -> None:
        try:
            self.output_root = Path(output_root).resolve()
        except TypeError as error:
            raise ValueError("输出根目录必须是有效路径") from error
        self.output_root.mkdir(parents=True, exist_ok=True)
        _validate_root(self.output_root)
        now = datetime.now(timezone.utc)
        run_id = f"{now:%Y%m%d_%H%M%S_%f}_{uuid4().hex}"
        self.path = self.output_root / f"photoye_run_{run_id}.undo.json"
        self._data = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": now.isoformat(),
            "output_root": str(self.output_root),
            "mode": "pending",
            "operations": [],
            "status": "in_progress",
            "created_directories": [],
        }
        _atomic_write(self.path, self._data)

    @property
    def operations(self) -> list[dict]:
        return deepcopy(self._data["operations"])

    def _persist(self, data: dict) -> None:
        _validate_manifest(data, self.path)
        _atomic_write(self.path, data)
        self._data = data

    def record_created(
        self, source: Path, target: Path, operation_type: str, is_companion: bool,
    ) -> None:
        if type(operation_type) is not str or operation_type not in _OPERATION_TYPES:
            raise ValueError("操作类型必须为 hardlink 或 copy")
        if type(is_companion) is not bool:
            raise ValueError("is_companion 必须为布尔值")
        source = Path(source).resolve()
        target = _safe_child(Path(target).absolute(), self.output_root)
        if target == source or target == self.path:
            raise ValueError("目标不能是源文件或事务清单本身")
        data = deepcopy(self._data)
        data["operations"].append({
            "source": str(source), "target": str(target), "type": operation_type,
            "is_companion": is_companion, "fingerprint": _fingerprint(target),
        })
        data["mode"] = _mode(data["operations"])
        self._persist(data)

    def record_directories(self, paths: list[Path]) -> None:
        if type(paths) is not list:
            raise ValueError("目录清单必须是列表")
        data = deepcopy(self._data)
        known = {entry["path"]: index for index, entry in enumerate(data["created_directories"])}
        for path in paths:
            path = _safe_child(Path(path).absolute(), self.output_root)
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"记录的目录必须真实存在: {path}")
            entry = {"path": str(path), **_identity(info)}
            if str(path) in known:
                data["created_directories"][known[str(path)]] = entry
            else:
                known[str(path)] = len(data["created_directories"])
                data["created_directories"].append(entry)
        self._persist(data)

    def finish(self, status: str = "completed") -> None:
        if type(status) is not str or not status.strip():
            raise ValueError("事务状态必须是非空字符串")
        data = deepcopy(self._data)
        data["status"] = status
        self._persist(data)


def _fields(value: object, fields: set[str], label: str) -> None:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label}字段不符合 schema_version=1")


def _integers(value: dict, fields: set[str]) -> None:
    for key in fields:
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"{key} 必须是非负整数")


def _validate_manifest(data: dict, manifest: Path) -> tuple[Path, list[dict], list[dict]]:
    _fields(data, _MANIFEST_FIELDS, "事务清单")
    if type(data["schema_version"]) is not int or data["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("不支持的事务清单版本")
    for key in ("run_id", "created_at", "status", "mode"):
        if type(data[key]) is not str or not data[key].strip():
            raise ValueError(f"{key} 必须是非空字符串")
    datetime.fromisoformat(data["created_at"])
    root = _absolute_path(data["output_root"])
    _validate_root(root)
    _safe_child(manifest, root)
    if not manifest.is_file():
        raise ValueError("事务清单必须是真实普通文件")
    operations = data["operations"]
    directories = data["created_directories"]
    if type(operations) is not list or type(directories) is not list:
        raise ValueError("operations 和 created_directories 必须是列表")

    targets = set()
    sources = set()
    for operation in operations:
        _fields(operation, {"source", "target", "type", "is_companion", "fingerprint"}, "操作")
        source = _absolute_path(operation["source"]).resolve()
        target = _absolute_path(operation["target"])
        _safe_child(target, root)
        if target in targets or target == manifest:
            raise ValueError(f"重复目标或目标为事务清单: {target}")
        if target.exists() and target.samefile(manifest):
            raise ValueError(f"目标与事务清单共享文件身份: {target}")
        if type(operation["type"]) is not str or operation["type"] not in _OPERATION_TYPES:
            raise ValueError("非法操作类型")
        if type(operation["is_companion"]) is not bool:
            raise ValueError("is_companion 必须为布尔值")
        fingerprint = operation["fingerprint"]
        _fields(fingerprint, _FINGERPRINT_FIELDS, "文件指纹")
        _integers(fingerprint, {"st_dev", "st_ino", "size", "mtime_ns"})
        if type(fingerprint["sha256"]) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", fingerprint["sha256"],
        ):
            raise ValueError("sha256 必须是 64 位小写十六进制字符串")
        targets.add(target)
        sources.add(source)
    if targets & sources:
        raise ValueError("目标不能是任何已记录的源文件")
    if data["mode"] != _mode(operations):
        raise ValueError("mode 与操作清单不一致")

    seen_directories = set()
    for directory in directories:
        _fields(directory, {"path", "st_dev", "st_ino"}, "创建目录")
        _integers(directory, {"st_dev", "st_ino"})
        path = _absolute_path(directory["path"])
        _safe_child(path, root)
        if path in seen_directories or path in targets or path == manifest or path in sources:
            raise ValueError(f"重复目录或目录与文件清单冲突: {path}")
        seen_directories.add(path)
        _check_directory(directory, root)
    return root, operations, directories


def _check_directory(directory: dict, root: Path) -> bool:
    path = _absolute_path(directory["path"])
    _safe_child(path, root)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(info.st_mode) or _identity(info) != {
        "st_dev": directory["st_dev"], "st_ino": directory["st_ino"],
    }:
        raise ValueError(f"目录身份发生变化: {path}")
    return True


def _json_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 含重复字段: {key}")
        result[key] = value
    return result


def undo_transaction(manifest_path: Path) -> UndoReport:
    """校验整批后反序撤销，全部成功才移除清单；失败时保留以供重试。"""
    report = UndoReport(total_count=0)
    try:
        manifest = Path(manifest_path).absolute()
        info = manifest.lstat()
        if _is_link(info) or not stat.S_ISREG(info.st_mode):
            raise ValueError("事务清单必须是真实普通文件")
        manifest_fingerprint = _fingerprint(manifest)
        with manifest.open("r", encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=_json_object)
        if type(data) is dict and type(data.get("operations")) is list:
            report.total_count = len(data["operations"])
        root, operations, directories = _validate_manifest(data, manifest)

        existing = []
        for operation in operations:
            target = Path(operation["target"])
            try:
                current = _fingerprint(target)
                if current != operation["fingerprint"]:
                    raise ValueError(f"目标已被外部修改或替换: {target}")
            except FileNotFoundError:
                report.missing_count += 1
            except (OSError, ValueError) as error:
                report.failed_count += 1
                report.errors.append(str(error))
            else:
                existing.append(operation)
        if report.failed_count:
            return report

        for operation in reversed(existing):
            target = Path(operation["target"])
            _safe_child(target, root)
            try:
                # 整批预检后仍再校验一次，缩短外部替换与删除之间的竞态窗口。
                if _fingerprint(target) != operation["fingerprint"]:
                    raise ValueError(f"删除前目标已发生变化: {target}")
                target.unlink()
            except FileNotFoundError:
                report.missing_count += 1
            else:
                report.removed_count += 1

        for directory in sorted(
            directories, key=lambda entry: len(Path(entry["path"]).parts), reverse=True,
        ):
            if _check_directory(directory, root):
                try:
                    Path(directory["path"]).rmdir()
                except OSError as error:
                    if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                        raise
        _safe_child(manifest, root)
        if _fingerprint(manifest) != manifest_fingerprint:
            raise ValueError(f"撤销期间清单已发生变化，保留文件：{manifest}")
        try:
            manifest.unlink()
        except OSError as error:
            raise OSError(f"撤销清单清理失败，已保留供重试：{manifest}；{error}") from error
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as error:
        report.failed_count += 1
        report.errors.append(str(error))
    return report