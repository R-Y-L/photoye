#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 物理存储操作执行器模块 (V3.0)

负责将分拣计划中的动作安全落地至物理磁盘。
特性:
1. 优先调用 os.link 原生硬链接 (同一分区 0 额外磁盘占用、毫秒级完成)
2. 跨磁盘/移动U盘自适应降级为独占创建的流式复制
3. 重名防灾强保护 (No Overwrite): 同名文件自动递增序号，伴侣文件保持严格同步更名
4. 结构化执行报告与持久化事务清单，支持受保护撤销
符合 zlog/REQUIREMENTS.md F2.2 与 F2.4 规范。
"""

from __future__ import annotations

import errno
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .pipeline import DispatchAction, DispatchPlan
from .transaction import TransactionJournal, fingerprint


@dataclass
class FileOpResult:
    """单个文件的物理操作结果"""
    source_path: Path
    target_path: Path                  # 最终落地的实际目标路径 (可能因防灾追加了序号)
    operation_type: str                # "hardlink" 或 "copy"
    is_companion: bool
    success: bool
    error_message: Optional[str] = None


@dataclass
class ExecutionReport:
    """整批物理分拣的落地报告"""
    output_root: Path
    total_actions: int = 0
    successful_count: int = 0
    failed_count: int = 0
    hardlink_count: int = 0
    copy_count: int = 0
    results: List[FileOpResult] = field(default_factory=list)
    manifest_path: Optional[Path] = None
    errors: List[str] = field(default_factory=list)


class FileOperationsExecutor:
    """物理文件安全落地执行器"""

    def __init__(self, prefer_hardlink: bool = True) -> None:
        """初始化执行器。

        Args:
            prefer_hardlink: 是否优先使用硬链接 (同一磁盘推荐 True，0 占空间)
        """
        self.prefer_hardlink = prefer_hardlink

    @staticmethod
    def resolve_non_colliding_path(target_path: Path) -> Path:
        """若目标文件已存在，安全计算递增防灾序号路径 (如 IMG_001_1.JPG)。"""
        if not os.path.lexists(target_path):
            return target_path

        folder = target_path.parent
        stem = target_path.stem
        suffix = target_path.suffix

        counter = 1
        while True:
            candidate = folder / f"{stem}_{counter}{suffix}"
            if not os.path.lexists(candidate):
                return candidate
            counter += 1

    def execute_single_action(
        self,
        action: DispatchAction,
        resolved_stems_map: Dict[Tuple[str, str], str],
    ) -> FileOpResult:
        """执行单个分拣动作并处理重名碰撞。

        Args:
            action: 分拣动作
            resolved_stems_map: 字典，记录主文件在此篮子下因重名产生的最终 stem 映射，
                               确保伴侣文件 (如 .mov, .xmp) 同步采用相同的序号！
        """
        group_key = (action.basket_id, action.source_path.stem.lower())
        if group_key in resolved_stems_map:
            final_target = action.target_path.with_stem(resolved_stems_map[group_key])
        else:
            final_target = self.resolve_non_colliding_path(action.target_path)
            resolved_stems_map[group_key] = final_target.stem
        try:
            final_target.parent.mkdir(parents=True, exist_ok=True)
            operation = self._create_file(action.source_path.resolve(), final_target)
            return FileOpResult(action.source_path.resolve(), final_target, operation, action.is_companion, True)
        except OSError as exc:
            return FileOpResult(action.source_path.resolve(), final_target, "copy", action.is_companion, False, str(exc))

    def _create_file(self, source: Path, target: Path) -> str:
        if self.prefer_hardlink:
            try:
                os.link(source, target)
                return "hardlink"
            except OSError as exc:
                # 仅文件系统能力与跨卷失败允许降级；重名和权限错误不得转为覆盖复制。
                if exc.errno not in (errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS) and getattr(exc, "winerror", None) not in (1, 17, 50):
                    raise
        created = False
        try:
            with source.open("rb") as reader, target.open("xb") as writer:
                created = True
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            shutil.copystat(source, target)
        except OSError:
            if created:
                target.unlink()
            raise
        return "copy"

    def execute_plan(self, plan: DispatchPlan) -> ExecutionReport:
        """按照片组落地并持久化审计；一组失败只回退本组刚创建的目标。"""
        root = plan.output_root.resolve()
        report = ExecutionReport(output_root=root, total_actions=len(plan.actions))
        try:
            sources = {a.source_path.resolve() for a in plan.actions}
            for action in plan.actions:
                target = action.target_path.resolve()
                if target == root or not target.is_relative_to(root) or target in sources:
                    raise ValueError(f"分拣目标越界或与源文件重叠：{target}")
            journal = TransactionJournal(root)
            report.manifest_path = journal.path
        except (OSError, ValueError) as exc:
            report.errors.append(str(exc))
            report.results = [FileOpResult(a.source_path, a.target_path, "copy", a.is_companion, False, str(exc)) for a in plan.actions]
            report.failed_count = len(plan.actions)
            return report

        groups = {}
        for action in plan.actions:
            identity = action.asset_id or str(action.source_path.resolve().with_suffix(""))
            key = (action.basket_id, identity, str(action.target_path.parent.resolve()), action.target_path.stem)
            groups.setdefault(key, []).append(action)

        handled_actions = set()
        for actions in groups.values():
            targets, created, directories, results = [], [], [], []
            fatal = False
            try:
                for action in actions:
                    if not action.source_path.is_file():
                        raise FileNotFoundError(f"源文件不存在：{action.source_path}")
                base = actions[0].target_path.stem
                counter = 0
                recorded_targets = {Path(operation["target"]) for operation in journal.operations}
                while True:
                    stem = base if counter == 0 else f"{base}_{counter}"
                    targets = [a.target_path.with_stem(stem).absolute() for a in actions]
                    if len(set(targets)) != len(targets):
                        raise ValueError("同一资产组包含重名目标")
                    if all(not os.path.lexists(target) and target not in recorded_targets for target in targets):
                        break
                    counter += 1
                for target in targets:
                    parent = target.parent
                    missing = []
                    while not parent.exists() and parent != root:
                        missing.append(parent)
                        parent = parent.parent
                    for directory in reversed(missing):
                        directory.mkdir()
                        directories.append(directory)
                try:
                    journal.record_directories(directories)
                except (OSError, ValueError):
                    fatal = True
                    raise
                for action, target in zip(actions, targets):
                    if not target.resolve().is_relative_to(root):
                        raise ValueError(f"目标目录在执行期间变化：{target}")
                    operation = self._create_file(action.source_path.resolve(), target)
                    created.append((target, None))
                    created[-1] = (target, fingerprint(target))
                    try:
                        journal.record_created(action.source_path, target, operation, action.is_companion)
                    except (OSError, ValueError):
                        fatal = True
                        raise
                    results.append(FileOpResult(action.source_path.resolve(), target, operation, action.is_companion, True))
            except (OSError, ValueError) as exc:
                error = str(exc)
                for target, expected in reversed(created):
                    try:
                        if target.exists():
                            if expected is None or target.is_symlink() or fingerprint(target) != expected:
                                raise OSError(f"回退目标已变化，保留文件：{target}")
                            target.unlink()
                    except (OSError, ValueError) as cleanup_error:
                        error += f"；回退失败：{cleanup_error}"
                for directory in reversed(directories):
                    try:
                        if directory.exists() and not any(directory.iterdir()):
                            directory.rmdir()
                    except OSError as cleanup_error:
                        error += f"；目录回退失败：{cleanup_error}"
                results = [FileOpResult(a.source_path.resolve(), targets[i] if targets else a.target_path, "copy", a.is_companion, False, error) for i, a in enumerate(actions)]
            report.results.extend(results)
            handled_actions.update(id(action) for action in actions)
            if fatal:
                report.errors.append("事务清单写入失败，已停止后续分拣")
                break

        if len(report.results) < report.total_actions:
            for action in plan.actions:
                if id(action) not in handled_actions:
                    report.results.append(FileOpResult(action.source_path, action.target_path, "copy", action.is_companion, False, "审计写入失败，未执行"))
        report.successful_count = sum(r.success for r in report.results)
        report.failed_count = sum(not r.success for r in report.results)
        report.hardlink_count = sum(r.success and r.operation_type == "hardlink" for r in report.results)
        report.copy_count = sum(r.success and r.operation_type == "copy" for r in report.results)
        try:
            journal.finish("completed" if report.failed_count == 0 else "partial_failure")
        except OSError as exc:
            report.errors.append(f"事务清单结束状态写入失败：{exc}")
        return report
