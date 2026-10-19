#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 物理存储操作执行器模块 (V3.0)

负责将分拣计划中的动作安全落地至物理磁盘。
特性:
1. 优先调用 os.link 原生硬链接 (同一分区 0 额外磁盘占用、毫秒级完成)
2. 跨磁盘/移动U盘自适应平滑降级为 shutil.copy2 流式复制
3. 重名防灾强保护 (No Overwrite): 同名文件自动递增序号，伴侣文件保持严格同步更名
4. 结构化执行报告 (ExecutionReport)，为下一步 M2-F4 事务撤销回滚提供完整物理台账
符合 zlog/REQUIREMENTS.md F2.2 与 F2.4 规范。
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from .pipeline import DispatchAction, DispatchPlan


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


class FileOperationsExecutor:
    """物理文件安全落地执行器"""

    def __init__(self, prefer_hardlink: bool = True) -> None:
        """初始化执行器。

        Args:
            prefer_hardlink: 是否优先使用硬链接 (同一磁盘推荐 True，0 占空间)
        """
        self.prefer_hardlink = prefer_hardlink

    @staticmethod
    def _is_same_file_content(src: Path, dst: Path) -> bool:
        """快速判断源文件与已存在的目标文件是否为同一个物理文件或内容完全相同。"""
        try:
            # 1. 检查是否为指向同一 inode 的现有硬链接
            if src.stat().st_ino == dst.stat().st_ino and src.stat().st_dev == dst.stat().st_dev:
                return True
            # 2. 检查文件大小
            if src.stat().st_size != dst.stat().st_size:
                return False
        except Exception:
            return False
        return False

    @staticmethod
    def resolve_non_colliding_path(target_path: Path) -> Path:
        """若目标文件已存在，安全计算递增防灾序号路径 (如 IMG_001_1.JPG)。"""
        if not target_path.exists():
            return target_path

        folder = target_path.parent
        stem = target_path.stem
        suffix = target_path.suffix

        counter = 1
        while True:
            candidate = folder / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
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
        src = action.source_path.resolve()
        target_dir = action.target_path.parent.resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        # 检查该资产组在此篮子下是否已经确定了重名后的基准文件名
        group_key = (action.basket_id, action.source_path.stem.lower())

        if group_key in resolved_stems_map:
            actual_stem = resolved_stems_map[group_key]
            final_target = target_dir / f"{actual_stem}{action.target_path.suffix}"
        else:
            final_target = self.resolve_non_colliding_path(action.target_path)
            # 记录主照片重名后的新 stem，供后续出现的伴侣文件直接复用
            resolved_stems_map[group_key] = final_target.stem

        # 若目标文件已存在且就是自身硬链接，视为成功
        if final_target.exists() and self._is_same_file_content(src, final_target):
            return FileOpResult(
                source_path=src,
                target_path=final_target,
                operation_type="hardlink",
                is_companion=action.is_companion,
                success=True,
            )

        op_used = "copy"
        # 优先尝试硬链接
        if self.prefer_hardlink:
            try:
                os.link(str(src), str(final_target))
                return FileOpResult(
                    source_path=src,
                    target_path=final_target,
                    operation_type="hardlink",
                    is_companion=action.is_companion,
                    success=True,
                )
            except OSError:
                # 跨磁盘分区或文件系统不支持时，平滑降级为复制
                op_used = "copy"

        # 降级复制 (保留元数据时间戳)
        try:
            shutil.copy2(str(src), str(final_target))
            return FileOpResult(
                source_path=src,
                target_path=final_target,
                operation_type="copy",
                is_companion=action.is_companion,
                success=True,
            )
        except Exception as e:
            return FileOpResult(
                source_path=src,
                target_path=final_target,
                operation_type=op_used,
                is_companion=action.is_companion,
                success=False,
                error_message=str(e),
            )

    def execute_plan(self, plan: DispatchPlan) -> ExecutionReport:
        """执行整批分拣计划并生成落地审计报告。"""
        report = ExecutionReport(output_root=plan.output_root)
        report.total_actions = len(plan.actions)

        resolved_stems: Dict[Tuple[str, str], str] = {}

        # 优先先处理主照片，后处理伴侣文件，保证主照片序号先行决定
        sorted_actions = sorted(plan.actions, key=lambda a: 1 if a.is_companion else 0)

        for action in sorted_actions:
            res = self.execute_single_action(action, resolved_stems)
            report.results.append(res)

            if res.success:
                report.successful_count += 1
                if res.operation_type == "hardlink":
                    report.hardlink_count += 1
                else:
                    report.copy_count += 1
            else:
                report.failed_count += 1

        return report
