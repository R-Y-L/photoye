#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 多维规则物理分拣管道与去向篮子模型 (V3.0)

支持左侧多维漏斗筛选 + 右侧去向篮子 (DispatchBasket) 架构。
支持:
1. 自动规则吸附 (人物定向、合照1对多扇出、题材归集、日期范围)
2. 手工显式拖拽/选入追加 (例如将特定风景或特定日期的照片手动加给某人)
3. 伴侣文件 (LivePhoto/Sidecar/RAW) 镜像级联绑定，同进同退
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Set, Tuple, Union

from core.ai.scene_classifier import PhotoMultiLabel
from core.metadata.companion_scanner import CompanionAsset, PhotoAssetGroup


@dataclass
class BasketFilterRule:
    """单个去向篮子的自动匹配过滤规则 (可选)"""
    person_names: List[str] = field(default_factory=list)      # 包含出镜人物 (如 ["Alice"])
    scene_tags: List[str] = field(default_factory=list)        # 包含场景题材 (如 ["风景"])
    date_start: Optional[str] = None                           # 拍摄日期起 (ISO 格式或 "YYYY-MM-DD")
    date_end: Optional[str] = None                             # 拍摄日期止
    require_all_persons: bool = False                          # 是否要求所有指定人物均出镜


@dataclass
class DispatchBasket:
    """分发去向篮子对象 (代表一个朋友相册、事件相册或分类归档)"""
    basket_id: str                                             # 篮子唯一 ID
    name: str                                                  # 显示名称 (用户可随时修改，如 "云南旅行_精选")
    output_subfolder: str                                      # 相对输出子目录 (用户可随时修改，如 "云南旅行/YouYu")
    rule: Optional[BasketFilterRule] = None                    # 自动吸附规则
    manual_photo_stems: Set[str] = field(default_factory=set)  # 用户手动拖入/追加的主照片主干名集合
    excluded_photo_stems: Set[str] = field(default_factory=set)# 用户显式移出/排除的主照片主干名集合 (黑名单机制)
    naming_pattern: str = "original"                           # 命名模板: "original" (原名), "date" (时间), "prefix" (自定义前缀)
    custom_prefix: str = ""                                    # 自定义前缀文本 (如 "YouYu_")


@dataclass
class DispatchAction:
    """单个物理文件分拣动作"""
    source_path: Path                                          # 源文件路径
    target_path: Path                                          # 目标文件路径
    basket_id: str                                             # 所属去向篮子
    is_companion: bool = False                                 # 是否为伴侣文件
    companion_type: Optional[str] = None                       # 伴侣类型
    reason: str = "auto_rule"                                  # 加入原因: "auto_rule" 或 "manual_drag"
    asset_id: Optional[str] = None                             # 主照片完整路径，标识伴侣原子事务


@dataclass
class DispatchPlan:
    """整批分拣执行计划"""
    output_root: Path
    baskets: List[DispatchBasket] = field(default_factory=list)
    actions: List[DispatchAction] = field(default_factory=list)
    basket_item_counts: Dict[str, int] = field(default_factory=dict)  # 统计各篮子照片组数量


class SortingPipeline:
    """多维规则分拣管道核心"""

    def __init__(self, output_root: Union[str, Path]) -> None:
        self.output_root = Path(output_root).resolve()

    def resolve_output_folder(self, subfolder: str) -> Path:
        """在输入边界拒绝绝对路径、上级跳转及链接到输出根目录外的路径。"""
        path = Path(subfolder)
        windows_path = PureWindowsPath(subfolder)
        if not subfolder.strip() or path.is_absolute() or windows_path.drive or windows_path.root or ".." in windows_path.parts:
            raise ValueError("输出子目录必须是根目录内的相对路径，不允许 .. 或绝对路径")
        target = (self.output_root / path).resolve()
        if not target.is_relative_to(self.output_root):
            raise ValueError("输出子目录不能越过所选根目录")
        return target

    @staticmethod
    def _is_asset_match_rule(
        group: PhotoAssetGroup,
        label: Optional[PhotoMultiLabel],
        rule: BasketFilterRule,
    ) -> bool:
        """判定单个资产组是否命中篮子的自动规则。"""
        # 1. 匹配人物维度
        if rule.person_names:
            if not label or not label.person_names:
                return False
            present = set(label.person_names)
            if rule.require_all_persons:
                if not set(rule.person_names).issubset(present):
                    return False
            else:
                if not any(p in present for p in rule.person_names):
                    return False

        # 2. 匹配题材维度
        if rule.scene_tags:
            if not label or not label.scene_tags:
                return False
            if not any(s in label.scene_tags for s in rule.scene_tags):
                return False

        # 3. 匹配日期维度
        if rule.date_start or rule.date_end:
            if not group.created_at:
                return False
            photo_date = group.created_at[:10]  # "YYYY-MM-DD"
            if rule.date_start and photo_date < rule.date_start:
                return False
            if rule.date_end and photo_date > rule.date_end:
                return False

        return True

    def build_default_baskets(
        self,
        asset_groups: List[PhotoAssetGroup],
        labels_map: Dict[str, PhotoMultiLabel],
    ) -> List[DispatchBasket]:
        """根据相册中的出镜人物与题材自动发现并生成推荐的基础篮子列表。"""
        all_persons: Set[str] = set()
        all_scenes: Set[str] = set()

        for g in asset_groups:
            l = labels_map.get(g.main_photo_path.name) or labels_map.get(str(g.main_photo_path))
            if l:
                all_persons.update(l.person_names)
                all_scenes.update(l.scene_tags)

        baskets: List[DispatchBasket] = []

        # 1. 为每个出镜人物自动建篮子 (合影1对多扇出基础)
        for p in sorted(list(all_persons)):
            if p:
                baskets.append(
                    DispatchBasket(
                        basket_id=f"person_{p}",
                        name=f"人物_{p}",
                        output_subfolder=f"人物/{p}",
                        rule=BasketFilterRule(person_names=[p]),
                    )
                )

        # 2. 为纯题材建篮子
        for s in sorted(list(all_scenes)):
            if s:
                baskets.append(
                    DispatchBasket(
                        basket_id=f"scene_{s}",
                        name=f"题材_{s}",
                        output_subfolder=f"题材/{s}",
                        rule=BasketFilterRule(scene_tags=[s]),
                    )
                )

        return baskets

    @staticmethod
    def _compute_target_filename(
        group: PhotoAssetGroup,
        suffix: str,
        naming_pattern: str,
        custom_prefix: str,
        seq_idx: int,
    ) -> str:
        """根据命名模式计算落地的目标文件名 (保证主图与伴侣文件共享相同前缀与序号)。"""
        if naming_pattern == "date" and group.created_at:
            try:
                # 转换为 "20260520_183000"
                dt = datetime.datetime.fromisoformat(group.created_at)
                base = dt.strftime("%Y%m%d_%H%M%S")
                return f"{base}{suffix}"
            except Exception:
                pass
        elif naming_pattern == "prefix" and custom_prefix.strip():
            clean_pre = custom_prefix.strip()
            return f"{clean_pre}_{seq_idx:03d}{suffix}"

        # 默认原名
        return f"{group.main_photo_path.stem}{suffix}"

    def plan_dispatch(
        self,
        asset_groups: List[PhotoAssetGroup],
        labels_map: Dict[str, PhotoMultiLabel],
        baskets: List[DispatchBasket],
    ) -> DispatchPlan:
        """结合自动规则、手工追加、显式排除与重命名模板，为所有指定篮子生成具体的物理文件分发动作。"""
        plan = DispatchPlan(output_root=self.output_root, baskets=baskets)
        actions: List[DispatchAction] = []
        counts: Dict[str, int] = {b.basket_id: 0 for b in baskets}

        for basket in baskets:
            target_folder = self.resolve_output_folder(basket.output_subfolder)
            handled_stems_for_basket: Set[str] = set()
            seq_counter = 1

            for group in asset_groups:
                stem_lower = group.main_photo_path.stem.lower()
                # 排除黑名单保护: 若用户手动移除了该照片，坚决不在本篮子落地
                if stem_lower in [s.lower() for s in basket.excluded_photo_stems]:
                    continue

                label = labels_map.get(group.main_photo_path.name) or labels_map.get(str(group.main_photo_path))

                is_manual = (stem_lower in [s.lower() for s in basket.manual_photo_stems])
                is_auto = False
                if basket.rule is not None:
                    is_auto = self._is_asset_match_rule(group, label, basket.rule)

                if is_manual or is_auto:
                    handled_stems_for_basket.add(stem_lower)
                    reason_desc = "manual_drag" if is_manual else "auto_rule"

                    # 计算目标主文件名
                    target_main_name = self._compute_target_filename(
                        group=group,
                        suffix=group.main_photo_path.suffix,
                        naming_pattern=basket.naming_pattern,
                        custom_prefix=basket.custom_prefix,
                        seq_idx=seq_counter,
                    )
                    if Path(target_main_name).name != target_main_name or PureWindowsPath(target_main_name).name != target_main_name:
                        raise ValueError("照片命名前缀不能包含目录路径")

                    # 1. 主照片动作
                    actions.append(
                        DispatchAction(
                            source_path=group.main_photo_path,
                            target_path=target_folder / target_main_name,
                            basket_id=basket.basket_id,
                            is_companion=False,
                            reason=reason_desc,
                            asset_id=str(group.main_photo_path.resolve()),
                        )
                    )

                    # 2. 伴侣文件镜像级联动作 (共享相同的新 stem，同进同退)
                    main_stem_new = Path(target_main_name).stem
                    for comp in group.companions:
                        comp_target_name = f"{main_stem_new}{comp.filepath.suffix}"
                        actions.append(
                            DispatchAction(
                                source_path=comp.filepath,
                                target_path=target_folder / comp_target_name,
                                basket_id=basket.basket_id,
                                is_companion=True,
                                companion_type=comp.file_type,
                                reason=reason_desc,
                                asset_id=str(group.main_photo_path.resolve()),
                            )
                        )

                    seq_counter += 1

            counts[basket.basket_id] = len(handled_stems_for_basket)

        plan.actions = actions
        plan.basket_item_counts = counts
        return plan

