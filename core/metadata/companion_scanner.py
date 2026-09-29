#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 伴侣文件探测与资产打包扫描模块 (V3.0)

负责同目录同名伴侣资产探测 (LivePhoto 动态视频、调色侧车、RAW 负片)，
将其与主照片绑定为原子资产组，确保物理分拣时同进同退。
符合 zlog/REQUIREMENTS.md F2.3 规范。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from .exif_reader import ExifReader


@dataclass
class CompanionAsset:
    """伴侣文件元数据对象"""
    filepath: Path                     # 伴侣文件绝对路径
    file_type: str                     # "live_video", "sidecar", "raw_negative"
    filesize: int                      # 文件大小 (字节)


@dataclass
class PhotoAssetGroup:
    """主照片及其绑定的全部伴侣文件组成的原子资产单元"""
    main_photo_path: Path              # 主照片绝对路径
    filesize: int                      # 主照片大小
    created_at: Optional[str] = None   # EXIF 拍摄时间 (ISO 格式)
    latitude: Optional[float] = None   # GPS 纬度
    longitude: Optional[float] = None  # GPS 经度
    orientation: Optional[int] = None  # 旋转方向
    companions: List[CompanionAsset] = field(default_factory=list)


class CompanionScanner:
    """伴侣文件探测与资产组装器"""

    # 主照片受支持的后缀
    PRIMARY_IMAGE_EXTENSIONS: Set[str] = {
        ".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff"
    }

    # 伴侣文件后缀映射分类
    COMPANION_TYPE_MAPPING: Dict[str, str] = {
        # 1. 动态照片视频轨 (LivePhoto)
        ".mov": "live_video",
        ".mp4": "live_video",
        # 2. 调色侧车与编辑数据 (Sidecar)
        ".xmp": "sidecar",
        ".aae": "sidecar",
        # 3. 相机原始负片双格式伴侣 (RAW)
        ".cr2": "raw_negative",
        ".cr3": "raw_negative",
        ".arw": "raw_negative",
        ".nef": "raw_negative",
        ".dng": "raw_negative",
        ".raf": "raw_negative",
        ".rw2": "raw_negative",
    }

    @classmethod
    def scan_directory(
        cls,
        directory_path: Union[str, Path],
        recursive: bool = False,
    ) -> List[PhotoAssetGroup]:
        """扫描指定目录，自动完成主照片探测、EXIF 元数据提取与伴侣文件绑定。

        Args:
            directory_path: 相册目录路径
            recursive: 是否递归扫描子目录 (默认 False)

        Returns:
            排好序的 PhotoAssetGroup 原子资产组列表
        """
        dir_p = Path(directory_path)
        if not dir_p.exists() or not dir_p.is_dir():
            return []

        # 1. 收集目录下所有文件，按目录分组
        files_by_dir: Dict[Path, List[Path]] = {}
        pattern = "**/*" if recursive else "*"
        for item in dir_p.glob(pattern):
            if item.is_file():
                parent = item.parent
                if parent not in files_by_dir:
                    files_by_dir[parent] = []
                files_by_dir[parent].append(item)

        asset_groups: List[PhotoAssetGroup] = []

        # 2. 对每个目录内部建立主干名索引探测伴侣
        for folder, files in files_by_dir.items():
            # 建立 stem -> List[Path] 索引 (小写匹配)
            stem_index: Dict[str, List[Path]] = {}
            for f in files:
                stem_lower = f.stem.lower()
                if stem_lower not in stem_index:
                    stem_index[stem_lower] = []
                stem_index[stem_lower].append(f)

            # 遍历找出所有主照片
            for f in files:
                ext_lower = f.suffix.lower()
                if ext_lower in cls.PRIMARY_IMAGE_EXTENSIONS:
                    stem_lower = f.stem.lower()
                    sibling_files = stem_index.get(stem_lower, [])

                    # 探测同主名伴侣
                    companions: List[CompanionAsset] = []
                    for sib in sibling_files:
                        if sib == f:
                            continue
                        sib_ext = sib.suffix.lower()
                        if sib_ext in cls.COMPANION_TYPE_MAPPING:
                            companions.append(
                                CompanionAsset(
                                    filepath=sib.resolve(),
                                    file_type=cls.COMPANION_TYPE_MAPPING[sib_ext],
                                    filesize=sib.stat().st_size,
                                )
                            )

                    # 提取 EXIF 元数据
                    exif_meta = ExifReader.read_metadata(f)

                    asset_groups.append(
                        PhotoAssetGroup(
                            main_photo_path=f.resolve(),
                            filesize=f.stat().st_size,
                            created_at=exif_meta.get("created_at"),
                            latitude=exif_meta.get("latitude"),
                            longitude=exif_meta.get("longitude"),
                            orientation=exif_meta.get("orientation"),
                            companions=companions,
                        )
                    )

        # 默认按文件名排序输出
        asset_groups.sort(key=lambda g: g.main_photo_path.name)
        return asset_groups
