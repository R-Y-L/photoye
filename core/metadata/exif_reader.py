#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye EXIF 元数据解析模块 (V3.0)

负责纯 Python 提取照片的拍摄时间 (DateTimeOriginal)、
GPS 经纬度坐标与相机旋转角度 (Orientation)。
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from PIL import ExifTags, Image


class ExifReader:
    """照片 EXIF 基础元数据提取器"""

    @staticmethod
    def _convert_gps_to_degrees(value: Tuple) -> float:
        """将 EXIF GPS 的度分秒格式转换为十进制浮点度数。"""
        # PIL 返回的可能是 IFDRational 或 float 元组
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)

    @classmethod
    def read_metadata(cls, image_path: Union[str, Path]) -> Dict[str, Optional[Union[str, float, int]]]:
        """读取单张照片的 EXIF 元数据。

        Returns:
            {
                "created_at": "2026-05-01T14:30:00" 或 None,
                "latitude": 30.123456 或 None,
                "longitude": 120.123456 或 None,
                "orientation": 1 或 None,
            }
        """
        result: Dict[str, Optional[Union[str, float, int]]] = {
            "created_at": None,
            "latitude": None,
            "longitude": None,
            "orientation": None,
        }

        p = Path(image_path)
        if not p.exists():
            return result

        try:
            with Image.open(p) as img:
                exif_data = img.getexif()
                if not exif_data:
                    return result

                # 1. 解析旋转方向 (Orientation)
                result["orientation"] = exif_data.get(ExifTags.Base.Orientation)

                # 2. 解析拍摄时间 (DateTimeOriginal / DateTimeDigitized / DateTime)
                # 检查主 EXIF 与 IFD 子树
                dt_str = exif_data.get(ExifTags.Base.DateTime)
                if hasattr(ExifTags, "IFD") and ExifTags.IFD.Exif in exif_data:
                    sub_exif = exif_data.get_ifd(ExifTags.IFD.Exif)
                    dt_str = sub_exif.get(ExifTags.Base.DateTimeOriginal) or dt_str

                if dt_str:
                    try:
                        # 格式通常为 "YYYY:MM:DD HH:MM:SS"
                        clean_dt = str(dt_str).strip()
                        parsed_dt = datetime.datetime.strptime(clean_dt, "%Y:%m:%d %H:%M:%S")
                        result["created_at"] = parsed_dt.isoformat()
                    except Exception:
                        pass

                # 3. 解析 GPS 经纬度
                if hasattr(ExifTags, "IFD") and ExifTags.IFD.GPSInfo in exif_data:
                    gps_info = exif_data.get_ifd(ExifTags.IFD.GPSInfo)
                    if gps_info:
                        gps_lat = gps_info.get(2)  # GPSLatitude
                        gps_lat_ref = gps_info.get(1)  # GPSLatitudeRef ('N' / 'S')
                        gps_lon = gps_info.get(4)  # GPSLongitude
                        gps_lon_ref = gps_info.get(3)  # GPSLongitudeRef ('E' / 'W')

                        if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
                            lat = cls._convert_gps_to_degrees(gps_lat)
                            if str(gps_lat_ref).upper() == "S":
                                lat = -lat
                            lon = cls._convert_gps_to_degrees(gps_lon)
                            if str(gps_lon_ref).upper() == "W":
                                lon = -lon
                            result["latitude"] = round(lat, 6)
                            result["longitude"] = round(lon, 6)

        except Exception as e:
            # 容错降级
            pass

        return result
