#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载并整理 Photoye 所需的模型文件

目标：保证在 Windows 上直接运行，所有模型统一落在 photoye/models/models 目录。
支持的模型：
- OpenCV YuNet (人脸检测)
- OpenCV SFace (人脸识别/特征向量)
- Dlib shape predictor + face recognition
- MobileNetV3-Large-224 (场景分类)
- ImageNet 类别文件（用于分类结果映射）
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, Optional


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    percent = min(100, (block_num * block_size * 100) // (total_size or 1))
    sys.stdout.write(f"\r下载进度: {percent:3d}%")
    sys.stdout.flush()


def download_file(url: str, dst: Path, sha256: Optional[str] = None) -> bool:
    """下载文件到 dst。存在且校验通过则跳过。"""
    try:
        if dst.exists() and (sha256 is None or _verify_sha256(dst, sha256)):
            print(f"✅ 已存在 {dst.name}")
            return True

        print(f"⬇️  正在下载 {dst.name} ...")
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dst, _progress_hook)
        print()

        if sha256 and not _verify_sha256(dst, sha256):
            print(f"❌ 校验失败: {dst.name}")
            dst.unlink(missing_ok=True)
            return False

        print(f"✅ 下载完成 {dst.name}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 下载失败 {dst.name}: {exc}")
        return False


def _verify_sha256(path: Path, sha256: str) -> bool:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        ok = h.hexdigest() == sha256
        if not ok:
            print(f"⚠️  SHA256 不匹配，期望 {sha256} 实际 {h.hexdigest()}")
        return ok
    except Exception:
        return False


def _maybe_decompress_bz2(path: Path) -> Path:
    if path.suffix != ".bz2":
        return path

    import bz2

    target = path.with_suffix("")
    if target.exists():
        print(f"✅ 已解压 {target.name}")
        return target

    print(f"📦 解压 {path.name} ...")
    try:
        with bz2.BZ2File(path, "rb") as src, target.open("wb") as dst:
            dst.write(src.read())
        print(f"✅ 解压完成 -> {target.name}")
        return target
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 解压失败 {path.name}: {exc}, 将删除损坏文件后重试下载")
        path.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        return path


def _fetch_all(items: Iterable[dict]) -> None:
    for info in items:
        url = info["url"]
        filename = info["filename"]
        sha256 = info.get("sha256")
        dst = MODEL_DIR / filename

        if not download_file(url, dst, sha256=sha256):
            continue

        if dst.suffix == ".bz2":
            dst = _maybe_decompress_bz2(dst)
        if info.get("post_copy_to"):
            # 复制一份到兼容路径（如 OpenCV 需要模型位于同目录下）
            for alias in info["post_copy_to"]:
                alias_path = MODEL_DIR / alias
                if not alias_path.exists():
                    alias_path.write_bytes(dst.read_bytes())
                    print(f"🔁 复制 {dst.name} -> {alias}")


def main() -> None:
    models_to_download = [
        {
            "name": "OpenCV YuNet 人脸检测",
            "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "filename": "face_detection_yunet_2023mar.onnx",
            # 官方仓库文件更新过，暂不校验哈希
        },
        {
            "name": "OpenCV SFace 人脸识别",
            "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            "filename": "face_recognition_sface_2021dec.onnx",
            # 官方仓库文件更新过，暂不校验哈希
        },
        {
            "name": "Dlib 68 点关键点预测器",
            "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            "filename": "shape_predictor_68_face_landmarks.dat.bz2",
            # 官方未提供稳定哈希，留空
        },
        {
            "name": "Dlib 人脸识别 ResNet",
            "url": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
            "filename": "dlib_face_recognition_resnet_model_v1.dat.bz2",
        },
        {
            "name": "MobileNetV2-224 分类",
            "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx",
            "filename": "image_classification_mobilenetv2_2022apr.onnx",
            # 暂不校验哈希
        },
        {
            "name": "ImageNet 类别列表",
            "url": "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            "filename": "imagenet_classes.txt",
        },
        {
            "name": "OpenCLIP ViT-B/32 视觉编码器",
            "url": "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/model.onnx",
            "filename": "openclip_vitb32_vision.onnx",
        },
        {
            "name": "OpenCLIP ViT-B/32 文本编码器",
            "url": "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx",
            "filename": "openclip_vitb32_text.onnx",
        },
        {
            "name": "OpenCLIP Tokenizer",
            "url": "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
            "filename": "openclip_tokenizer.json",
        },
        {
            "name": "DINOv2 Base ONNX",
            "url": "https://huggingface.co/Xenova/dinov2-base/resolve/main/onnx/model.onnx",
            "filename": "dinov2_base.onnx",
        },
        {
            "name": "DINOv2 预处理配置",
            "url": "https://huggingface.co/Xenova/dinov2-base/resolve/main/preprocessor_config.json",
            "filename": "dinov2_preprocessor_config.json",
        },
    ]

    print("开始下载 AI 模型文件，目标目录:", MODEL_DIR)
    _fetch_all(models_to_download)
    print("\n✅ 模型下载脚本执行完毕！")


if __name__ == "__main__":
    main()