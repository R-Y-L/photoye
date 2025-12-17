#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Centralized model profile definitions for Photoye.

These presets let us switch between lightweight and accuracy-first
configurations without touching the analyzer code. Each profile specifies
which detector, recognizer, and scene classifier to load.
"""

from __future__ import annotations

from typing import Dict

MODEL_PROFILE_ENV_VAR = "PHOTOYE_MODEL_PROFILE"
DEFAULT_MODEL_PROFILE = "balanced"

MODEL_PROFILES: Dict[str, Dict[str, str]] = {
    # Focus on responsiveness; pure OpenCV pipeline keeps dependencies light.
    "speed": {
        "detector": "yunet",
        "recognizer": "sface",
        "classifier": "mobilenetv2",
        "description": "最快组合：YuNet + SFace + MobileNetV2",
    },
    # Default mix: fast YuNet detector + more accurate dlib embeddings.
    "balanced": {
        "detector": "yunet",
        "recognizer": "dlib",
        "classifier": "mobilenetv2",
        "description": "默认组合：YuNet + dlib + MobileNetV2",
    },
    # Heavyweight option: rely entirely on dlib for detection+recognition.
    "accuracy": {
        "detector": "dlib",
        "recognizer": "dlib",
        "classifier": "mobilenetv2",
        "description": "高精度组合：dlib 检测 + dlib 识别",
    },
    # Zero-shot / fine-grained scene understanding with OpenCLIP.
    "zeroshot": {
        "detector": "yunet",
        "recognizer": "sface",
        "classifier": "openclip",
        "description": "零样本组合：YuNet + SFace + OpenCLIP",
    },
}


def list_available_profiles() -> Dict[str, Dict[str, str]]:
    """Expose a shallow copy for UI/CLI listing."""
    return {name: config.copy() for name, config in MODEL_PROFILES.items()}
