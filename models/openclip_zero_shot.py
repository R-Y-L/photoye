#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenCLIP zero-shot scene classifier using ONNX Runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .model_interfaces import SceneClassifier

try:  # tokenizers is lightweight but may be missing if用户尚未安装
    from tokenizers import Tokenizer
except Exception:  # noqa: BLE001
    Tokenizer = None  # type: ignore

try:
    import onnxruntime as ort
except Exception:  # noqa: BLE001
    ort = None  # type: ignore

MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_LABELS = {
    "风景": "a wide landscape photograph, nature scenery",
    "建筑": "a photo of impressive architecture or buildings",
    "人物自拍": "a selfie portrait of a single person",
    "人物合照": "a group photo of multiple friends smiling",
    "室内": "a cozy indoor scene",
    "文档": "a scanned document or sheet of paper",
    "美食": "a delicious food photograph with close up dish",
}


class OpenCLIPZeroShotClassifier(SceneClassifier):
    """Zero-shot classifier that maps prompts directly to probabilities."""

    def __init__(
        self,
        vision_model_path: Optional[Path] = None,
        text_model_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
        labels_file: Optional[Path] = None,
    ) -> None:
        self.vision_path = Path(vision_model_path) if vision_model_path else MODEL_DIR / "openclip_vitb32_vision.onnx"
        self.text_path = Path(text_model_path) if text_model_path else MODEL_DIR / "openclip_vitb32_text.onnx"
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else MODEL_DIR / "openclip_tokenizer.json"
        self.labels_file = Path(labels_file) if labels_file else MODEL_DIR / "openclip_labels.json"

        self.context_length = 77
        self.vision_session = None
        self.text_session = None
        self.tokenizer = None
        self._label_prompts: Dict[str, str] = {}
        self._vision_input_name: Optional[str] = None
        self._text_input_names: List[str] = []
        self._logit_scale = 100.0  # 可调节，值越大分布越尖锐
        self._vision_inputs = []

        self._init_resources()

    # ------------------------------------------------------------------
    def _init_resources(self) -> None:
        if ort is None:
            print("⚠️ 未安装 onnxruntime，OpenCLIP 将使用占位输出")
            return

        if not self.vision_path.exists() or not self.text_path.exists():
            print("⚠️ OpenCLIP 模型文件缺失，路径:", self.vision_path, self.text_path)
            return

        try:
            self.vision_session = ort.InferenceSession(str(self.vision_path), providers=["CPUExecutionProvider"])
            self.text_session = ort.InferenceSession(str(self.text_path), providers=["CPUExecutionProvider"])
            self._vision_inputs = self.vision_session.get_inputs()
            self._vision_input_name = self._vision_inputs[0].name
            self._text_input_names = [inp.name for inp in self.text_session.get_inputs()]
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ 无法创建 OpenCLIP onnx session: {exc}")
            self.vision_session = None
            self.text_session = None

        if Tokenizer is None:
            print("⚠️ 未安装 tokenizers，OpenCLIP 将无法进行真正的文本编码")
        elif not self.tokenizer_path.exists():
            print("⚠️ 缺少 tokenizer.json: ", self.tokenizer_path)
        else:
            try:
                self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ 载入 tokenizer 失败: {exc}")
                self.tokenizer = None

        self._label_prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        prompts = DEFAULT_LABELS.copy()
        if self.labels_file.exists():
            try:
                user_config = json.loads(self.labels_file.read_text(encoding="utf-8"))
                for label, prompt in user_config.items():
                    prompts[label] = prompt
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ 读取 {self.labels_file} 失败: {exc}")
        return prompts

    # ------------------------------------------------------------------
    def classify(self, image_path: str) -> Dict[str, float]:
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return {}

        if not (self.vision_session and self.tokenizer):
            return self._mock_classification()

        # 这个模型是融合模型，同时需要图像和文本输入
        result = self._encode_image_with_text(image_path)
        if result is None:
            return self._mock_classification()
        
        return result

    # ------------------------------------------------------------------
    def _encode_image_with_text(self, image_path: str) -> Optional[Dict[str, float]]:
        """融合模型：同时输入图像和所有文本提示，直接获取相似度"""
        try:
            # 准备图像
            image = Image.open(image_path).convert("RGB")
            image = image.resize((224, 224))
            img = np.array(image).astype(np.float32) / 255.0
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
            img = (img - mean) / std
            img = img.transpose(2, 0, 1)[None, ...]  # [1, 3, 224, 224]
            
            # 准备文本
            labels = list(self._label_prompts.keys())
            prompts = [self._label_prompts[label] for label in labels]
            
            tokenized = [self._encode_single_prompt(p) for p in prompts]
            input_ids = np.stack([tk[0] for tk in tokenized]).astype(np.int64)  # [N, 77]
            attention = np.stack([tk[1] for tk in tokenized]).astype(np.int64)  # [N, 77]
            
            # 构造融合输入
            inputs = {
                "pixel_values": img,
                "input_ids": input_ids,
                "attention_mask": attention,
            }
            
            # 运行推理，获取 logits_per_image
            outputs = self.vision_session.run(None, inputs)
            # 输出顺序: logits_per_image, logits_per_text, text_embeds, image_embeds
            logits_per_image = outputs[0]  # [1, N]
            
            # softmax 转换为概率
            probs = self._softmax(logits_per_image[0])
            
            return {label: float(prob) for label, prob in zip(labels, probs)}
            
        except Exception as exc:
            print(f"OpenCLIP 推理失败: {exc}")
            import traceback
            traceback.print_exc()
            return None

    def _encode_prompts(self) -> tuple[Optional[np.ndarray], List[str]]:
        labels = list(self._label_prompts.keys())
        prompts = [self._label_prompts[label] for label in labels]
        if self.tokenizer is None:
            return None, labels
        try:
            tokenized = [self._encode_single_prompt(p) for p in prompts]
            input_ids = np.stack([tk[0] for tk in tokenized]).astype(np.int64)
            attention = np.stack([tk[1] for tk in tokenized]).astype(np.int64)
            ort_inputs = {}
            text_inputs = self.text_session.get_inputs()
            if len(text_inputs) == 1:
                ort_inputs[text_inputs[0].name] = input_ids
            else:
                ort_inputs[text_inputs[0].name] = input_ids
                ort_inputs[text_inputs[1].name] = attention
            text_out = self.text_session.run(None, ort_inputs)[0]
            text_feats = text_out / (np.linalg.norm(text_out, axis=-1, keepdims=True) + 1e-6)
            return text_feats, labels
        except Exception as exc:  # noqa: BLE001
            print(f"OpenCLIP 文本编码失败: {exc}")
            return None, labels

    def _encode_single_prompt(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        encoding = self.tokenizer.encode(prompt)
        ids = encoding.ids[: self.context_length]
        attention = getattr(encoding, "attention_mask", [1] * len(ids))
        mask = attention[: self.context_length]
        if len(ids) < self.context_length:
            pad_len = self.context_length - len(ids)
            ids += [0] * pad_len
            mask += [0] * pad_len
        return np.array(ids, dtype=np.int64), np.array(mask, dtype=np.int64)

    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / (np.sum(exp) + 1e-6)

    def _mock_classification(self) -> Dict[str, float]:
        labels = list(DEFAULT_LABELS.keys())
        weights = np.random.rand(len(labels))
        weights /= weights.sum()
        return dict(zip(labels, weights.tolist()))

