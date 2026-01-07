#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - MobileNetV2 场景分类模型适配器（真实推理版）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from .model_interfaces import SceneClassifier


MODEL_DIR = Path(__file__).resolve().parent / "models"


class MobileNetV2SceneClassifier(SceneClassifier):
    """使用 ONNX Runtime 运行 MobileNetV2"""

    def __init__(self, model_path: str | None = None, class_file: str | None = None):
        self.model_path = Path(model_path) if model_path else MODEL_DIR / "image_classification_mobilenetv2_2022apr.onnx"
        self.class_file = Path(class_file) if class_file else MODEL_DIR / "imagenet_classes.txt"
        self.session = None
        self.labels: List[str] = []

        self._load_labels()
        self._load_model()

    def _load_labels(self) -> None:
        if not self.class_file.exists():
            print(f"⚠️ 未找到类别文件: {self.class_file}")
            return
        try:
            # 文件是每行一个类别
            text = self.class_file.read_text(encoding="utf-8")
            self.labels = [line.strip() for line in text.splitlines() if line.strip()]
            print(f"✅ 加载 {len(self.labels)} 个 ImageNet 类别")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ 读取类别文件失败: {exc}")

    def _load_model(self) -> None:
        try:
            import onnxruntime as ort

            if not self.model_path.exists():
                print(f"⚠️ 未找到 MobileNetV2 模型: {self.model_path}")
                return

            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            print(f"✅ 加载 MobileNetV2 模型: {self.model_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ 加载 MobileNetV2 失败: {exc}")
            self.session = None

    def _preprocess(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        img = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # CHW
        return np.expand_dims(img, axis=0)

    def classify(self, image_path: str) -> Dict[str, float]:
        print(f"使用 MobileNetV2 进行场景分类: {image_path}")

        try:
            import os

            if not os.path.exists(image_path):
                print(f"图片文件不存在: {image_path}")
                return {}

            if self.session is None:
                return self._mock_classification()

            input_tensor = self._preprocess(image_path)
            input_name = self.session.get_inputs()[0].name
            logits = self.session.run(None, {input_name: input_tensor})[0]
            probs = self._softmax(logits[0])

            return self._map_to_coarse_categories(probs)

        except Exception as exc:  # noqa: BLE001
            print(f"场景分类出错: {exc}")
            return self._mock_classification()

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _map_to_coarse_categories(self, probs: np.ndarray) -> Dict[str, float]:
        """将 ImageNet 1000 类概率映射到 Photoye 的七个大类"""
        # ImageNet 没有直接的 "person" 类，需要通过服饰、活动、场景间接推断
        categories = {
            "人物": [
                # 服饰相关
                "suit", "groom", "jersey", "jean", "sweatshirt", "cardigan",
                "pajama", "kimono", "bikini", "swimming trunks", "miniskirt",
                "hoopskirt", "sarong", "mortarboard", "stole", "wig",
                # 人物活动/姿态相关  
                "baseball", "basketball", "volleyball", "tennis ball", "ping-pong ball",
                "football helmet", "ski", "snowboard", "golf ball",
                # 人物物品
                "lipstick", "sunglasses", "sunglass", "bow tie", "neck brace",
                "hair spray", "perfume", "lotion", "face powder",
            ],
            "动物": [
                "dog", "cat", "bird", "fish", "shark", "whale", "dolphin",
                "monkey", "ape", "bear", "zebra", "lion", "tiger", "elephant",
                "horse", "cow", "sheep", "pig", "goat", "rabbit", "hamster",
                "fox", "wolf", "deer", "giraffe", "hippo", "rhino",
                "snake", "lizard", "turtle", "frog", "spider", "scorpion",
                "butterfly", "bee", "ant", "beetle", "dragonfly",
                "parrot", "owl", "eagle", "hawk", "penguin", "flamingo",
                "retriever", "shepherd", "terrier", "poodle", "bulldog", "collie",
                "tabby", "persian", "siamese", "egyptian cat",
            ],
            "美食": [
                "pizza", "hamburger", "hotdog", "sandwich", "taco", "burrito",
                "sushi", "soup", "salad", "pasta", "noodle", "rice",
                "ice cream", "cake", "pie", "donut", "cookie", "chocolate",
                "coffee", "espresso", "cappuccino", "tea", "beer", "wine",
                "guacamole", "carbonara", "meat loaf", "bagel", "pretzel",
                "cheeseburger", "french fries", "waffle", "pancake",
                "plate", "bowl", "cup", "mug", "goblet", "wine bottle",
            ],
            "建筑": [
                "castle", "palace", "monastery", "mosque", "church", "cathedral",
                "library", "barn", "tower", "viaduct", "bridge", "dam",
                "skyscraper", "dome", "obelisk", "fountain", "triumphal arch",
                "bell cote", "beacon", "lighthouse", "steel arch bridge",
                "suspension bridge", "pier", "boathouse", "cinema", "theater",
            ],
            "室内": [
                "sofa", "couch", "television", "bookcase", "desk", "table",
                "dining table", "bed", "wardrobe", "cabinet", "chest",
                "kitchen", "oven", "stove", "refrigerator", "microwave",
                "chair", "throne", "studio couch", "four-poster",
                "entertainment center", "home theater", "window shade",
                "washbasin", "bathtub", "shower curtain", "toilet seat",
            ],
            "文档": [
                "book", "notebook", "binder", "menu", "comic book", "crossword",
                "envelope", "letter", "packet", "paper towel", "newspaper",
                "scoreboard", "web site", "monitor", "screen", "laptop",
                "notebook computer", "desktop computer", "keyboard", "mouse",
            ],
            "风景": [
                "mountain", "alp", "valley", "lakeside", "lakeshore", "seashore",
                "cliff", "promontory", "volcano", "desert", "sandbar",
                "forest", "jungle", "rainforest", "waterfall", "geyser",
                "coral reef", "sea", "ocean", "river", "stream", "pond",
                "sky", "cloud", "sunset", "sunrise", "rainbow",
            ],
        }

        if not self.labels:
            # 没有标签时直接返回 top1 作为风景
            return {"风景": float(np.max(probs))}

        # 构建 label->index 映射
        scores = {key: 0.0 for key in categories}
        for idx, label in enumerate(self.labels):
            for cat, keywords in categories.items():
                if any(kw in label for kw in keywords):
                    scores[cat] += float(probs[idx])

        # 如果全为 0，则将最大概率归为风景（默认安全）
        if all(score == 0 for score in scores.values()):
            scores["风景"] = float(np.max(probs))

        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    def _mock_classification(self) -> Dict[str, float]:
        weights = np.random.rand(7)
        weights = weights / weights.sum()
        return dict(zip(["风景", "建筑", "动物", "文档", "室内", "美食", "人物"], weights.tolist()))