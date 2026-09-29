#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 交互式自然语言搜图体验脚本"""

import os
import sys
from pathlib import Path
import numpy as np

# 确保能检索到 core/ 和 data/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ai.scene_classifier import SceneClassifier
from core.ai.semantic_search import SemanticSearchEngine

def interactive_search():
    print("======================================================================")
    print("        Photoye 纯本地自然语言以文搜图 - 交互式体验终端")
    print("======================================================================\n")

    test_dir = Path(__file__).resolve().parent / "test_images"
    image_paths = sorted(list(test_dir.glob("*.jpg")))
    
    if not image_paths:
        print(f"❌ 未在 {test_dir} 找到测试照片")
        return

    print("⏳ 正在初始化 OpenCLIP 多模态神经网络引擎...")
    scene_classifier = SceneClassifier()
    search_engine = SemanticSearchEngine()

    print(f"🖼️ 正在提取 {len(image_paths)} 张测试图片的 512 维特征向量...")
    identifiers = []
    embeddings = []

    for img_p in image_paths:
        emb = scene_classifier.extract_image_embedding(img_p)
        if emb is not None:
            identifiers.append(img_p.name)
            embeddings.append(emb)

    matrix = np.array(embeddings, dtype=np.float32)
    print(f"✅ 特征矩阵构建完成！共索引 {len(identifiers)} 张照片。\n")
    print("💡 提示：您可以输入任意自然语言生活描述（支持中英文，如：")
    print("   - 海滩风景 / beach sunset")
    print("   - 树林树木 / forest green trees")
    print("   - 两个人在一起 / two friends together")
    print("   - 很多人大合影 / a large crowd of people")
    print("   - 女生特写肖像 / girl headshot portrait")
    print("   输入 'q' 或 'exit' 退出交互。\n")

    while True:
        try:
            query = input("🔍 请输入您的检索词: ").strip()
            if not query:
                continue
            if query.lower() in ("q", "exit", "quit"):
                print("\n👋 退出搜图体验。")
                break

            results = search_engine.search(query, matrix, identifiers, top_k=5)
            print(f"\n📊 检索结果 Top 5 (针对查询: '{query}'):")
            print("----------------------------------------------------------------------")
            for idx, item in enumerate(results):
                bar_len = int(max(0, item.similarity) * 40)
                bar = "█" * bar_len
                print(f"  #{idx+1}: {item.identifier:<28} 匹配分: {item.similarity:.4f} | {bar}")
            print("----------------------------------------------------------------------\n")

        except KeyboardInterrupt:
            print("\n👋 退出搜图体验。")
            break
        except Exception as e:
            print(f"⚠️ 检索发生异常: {e}\n")

if __name__ == "__main__":
    interactive_search()
