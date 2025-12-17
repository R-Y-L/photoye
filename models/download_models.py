#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import urllib.request
import zipfile
import sys

def download_file(url, filename):
    """下载文件并显示进度"""
    def progress_hook(block_num, block_size, total_size):
        percent = min(100, (block_num * block_size * 100) // total_size)
        sys.stdout.write(f'\r下载进度: {percent}%')
        sys.stdout.flush()
    
    print(f"正在下载 {filename}...")
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n{filename} 下载完成!")
        return True
    except Exception as e:
        print(f"\n下载失败: {e}")
        return False

def extract_zip(filename, extract_to="./"):
    """解压zip文件"""
    print(f"正在解压 {filename}...")
    try:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"{filename} 解压完成!")
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False

def main():
    # 创建models目录
    if not os.path.exists("models"):
        os.makedirs("models")
    
    os.chdir("models")
    
    # YOLOv8-Face 模型下载链接 (示例链接，需要替换为实际可用链接)
    # 注意：这里使用占位链接，实际使用时请替换为真实模型链接
    yolov8_url = "https://github.com/clovaai/yolov8/releases/download/v1.0/yolov8n-face.onnx"
    yolov8_filename = "yolov8n-face.onnx"
    
    # ArcFace 模型下载链接 (示例链接，需要替换为实际可用链接)
    # 注意：这里使用占位链接，实际使用时请替换为真实模型链接
    arcface_url = "https://github.com/clovaai/arcface/releases/download/v1.0/arcface_resnet18.onnx"
    arcface_filename = "arcface_resnet18.onnx"
    
    print("开始下载AI模型文件...")
    
    # 下载YOLOv8-Face模型
    if not os.path.exists(yolov8_filename):
        if download_file(yolov8_url, yolov8_filename):
            print(f"YOLOv8-Face 模型已保存为 {yolov8_filename}")
        else:
            print("YOLOv8-Face 模型下载失败")
    else:
        print(f"YOLOv8-Face 模型已存在: {yolov8_filename}")
    
    # 下载ArcFace模型
    if not os.path.exists(arcface_filename):
        if download_file(arcface_url, arcface_filename):
            print(f"ArcFace 模型已保存为 {arcface_filename}")
        else:
            print("ArcFace 模型下载失败")
    else:
        print(f"ArcFace 模型已存在: {arcface_filename}")
    
    print("模型下载脚本执行完毕!")

if __name__ == "__main__":
    main()