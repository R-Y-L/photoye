#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - 后台任务管理模块
负责处理所有耗时操作，避免UI线程阻塞

版本: 2.1 (CLIP Embedding 支持)
"""

import os
import time
from pathlib import Path
from typing import List, Callable, Optional, Dict, Any
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap
from database import (
    add_photo,
    add_photos_batch,
    is_photo_exist,
    update_photo_status,
    add_face_data,
    add_faces_batch,
    get_photo_status,
    get_photos_without_faces,
    update_photo_embedding_by_path,
    batch_update_photo_embeddings,
    get_photos_without_embedding,
    search_photos_by_embedding,
)
from analyzer import AIAnalyzer


# ==================== 缩略图异步加载 ====================

class ThumbnailWorker(QThread):
    """
    缩略图生成工作线程
    
    在后台生成缩略图，避免UI卡顿
    """
    
    # 信号：(文件路径, QPixmap缩略图)
    thumbnail_ready = pyqtSignal(str, object)
    # 信号：批量完成
    batch_completed = pyqtSignal()
    
    def __init__(self, thumbnail_size: int = 150):
        super().__init__()
        self.thumbnail_size = thumbnail_size
        self.pending_paths: List[str] = []
        self.is_running = False
        self.should_stop = False
        self._lock = False  # 简单锁
    
    def add_paths(self, paths: List[str]):
        """添加待处理的图片路径"""
        # 去重添加
        existing = set(self.pending_paths)
        for p in paths:
            if p not in existing:
                self.pending_paths.append(p)
    
    def run(self):
        """线程主执行函数"""
        self.is_running = True
        self.should_stop = False
        
        while not self.should_stop:
            if not self.pending_paths:
                # 没有待处理的，休眠一下
                time.sleep(0.05)
                continue
            
            # 取出一个路径
            path = self.pending_paths.pop(0)
            
            try:
                pixmap = self._create_thumbnail(path)
                if pixmap:
                    self.thumbnail_ready.emit(path, pixmap)
            except Exception as e:
                print(f"生成缩略图失败: {path}, 错误: {e}")
            
            # 如果队列空了，发送批量完成信号
            if not self.pending_paths:
                self.batch_completed.emit()
        
        self.is_running = False
    
    def _create_thumbnail(self, image_path: str) -> Optional[QPixmap]:
        """创建缩略图"""
        if not os.path.exists(image_path):
            return None
        
        image = QImage(image_path)
        if image.isNull():
            return None
        
        from PyQt6.QtCore import Qt
        thumbnail = image.scaled(
            self.thumbnail_size, self.thumbnail_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation  # 使用快速变换提升性能
        )
        return QPixmap.fromImage(thumbnail)
    
    def stop(self):
        """停止线程"""
        self.should_stop = True
        self.pending_paths.clear()


class ScanWorker(QThread):
    """
    文件扫描工作线程
    
    导入时自动进行场景分类（不做人脸检测），为照片打上基础标签
    """
    
    # 定义信号
    progress_updated = pyqtSignal(int, int)  # (current, total)
    file_found = pyqtSignal(str)  # filepath
    scan_completed = pyqtSignal(int)  # total_files
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self, root_path: str, supported_extensions: List[str] = None, model_profile: Optional[str] = None):
        """
        初始化扫描工作线程
        
        Args:
            root_path: 要扫描的根目录路径
            supported_extensions: 支持的文件扩展名列表
            model_profile: 模型档位（用于场景分类）
        """
        super().__init__()
        
        self.root_path = root_path
        self.supported_extensions = supported_extensions or [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'
        ]
        self.is_running = False
        self.should_stop = False
        self.model_profile = model_profile

        # 初始化场景分类器（轻量，仅用于分类）
        self.scene_classifier = None
        self._init_classifier()
        
        print(f"扫描工作线程初始化")
        print(f"根目录: {root_path}")
        print(f"支持格式: {self.supported_extensions}")
    
    def _init_classifier(self):
        """初始化场景分类器和 CLIP 编码器"""
        # 初始化 CLIP 编码器 (用于语义 embedding)
        try:
            from models.clip_embedding import CLIPEmbeddingEncoder
            self.clip_encoder = CLIPEmbeddingEncoder()
            if self.clip_encoder.is_available():
                print("✅ CLIP Embedding 编码器初始化成功")
            else:
                print("⚠️ CLIP Embedding 编码器不可用")
                self.clip_encoder = None
        except Exception as e:
            print(f"⚠️ CLIP 编码器初始化失败: {e}")
            self.clip_encoder = None
        
        # 初始化场景分类器 (可选，用于向后兼容)
        try:
            from models.mobilenetv2_classifier import MobileNetV2SceneClassifier
            self.scene_classifier = MobileNetV2SceneClassifier()
            print("✅ 场景分类器初始化成功")
        except Exception as e:
            print(f"⚠️ 场景分类器初始化失败: {e}")
            self.scene_classifier = None
    
    def run(self):
        """线程主执行函数"""
        self.is_running = True
        self.should_stop = False
        
        try:
            print(f"开始扫描目录: {self.root_path}")
            
            if not os.path.exists(self.root_path):
                self.error_occurred.emit(f"目录不存在: {self.root_path}")
                return
            
            self._scan_and_classify_directory()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"扫描过程中发生错误: {str(e)}")
        finally:
            self.is_running = False
    
    def _scan_and_classify_directory(self):
        """扫描目录并对每张照片提取 CLIP embedding 和进行场景分类"""
        # 收集所有支持的图片文件
        image_files = []
        for root, dirs, files in os.walk(self.root_path):
            if self.should_stop:
                break
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.supported_extensions):
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
        
        if self.should_stop:
            return
        
        total_files = len(image_files)
        print(f"发现 {total_files} 个图片文件")
        
        # 批量添加到数据库
        new_files = [f for f in image_files if not is_photo_exist(f)]
        if new_files:
            add_photos_batch(new_files)
            print(f"批量添加 {len(new_files)} 张新照片")
        
        # 逐个提取 CLIP embedding 和进行场景分类
        processed_files = 0
        for file_path in image_files:
            if self.should_stop:
                break
            
            # 获取照片状态
            status_row = get_photo_status(file_path)
            if status_row:
                photo_id, status, category = status_row
                
                # 提取 CLIP embedding (优先)
                if self.clip_encoder:
                    try:
                        embedding = self.clip_encoder.encode_image(file_path)
                        if embedding is not None:
                            update_photo_embedding_by_path(file_path, embedding)
                    except Exception as e:
                        print(f"CLIP embedding 提取失败: {file_path}, 错误: {e}")
                
                # 如果没有分类，进行场景分类（向后兼容）
                if not category and self.scene_classifier:
                    try:
                        classification = self.scene_classifier.classify(file_path)
                        if classification:
                            best_category = max(classification.items(), key=lambda x: x[1])[0]
                            update_photo_status(photo_id, 'done', best_category)
                    except Exception as e:
                        print(f"分类失败: {file_path}, 错误: {e}")
                
                self.file_found.emit(file_path)
            
            processed_files += 1
            self.progress_updated.emit(processed_files, total_files)
        
        if not self.should_stop:
            self.scan_completed.emit(processed_files)
    
    def stop_scan(self):
        """停止扫描"""
        print("请求停止扫描")
        self.should_stop = True


class FaceAnalysisWorker(QThread):
    """
    人脸分析工作线程
    
    专门用于人脸检测与识别，独立于照片导入流程
    """
    
    # 定义信号
    progress_updated = pyqtSignal(int, int)  # (current, total)
    face_detected = pyqtSignal(str, int)  # (filepath, face_count)
    analysis_completed = pyqtSignal(int, int)  # (total_photos, total_faces)
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self, library_path: str = None, model_profile: Optional[str] = None):
        """
        初始化人脸分析工作线程
        
        Args:
            library_path: 限制在某个目录下分析
            model_profile: 模型档位
        """
        super().__init__()
        
        self.library_path = library_path
        self.model_profile = model_profile
        self.is_running = False
        self.should_stop = False
        
        # 初始化AI分析器
        self.ai_analyzer = AIAnalyzer(model_profile=model_profile)
        
        print(f"人脸分析工作线程初始化")
        if library_path:
            print(f"分析目录: {library_path}")
    
    def run(self):
        """线程主执行函数"""
        self.is_running = True
        self.should_stop = False
        
        try:
            print("开始人脸分析...")
            self._analyze_faces()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"人脸分析错误: {str(e)}")
        finally:
            self.is_running = False
    
    def _analyze_faces(self):
        """分析所有需要人脸检测的照片"""
        # 获取需要人脸检测的照片（分类为人物相关但尚无人脸数据）
        photos = get_photos_without_faces(self.library_path)
        
        if not photos:
            print("没有需要人脸分析的照片")
            self.analysis_completed.emit(0, 0)
            return
        
        total_photos = len(photos)
        total_faces = 0
        processed = 0
        
        print(f"需要分析 {total_photos} 张照片")
        
        # 收集所有人脸数据用于批量插入
        faces_batch = []
        
        for photo in photos:
            if self.should_stop:
                break
            
            photo_id = photo['id']
            file_path = photo['filepath']
            
            try:
                # 检测人脸
                faces = self.ai_analyzer.detect_faces(file_path)
                
                if faces:
                    # 为每个人脸提取特征
                    for face in faces:
                        embedding = self.ai_analyzer.get_face_embedding(
                            file_path, 
                            face['bbox'],
                            face.get('landmarks')
                        )
                        if embedding is not None:
                            faces_batch.append({
                                'photo_id': photo_id,
                                'bbox': face['bbox'],
                                'embedding': embedding,
                                'confidence': face.get('confidence', 0.0)
                            })
                            total_faces += 1
                    
                    self.face_detected.emit(file_path, len(faces))
                    
                    # 根据人脸数量更新分类
                    if len(faces) == 1:
                        update_photo_status(photo_id, 'done', '单人照')
                    elif len(faces) > 1:
                        update_photo_status(photo_id, 'done', '合照')
                
            except Exception as e:
                print(f"人脸分析失败: {file_path}, 错误: {e}")
            
            processed += 1
            self.progress_updated.emit(processed, total_photos)
            
            # 每50张批量插入一次
            if len(faces_batch) >= 50:
                add_faces_batch(faces_batch)
                faces_batch.clear()
        
        # 插入剩余的人脸数据
        if faces_batch:
            add_faces_batch(faces_batch)
        
        if not self.should_stop:
            self.analysis_completed.emit(processed, total_faces)
            print(f"人脸分析完成: {processed} 张照片, {total_faces} 个人脸")
    
    def stop(self):
        """停止分析"""
        print("请求停止人脸分析")
        self.should_stop = True
    def stop_scan(self):
        """
        停止扫描
        """
        print("请求停止扫描")
        self.should_stop = True


class AnalysisWorker(QThread):
    """
    AI分析工作线程
    
    在阶段0中，这是一个占位类
    实际的AI分析功能将在阶段3中实现
    """
    
    # 定义信号
    progress_updated = pyqtSignal(int, int)  # (current, total)
    photo_analyzed = pyqtSignal(str, dict)  # (filepath, analysis_result)
    analysis_completed = pyqtSignal(int)  # total_analyzed
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self, photo_list: List[str]):
        """
        初始化分析工作线程
        
        Args:
            photo_list: 待分析的照片路径列表
        """
        super().__init__()
        
        self.photo_list = photo_list
        self.is_running = False
        self.should_stop = False
        
        print(f"[占位] 分析工作线程初始化，待分析照片: {len(photo_list)} 张")
    
    def run(self):
        """
        线程主执行函数
        """
        self.is_running = True
        self.should_stop = False
        
        try:
            print(f"[占位] 开始AI分析")
            
            # 在实际实现中，这里会:
            # 1. 创建AI分析器实例
            # 2. 逐一分析每张照片
            # 3. 将分析结果存入数据库
            # 4. 发送进度更新信号
            
            # 占位实现 - 模拟分析过程
            self._simulate_analysis()
            
        except Exception as e:
            self.error_occurred.emit(f"分析过程中发生错误: {str(e)}")
        finally:
            self.is_running = False
    
    def _simulate_analysis(self):
        """
        模拟分析过程 (占位函数)
        """
        total_photos = len(self.photo_list)
        analyzed_count = 0
        
        for i, photo_path in enumerate(self.photo_list):
            if self.should_stop:
                break
            
            # 模拟分析时间
            time.sleep(0.5)
            
            # 模拟分析结果
            mock_result = {
                'category': '单人照' if i % 3 == 0 else ('合照' if i % 3 == 1 else '风景'),
                'faces_count': i % 3 if i % 3 != 2 else 0,
                'confidence': 0.85 + (i % 10) * 0.01
            }
            
            # 发送分析结果信号
            self.photo_analyzed.emit(photo_path, mock_result)
            
            analyzed_count += 1
            
            # 发送进度更新信号
            self.progress_updated.emit(analyzed_count, total_photos)
        
        # 发送完成信号
        if not self.should_stop:
            self.analysis_completed.emit(analyzed_count)
    
    def stop_analysis(self):
        """
        停止分析
        """
        print("[占位] 请求停止分析")
        self.should_stop = True


class ClusteringWorker(QThread):
    """
    人脸聚类工作线程
    
    在阶段0中，这是一个占位类
    实际的聚类功能将在阶段5中实现
    """
    
    # 定义信号
    progress_updated = pyqtSignal(int, int)  # (current, total)
    clustering_completed = pyqtSignal(list)  # clusters_result
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self, similarity_threshold: float = 0.6):
        """
        初始化聚类工作线程
        
        Args:
            similarity_threshold: 相似度阈值
        """
        super().__init__()
        
        self.similarity_threshold = similarity_threshold
        self.is_running = False
        self.should_stop = False
        
        print(f"[占位] 聚类工作线程初始化，相似度阈值: {similarity_threshold}")
    
    def run(self):
        """
        线程主执行函数
        """
        self.is_running = True
        self.should_stop = False
        
        try:
            print(f"[占位] 开始人脸聚类")
            
            # 在实际实现中，这里会:
            # 1. 从数据库读取所有未命名的人脸特征向量
            # 2. 使用聚类算法进行分组
            # 3. 发送进度更新信号
            # 4. 返回聚类结果
            
            # 占位实现 - 模拟聚类过程
            self._simulate_clustering()
            
        except Exception as e:
            self.error_occurred.emit(f"聚类过程中发生错误: {str(e)}")
        finally:
            self.is_running = False
    
    def _simulate_clustering(self):
        """
        模拟聚类过程 (占位函数)
        """
        # 模拟聚类进度
        for i in range(10):
            if self.should_stop:
                break
            
            time.sleep(0.1)
            self.progress_updated.emit(i + 1, 10)
        
        # 模拟聚类结果
        if not self.should_stop:
            mock_clusters = [
                {'cluster_id': 0, 'face_ids': [1, 5, 12, 18], 'representative_face_id': 1},
                {'cluster_id': 1, 'face_ids': [3, 8, 15], 'representative_face_id': 3},
                {'cluster_id': 2, 'face_ids': [7, 11, 20, 25, 30], 'representative_face_id': 7},
            ]
            self.clustering_completed.emit(mock_clusters)
    
    def stop_clustering(self):
        """
        停止聚类
        """
        print("[占位] 请求停止聚类")
        self.should_stop = True


# ==================== 语义搜索 ====================

class SemanticSearchWorker(QThread):
    """
    语义搜索工作线程
    
    使用 CLIP 文本编码器将查询转换为向量，
    然后与数据库中的图片向量计算相似度
    """
    
    # 信号
    search_completed = pyqtSignal(list)  # List of (photo_id, filepath, similarity)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, query: str, top_k: int = 50):
        """
        初始化语义搜索
        
        Args:
            query: 搜索查询文本
            top_k: 返回结果数量
        """
        super().__init__()
        self.query = query
        self.top_k = top_k
        self.clip_encoder = None
    
    def run(self):
        """执行语义搜索"""
        try:
            # 初始化 CLIP 编码器
            from models.clip_embedding import CLIPEmbeddingEncoder
            self.clip_encoder = CLIPEmbeddingEncoder()
            
            if not self.clip_encoder.is_available():
                self.error_occurred.emit("CLIP 编码器不可用")
                return
            
            # 编码查询文本
            query_embedding = self.clip_encoder.encode_text(self.query)
            if query_embedding is None:
                self.error_occurred.emit("文本编码失败")
                return
            
            # 搜索相似照片
            results = search_photos_by_embedding(
                query_embedding,
                top_k=self.top_k,
                threshold=0.1  # 最低相似度阈值
            )
            
            self.search_completed.emit(results)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"搜索错误: {str(e)}")


def main():
    """
    主函数 - 用于独立测试后台任务模块
    """
    print("=" * 50)
    print("Photoye 后台任务模块测试 (阶段0)")
    print("=" * 50)
    
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # 测试扫描工作线程
    print("\n测试文件扫描工作线程...")
    scan_worker = ScanWorker("/test/path")
    
    def on_progress(current, total):
        print(f"扫描进度: {current}/{total}")
    
    def on_file_found(filepath):
        print(f"发现文件: {filepath}")
    
    def on_scan_completed(total):
        print(f"扫描完成，共发现 {total} 个文件")
        app.quit()
    
    def on_error(error):
        print(f"发生错误: {error}")
        app.quit()
    
    # 连接信号
    scan_worker.progress_updated.connect(on_progress)
    scan_worker.file_found.connect(on_file_found)
    scan_worker.scan_completed.connect(on_scan_completed)
    scan_worker.error_occurred.connect(on_error)
    
    # 启动线程
    scan_worker.start()
    
    print("\n后台任务模块测试完成！")
    print("注意: 当前为占位实现，实际功能将在后续阶段开发")
    
    # 运行事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main()