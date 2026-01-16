#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoye - 后台任务管理模块
负责处理所有耗时操作，避免UI线程阻塞

版本: 2.2 (自动化流水线)
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
    get_all_face_embeddings_for_clustering,
    update_face_cluster_assignments,
    create_person_for_cluster,
    clear_all_ai_data,
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
    智能扫描工作线程 (V2.2 自动化流水线)
    
    完整流程: 扫描 → CLIP分类 → 人脸检测 → 交叉验证 → 自动聚类
    用户只需选择文件夹，后台自动完成所有AI处理
    """
    
    # 定义信号
    progress_updated = pyqtSignal(int, int)  # (current, total)
    stage_changed = pyqtSignal(str)  # 当前阶段描述
    file_found = pyqtSignal(str)  # filepath
    scan_completed = pyqtSignal(int)  # total_files
    pipeline_completed = pyqtSignal(dict)  # 完整流水线结果
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self, root_path: str, supported_extensions: List[str] = None, model_profile: Optional[str] = None):
        """
        初始化扫描工作线程
        
        Args:
            root_path: 要扫描的根目录路径
            supported_extensions: 支持的文件扩展名列表
            model_profile: 模型档位（用于AI分析）
        """
        super().__init__()
        
        self.root_path = root_path
        self.supported_extensions = supported_extensions or [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'
        ]
        self.is_running = False
        self.should_stop = False
        self.model_profile = model_profile
        
        # 统计数据
        self.stats = {
            'total_files': 0,
            'new_files': 0,
            'faces_detected': 0,
            'categories_corrected': 0,
            'clusters_created': 0,
            'noise_faces': 0
        }

        # 初始化AI组件
        self.clip_encoder = None
        self.scene_classifier = None
        self.ai_analyzer = None
        self._init_ai_components()
        
        print(f"扫描工作线程初始化 (V2.2 自动化流水线)")
        print(f"根目录: {root_path}")
        print(f"支持格式: {self.supported_extensions}")
    
    def _init_ai_components(self):
        """初始化所有AI组件"""
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
        
        # 初始化 OpenCLIP 零样本分类器 (替代 MobileNetV2，更准确)
        try:
            from models.openclip_zero_shot import OpenCLIPZeroShotClassifier
            self.scene_classifier = OpenCLIPZeroShotClassifier()
            print("✅ OpenCLIP 零样本分类器初始化成功")
        except Exception as e:
            print(f"⚠️ OpenCLIP 零样本分类器初始化失败: {e}")
            # 回退到 MobileNetV2
            try:
                from models.mobilenetv2_classifier import MobileNetV2SceneClassifier
                self.scene_classifier = MobileNetV2SceneClassifier()
                print("⚠️ 回退到 MobileNetV2 分类器")
            except Exception as e2:
                print(f"⚠️ 场景分类器初始化失败: {e2}")
                self.scene_classifier = None
        
        # 初始化AI分析器 (用于人脸检测和识别)
        try:
            self.ai_analyzer = AIAnalyzer(model_profile=self.model_profile)
            print("✅ AI分析器初始化成功 (人脸检测+识别)")
        except Exception as e:
            print(f"⚠️ AI分析器初始化失败: {e}")
            self.ai_analyzer = None
    
    def run(self):
        """线程主执行函数 - 完整的自动化流水线"""
        self.is_running = True
        self.should_stop = False
        self.stats = {k: 0 for k in self.stats}
        
        try:
            print(f"开始自动化流水线: {self.root_path}")
            
            if not os.path.exists(self.root_path):
                self.error_occurred.emit(f"目录不存在: {self.root_path}")
                return
            
            # Stage 0: 清空旧的 AI 分析数据
            self.stage_changed.emit("🗑️ 清空旧数据...")
            clear_all_ai_data()
            
            # Stage 1: 扫描文件
            self.stage_changed.emit("📂 扫描文件...")
            image_files = self._scan_files()
            if self.should_stop:
                return
            
            # Stage 2: CLIP 分类与 Embedding
            self.stage_changed.emit("🏷️ 场景分类中...")
            self._classify_and_embed(image_files)
            if self.should_stop:
                return
            
            # Stage 3: 人脸检测与特征提取
            self.stage_changed.emit("👤 检测人脸...")
            self._detect_faces(image_files)
            if self.should_stop:
                return
            
            # Stage 4: 自动聚类
            self.stage_changed.emit("🔗 人脸聚类...")
            self._auto_clustering()
            if self.should_stop:
                return
            
            # 完成
            self.stage_changed.emit("✅ 处理完成")
            self.pipeline_completed.emit(self.stats)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"流水线错误: {str(e)}")
        finally:
            self.is_running = False
    
    def _scan_files(self) -> List[str]:
        """Stage 1: 扫描目录收集图片文件"""
        image_files = []
        for root, dirs, files in os.walk(self.root_path):
            if self.should_stop:
                break
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.supported_extensions):
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
        
        self.stats['total_files'] = len(image_files)
        print(f"发现 {len(image_files)} 个图片文件")
        
        # 批量添加新文件到数据库
        new_files = [f for f in image_files if not is_photo_exist(f)]
        if new_files:
            add_photos_batch(new_files)
            self.stats['new_files'] = len(new_files)
            print(f"批量添加 {len(new_files)} 张新照片")
        
        self.scan_completed.emit(len(image_files))
        return image_files
    
    def _classify_and_embed(self, image_files: List[str]):
        """Stage 2: CLIP 分类与 Embedding 提取 (V2.3: Multi-Crop)"""
        total = len(image_files)
        
        for i, file_path in enumerate(image_files):
            if self.should_stop:
                break
            
            status_row = get_photo_status(file_path)
            if not status_row:
                continue
                
            photo_id, status, category = status_row
            
            # 提取 CLIP embedding (V2.3: 使用 Multi-Crop)
            if self.clip_encoder:
                try:
                    # 优先使用 multi-crop，回退到单一裁剪
                    if hasattr(self.clip_encoder, 'encode_image_multicrop'):
                        embedding = self.clip_encoder.encode_image_multicrop(file_path, n_crops=5)
                    else:
                        embedding = self.clip_encoder.encode_image(file_path)
                    
                    if embedding is not None:
                        update_photo_embedding_by_path(file_path, embedding)
                except Exception as e:
                    print(f"CLIP embedding 提取失败: {file_path}, 错误: {e}")
            
            # 场景分类（如果尚未分类）
            if not category and self.scene_classifier:
                try:
                    classification = self.scene_classifier.classify(file_path)
                    if classification:
                        best_category = max(classification.items(), key=lambda x: x[1])[0]
                        update_photo_status(photo_id, 'done', best_category)
                except Exception as e:
                    print(f"分类失败: {file_path}, 错误: {e}")
            
            self.file_found.emit(file_path)
            self.progress_updated.emit(i + 1, total)
    
    def _detect_faces(self, image_files: List[str]):
        """Stage 3: 人脸检测与特征提取 + 交叉验证"""
        if not self.ai_analyzer:
            print("⚠️ AI分析器不可用，跳过人脸检测")
            return
        
        import json
        
        # 获取所有照片进行人脸检测（不再限制分类）
        total = len(image_files)
        faces_batch = []
        
        for i, file_path in enumerate(image_files):
            if self.should_stop:
                break
            
            status_row = get_photo_status(file_path)
            if not status_row:
                continue
                
            photo_id, status, category = status_row
            
            try:
                # 检测人脸
                faces = self.ai_analyzer.detect_faces(file_path)
                
                if faces:
                    # 交叉验证：如果检测到人脸，但分类不是人物相关，则修正分类
                    non_person_categories = ['风景', '建筑', '美食', '动物', '文档', '室内']
                    if category in non_person_categories:
                        # 修正分类
                        new_category = '合照' if len(faces) > 1 else '单人照'
                        update_photo_status(photo_id, 'done', new_category)
                        self.stats['categories_corrected'] += 1
                        print(f"📝 交叉验证修正: {os.path.basename(file_path)} [{category}] → [{new_category}]")
                    
                    # 为每个人脸提取特征
                    for face in faces:
                        embedding = self.ai_analyzer.get_face_embedding(
                            file_path, 
                            face['bbox'],
                            face.get('landmarks')
                        )
                        if embedding is not None:
                            landmarks_json = None
                            if face.get('landmarks'):
                                landmarks_json = json.dumps(face['landmarks'])
                            
                            faces_batch.append({
                                'photo_id': photo_id,
                                'bbox': face['bbox'],
                                'embedding': embedding,
                                'confidence': face.get('confidence', 0.0),
                                'landmarks': landmarks_json
                            })
                            self.stats['faces_detected'] += 1
                    
                    # 更新分类（如果尚未分类）
                    if not category:
                        new_cat = '单人照' if len(faces) == 1 else '合照'
                        update_photo_status(photo_id, 'done', new_cat)
                
            except Exception as e:
                print(f"人脸检测失败: {file_path}, 错误: {e}")
            
            self.progress_updated.emit(i + 1, total)
            
            # 每50张批量插入一次
            if len(faces_batch) >= 50:
                add_faces_batch(faces_batch)
                faces_batch.clear()
        
        # 插入剩余的人脸数据
        if faces_batch:
            add_faces_batch(faces_batch)
        
        print(f"人脸检测完成: {self.stats['faces_detected']} 个人脸, {self.stats['categories_corrected']} 个分类修正")
    
    def _auto_clustering(self):
        """Stage 4: 自动聚类"""
        from clustering import cluster_faces_dbscan
        from database import create_person_for_single_face
        
        # 获取所有未分配的人脸 embedding
        face_embeddings = get_all_face_embeddings_for_clustering()
        
        if not face_embeddings:
            print("没有需要聚类的人脸")
            return
        
        print(f"开始聚类 {len(face_embeddings)} 个人脸...")
        
        # 执行 DBSCAN 聚类 (min_samples=2: 至少2张才能形成簇)
        result = cluster_faces_dbscan(
            face_embeddings,
            eps=0.6,  # 调整：更严格的阈值，避免把不同人聚在一起
            min_samples=2
        )
        
        # 为每个聚类创建人物并分配人脸
        assignments = {}  # face_id -> person_id
        
        for cluster_id, face_ids in result['clusters'].items():
            person_id = create_person_for_cluster(cluster_id)
            if person_id > 0:
                for face_id in face_ids:
                    assignments[face_id] = person_id
        
        # 为噪声点（只出现一次的人脸）创建独立人物
        # 这样确保每个检测到的人脸都有对应的人物记录
        noise_persons_created = 0
        for face_id in result['noise_ids']:
            person_id = create_person_for_single_face(face_id)
            if person_id > 0:
                assignments[face_id] = person_id
                noise_persons_created += 1
        
        # 更新数据库
        update_face_cluster_assignments(assignments, [])  # 不再有真正的噪声
        
        total_persons = result['n_clusters'] + noise_persons_created
        self.stats['clusters_created'] = total_persons
        self.stats['noise_faces'] = 0  # 噪声已转为独立人物
        
        print(f"聚类完成: {result['n_clusters']} 个人物, {result['n_noise']} 个噪声")
    
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
                            # 序列化 landmarks 为 JSON 字符串
                            import json
                            landmarks_json = None
                            if face.get('landmarks'):
                                landmarks_json = json.dumps(face['landmarks'])
                            
                            faces_batch.append({
                                'photo_id': photo_id,
                                'bbox': face['bbox'],
                                'embedding': embedding,
                                'confidence': face.get('confidence', 0.0),
                                'landmarks': landmarks_json
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
    
    使用 DBSCAN 算法对人脸特征进行聚类，
    能更好地处理噪声点（离群人脸）
    """
    
    # 定义信号
    progress_updated = pyqtSignal(int, int)  # (current, total)
    clustering_completed = pyqtSignal(dict)  # clustering result
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self, eps: float = 0.7, min_samples: int = 2):
        """
        初始化聚类工作线程
        
        Args:
            eps: DBSCAN 邻域半径（余弦距离），推荐 0.5-0.8
            min_samples: 形成簇的最小样本数
        """
        super().__init__()
        
        self.eps = eps
        self.min_samples = min_samples
        self.is_running = False
        self.should_stop = False
        
        print(f"聚类工作线程初始化: eps={eps}, min_samples={min_samples}")
    
    def run(self):
        """线程主执行函数"""
        self.is_running = True
        self.should_stop = False
        
        try:
            print("开始 DBSCAN 人脸聚类...")
            self._perform_clustering()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"聚类过程中发生错误: {str(e)}")
        finally:
            self.is_running = False
    
    def _perform_clustering(self):
        """执行 DBSCAN 聚类"""
        from database import (
            get_all_face_embeddings_for_clustering,
            update_face_cluster_assignments,
            create_person_for_cluster
        )
        from clustering import cluster_faces_dbscan
        
        # 获取所有未分配的人脸 embedding
        self.progress_updated.emit(0, 100)
        face_embeddings = get_all_face_embeddings_for_clustering()
        
        if not face_embeddings:
            print("没有需要聚类的人脸")
            self.clustering_completed.emit({
                'n_clusters': 0,
                'n_noise': 0,
                'n_faces': 0
            })
            return
        
        print(f"获取到 {len(face_embeddings)} 个待聚类人脸")
        self.progress_updated.emit(20, 100)
        
        if self.should_stop:
            return
        
        # 执行 DBSCAN 聚类
        result = cluster_faces_dbscan(
            face_embeddings,
            eps=self.eps,
            min_samples=self.min_samples
        )
        
        self.progress_updated.emit(60, 100)
        
        if self.should_stop:
            return
        
        # 为每个聚类创建人物并分配人脸
        assignments = {}  # face_id -> person_id
        
        for cluster_id, face_ids in result['clusters'].items():
            # 创建新人物
            person_id = create_person_for_cluster(cluster_id)
            if person_id > 0:
                for face_id in face_ids:
                    assignments[face_id] = person_id
        
        self.progress_updated.emit(80, 100)
        
        if self.should_stop:
            return
        
        # 更新数据库
        update_face_cluster_assignments(assignments, result['noise_ids'])
        
        self.progress_updated.emit(100, 100)
        
        # 发送完成信号
        final_result = {
            'n_clusters': result['n_clusters'],
            'n_noise': result['n_noise'],
            'n_faces': len(face_embeddings),
            'clusters': result['clusters']
        }
        
        print(f"聚类完成: {result['n_clusters']} 个人物, {result['n_noise']} 个噪声")
        self.clustering_completed.emit(final_result)
    
    def stop_clustering(self):
        """停止聚类"""
        print("请求停止聚类")
        self.should_stop = True


# ==================== 语义搜索 ====================

class SemanticSearchWorker(QThread):
    """
    语义搜索工作线程 (V2.3: Prompt Ensemble)
    
    使用 CLIP 文本编码器将查询转换为向量，
    然后与数据库中的图片向量计算相似度
    """
    
    # 信号
    search_completed = pyqtSignal(list)  # List of (photo_id, filepath, similarity)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, query: str, top_k: int = 20, threshold: float = 0.25, use_ensemble: bool = True):
        """
        初始化语义搜索
        
        Args:
            query: 搜索查询文本
            top_k: 返回结果数量
            threshold: 最低相似度阈值（V2.3 降低阈值以适配 ensemble）
            use_ensemble: 是否使用 Prompt Ensemble (V2.3)
        """
        super().__init__()
        self.query = query
        self.top_k = top_k
        self.threshold = threshold
        self.use_ensemble = use_ensemble
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
            
            # 编码查询文本 (V2.3: 使用 Prompt Ensemble)
            if self.use_ensemble and hasattr(self.clip_encoder, 'encode_text_ensemble'):
                query_embedding = self.clip_encoder.encode_text_ensemble(self.query)
            else:
                query_embedding = self.clip_encoder.encode_text(self.query)
                
            if query_embedding is None:
                self.error_occurred.emit("文本编码失败")
                return
            
            # 搜索相似照片
            results = search_photos_by_embedding(
                query_embedding,
                top_k=self.top_k,
                threshold=self.threshold
            )
            
            # 过滤低相似度结果
            filtered_results = [
                r for r in results if r[2] >= self.threshold
            ]
            
            self.search_completed.emit(filtered_results)
            
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