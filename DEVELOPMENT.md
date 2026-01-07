# Photoye 开发规划文档

> **当前版本:** 2.0 → 2.1 升级中  
> **最后更新:** 2026年01月07日

---

## 📋 目录

1. [版本历史](#版本历史)
2. [V2.1 升级计划 - AI引擎重构](#v21-升级计划---ai引擎重构)
3. [技术方案详解](#技术方案详解)
4. [开发任务清单](#开发任务清单)
5. [已完成阶段回顾](#已完成阶段回顾)
6. [未来路线图](#未来路线图)

---

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| V1.0 | 2025-08 | 基础功能：扫描、分类、缩略图 |
| V2.0 | 2026-01-06 | 重构：异步缩略图、即时筛选、批量DB操作 |
| V2.1 | 2026-01-07 | **当前** - AI升级：CLIP Embedding + InsightFace |

---

## V2.1 升级计划 - AI引擎重构

### 🎯 升级目标

1. **场景分类准确率提升：** 从 MobileNetV2 分类器 → CLIP Embedding 语义搜索
2. **人脸识别准确率提升：** 从 YuNet+SFace → RetinaFace+ArcFace (InsightFace)
3. **聚类算法优化：** 从 Union-Find → DBSCAN (处理噪声)

### 📊 预期效果

| 指标 | V2.0 (当前) | V2.1 (目标) |
|------|------------|-------------|
| 场景分类准确率 | ~70% | ~90% |
| 人脸检测召回率 | ~85% | ~95% |
| 人脸识别准确率 | ~80% | ~95% |
| 聚类纯度 | ~75% | ~90% |

---

## 技术方案详解

### 方案一：场景理解升级 (Classifier → Embedding)

#### 问题分析
MobileNetV2 基于 ImageNet 1000 类训练，其局限性：
- 没有"人物"、"合照"等类别
- 依赖关键词映射，容易误判
- 无法处理自定义查询

#### 解决方案：CLIP Embedding

```
旧方案 (MobileNetV2):
图片 → MobileNetV2 → "seashore (0.9)" → 映射 → "风景"
                                          ↑
                                      关键词匹配容易失败

新方案 (CLIP Embedding):
图片 → CLIP Vision Encoder → 512维向量 → 存入 photos.embedding

查询时:
"海边的合照" → CLIP Text Encoder → 512维向量
            ↓
    与所有图片向量计算余弦相似度
            ↓
    返回最相似的照片
```

#### 模型选型

| 模型 | 尺寸 | 速度 | 精度 | 推荐 |
|------|------|------|------|------|
| CLIP ViT-B/32 | 350MB | 慢 | 高 | ❌ 太重 |
| OpenCLIP ViT-B/32 | 350MB | 慢 | 高 | ⚠️ 备选 |
| **MobileCLIP-S0** | 50MB | 快 | 中高 | ✅ 推荐 |
| MobileCLIP-S1 | 80MB | 中 | 高 | ⚠️ 备选 |

#### 数据流变更

```
ScanWorker (新版):
1. 批量添加照片到数据库
2. MobileCLIP 提取 512维向量
3. 存入 photos.embedding (BLOB)
4. category 字段改为缓存最近搜索结果

SemanticSearchWorker (新增):
1. 用户输入查询文本
2. CLIP Text Encoder 编码为向量
3. 与 photos.embedding 计算余弦相似度
4. 返回 Top-K 结果
```

---

### 方案二：人脸识别升级 (InsightFace 管线)

#### 问题分析
当前 YuNet + SFace 组合的问题：
- YuNet 对小脸、侧脸检测较弱
- SFace 128维特征区分度不足
- 无人脸对齐导致识别偏差

#### 解决方案：RetinaFace + ArcFace

```
旧方案:
图片 → YuNet检测 → 简单裁剪 → SFace → 128维向量

新方案:
图片 → RetinaFace检测 → 5点landmarks
                           ↓
              仿射变换对齐 (Affine Alignment)
                           ↓
              ArcFace → 512维向量
```

#### 人脸对齐算法

```python
# 标准人脸模板 (112x112)
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],  # 左眼
    [73.5318, 51.5014],  # 右眼
    [56.0252, 71.7366],  # 鼻尖
    [41.5493, 92.3655],  # 左嘴角
    [70.7299, 92.2041],  # 右嘴角
], dtype=np.float32)

def align_face(image, landmarks_5):
    """
    使用仿射变换将人脸对齐到标准模板
    """
    M = cv2.estimateAffinePartial2D(landmarks_5, ARCFACE_TEMPLATE)[0]
    aligned = cv2.warpAffine(image, M, (112, 112))
    return aligned
```

#### 模型选型

| 模型 | 检测/识别 | 尺寸 | 特点 |
|------|----------|------|------|
| RetinaFace MobileNet0.25 | 检测 | 1.6MB | 快速，适合实时 |
| RetinaFace ResNet50 | 检测 | 100MB | 更准，较慢 |
| **ArcFace-R50** | 识别 | 166MB | 512维，高精度 |
| ArcFace-R100 | 识别 | 250MB | 更高精度 |

---

### 方案三：聚类算法升级 (Union-Find → DBSCAN)

#### 问题分析
Union-Find 的问题：
- 二元判断：要么同一人，要么不是
- 无法处理"路人甲"、模糊人脸
- 可能把路人强行归类给家人

#### 解决方案：DBSCAN

```python
from sklearn.cluster import DBSCAN

def cluster_faces_dbscan(embeddings, eps=0.5, min_samples=3):
    """
    DBSCAN 聚类
    - eps: 邻域半径 (越小越严格)
    - min_samples: 核心点最小邻居数
    - 返回: labels, -1 表示噪声点 (离群的脸)
    """
    # 使用余弦距离
    from sklearn.metrics.pairwise import cosine_distances
    distance_matrix = cosine_distances(embeddings)
    
    clustering = DBSCAN(
        eps=eps, 
        min_samples=min_samples,
        metric='precomputed'
    )
    labels = clustering.fit_predict(distance_matrix)
    return labels
```

#### DBSCAN 优势

| 特性 | Union-Find | DBSCAN |
|------|-----------|--------|
| 噪声处理 | ❌ 无 | ✅ 标记为 -1 |
| 参数敏感度 | 单一阈值 | eps + min_samples |
| 聚类形状 | 球形 | 任意形状 |
| 路人处理 | 强行归类 | 自动排除 |

---

## 开发任务清单

### 🔴 Sprint 1: 场景理解升级 (CLIP Embedding)

- [x] S1-1: 下载 MobileCLIP ONNX 模型 *(使用现有 OpenCLIP)*
- [x] S1-2: 创建 `models/clip_embedding.py` 适配器
- [x] S1-3: 数据库迁移: `photos` 表添加 `embedding` 字段
- [x] S1-4: 修改 `ScanWorker`: 提取 CLIP 向量
- [x] S1-5: 创建 `SemanticSearchWorker` 语义搜索
- [x] S1-6: UI: 添加语义搜索输入框
- [x] S1-7: 单元测试: CLIP 编码器

---

### 🟡 Sprint 2: 人脸识别升级 (InsightFace)

- [ ] S2-1: 下载 RetinaFace ONNX 模型
- [ ] S2-2: 下载 ArcFace ONNX 模型
- [ ] S2-3: 创建 `models/retinaface_detector.py`
- [ ] S2-4: 创建 `models/arcface_recognizer.py`
- [ ] S2-5: 实现人脸对齐 `align_face()` 函数
- [ ] S2-6: 数据库迁移: `faces` 表添加 `landmarks`, `is_noise` 字段
- [ ] S2-7: 修改 `FaceAnalysisWorker`: 使用新模型
- [ ] S2-8: 更新 `model_profiles.py` 配置
- [ ] S2-9: 单元测试: InsightFace 管线

---

### 🟢 Sprint 3: 聚类算法升级 (DBSCAN)

- [ ] S3-1: 添加 `scikit-learn` 依赖
- [ ] S3-2: 实现 `cluster_faces_dbscan()` 函数
- [ ] S3-3: 修改 `ClusteringWorker`: 使用 DBSCAN
- [ ] S3-4: UI: 噪声人脸标记与手动归类
- [ ] S3-5: 单元测试: DBSCAN 聚类

---

### 🔵 Sprint 4: 集成与优化

- [ ] S4-1: 数据迁移脚本: 旧版 → V2.1
- [ ] S4-2: 性能测试: 2500张照片基准
- [ ] S4-3: 引入 FAISS 向量索引 (可选)
- [ ] S4-4: 端到端测试

---

## 已完成阶段回顾

### ✅ V1.0 阶段 (2025-08)

- [x] 项目初始化与环境搭建
- [x] 数据库设计与实现
- [x] ScanWorker 文件扫描
- [x] MobileNetV2 场景分类
- [x] YuNet 人脸检测
- [x] SFace/Dlib 人脸识别
- [x] 缩略图视图
- [x] 基础筛选功能

### ✅ V2.0 阶段 (2026-01-06)

- [x] ThumbnailWorker 异步缩略图
- [x] 批量数据库操作 (add_photos_batch, add_faces_batch)
- [x] ScanWorker 重构 (自动分类)
- [x] 即时筛选 (移除"应用筛选"按钮)
- [x] FaceAnalysisWorker 独立人脸分析
- [x] SFace 对齐修复 (alignCrop)
- [x] Union-Find 聚类算法

---

## 未来路线图

| 版本 | 目标 | 主要功能 |
|------|------|---------|
| V2.1 | AI升级 | CLIP Embedding + InsightFace + DBSCAN |
| V2.2 | 人物管理 | 人脸命名UI、合并/拆分人物 |
| V2.3 | 高级搜索 | 自然语言搜索、组合条件 |
| V3.0 | 多媒体 | 视频关键帧提取与分析 |
| V3.1 | 地理信息 | GPS坐标解析、地图展示 |

---

## 开发指南

### 模型下载

```bash
# 下载所有模型
python models/download_models.py

# 仅下载 InsightFace 模型
python models/download_models.py --insightface

# 仅下载 MobileCLIP 模型
python models/download_models.py --mobileclip
```

### 数据库迁移

```bash
# 从 V2.0 迁移到 V2.1
python scripts/migrate_v21.py

# 检查迁移状态
python scripts/migrate_v21.py --check
```

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 仅运行 InsightFace 测试
python -m pytest tests/test_insightface.py

# 仅运行 CLIP 测试
python -m pytest tests/test_mobileclip.py
```

---

## 附录：关键代码位置

| 功能 | 文件 | 函数/类 |
|------|------|--------|
| CLIP 编码器 | `models/mobileclip_encoder.py` | `MobileCLIPEncoder` |
| RetinaFace 检测 | `models/retinaface_detector.py` | `RetinaFaceDetector` |
| ArcFace 识别 | `models/arcface_recognizer.py` | `ArcFaceRecognizer` |
| 人脸对齐 | `models/face_align.py` | `align_face()` |
| DBSCAN 聚类 | `analyzer.py` | `cluster_faces_dbscan()` |
| 语义搜索 | `worker.py` | `SemanticSearchWorker` |
| 数据库迁移 | `database.py` | `migrate_to_v21()` |
