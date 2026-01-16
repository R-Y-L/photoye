# Photoye 开发规划文档

> **当前版本:** 2.3 (Open-Vocabulary)  
> **最后更新:** 2026年01月16日

---

## 📋 目录

1. [V2.3 升级计划 - Open-Vocabulary 语义检索](#v23-升级计划---open-vocabulary-语义检索)
2. [V2.2 升级计划 - 自动化流水线](#v22-升级计划---自动化流水线)
3. [版本历史](#版本历史)
4. [技术方案详解](#技术方案详解)
5. [已完成阶段回顾](#已完成阶段回顾)

---

## V2.3 升级计划 - Open-Vocabulary 语义检索 ✅ 已完成

### 🎯 升级目标

放弃固定类别分类，走开放词汇语义检索：

1. **Prompt Ensemble (文本端):** 多模板平均，提升文本 embedding 质量
2. **Multi-Crop (图像端):** 多裁剪融合，捕获更完整的图像信息
3. **Open-Vocabulary 搜索:** 用户输入任意自然语言查询

### 📊 升级对比

| 方向 | V2.2 (之前) | V2.3 (当前) |
|------|-------------|---------------|
| 文本 Embedding | 单一提示 | **Prompt Ensemble** (7 模板平均) ✅ |
| 图像 Embedding | 单一 center crop | **Multi-crop** (5 裁剪融合) ✅ |
| 检索方式 | 固定 8 类别分类 | **Open-vocabulary** 语义搜索 ✅ |

### 📋 任务清单

#### Phase 1: 核心算法 (clip_embedding.py) ✅

| ID | 任务 | 状态 |
|----|------|------|
| P1-1 | 实现 PROMPT_TEMPLATES 和 encode_text_ensemble() | ✅ 完成 |
| P1-2 | 实现 _generate_crops() 和 encode_image_multicrop() | ✅ 完成 |
| P1-3 | 添加 embedding_version 参数支持回退 | ✅ 完成 |

#### Phase 2: 工作流整合 (worker.py) ✅

| ID | 任务 | 状态 |
|----|------|------|
| P2-1 | ScanWorker 使用 multi-crop embedding | ✅ 完成 |
| P2-2 | SemanticSearchWorker 使用 Prompt Ensemble | ✅ 完成 |
| P2-3 | 保留交叉验证 (检测到人脸 → 标记含人物) | ✅ 保留 |

#### Phase 3: UI (main.py) ✅

| ID | 任务 | 状态 |
|----|------|------|
| P3-1 | 语义搜索输入框 | ✅ 已有 |
| P3-2 | 搜索结果显示相似度 | ✅ 已有 |
| P3-3 | 分类筛选作为辅助 | ✅ 保留 |

### 🔧 技术方案

**Prompt Ensemble 模板:**
```python
PROMPT_TEMPLATES = [
    "a photo of {}",
    "a photograph of {}",
    "an image of {}",
    "{} in a photo",
    "a picture of {}",
    "a good photo of {}",
    "a photo showing {}",
]
```

**Multi-Crop 策略:**
```python
# 5-crop: 中心 + 4角
CROPS = ["center", "top_left", "top_right", "bottom_left", "bottom_right"]
# 中心权重更高
WEIGHTS = [2.0, 1.0, 1.0, 1.0, 1.0]
```

---

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| V1.0 | 2025-08 | 基础功能：扫描、分类、缩略图 |
| V2.0 | 2026-01-06 | 重构：异步缩略图、即时筛选、批量DB操作 |
| V2.1 | 2026-01-07 | AI升级：CLIP Embedding + InsightFace + DBSCAN |
| V2.2 | 2026-01-16 | 自动化流水线 + OpenCLIP 零样本分类 |
| V2.3 | 2026-01-16 | **Open-Vocabulary** 语义检索 (Prompt Ensemble + Multi-Crop) |

---

## V2.2 升级计划 - 自动化流水线 ✅ 已完成

### 🎯 升级目标

1. **全自动 AI 处理：** 用户只需选择文件夹，后台自动完成扫描→分类→人脸检测→聚类
2. **交叉验证分类：** CLIP 分类与人脸检测结果互相验证，提高准确度
3. **人物命名功能：** 聚类完成后，用户可以为人物分组命名
4. **高级复合筛选：** 支持按人物+场景的组合条件筛选照片

### 📊 核心流程图

```
用户选择文件夹
       ↓
┌─────────────────────────────────────────────────────────┐
│                   ScanWorker (后台自动)                  │
├─────────────────────────────────────────────────────────┤
│  Step 1: 扫描文件                                        │
│  ├── 遍历目录收集图片文件                                │
│  └── 批量写入 photos 表 (status='pending')              │
│                                                         │
│  Step 2: 场景分类 (OpenCLIP)                            │
│  ├── 提取 512维 embedding → photos.embedding            │
│  └── 初步分类 → photos.category                         │
│                                                         │
│  Step 3: 人脸检测 (InsightFace)                         │
│  ├── 检测人脸 + 5点 landmarks                           │
│  └── 提取人脸 embedding → faces 表                      │
│                                                         │
│  Step 4: 交叉验证 ⚡                                     │
│  └── CLIP分类="风景" 但检测到人脸 → 修正为"合照"        │
│                                                         │
│  Step 5: 自动聚类 (DBSCAN)                              │
│  ├── 对所有未归属人脸聚类                                │
│  ├── 创建 persons 记录                                   │
│  └── 更新 faces.person_id                               │
└─────────────────────────────────────────────────────────┘
       ↓
UI 自动刷新，显示分类结果和人物分组
       ↓
用户为人物命名 (双击编辑)
       ↓
高级筛选：按人物+场景组合查询
```

### 📋 任务清单

#### Sprint 5: 自动化流水线

| ID | 任务 | 状态 |
|----|------|------|
| S5-1 | 重构 ScanWorker：整合扫描+分类+人脸检测 | ✅ 完成 |
| S5-2 | 实现交叉验证：CLIP分类与人脸检测互相校正 | ✅ 完成 |
| S5-3 | 自动触发聚类：人脸分析完成后自动运行 | ✅ 完成 |
| S5-4 | 简化 UI：移除手动人脸分析/聚类按钮 | ✅ 完成 |
| S5-5 | 强化状态反馈：实时显示后台任务进度 | ✅ 完成 |
| S5-6 | 升级分类器：MobileNetV2 → OpenCLIP 零样本 | ✅ 完成 |

**分类准确率提升 (2026-01-16):**
| 图片 | MobileNetV2 | OpenCLIP 零样本 |
|------|-------------|-----------------|
| food.jpg | ~0.25 | **0.996** (美食) |
| group_photo.jpg | ~0.23 | **0.968** (合照) |
| single_face.jpg | ~0.25 | **0.803** (单人照) |

#### Sprint 6: 人物管理

| ID | 任务 | 状态 |
|----|------|------|
| S6-1 | 人物命名：双击编辑人物名称 | ⏳ 待开始 |
| S6-2 | 数据库：添加 update_person_name() 函数 | ⏳ 待开始 |
| S6-3 | 自动刷新：命名后刷新筛选下拉列表 | ⏳ 待开始 |
| S6-4 | 人物合并/拆分功能 | ⏳ 未来 |

#### Sprint 7: 高级筛选

| ID | 任务 | 状态 |
|----|------|------|
| S7-1 | 复合筛选 UI：多条件组合面板 | ⏳ 未来 |
| S7-2 | 数据库：支持 AND/OR 组合查询 | ⏳ 未来 |
| S7-3 | 保存筛选条件为智能相册 | ⏳ 未来 |

---

## V2.1 已完成 - AI引擎重构

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

- [x] S2-1: 下载 RetinaFace ONNX 模型 *(buffalo_sc: det_500m.onnx)*
- [x] S2-2: 下载 ArcFace ONNX 模型 *(w600k_mbf.onnx, 512维)*
- [x] S2-3: 创建 `models/insightface_detector.py`
- [x] S2-4: 创建 `models/insightface_recognizer.py`
- [x] S2-5: 实现人脸对齐 `align_face()` 函数 *(5点landmarks → 112x112)*
- [x] S2-6: 数据库迁移: `faces` 表添加 `landmarks`, `is_noise` 字段
- [x] S2-7: 修改 `FaceAnalysisWorker`: 使用新模型、保存landmarks
- [x] S2-8: 更新 `model_profiles.py`: 添加 `insightface` 和 `best` 配置
- [x] S2-9: 单元测试: InsightFace 管线 *(11/11 passed)*

---

### 🟢 Sprint 3: 聚类算法升级 (DBSCAN)

- [x] S3-1: 添加 `scikit-learn` 依赖 *(已安装)*
- [x] S3-2: 实现 `cluster_faces_dbscan()` 函数 *(clustering.py)*
- [x] S3-3: 修改 `ClusteringWorker`: 使用 DBSCAN
- [x] S3-4: 数据库: 噪声人脸标记 `is_noise` 及相关函数
- [x] S3-5: 单元测试: DBSCAN 聚类 *(9/9 passed)*

---

### 🔵 Sprint 4: 集成与优化

- [x] S4-1: 数据迁移脚本: `migrate_to_v21()` 函数
- [x] S4-2: 端到端测试: `tests/test_e2e.py`
- [ ] S4-3: 引入 FAISS 向量索引 *(可选，大规模库时启用)*
- [ ] S4-4: 性能测试: 2500张照片基准

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

### ✅ V2.1 阶段 (2026-01-07 ~ 2026-01-16)

**Sprint 1-3: AI引擎升级**
- [x] CLIP Embedding: OpenCLIP ViT-B/32 语义搜索
- [x] InsightFace: buffalo_sc (det_500m + w600k_mbf)
- [x] DBSCAN 聚类: eps=0.7, min_samples=2
- [x] 数据库迁移: photos.embedding, faces.landmarks, faces.is_noise
- [x] 单元测试: 11/11 InsightFace, 9/9 DBSCAN

---

## 未来路线图

| 版本 | 目标 | 主要功能 | 状态 |
|------|------|---------|------|
| V2.1 | AI升级 | CLIP Embedding + InsightFace + DBSCAN | ✅ 完成 |
| V2.2 | 自动化 | 全自动流水线 + 交叉验证 + 人物命名 | 🔄 进行中 |
| V2.3 | 高级搜索 | 自然语言搜索、复合条件筛选 | ⏳ 计划中 |
| V3.0 | 多媒体 | 视频关键帧提取与分析 | 💡 构想 |
| V3.1 | 地理信息 | GPS坐标解析、地图展示 | 💡 构想 |

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
