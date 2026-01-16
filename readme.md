# Photoye - 本地智能照片管理助手

> **版本:** 2.3 (Open-Vocabulary 语义检索)  
> **更新日期:** 2026年01月16日

---

## 1. 项目概述

### 1.1 项目愿景

打造一款以隐私保护为核心、运行于本地的、智能化的照片与视频管理工具。让每一位用户都能轻松驾驭海量的数字记忆，高效地整理、发现和重温生活中的美好瞬间。

### 1.2 核心问题

本项目旨在解决现代用户面临的数字资产管理困境：

- **存储混乱:** 照片和视频散落在不同文件夹，缺乏统一管理
- **检索困难:** 无法快速找到特定人物、事件或场景的照片
- **利用率低:** 大量珍贵照片在存储后被遗忘，无法发挥其情感价值
- **隐私担忧:** 不愿将包含家庭隐私的照片上传至公共云服务

### 1.3 核心原则

- **本地优先 (Local-First):** 所有数据处理和存储均在用户本地设备完成
- **隐私至上 (Privacy-First):** 用户的照片和分析出的元数据永远属于用户自己
- **非破坏性操作 (Non-Destructive):** 绝不修改、移动或删除用户的原始文件
- **智能自动化 (Smart Automation):** 用户只需选择文件夹，AI自动完成所有分析

---

## 2. 系统架构

### 2.1 架构图

```
+------------------------------------------------------+
|                  表现层 (Presentation Layer)           |
|                (PyQt6 - 用户图形界面 UI)               |
+----------------------+-------------------------------+
                       | (用户交互, 数据展示)
+----------------------v-------------------------------+
|                业务逻辑层 (Business Logic Layer)       |
|    (应用主逻辑, 任务调度, 工作流控制 - main.py)        |
+----------------------+-------------------------------+
      | (调用AI分析)         | (读写元数据)
+-----v----------------+-----v--------------------------+
| AI引擎层 (AI Engine) | 数据服务层 (Data Service Layer)  |
| (analyzer.py)        | (database.py)                  |
+----------------------+-------------------------------+
      | (模型推理)           | (数据库操作)
+-----v----------------+-----v--------------------------+
|      硬件/系统层 (Hardware/System Layer)               |
|   (CPU/GPU, ONNX RT) | (SQLite3 数据库文件)            |
+------------------------------------------------------+
```

### 2.2 项目文件结构

```
Photoye/
├── main.py                 # 应用主入口和UI (PyQt6)
├── database.py             # 数据库交互模块 (SQLite3)
├── analyzer.py             # AI分析模块 (统一调度)
├── worker.py               # 后台工作线程模块 (V2.3 Open-Vocabulary)
├── clustering.py           # DBSCAN 人脸聚类模块
├── model_profiles.py       # 模型配置档定义
├── run.py                  # 快速启动脚本
├── start_photoye.bat       # Windows批处理启动
├── requirements.txt        # Python依赖
├── models/                 # 模型适配器代码
│   ├── insightface_detector.py     # InsightFace人脸检测 
│   ├── insightface_recognizer.py   # InsightFace人脸识别 
│   ├── openclip_zero_shot.py       # OpenCLIP零样本分类 
│   ├── clip_embedding.py           # CLIP语义编码器 (V2.3 Multi-Crop + Ensemble)
│   ├── opencv_yunet_detector.py    # YuNet人脸检测 (兼容)
│   ├── opencv_sface_recognizer.py  # SFace人脸识别 (兼容)
│   ├── mobilenetv2_classifier.py   # MobileNetV2场景分类 (兼容)
│   ├── model_interfaces.py         # 模型接口定义
│   ├── download_models.py          # 模型下载脚本
│   └── models/                     # ONNX模型文件目录
└── tests/                  # 单元测试
    ├── test_*.py           # 各模块测试文件
    └── test_images/        # 测试图片 (不纳入git)
```

---

## 3. 功能模块

### 3.0 自动化 AI 流水线 (V2.3 Open-Vocabulary)

用户只需选择照片文件夹，后台自动完成所有AI分析：

```
用户选择文件夹
       ↓
┌─────────────────────────────────────────────────────────┐
│              ScanWorker 自动化流水线                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  🗑️ Step 0: 清空旧数据                                   │
│  └── 清空 faces, persons, photos 表，确保全新分析        │
│                                                         │
│  📂 Step 1: 扫描文件                                     │
│  └── 遍历目录，收集图片，写入数据库                       │
│                                                         │
│  🏷️ Step 2: Embedding 提取 (V2.3 Multi-Crop)            │
│  ├── 5-crop 融合: 中心 + 4角 (加权平均)                  │
│  └── 提取 512维 embedding 用于语义搜索                  │
│                                                         │
│  👤 Step 3: 人脸检测 (InsightFace buffalo_sc)            │
│  ├── RetinaFace 检测人脸 + 5点关键点                     │
│  └── ArcFace 提取 512维人脸 embedding                   │
│                                                         │
│  ⚡ Step 4: 交叉验证                                     │
│  └── 检测到人脸 → 自动标记为含人物照片                  │
│                                                         │
│  🔗 Step 5: 自动聚类 (DBSCAN)                            │
│  ├── 将相似人脸分组为"人物" (eps=0.6, cosine距离)        │
│  └── 噪声点自动创建为"路人 #X"                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
       ↓
UI 自动刷新，显示分类结果和人物分组
       ↓
用户为人物命名 → **Open-Vocabulary 语义搜索**
```

### 3.1 V2.3 核心特性: Open-Vocabulary 语义检索

| 特性 | 说明 |
|------|------|
| **Prompt Ensemble** | 7 个模板平均，提升文本 embedding 质量 |
| **Multi-Crop** | 5 个裁剪融合 (中心+4角)，捕获更完整图像信息 |
| **Open-Vocabulary** | 用户输入任意自然语言查询，不受固定类别限制 |

**搜索示例:**
- "一个人在海边" → 返回海边单人照
- "美食摆盘" → 返回精致摆盘的食物照片
- "古建筑" → 返回古典建筑照片

### 3.2 AI模型清单

项目集成的模型文件（位于 `models/models/` 目录）：

**场景理解模型：**
| 模型文件 | 用途 | 来源 | 状态 |
|---------|------|------|------|
| `openclip_vitb32_vision.onnx` | OpenCLIP ViT-B/32 视觉编码器 (512维) | OpenCLIP | ✅ **主力** |
| `openclip_vitb32_text.onnx` | OpenCLIP ViT-B/32 文本编码器 | OpenCLIP | ✅ **主力** |
| `image_classification_mobilenetv2_2022apr.onnx` | MobileNetV2 分类 | OpenCV Zoo | ⚠️ 向后兼容 |

**人脸识别模型 (InsightFace buffalo_sc)：**
| 模型文件 | 用途 | 来源 | 状态 |
|---------|------|------|------|
| `det_500m.onnx` | RetinaFace 人脸检测 (2.5MB) | InsightFace | ✅ **主力** |
| `w600k_mbf.onnx` | ArcFace 人脸识别 (512维, 13.6MB) | InsightFace | ✅ **主力** |
| `face_detection_yunet_2023mar.onnx` | YuNet 人脸检测 | OpenCV Zoo | ⚠️ 向后兼容 |
| `face_recognition_sface_2021dec.onnx` | SFace 人脸识别 | OpenCV Zoo | ⚠️ 向后兼容 |

### 3.3 模型配置档 (Profiles)

在 `model_profiles.py` 中定义预设组合：

| 配置名 | 人脸检测 | 人脸识别 | 场景理解 | 特点 |
|--------|---------|---------|---------|------|
| `legacy` | YuNet | SFace | MobileNetV2 | 旧版兼容 |
| `insightface` | InsightFace | InsightFace | OpenCLIP | **默认**，推荐使用 |
| `best` | InsightFace | InsightFace | OpenCLIP | 最高精度 |

**切换方式：**
```bash
# 方式1: 环境变量
set PHOTOYE_MODEL_PROFILE=insightface

# 方式2: 代码中指定
analyzer = AIAnalyzer(model_profile="insightface")
```

### 3.4 后台工作线程

| Worker 类 | 功能 | 状态 |
|-----------|------|------|
| `ThumbnailWorker` | 异步生成缩略图，避免UI卡顿 | ✅ 已实现 |
| `ScanWorker` | **自动化流水线**: 清空→扫描→Multi-Crop→人脸→聚类 | ✅ **V2.3 核心** |
| `ClusteringWorker` | DBSCAN 人脸聚类 (eps=0.6, min_samples=2, cosine) | ✅ 已实现 |
| `SemanticSearchWorker` | CLIP 语义搜索 (Prompt Ensemble) | ✅ **V2.3 核心** |
| `FaceAnalysisWorker` | 独立人脸分析（保留用于手动触发） | ✅ 已实现 |

### 3.5 数据库结构

使用 SQLite3，数据库文件：`photoye_library.db`

**注意**: 每次扫描新文件夹时会清空旧数据，程序退出时也会清空，确保每次都是全新分析。

```sql
-- 照片表
CREATE TABLE photos (
    id INTEGER PRIMARY KEY,
    filepath TEXT NOT NULL UNIQUE,
    filesize INTEGER,
    created_at TEXT,
    category TEXT,                    -- 语义搜索结果缓存
    embedding BLOB,                   -- CLIP 512维特征向量 (新增)
    status TEXT DEFAULT 'pending'     -- pending/processing/done
);

-- 人物表
CREATE TABLE persons (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    cover_face_id INTEGER
);

-- 人脸数据表 (V2.1 升级版)
CREATE TABLE faces (
    id INTEGER PRIMARY KEY,
    photo_id INTEGER NOT NULL,
    person_id INTEGER,               -- NULL表示未命名，-1表示噪声点
    bbox TEXT NOT NULL,              -- JSON: [x1,y1,x2,y2]
    landmarks TEXT,                  -- JSON: 5点关键点坐标 (新增)
    embedding BLOB NOT NULL,         -- ArcFace 512维特征向量
    confidence REAL DEFAULT 0.0,
    is_noise INTEGER DEFAULT 0,      -- DBSCAN噪声标记 (新增)
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
);

-- 向量索引表 (可选，用于FAISS)
CREATE TABLE vector_index_meta (
    id INTEGER PRIMARY KEY,
    index_type TEXT,                 -- 'photo_clip' / 'face_arcface'
    last_updated TEXT,
    count INTEGER
);
```

---

## 4. 技术栈

| 类别 | 技术/库 | 版本要求 |
|------|---------|---------|
| 编程语言 | Python | 3.9+ |
| GUI框架 | PyQt6 | ≥6.4.0 |
| AI推理 | ONNX Runtime | ≥1.15.0 |
| 图像处理 | OpenCV-Contrib | ≥4.7.0 |
| 科学计算 | NumPy | ≥1.21.0 |
| 图像读取 | Pillow | ≥9.0.0 |
| 分词器 | tokenizers | ≥0.13.0 |
| 数据库 | SQLite3 | Python内置 |
| 聚类 | scikit-learn | ≥1.0.0 |

---

## 5. 快速开始

### 5.1 安装依赖

```bash
# 创建虚拟环境
conda create -n photoye python=3.10
conda activate photoye

# 安装依赖
pip install -r requirements.txt

# 如果有NVIDIA GPU
pip install onnxruntime-gpu
```

### 5.2 下载模型

```bash
python models/download_models.py
```

### 5.3 运行应用

```bash
python main.py
# 或
python run.py
# 或 Windows
start_photoye.bat
```

---

## 6. 使用流程

1. **启动应用** → 点击"选择文件夹"
2. **等待自动分析** → 观察左侧"🤖 AI 分析状态"面板
3. **浏览分类结果** → 使用分类筛选器（风景/美食/合照等）
4. **管理人物** → 切换到"按人物"筛选，为人物命名
5. **语义搜索** → 在搜索框输入自然语言描述

---

## 7. 测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行单个测试
python tests/test_models.py
python tests/test_analyzer.py
```

---

## 8. 开发文档

详细的开发规划、任务进度和技术方案请参阅：**[DEVELOPMENT.md](DEVELOPMENT.md)**

---

## 许可证

本项目仅供学习和个人使用。
