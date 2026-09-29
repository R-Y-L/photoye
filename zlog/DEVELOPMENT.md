# Photoye 当前冲刺看板 (Sprint 1: AI理解与数据基建)

> **当前聚焦阶段:** Milestone 1 (AI 理解与数据基建)  
> **全局规划来源:** `zlog/ROADMAP.md`  
> **核心架构标准:** `zlog/ARCHITECTURE.md`

---

## 📌 当前 Sprint 任务执行看板 (高频变动)

| 任务编码 | 任务名称 | 范围与关键交付 | 状态 | 关联模块 / 测试 |
| :--- | :--- | :--- | :--- | :--- |
| **M1-F1** | SQLite V3.0 Schema | 新增 photos, companion_files, persons, faces, backup_ledger 五张表及索引 | ✅ 已完成 | `data/database.py`<br/>`tests/test_database_v3.py` |
| **M1-F2** | 熟人底库多姿态质心 | 支持主质心 $\vec{C}_{\text{front}}$ 与侧脸模板 $\vec{C}_{\text{profile}}$ 存储、在线更新与增量聚类 | ✅ 已完成 | `core/ai/face_library.py`<br/>`tests/test_face_library.py` |
| **M1-F3** | 自适应人像感知引擎 | 基于 5 点关键点偏航角(Yaw)与俯仰角(Pitch)评估、多姿态人脸抽取 | ✅ 已完成 | `core/ai/face_engine.py`<br/>`tests/test_real_face_engine.py` |
| **M1-F4** | OpenCLIP 场景分类与多标签融合 | 零样本多题材分类、人像主体与场景题材正交多标签聚合引擎 | ✅ 已完成 | `core/ai/scene_classifier.py`<br/>`tests/test_scene_classifier.py` |
| **M1-F5** | 本地开放式语义搜索 | 纯 ONNX 文本编码、Prompt Ensemble 模板平均、自然语言搜图 | ✅ 已完成 | `core/ai/semantic_search.py`<br/>`tests/test_semantic_search.py` |

---

## 📝 当前模块开发说明 (M1-F5)
- **目标**: 封装 `core/ai/semantic_search.py`，实现 `SemanticSearchEngine` 纯本地自然语言搜图服务。
- **输入**: 任意自然语言生活化查询词（中英描述如“自然森林风景”、“两个人合照”、“特写人像”等）。
- **计算**: 
  - 文本端: 基于预置 Prompt Ensemble 多模板计算 512 维平均文本向量；
  - 矩阵计算: 文本向量与数据库/内存中全量照片的 512 维图像特征向量进行批量余弦相似度矩阵点积；
  - 排序过滤: 输出 Top-K 最匹配照片路径、相似度分值与多标签属性。
- **验收标准**: 编写 `tests/test_semantic_search.py`，对测试集 16 张真实图片进行自然语言查图断言，保证搜“自然风景/树林”排名前列全为风光图，搜“合影”排名前列全为合影图。


```sql
-- 照片基础资产表
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT NOT NULL UNIQUE,          -- 主文件绝对路径
    filesize INTEGER NOT NULL,              -- 文件大小 (字节)
    sha256 TEXT NOT NULL,                   -- 文件内容哈希指纹
    created_at TEXT,                        -- EXIF拍摄原始时间戳 (ISO格式)
    category TEXT,                          -- 题材分类: 风景/美食/单人照/合照等
    embedding BLOB,                         -- OpenCLIP 512维图像特征向量
    latitude REAL,                          -- EXIF GPS 纬度
    longitude REAL,                         -- EXIF GPS 经度
    location_name TEXT,                     -- 离线逆地理编码地名 (省/市/区/景点)
    status TEXT DEFAULT 'pending',          -- 处理状态: pending, processed
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 伴侣文件关联表 (LivePhoto, RAW, Sidecar 等)
CREATE TABLE IF NOT EXISTS companion_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,              -- 关联的主照片ID
    filepath TEXT NOT NULL UNIQUE,          -- 伴侣文件绝对路径
    file_type TEXT NOT NULL,                -- 伴侣类型: live_video, sidecar, raw_negative
    filesize INTEGER NOT NULL,
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);

-- 熟人底库表 (支持多姿态质心持久化)
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,              -- 人物姓名
    cover_face_id INTEGER,                  -- 代表头像人脸ID
    centroid_front BLOB,                    -- 正脸/主质心 512维向量 (L2归一化)
    centroid_profile BLOB,                  -- 侧脸质心 512维向量 (L2归一化)
    face_count INTEGER DEFAULT 0,           -- 已归纳人脸样本总数
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 人脸特征表
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,              -- 关联的照片ID
    person_id INTEGER,                      -- 关联的人物ID (未命名或噪声为NULL)
    bbox TEXT NOT NULL,                     -- 边界框坐标 JSON: [x1, y1, x2, y2]
    landmarks TEXT NOT NULL,                -- 5点关键点 JSON: [[x,y],...]
    embedding BLOB NOT NULL,                -- 512维 ArcFace 特征向量
    confidence REAL DEFAULT 0.0,            -- 检测置信度
    yaw_angle REAL,                         -- 人脸偏航姿态角 (估算值)
    is_noise INTEGER DEFAULT 0,             -- 噪声点标记
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
);

-- 云备份台账表
CREATE TABLE IF NOT EXISTS backup_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_sha256 TEXT NOT NULL,             -- 文件指纹
    target_platform TEXT NOT NULL,          -- 目标平台 (如 "QQ相册", "百度网盘")
    album_name TEXT,                        -- 目标相册名称
    is_uploaded INTEGER DEFAULT 0,          -- 备份状态 (0: 未完成, 1: 已完成)
    uploaded_at TIMESTAMP,                  -- 完成时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(photo_sha256, target_platform, album_name)
);
```

---

## 核心算法与流程

### 1. 人脸姿态评估与识别匹配
- **偏航角比率计算**：
  $$R = \frac{|x_{\text{nose}} - x_{\text{left\_eye}}|}{|x_{\text{right\_eye}} - x_{\text{nose}}|}$$
  当 $R \ge 1.8$ 或 $R \le 0.55$ 时判定为大角度侧脸，调用高精度模型提取特征。
- **相似度计算与质心更新**：
  $$\text{Sim}(f, P) = \max\left(\vec{f} \cdot \vec{C}_{\text{front}}, \; \vec{f} \cdot \vec{C}_{\text{profile}}\right)$$
  当 $\text{Sim} \ge 0.65$ 时归入该人物，并更新对应姿态的质心滑动平均值：
  $$\vec{C} \leftarrow \text{Normalize}(\alpha \vec{C} + (1 - \alpha) \vec{f})$$

### 2. 分拣事务与伴侣文件
- **原子操作**：主文件与伴侣文件作为一个事务单元处理。
- **链接与复制**：同磁盘优先使用 `os.link`，跨磁盘使用文件复制。
- **事务记录与撤销**：分拣过程将操作记录写入 `photoye_run_<timestamp>.undo.json`，撤销操作根据记录删除目标文件。

