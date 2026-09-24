# Photoye 开发规划与任务看板

---

## 当前开发功能看板

| 模块编码 | 任务名称 | 范围与交付项 | 状态 | 关联模块 |
| :--- | :--- | :--- | :--- | :--- |
| **M1-F1** | SQLite V3.0 Schema | 新增 `sha256`, `gps`, `companion_files`, `backup_ledger` 表与字段 | ✅ 已完成 | `data/database.py` |
| **M1-F2** | 熟人底库多姿态质心 | 支持主质心 $\vec{C}_{\text{front}}$ 与侧脸模板 $\vec{C}_{\text{profile}}$ 存储与微调 | 🔄 进行中 | `core/ai/`, `data/database.py` |
| **M1-F3** | 侧脸级联识别 | 基于 5 点关键点偏航角比率，动态调用高精度模型提取特征 | ⏳ 待开始 | `core/ai/` |
| **M2-F1** | 伴侣文件探测与事务记录 | 识别同名 `.MOV`/`.XMP`/`.CR3` 等文件，写入 `undo.json` | ⏳ 待开始 | `sorter/` |
| **M2-F2** | 规则分拣管道 | 按人物 1对多扇出分拣、优先硬链接 (`os.link`)、跨盘复制、ZIP 导出 | ⏳ 待开始 | `sorter/` |
| **M3-F1** | 本地备份台账 | 基于分块 SHA-256 建立指纹账本，漏传对账与唤起资源管理器定位 | ⏳ 待开始 | `ledger/` |
| **M4-F1** | 展厅与画卷导出 | 时间线索引、那年今日查询、离线逆地理编码、独立 HTML 故事导出 | ⏳ 待开始 | `gallery/` |
| **M5-F1** | UI 视图重构 | 分拣工坊与时光展厅双入口分流，拆分原 `main.py` 单体逻辑 | ⏳ 待开始 | `ui/` |

---

## 架构分层

```
photoye/
├── app.py                      # 应用程序入口
├── core/                       # 业务逻辑层
│   ├── ai/                     # 模型推理 (人脸检测识别、CLIP语义特征、场景分类)
│   ├── sorter/                 # 文件分拣 (硬链接/复制、伴侣文件处理、事务与撤销)
│   ├── ledger/                 # 云备份台账 (SHA-256 计算、备份状态对账)
│   ├── gallery/                # 展厅服务 (时间线索引、离线逆地理编码、HTML故事导出)
│   └── metadata/               # 元数据提取 (EXIF 解析、伴侣文件关联)
├── data/                       # 数据持久层 (SQLite Schema 与数据访问接口)
├── ui/                         # 界面交互层 (PyQt6 视图组件与后台任务线程)
└── tests/                      # 自动化测试用例
```

---

## 核心数据结构

### 1. 数据库 Schema (SQLite 3)

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

