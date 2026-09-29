# Photoye 当前冲刺看板 (Sprint 2: 多维规则物理分拣引擎)

> **当前聚焦阶段:** Milestone 2 (多维规则物理分拣引擎)  
> **全局规划来源:** `zlog/ROADMAP.md`  
> **核心架构标准:** `zlog/ARCHITECTURE.md`

---

## 📌 当前 Sprint 任务执行看板 (高频变动)

| 任务编码 | 任务名称 | 范围与关键交付 | 状态 | 关联模块 / 测试 |
| :--- | :--- | :--- | :--- | :--- |
| **M2-F1** | EXIF 与伴侣文件探测 | EXIF 拍摄时间/GPS 解析、同名动态视频(.MOV)及调色/RAW文件配对 | ✅ 已完成 | `core/metadata/`<br/>`tests/test_companion_scanner.py` |
| **M2-F2** | 多维规则分发管道 | 按人物定向分拣 (合影1对多扇出)、按题材归集、按年月层级归档 | ✅ 已完成 | `core/sorter/pipeline.py`<br/>`tests/test_sort_pipeline.py` |
| **M2-F3** | 存储操作模式 | 同磁盘优先 `os.link` 硬链接 (0空间占用)、跨磁盘流式复制、重名防灾 | 🔄 进行中 | `core/sorter/file_ops.py`<br/>`tests/test_file_ops.py` |
| **M2-F4** | 事务审计与一键撤销 | 写入 `photoye_run_<timestamp>.undo.json` 事务清单、回滚引擎 | ⏳ 待开始 | `core/sorter/transaction.py`<br/>`tests/test_transaction.py` |
| **M2-F5** | 独立分包打包导出 | 整理产物按人物/分类分别压缩为独立 ZIP 文件 | ⏳ 待开始 | `core/sorter/exporter.py`<br/>`tests/test_exporter.py` |

---

## 📝 当前模块开发说明 (M2-F2)
- **目标**: 实现 `core/sorter/pipeline.py`，构建 `SortingPipeline` 多维规则物理分拣路由器。
- **输入**: `PhotoAssetGroup` 资产单元、AI 多标签分析结果（包含人物出镜名单、场景题材标签与拍摄时间戳）。
- **计算**:
  1. **按人物 1 对多扇出分发 (Fan-out Dispatch)**: 若合照中同时出镜人物 Alice 和 Bob，自动路由并分发至 `人物_相册/Alice/` 与 `人物_相册/Bob/` 两个独立目标；
  2. **按场景题材归集**: 无人物照片按题材标签路由至对应子目录（如 `题材_相册/风景/`, `题材_相册/美食/`）；
  3. **按拍摄时间层级归档**: 若启用时间归档，依据 EXIF 拍摄时间自动构建 `时间_相册/YYYY/MM/` 目录结构；
  4. **伴侣文件级联路由**: 每一个目标去向中，主照片的全部伴侣文件（`.mov`, `.xmp` 等）同步生成对应的伴侣目标路径，同进同退。
- **输出对象**: `DispatchPlan`（结构化分发计划，包含每一对源文件与目标文件的完整映射，供底层存储与事务执行）。
- **验收标准**: 编写 `tests/test_sort_pipeline.py`，模拟单人照、多人合照（1对多扇出）、纯风景照与无 EXIF 照片，验证生成的物理分发路由映射表 100% 准确无错乱。


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

