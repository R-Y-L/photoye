# Photoye 全局技术架构与核心数据结构标准 (V3.0)

> **定位:** 全项目技术实现、分层设计、数据模型与核心算法的权威基准  
> **设计约束:** 纯 Python + ONNX Runtime · 本地优先 · 强内聚低耦合 · 零平台编译依赖

---

## 1. 系统分层架构 (System Architecture)

系统严格采用分层解耦架构，核心业务层不依赖任何 UI 库，支持独立自动化测试：

```
photoye/
├── app.py                      # 应用程序顶层入口 (极简路由)
├── core/                       # 核心业务逻辑层 (纯 Python / 零 UI 依赖)
│   ├── ai/                     # AI 推理与特征服务 (人脸感知、底库质心、OpenCLIP检索、场景纠偏)
│   ├── sorter/                 # 物理分拣引擎 (硬链接/复制、伴侣文件配对、undo事务回滚、ZIP导出)
│   ├── ledger/                 # 云备份台账服务 (分块 SHA-256 计算、漏传对账、系统资源管理器调起)
│   ├── gallery/                # 回忆展厅服务 (时间线索引、离线逆地理地名解析、HTML故事生成)
│   └── metadata/               # 基础元数据解析 (EXIF 解析、GPS 坐标提取、伴侣文件同名扫描)
├── data/                       # 数据持久层 (SQLite 3 Schema 权威定义与 DAO 读写接口)
├── ui/                         # 桌面表现层 (PyQt6 双模开屏、分拣工坊视图、时光展厅视图、异步 Worker)
└── tests/                      # 自动化测试套件 (包含算法单测与真实图片行为测试)
```

### 分层依赖约束准则
1. **单向依赖流**: `ui` $\to$ `core` $\to$ `data`。
2. **严禁逆向与跨层泄露**: `core` 与 `data` 禁止导入 `PyQt6` 或任何 UI 组件。
3. **数据传递解耦**: 层间交互采用 Python 原生基础类型、`dataclass` 或标准 `dict`，不传递私有复杂对象。

---

## 2. 核心数据结构与 Schema 标准 (SQLite 3)

数据库是系统状态持久化的唯一权威实体，由 `data/database.py` 严格按照以下标准实现：

### 2.1 照片基础资产表 (`photos`)
记录相册内所有主照片的基础物理信息、拍摄时间与全模态特征。
```sql
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT NOT NULL UNIQUE,          -- 主照片绝对路径
    filesize INTEGER NOT NULL,              -- 文件大小 (字节)
    sha256 TEXT NOT NULL,                   -- 文件内容分块哈希指纹 (用于移动/改名后对账)
    created_at TEXT,                        -- EXIF拍摄原始时间戳 (ISO格式)
    category TEXT,                          -- 题材分类: 风景/美食/单人照/合照/夜景等
    embedding BLOB,                         -- OpenCLIP 512维图像特征向量 (float32 byte-stream)
    latitude REAL,                          -- EXIF GPS 纬度
    longitude REAL,                         -- EXIF GPS 经度
    location_name TEXT,                     -- 离线逆地理编码地名 (如 "四川省 · 阿坝州 · 九寨沟")
    status TEXT DEFAULT 'pending',          -- 处理状态: pending, processed
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.2 伴侣文件关联表 (`companion_files`)
追踪与主照片关联的同名伴侣资产，保持物理文件生命周期同步。
```sql
CREATE TABLE IF NOT EXISTS companion_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,              -- 关联的主照片 ID
    filepath TEXT NOT NULL UNIQUE,          -- 伴侣文件绝对路径
    file_type TEXT NOT NULL,                -- 伴侣类型: live_video (.mov/.mp4), sidecar (.xmp/.aae), raw_negative (.cr3/.arw)
    filesize INTEGER NOT NULL,              -- 文件大小 (字节)
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);
```

### 2.3 熟人底库表 (`persons`)
持久化记录已知人物的身份标识与多姿态质心特征，跨批次永久记忆。
```sql
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,              -- 人物姓名
    cover_face_id INTEGER,                  -- 代表头像人脸 ID
    centroid_front BLOB,                    -- 正脸主质心 512维归一化向量
    centroid_profile BLOB,                  -- 侧脸质心 512维归一化向量
    face_count INTEGER DEFAULT 0,           -- 已归纳人脸样本总数 (用于加权更新)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.4 人脸特征表 (`faces`)
记录在各照片中检测到的每一张人脸、精确坐标与姿态元数据。
```sql
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,              -- 关联的照片 ID
    person_id INTEGER,                      -- 关联的人物 ID (未命名或噪声为 NULL)
    bbox TEXT NOT NULL,                     -- 边界框坐标 JSON: [x1, y1, x2, y2]
    landmarks TEXT NOT NULL,                -- 5点关键点坐标 JSON: [[x,y],...]
    embedding BLOB NOT NULL,                -- 512维 ArcFace 特征向量
    confidence REAL DEFAULT 0.0,            -- 检测置信度
    yaw_angle REAL,                         -- 水平偏航姿态角比率
    is_noise INTEGER DEFAULT 0,             -- DBSCAN 噪声点标记 (0: 否, 1: 是)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
);
```

### 2.5 本地云备份台账表 (`backup_ledger`)
建立基于 SHA-256 数字指纹的云备份对账账本。
```sql
CREATE TABLE IF NOT EXISTS backup_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_sha256 TEXT NOT NULL,             -- 文件指纹 (改名或移位依然稳定关联)
    target_platform TEXT NOT NULL,          -- 目标云平台 (如 "QQ相册", "百度网盘")
    album_name TEXT,                        -- 目标相册名称
    is_uploaded INTEGER DEFAULT 0,          -- 备份状态 (0: 未完成, 1: 已完成)
    uploaded_at TIMESTAMP,                  -- 完成时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(photo_sha256, target_platform, album_name)
);
```

---

## 3. 核心算法模型与计算规范

### 3.1 人脸双向几何姿态评估
利用 5 点关键点（双眼、鼻尖、双嘴角）计算几何比率：
- **水平偏航比率 (Yaw Ratio)**:
  $$R_{\text{yaw}} = \frac{|x_{\text{nose}} - x_{\text{left\_eye}}|}{|x_{\text{right\_eye}} - x_{\text{nose}}| + 10^{-6}}$$
  - $0.55 < R_{\text{yaw}} < 1.8$: 判定为 **正脸 / 微侧 (front)**；
  - $R_{\text{yaw}} \le 0.55$ 或 $R_{\text{yaw}} \ge 1.8$: 判定为 **大角度侧脸 (profile)**。

- **垂直俯仰比率 (Pitch Ratio)**:
  $$R_{\text{pitch}} = \frac{|y_{\text{nose}} - y_{\text{eye\_center}}|}{|y_{\text{mouth\_center}} - y_{\text{nose}}| + 10^{-6}}$$
  - $R_{\text{pitch}} \le 0.35$ 或 $R_{\text{pitch}} \ge 1.6$: 判定为 **大角度仰俯姿态 (pitch)**。

### 3.2 熟人底库多姿态比对与在线微调
- **综合相似度计算**:
  $$\text{Sim}(f, P) = \max\left(\vec{f} \cdot \vec{C}_{\text{front}}, \; \vec{f} \cdot \vec{C}_{\text{profile}}\right)$$
  阈值标准: $\text{Sim} \ge 0.60$ 即确认为该人物。
- **在线滑动平均质心微调**:
  $$\vec{C}_{\text{new}} = \text{Normalize}\left(\alpha \vec{C}_{\text{old}} + (1 - \alpha) \vec{f}\right) \quad (\alpha = 0.85)$$

### 3.3 场景题材分类与多维度多标签融合规范
放弃互斥的单一分类覆盖逻辑，全面采用正交多标签（Multi-Label Tags）聚合体系：
- **人像主体维度 (Subject Dimension)**:
  - 检出人脸数 $= 0$: 输出标签 `无人物`
  - 检出人脸数 $= 1$: 输出标签 `单人照`，并追加 `人物:{name}` (若已确认)
  - 检出人脸数 $\ge 2$: 输出标签 `合照`，并分别追加出镜各人物 `人物:{name}`
- **场景题材维度 (Scene Dimension)**:
  - 由 OpenCLIP 计算各候选题材 (如 `风景`, `美食`, `建筑`, `室内`, `宠物`, `夜景`, `文档`) 的余弦相似度
  - 概率达标的题材标签全部保留，与人像主体标签正交叠加，绝不相互覆盖
- **最终合成输出**:
  输出标准的字符串标签列表，如 `["风景", "合照", "人物:YouYu"]`，既保留自然风光检索属性，又具备合影分拣去向。

### 3.4 分拣原子事务与回滚清单格式
分拣过程生成唯一事务编号，并在目标目录写入 `photoye_run_<timestamp>.undo.json`：
```json
{
  "run_id": "20260928_120000",
  "created_at": "2026-09-28T12:00:00",
  "mode": "hardlink",
  "operations": [
    {
      "source": "D:/Photos/IMG_01.JPG",
      "target": "D:/Output/YouYu/IMG_01.JPG",
      "type": "hardlink"
    },
    {
      "source": "D:/Photos/IMG_01.MOV",
      "target": "D:/Output/YouYu/IMG_01.MOV",
      "type": "hardlink",
      "is_companion": true
    }
  ]
}
```
撤销引擎根据操作清单反向解除链接或删除目标文件，若目标文件在分拣后已被外部程序修改则强制告警并停止删除。
