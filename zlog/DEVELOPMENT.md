# Photoye 当前冲刺看板 (Sprint 2: 整理工坊 UI 装配与物理分拣引擎)

> **当前聚焦阶段:** Milestone 2 (整理工坊 UI 界面装配 + 物理分拣落地)  
> **全局规划来源:** `zlog/ROADMAP.md`  
> **核心架构标准:** `zlog/ARCHITECTURE.md`

---

## 📌 当前 Sprint 任务执行看板 (高频变动)

| 任务编码 | 任务名称 | 范围与关键交付 | 状态 | 关联模块 / 测试 |
| :--- | :--- | :--- | :--- | :--- |
| **M2-F1** | EXIF 与伴侣文件探测 | EXIF 拍摄时间/GPS 解析、同名动态视频(.MOV)及调色/RAW文件配对 | ✅ 已完成 | `core/metadata/`<br/>`tests/test_companion_scanner.py` |
| **M2-F2** | 多维规则分发管道 | 按人物定向分拣 (合影1对多扇出)、按题材归集、按年月层级归档 | ✅ 已完成 | `core/sorter/pipeline.py`<br/>`tests/test_sort_pipeline.py` |
| **M2-UI** | 智能整理工坊桌面界面 | PyQt6 Fluent 控件、照片网格、去向卡片、大图预览、空状态与加载反馈；新建/编辑、删除确认及结果提示统一 Fluent 遮罩弹窗；系统目录选择器保留原生，分拣规则不变 | 🔄 Fluent 改造已实现，视觉待用户确认 | `app.py`, `ui/views/sorter_view.py`, `ui/views/preview_dialog.py`, `ui/views/workshop_dialogs.py`<br/>`tests/test_fluent_workshop.py`：启动、搜索/筛选、Qt 拖放、移出/编辑、真实模态弹窗确认/取消、两步编辑取消原子性、空白输入/回车/Esc、长文本滚动、预览缩放、异步失败与临时目录真实分拣；包含紧凑窗口与 150% Qt 缩放检查。验收不代表真实 AI 推理精度已重新验证 |
| **M2-F3** | 存储操作模式 | 同磁盘优先 `os.link` 硬链接 (0空间占用)、跨磁盘流式复制、重名防灾 | ✅ 已完成 | `core/sorter/file_ops.py`<br/>`tests/test_file_ops.py` |
| **M2-F4** | 事务审计与一键撤销 | 唯一 JSON 审计清单、文件身份与 SHA-256 校验、伴侣组失败回退；后台分拣、上次/历史清单撤销、Fluent 确认及关闭保护 | ✅ 已实现并通过自动化与样片验证，用户体验待确认 | `core/sorter/transaction.py`, `core/sorter/file_ops.py`<br/>`tests/test_transaction.py`, `tests/test_file_ops.py`, `tests/test_fluent_workshop.py` |
| **M2-F5** | 独立分包打包导出 | 整理产物按人物/分类分别压缩为独立 ZIP 文件 | ⏳ 待开始 | `core/sorter/exporter.py`<br/>`tests/test_exporter.py` |

**详情浏览交互补充**：图像区滚轮向下/向上切换下一张/上一张，Ctrl + 滚轮以鼠标位置为中心缩放；保留左右方向键、放大拖动、双击原图/适应、缩放工具栏。列表预览按当前筛选与搜索排序浏览，篮子预览仅浏览该篮子；首尾不循环，切图同步标签及伴侣文件并恢复适应窗口。右侧详情区滚轮仍用于滚动文本。回归使用真实 Qt 滚轮、按键、拖动及双击事件，不代表真实 AI 推理精度验收。

---

## 📝 当前模块开发说明 (M2-F4)
- **安全落地**：输出目录必须在所选根目录内；按完整主照片路径绑定原子组，整组共同选择无冲突序号；硬链接仅在跨卷或文件系统不支持时降级为独占创建的流式复制，重名和权限错误不能触发覆盖。
- **事务记录**：每次分拣在输出根目录生成唯一 `photoye_run_<timestamp>_<uuid>.undo.json`。原子替换写入 UTF-8 清单，登记源/目标、操作类型、伴侣标记、文件身份、大小、修改时间及分块 SHA-256；日志写入失败停止后续分拣并报告原因。
- **受保护撤销**：先校验整批现存目标，任一文件被修改或替换则整批停止；之后反序移除产物，源文件和既有目标不动。只清理本次登记且身份未变的空目录，保留非空目录。全部撤销成功后删除对应 JSON 清单并禁用该批次的撤销入口；任一步失败则保留清单供重试。旧版已撤销但保留的清单可再次选择清理，缺失目标安全跳过。
- **界面**：分拣和撤销在后台线程执行，期间禁用冲突操作并禁止关闭主窗口；提供“撤销上次分拣”和“选择撤销清单”，确认后才执行，结果包含逐文件失败原因。
- **本轮验证**：撤销清单清理修正后，相关回归 160 通过、5 跳过（Windows 当前账户无符号链接创建权限），150% Qt 缩放界面回归 46 通过；覆盖清单占用/目录清理失败后保留重试、清单替换保护、旧版已撤销清单清理及历史批次与最新批次按钮隔离。未重新验收 AI 精度，未操作真实相册或数据库。
- **工程边界**：撤销清单是受信本地审计文件，不是防篡改签名。普通失败可回退，但断电/强杀发生在文件创建与日志落盘之间时，仍可能留下未登记产物；未做崩溃恢复验收。撤销期间不要由其他程序编辑或替换输出文件。硬链接与源共享内容，编辑任一路径会影响另一条路径，撤销会因此拒绝删除。
- **审查待办（本轮未扩展）**：篮子手动拖入/移出仍按 stem 标识，同 stem 不同扩展名主图可能串扰；建议下一小阶段修正为完整路径身份并回归，之后再做 ZIP 导出。扫描线程仍使用内存人物底库，未在本轮重构。


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

