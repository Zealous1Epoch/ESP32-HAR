# ESP32-HAR 人体活动识别系统

> 基于 ESP32-S3 + MPU6050 + HMC5883L 的实时人体活动识别（6 类）
> 福建理工大学 · 智能科学与技术 · 模式识别专周实训
> GitHub: https://github.com/Zealous1Epoch/ESP32-HAR

---

## 项目结构

```
cc/
├── README.md                 ← 你在这里
├── .gitignore
├── portable.ini              # MicroPython 运行配置
│
├── esp32/                    # 🔵 ESP32 端 MicroPython 代码（烧录到开发板）
│   ├── main_serial.py        # 【主力】USB 串口版：采集→推理→JSON 输出
│   ├── main_wifi.py          # WiFi 热点版：自建 AP，手机浏览器看网页
│   ├── main.py               # 原始版本（参考用）
│   ├── collect_data.py       # 数据采集：串口输入动作名，倒计时 30s 采集 CSV
│   ├── scan_i2c.py           # I²C 总线扫描：检查 0x68(MPU) / 0x1E(HMC) 是否在线
│   ├── test_mag.py           # 磁力计 HMC5883L 功能测试
│   ├── wifi_only.py          # WiFi 纯连接测试
│   ├── wifi_test.py          # WiFi 功能测试
│   └── debug_test.py         # 调试脚本：直接读传感器原始值
│
├── pc/                       # 🟢 PC 端 Python 代码（在电脑上运行）
│   ├── train_v2.py           # 【主力】训练脚本：51维特征 + 随机森林 → rf_params.json
│   ├── train.py              # 原始训练脚本（30维特征，参考用）
│   ├── web_display.py        # Web 仪表盘 HTTP 服务器（localhost:8080）
│   ├── pc_display.py         # 终端命令行仪表盘
│   ├── collect_pc.py         # PC 端数据采集交互（连接 ESP32 串口转发命令）
│   ├── download_uci_har.py   # 下载并转换 UCI HAR 公开数据集
│   ├── generate_plots.py     # 生成报告图表（混淆矩阵、特征重要性、波形图等）
│   ├── generate_dashboard_charts.py  # 生成仪表盘截图
│   ├── verify_inference.py   # PC 端 vs ESP32 端推理一致性验证
│   └── test_infer.py         # 推理功能测试
│
├── models/                   # 🟡 训练好的模型文件
│   ├── rf_model.pkl          # sklearn RandomForest 模型（pickle，76KB）
│   ├── rf_params.json        # 模型参数 JSON（292KB → 上传到 ESP32 用）
│   └── scaler.pkl            # StandardScaler 标准化参数（pickle）
│
├── data/                     # 📂 数据集（.gitignore 排除，不上传 GitHub）
│   ├── dataset/              # 原始采集数据 + UCI HAR 转换后的 CSV
│   └── dataset_uci/          # UCI HAR 原始下载文件
│
├── plots/                    # 📊 生成的图表
│   ├── 01_confusion_matrix.png      # 混淆矩阵热力图
│   ├── 02_per_class_metrics.png     # 各类别 P/R/F1 柱状图
│   ├── 03_feature_importance.png    # 特征重要性排名
│   ├── 04_raw_signals.png           # 六类活动原始信号波形
│   ├── 05_feature_boxplot.png       # 关键特征各类别箱线图
│   ├── 06_system_pipeline.png       # 系统全链路流程图
│   ├── chart_donut.png              # 仪表盘：投票分布环形图
│   ├── chart_gauge.png              # 仪表盘：置信度仪表盘
│   ├── chart_timeline.png           # 仪表盘：活动历史时间线
│   ├── chart_votes.png              # 仪表盘：15棵树投票柱状图
│   └── dashboard_overview.png       # Web 仪表盘整体截图
│
└── docs/                     # 📄 文档
    ├── PROJECT_CONTEXT.md    # 项目完整上下文（硬件、模型、踩坑记录）
    ├── speech.md             # 答辩演讲稿（10 分钟）
    ├── presentation.html     # 答辩 PPT（HTML 版，21 页，浏览器打开）
    └── sk-ca7abad...txt      # (忽略)
```

---

## 快速开始

### 1. 训练模型（PC 端）

```bash
cd pc/
python train_v2.py
# 输出: ../models/rf_params.json (292KB)
```

需要先准备好数据（`data/dataset/` 目录下的 CSV 文件）。

### 2. 部署到 ESP32

```bash
# 1. 上传模型
mpremote fs cp models/rf_params.json :rf_params.json

# 2. 上传推理代码
mpremote fs cp esp32/main_serial.py :main.py

# 3. 复位运行
mpremote reset
```

### 3. 启动 Web 仪表盘

```bash
cd pc/
python web_display.py
# 浏览器打开 → http://localhost:8080
```

---

## 技术栈

| 层 | 技术 |
|----|------|
| 硬件 | ESP32-S3 N16R8 (240MHz, 8MB PSRAM) |
| 传感器 | MPU6050 (6轴 IMU) + HMC5883L (3轴磁力计)，I²C 总线 |
| 固件 | MicroPython v1.28.0 |
| 训练 | Python 3 + scikit-learn (RandomForest) |
| 特征 | 51 维（30 时域 + 4 方向感知 + 15 磁力计 + 2 幅值） |
| 评估 | 5-fold CV, 混淆矩阵, 标定增益, 消融实验 |
| 显示 | Web 仪表盘 (HTTP 8080) + WiFi 手机网页 + RGB LED |

---

## 核心数字

| 指标 | 数值 |
|------|------|
| 测试准确率 | **88.4%** |
| 5-fold CV | **78.4%** (±3.9%) |
| 方向感知特征贡献 | **+5.3%** 准确率 |
| 标定增益 | **+2.2%** 准确率 |
| 数据集 | 32 人 / 310+ 文件 / ~17K 样本 |
| 单次推理 | **< 5 ms** (ESP32) |
| 模型体积 | 292 KB JSON / 15 棵树 / depth=8 |

---

## 常用命令速查

```bash
# 列出 ESP32 文件
mpremote fs ls

# 上传文件到 ESP32
mpremote fs cp <本地文件> :<远程路径>

# 进入 ESP32 REPL
mpremote repl

# 软复位
mpremote reset

# 生成报告图表
cd pc/ && python generate_plots.py

# 生成仪表盘截图
cd pc/ && python generate_dashboard_charts.py

# 验证 PC vs ESP32 推理一致性
cd pc/ && python verify_inference.py
```

---

*最后更新：2026-06-26*
