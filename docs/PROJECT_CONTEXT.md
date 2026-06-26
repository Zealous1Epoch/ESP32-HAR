# ESP32-S3 MPU6050 HAR 项目上下文

## 项目概述
基于 ESP32-S3 + MPU6050 的人体活动识别（HAR）系统，实时检测 6 类动作：
**静坐(sit)、站立(stand)、走路(walk)、上楼(upstairs)、下楼(downstairs)、跑步(run)**

## GitHub 仓库
https://github.com/Zealous1Epoch/ESP32-HAR

## 课程背景
福建理工大学 智能科学与技术（大三）
《模式识别与统计学习》专周实训
任务书：从传感器数据到模式识别——可穿戴智能感知系统的全链路工程实践

---

## 硬件

### 老设备（当前聊天窗口使用）
- ESP32-S3 (YD-ESP32-S3 N16R8)
- MPU6050 六轴传感器 (I2C 0x68)
- 接线: SDA→GPIO8, SCL→GPIO9, LED→GPIO48
- USB 串口连接电脑

### 新设备（新聊天窗口）
- ESP32-S3（另一块板子/PCB扩展板）
- MPU6050 (0x68) + **HMC5883L 磁力计 (0x1E)** ← 新增！
- 两个传感器共享 I2C 总线
- 磁力计已验证正常工作

---

## 当前模型 v3

| 项目 | 详情 |
|------|------|
| 算法 | 随机森林 (Random Forest) |
| 参数 | 15棵树, max_depth=8, min_samples_leaf=5 |
| 特征 | **34维** (30时域基础 + 4方向感知) |
| 数据 | 31人 (UCI HAR 30人 + 朋友p0 + 自己p1) |
| 样本 | 16,987 个窗口 |
| 准确率 | 测试集 88.4%, 5-fold CV 78.4% |
| 模型文件 | rf_params.json (292KB) |

### 34维特征详情
```
索引 0-29:  6轴 × 5统计量 (mean/std/max/min/ptp)
           acc_x/acc_y/acc_z/gyro_x/gyro_y/gyro_z

索引 30:   acc_mag_mean    - 加速度幅值均值 (静止≈1g)
索引 31:   acc_mag_std     - 加速度幅值标准差 (运动波动)
索引 32:   tilt_angle      - 传感器倾斜角 (区分坐/站的核心!)
索引 33:   gyro_mag_mean   - 陀螺仪幅值均值 (静止≈0)
```

---

## 代码文件说明

### ESP32 端 (MicroPython)
| 文件 | 用途 |
|------|------|
| `main_serial.py` | **USB串口版** — 采集→推理→串口JSON输出 (稳定版) |
| `main_wifi.py` | **WiFi热点版** — ESP32自建AP，手机浏览器看网页 (有问题，连不上热点) |
| `collect_data.py` | **数据采集** — 串口输入动作名，倒计时30秒采集CSV |
| `rf_params.json` | 训练好的模型参数 (JSON) |

### PC 端 (Python)
| 文件 | 用途 |
|------|------|
| `web_display.py` | **网页仪表盘** — 读串口→HTTP服务器→浏览器实时显示 (localhost:8080) |
| `pc_display.py` | **终端仪表盘** — 命令行彩色实时显示 |
| `collect_pc.py` | **PC端采集交互** — 连接ESP32串口，转发采集命令 |
| `train_v2.py` | **训练脚本** — 34维特征 + 随机森林 → rf_params.json |
| `train.py` | 原始训练脚本 (30维, 朋友版) |
| `generate_plots.py` | 生成 PPT 图表 (混淆矩阵、特征重要性、波形图等) |
| `verify_inference.py` | PC vs ESP32 推理一致性验证 |
| `test_mag.py` | HMC5883L 磁力计测试脚本 |
| `download_uci_har.py` | 下载并转换 UCI HAR 数据集 |

### 数据集
- `dataset/` — 原始数据 + UCI HAR (335个CSV, 由 .gitignore 排除)
- `plots/` — 生成的图表

---

## ESP32-S3 特殊注意事项

1. **USB-Serial/JTAG**: 这个板子用内置USB，`mpremote` 的 raw REPL 经常连不上
2. **上传文件**: 需要在 ESP32 处于 REPL 模式时用 mpremote，不能用 `main.py` 运行时上传
3. **拔插USB**: 可靠的复位方式。打开串口时用 `dsrdtr=False` 避免意外复位
4. **残留进程**: Python 串口脚本如果崩溃会占着端口，需要 `kill` 清理
5. **MicroPython v1.28.0** with Octal-SPIRAM

---

## 新设备需要做的事

1. **采集磁力计数据**: 修改 `collect_data.py`，在 CSV 中加入 `mag_x, mag_y, mag_z` 三列
2. **扩展特征**: 磁力计3轴 × 5统计量 = 15维新特征，加上航向稳定性、磁场幅值等可能再加 3-4 维，总共约 **52-53维**
3. **重新采集**: 用新设备采集全部 6 类动作（每类 30 秒）
4. **重新训练**: 更新 `train_v2.py` 的特征提取（加磁力计特征）
5. **更新 ESP32 推理**: 更新 `main_serial.py` 的 `extract_features()` 和 `read_mpu_raw()`（加入磁力计读取）
6. **WiFi 调试**: 如果要用无线模式，需要排查 why ESP32-HAR 热点连不上

---

## 重要：串口通信的坑

```
打开串口 → dsrdtr=False (不自动复位)
上传文件 → 先 Ctrl-C 中断 main.py → 进入 REPL → mpremote fs cp
读取数据 → ser.readline() 有时会乱码，需要在 JSON 前后找 { }
进程管理 → 脚本出错后要 kill 残留进程
```
