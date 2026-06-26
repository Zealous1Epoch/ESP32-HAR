#!/usr/bin/env python3
"""
一键生成答辩 PPT 所需全部图表
依赖: pip install matplotlib seaborn pandas numpy scikit-learn joblib
运行: python generate_plots.py
输出: ./plots/ 目录下的 PNG 图片
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无 GUI 也能跑
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import json

# ===== 配置 =====
DATA_DIR = "./dataset"
WINDOW_SIZE = 128
STEP = 64
RANDOM_SEED = 42
OUT_DIR = "./plots"

LABEL_MAP = {
    "sit": 0, "stand": 1, "walk": 2,
    "upstairs": 3, "downstairs": 4, "run": 5
}
CLASS_NAMES = ["sit", "stand", "walk", "upstairs", "downstairs", "run"]
CLASS_NAMES_CN = ["静坐", "站立", "走路", "上楼", "下楼", "跑步"]

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs(OUT_DIR, exist_ok=True)
print(f"输出目录: {OUT_DIR}")


# ===================== 1. 加载数据 =====================
def load_data():
    X, y = [], []
    raw_signals = {}  # 每类存一段原始信号用于波形图

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    print(f"找到 {len(files)} 个 CSV 文件")

    for fname in sorted(files):
        parts = fname.split("_")
        if len(parts) < 3 or parts[2] not in LABEL_MAP:
            continue
        label = LABEL_MAP[parts[2]]
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        data = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].values

        # 保存第一份完整信号用于波形图
        if label not in raw_signals:
            raw_signals[label] = data[:512]  # 存 512 个采样点 (约 10 秒)

        for i in range(0, len(data) - WINDOW_SIZE + 1, STEP):
            X.append(data[i:i + WINDOW_SIZE])
            y.append(label)

    X, y = np.array(X), np.array(y)
    print(f"总样本数: {len(X)}")
    return X, y, raw_signals


def extract_feat(windows):
    feats = []
    for w in windows:
        feat = []
        for col in range(6):
            col_data = w[:, col]
            feat.append(np.mean(col_data))
            feat.append(np.std(col_data))
            feat.append(np.max(col_data))
            feat.append(np.min(col_data))
            feat.append(np.ptp(col_data))
        feats.append(feat)
    return np.array(feats)


X, y, raw_signals = load_data()
X_feat = extract_feat(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=15, max_depth=8, random_state=RANDOM_SEED)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

accuracy = (y_pred == y_test).mean()
print(f"\n模型准确率: {accuracy:.4f}")

# ===================== 2. 混淆矩阵 =====================
print("\n[1/6] 生成混淆矩阵...")
fig, ax = plt.subplots(figsize=(8, 6.5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES_CN)
disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=True)
ax.set_title("Confusion Matrix — Random Forest (15 trees, max_depth=8)", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/01_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✓ 01_confusion_matrix.png")

# ===================== 3. 各类别 F1 柱状图 =====================
print("\n[2/6] 生成各类别 Precision/Recall/F1 柱状图...")
from sklearn.metrics import precision_recall_fscore_support

p, r, f1, s = precision_recall_fscore_support(y_test, y_pred, labels=range(6))

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(CLASS_NAMES))
w = 0.25
bars1 = ax.bar(x - w, p, w, label="Precision", color="#4ECDC4", edgecolor="white")
bars2 = ax.bar(x, r, w, label="Recall", color="#FF6B6B", edgecolor="white")
bars3 = ax.bar(x + w, f1, w, label="F1-Score", color="#556270", edgecolor="white")

for bar in [bars1, bars2, bars3]:
    for rect in bar:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES_CN, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Per-Class Precision / Recall / F1-Score", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/02_per_class_metrics.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✓ 02_per_class_metrics.png")

# ===================== 4. 特征重要性 Top-15 =====================
print("\n[3/6] 生成特征重要性图...")
feature_names = []
for axis in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
    for stat in ["mean", "std", "max", "min", "ptp"]:
        feature_names.append(f"{axis}_{stat}")

importances = rf.feature_importances_
idx = np.argsort(importances)[::-1][:15]

fig, ax = plt.subplots(figsize=(10, 5.5))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, 15))
ax.barh(range(15), importances[idx][::-1], color=colors[::-1], edgecolor="white")
ax.set_yticks(range(15))
ax.set_yticklabels([feature_names[i] for i in idx][::-1], fontsize=9)
ax.set_xlabel("Importance", fontsize=12)
ax.set_title("Random Forest Feature Importance (Top 15/30)", fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/03_feature_importance.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✓ 03_feature_importance.png")

# ===================== 5. 原始信号波形对比（6 类动作各一例） =====================
print("\n[4/6] 生成原始信号波形对比图...")
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (label_id, label_name) in enumerate(zip(range(6), CLASS_NAMES)):
    ax = axes[idx]
    if label_id in raw_signals:
        signal = raw_signals[label_id]
        t = np.arange(len(signal)) / 50.0  # 50 Hz
        ax.plot(t, signal[:, 0], label="acc_x", alpha=0.8, linewidth=0.8)
        ax.plot(t, signal[:, 1], label="acc_y", alpha=0.8, linewidth=0.8)
        ax.plot(t, signal[:, 2], label="acc_z", alpha=0.8, linewidth=0.8)
    ax.set_title(f"{CLASS_NAMES_CN[label_id]} ({label_name})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Acceleration (raw)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

fig.suptitle("Raw MPU6050 Acceleration Signals — 6 Activities", fontsize=15, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/04_raw_signals.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✓ 04_raw_signals.png")

# ===================== 6. 特征分布箱线图（选 4 个关键特征） =====================
print("\n[5/6] 生成特征分布箱线图...")
key_features = [
    (0, "acc_x_mean (加速度X均值)"),
    (5, "acc_y_std (加速度Y标准差)"),
    (25, "gyro_x_ptp (陀螺X峰峰值)"),
    (29, "gyro_z_ptp (陀螺Z峰峰值)"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for i, (feat_idx, feat_name) in enumerate(key_features):
    ax = axes[i]
    data_by_class = [X_feat[y == c, feat_idx] for c in range(6)]
    bp = ax.boxplot(data_by_class, patch_artist=True, tick_labels=CLASS_NAMES_CN)
    palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
    ax.set_title(feat_name, fontsize=11, fontweight="bold")
    ax.set_ylabel("Feature Value", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Key Feature Distributions Across 6 Activities", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/05_feature_boxplot.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✓ 05_feature_boxplot.png")

# ===================== 7. 系统架构流程图 =====================
print("\n[6/6] 生成系统架构流程图...")
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis("off")

blocks = [
    (1, 2.5, "MPU6050\nData\nAcquisition", "#FF6B6B"),
    (3, 2.5, "Calibration\n& Preprocessing", "#FFEAA7"),
    (4.8, 2.5, "Sliding\nWindow\n(128×6)", "#4ECDC4"),
    (6.5, 2.5, "Feature\nExtraction\n(30 dims)", "#45B7D1"),
    (8.2, 2.5, "Random Forest\nInference\n(15 trees)", "#96CEB4"),
    (10.2, 2.5, "Smoothing\n(3-frame\nmajority)", "#DDA0DD"),
    (12.2, 2.5, "Output\nLED+Serial\n+Web", "#556270"),
]

for x, y, label, color in blocks:
    rect = plt.Rectangle((x - 0.7, y - 1.2), 1.4, 2.4, facecolor=color, edgecolor="white",
                          linewidth=2, alpha=0.85, transform=ax.transData)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold", color="white")

# 箭头
for i in range(len(blocks) - 1):
    x1 = blocks[i][0] + 0.7
    x2 = blocks[i + 1][0] - 0.7
    y = 2.5
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", color="white", lw=2.5,
                                connectionstyle="arc3,rad=0"))

ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
ax.set_title("ESP32-S3 HAR System Pipeline", fontsize=16, fontweight="bold", color="white", pad=20)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/06_system_pipeline.png", dpi=200, bbox_inches="tight",
            facecolor="#1a1a2e")
plt.close()
print("  ✓ 06_system_pipeline.png")

# ===================== 完成 =====================
print(f"\n{'='*50}")
print(f"全部完成! 图片输出到: {os.path.abspath(OUT_DIR)}/")
print(f"共生成 6 张图:")
print(f"  01_confusion_matrix.png   — 混淆矩阵热力图")
print(f"  02_per_class_metrics.png   — 各类别 P/R/F1 柱状图")
print(f"  03_feature_importance.png  — 特征重要性 Top-15")
print(f"  04_raw_signals.png         — 6 类动作原始信号波形")
print(f"  05_feature_boxplot.png     — 关键特征分布箱线图")
print(f"  06_system_pipeline.png     — 系统架构流程图")
print(f"{'='*50}")
