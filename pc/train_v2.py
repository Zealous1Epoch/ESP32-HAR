#!/usr/bin/env python3
"""
HAR 模型训练 v3 — 随机森林 + 磁力计特征 (51维)
v2→v3: 新增 17 维磁力计特征 (mag×3轴 + 磁场幅值)
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = "../data/dataset"
WINDOW_SIZE = 128
STEP = 64
RANDOM_SEED = 42

LABEL_MAP = {
    "sit": 0, "stand": 1, "walk": 2,
    "upstairs": 3, "downstairs": 4, "run": 5
}
CLASS_NAMES = list(LABEL_MAP.keys())
CLASS_NAMES_CN = ["静坐", "站立", "走路", "上楼", "下楼", "跑步"]

# ===== v3 参数 =====
N_ESTIMATORS = 15         # 树数量 (ESP32内存限制)
MAX_DEPTH = 8             # 最大深度
MIN_SAMPLES_LEAF = 5      # 叶节点最小样本数
N_FEATURES = 51           # 30基础 + 4方向感知 + 17磁力计

# ===== 加载数据 =====
def load_data():
    X, y, weights = [], [], []
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    print(f"找到 {len(files)} 个数据文件")

    for fname in files:
        parts = fname.split("_")
        if len(parts) < 3 or parts[2] not in LABEL_MAP:
            continue
        label = LABEL_MAP[parts[2]]
        subject = parts[1]  # p0, p1, p2, uci
        df = pd.read_csv(os.path.join(DATA_DIR, fname))

        # 读取 MPU 6 列 (必须)
        mpu_cols = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
        mpu_data = df[mpu_cols].values

        # 读取磁力计 3 列 (可选, 兼容旧数据)
        mag_cols = ["mag_x","mag_y","mag_z"]
        mag_data = df[mag_cols].values if all(c in df.columns for c in mag_cols) else np.zeros((len(df), 3))

        # 权重: 自己的数据(p1) ×5, p0/p2 ×3, UCI ×1
        if subject == "p1":
            w = 5.0
        elif subject in ("p0", "p2"):
            w = 3.0
        else:
            w = 1.0

        data = np.hstack([mpu_data, mag_data])

        for i in range(0, len(data)-WINDOW_SIZE+1, STEP):
            X.append(data[i:i+WINDOW_SIZE])
            y.append(label)
            weights.append(w)

    X, y = np.array(X), np.array(y)
    weights = np.array(weights)
    print(f"加载完成：{len(X)} 个样本")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}: {(y==i).sum():4d} 样本")
    return X, y, weights

# ===== 特征提取 v3 — 51维 (34维 + 17维磁力计特征) =====
def extract_feat(windows):
    feats = []
    for w in windows:
        feat = []

        # 1. MPU 原始30维: 每轴 mean/std/max/min/ptp (索引 0-29)
        for col in range(6):
            col_data = w[:, col]
            feat.append(np.mean(col_data))
            feat.append(np.std(col_data))
            feat.append(np.max(col_data))
            feat.append(np.min(col_data))
            feat.append(np.ptp(col_data))

        # 2. 方向感知4维 (索引 30-33)
        ax, ay, az = w[:, 0], w[:, 1], w[:, 2]     # 加速度
        gx, gy, gz = w[:, 3], w[:, 4], w[:, 5]     # 陀螺仪

        # 加速度幅值 (静止≈1g, 运动时波动大)
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        feat.append(np.mean(acc_mag))                # 30: acc_mag_mean
        feat.append(np.std(acc_mag))                 # 31: acc_mag_std

        # 重力方向角度 (坐/站的核心区别)
        ax_m, ay_m, az_m = np.mean(ax), np.mean(ay), np.mean(az)
        horiz_mag = np.sqrt(ax_m**2 + ay_m**2) + 1e-10
        feat.append(np.arctan2(horiz_mag, abs(az_m)))  # 32: tilt_angle

        # 陀螺仪幅值 (静止≈0, 运动时大)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        feat.append(np.mean(gyro_mag))               # 33: gyro_mag_mean

        # 3. 磁力计17维 (索引 34-50) — v3 新增
        mx, my, mz = w[:, 6], w[:, 7], w[:, 8]     # 磁力计3轴

        # 每轴 5 统计量 (索引 34-48)
        for mag_axis in [mx, my, mz]:
            feat.append(np.mean(mag_axis))          # mean
            feat.append(np.std(mag_axis))           # std
            feat.append(np.max(mag_axis))           # max
            feat.append(np.min(mag_axis))           # min
            feat.append(np.ptp(mag_axis))           # ptp

        # 磁场幅值统计 (索引 49-50)
        mag_mag = np.sqrt(mx**2 + my**2 + mz**2)
        feat.append(np.mean(mag_mag))               # 49: mag_mag_mean
        feat.append(np.std(mag_mag))                # 50: mag_mag_std

        feats.append(feat)
    return np.array(feats)

# ===== 训练 =====
if __name__ == "__main__":
    t0 = time.time()
    X, y, sample_w = load_data()
    X_feat = extract_feat(X)
    print(f"特征矩阵: {X_feat.shape} (应为 N×{N_FEATURES})")

    # 划分 (保留样本权重)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_feat, y, sample_w, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"训练集: {len(X_train)}  测试集: {len(X_test)}")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练
    print(f"\n训练 RandomForest (n={N_ESTIMATORS}, depth={MAX_DEPTH}, leaf={MIN_SAMPLES_LEAF})...")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train, sample_weight=w_train)

    # 评估
    y_pred = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*55}")
    print(f"  测试集准确率: {acc:.4f} ({acc*100:.1f}%)")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES_CN))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(f"{'':>8}", end="")
    for name in CLASS_NAMES_CN:
        print(f"{name:>6}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES_CN):
        print(f"  {name:6s}", end="")
        for j in range(6):
            print(f"{cm[i][j]:6d}", end="")
        print()

    # 交叉验证 (带权重)
    print("\n5-fold CV 准确率: ", end="")
    try:
        cv_scores = cross_val_score(rf, scaler.fit_transform(X_feat), y, cv=5,
                                     fit_params={'sample_weight': sample_w})
        print(f"{cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    except:
        cv_scores = cross_val_score(rf, scaler.fit_transform(X_feat), y, cv=5)
        print(f"{cv_scores.mean():.4f} (无权重)")

    # 模型大小估算
    total_nodes = sum(t.tree_.node_count for t in rf.estimators_)
    print(f"\n模型规模: {N_ESTIMATORS} 棵树, 总计约 {total_nodes} 个节点")

    # 导出
    print("\n导出模型...")
    params = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "n_estimators": rf.n_estimators,
        "n_classes": rf.n_classes_,
        "classes": rf.classes_.tolist(),
        "n_features": N_FEATURES,
        "trees": []
    }
    for tree in rf.estimators_:
        t = tree.tree_
        params["trees"].append({
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "value": t.value.tolist()
        })

    out_path = "../models/rf_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)

    file_size = os.path.getsize(out_path)
    print(f"  ✅ rf_params.json 已保存 ({file_size/1024:.1f} KB)")

    # 备份旧模型
    bak = "../models/rf_params_old.json"
    if os.path.exists(bak):
        os.remove(bak)

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s")
    print(f"\n下一步: 将 rf_params.json 上传到 ESP32 替换旧模型")
