#!/usr/bin/env python3
"""
HAR 模型训练 v2 — 更强的随机森林
改动: 更多树(25)、更深(max_depth=12)、类别平衡、输出模型大小信息
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = "./dataset"
WINDOW_SIZE = 128
STEP = 64
RANDOM_SEED = 42

LABEL_MAP = {
    "sit": 0, "stand": 1, "walk": 2,
    "upstairs": 3, "downstairs": 4, "run": 5
}
CLASS_NAMES = list(LABEL_MAP.keys())
CLASS_NAMES_CN = ["静坐", "站立", "走路", "上楼", "下楼", "跑步"]

# ===== v2 参数 =====
N_ESTIMATORS = 15      # 树数量 (ESP32内存限制)
MAX_DEPTH = 8           # 最大深度 (控制模型大小)
MIN_SAMPLES_LEAF = 5    # 叶节点最小样本数 (防过拟合)

# ===== 加载数据 =====
def load_data():
    X, y = [], []
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    print(f"找到 {len(files)} 个数据文件")

    for fname in files:
        parts = fname.split("_")
        if len(parts) < 3 or parts[2] not in LABEL_MAP:
            continue
        label = LABEL_MAP[parts[2]]
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        data = df[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]].values

        for i in range(0, len(data)-WINDOW_SIZE+1, STEP):
            X.append(data[i:i+WINDOW_SIZE])
            y.append(label)

    X, y = np.array(X), np.array(y)
    print(f"加载完成：{len(X)} 个样本")
    # 打印各类别分布
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}: {(y==i).sum():4d} 样本")
    return X, y

# ===== 特征提取 v3 — 34维 (原30维 + 4维方向感知特征) =====
def extract_feat(windows):
    feats = []
    for w in windows:
        feat = []
        # 1. 原始30维: 每轴 mean/std/max/min/ptp
        for col in range(6):
            col_data = w[:,col]
            feat.append(np.mean(col_data))
            feat.append(np.std(col_data))
            feat.append(np.max(col_data))
            feat.append(np.min(col_data))
            feat.append(np.ptp(col_data))

        # 2. 新增4维方向感知特征 (区分坐/站/下楼的关键)
        ax, ay, az = w[:,0], w[:,1], w[:,2]           # 加速度
        gx, gy, gz = w[:,3], w[:,4], w[:,5]           # 陀螺仪

        # 加速度幅值 (静止≈1g, 运动时波动大)
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        feat.append(np.mean(acc_mag))                  # 31: 平均幅值
        feat.append(np.std(acc_mag))                   # 32: 幅值波动

        # 重力方向角度 (坐/站的核心区别)
        ax_m, ay_m, az_m = np.mean(ax), np.mean(ay), np.mean(az)
        horiz_mag = np.sqrt(ax_m**2 + ay_m**2) + 1e-10
        feat.append(np.arctan2(horiz_mag, abs(az_m)))  # 33: 倾斜角(相对垂直)

        # 陀螺仪幅值 (静止≈0, 运动时大)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        feat.append(np.mean(gyro_mag))                 # 34: 平均角速度

        feats.append(feat)
    return np.array(feats)

# ===== 训练 =====
if __name__ == "__main__":
    t0 = time.time()
    X, y = load_data()
    X_feat = extract_feat(X)
    print(f"特征矩阵: {X_feat.shape}")

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
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
        class_weight="balanced",  # 类别平衡
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)

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

    # 交叉验证
    print("\n5-fold CV 准确率: ", end="")
    cv_scores = cross_val_score(rf, scaler.fit_transform(X_feat), y, cv=5)
    print(f"{cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # 模型大小估算
    total_nodes = sum(t.tree_.node_count for t in rf.estimators_)
    print(f"\n模型规模: {N_ESTIMATORS} 棵树, 总计约 {total_nodes} 个节点")

    # 保存
    print("\n导出模型...")
    params = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "n_estimators": rf.n_estimators,
        "n_classes": rf.n_classes_,
        "classes": rf.classes_.tolist(),
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

    out_path = "rf_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)

    file_size = os.path.getsize(out_path)
    print(f"  ✅ rf_params.json 已保存 ({file_size/1024:.1f} KB)")

    # 备份旧模型
    bak = "rf_params_old.json"
    if os.path.exists(bak):
        os.remove(bak)

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s")
    print(f"\n下一步: 将 rf_params.json 上传到 ESP32 替换旧模型")
