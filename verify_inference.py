"""验证脚本：对比 sklearn 和 ESP32 MicroPython 推理结果"""
import os
import numpy as np
import pandas as pd
import json
import joblib

# 数据和模型路径
DATA_DIR = "./dataset"
WINDOW_SIZE = 128
STEP = 64

LABEL_MAP = {
    "sit": 0, "stand": 1, "walk": 2,
    "upstairs": 3, "downstairs": 4, "run": 5
}
CLASS_NAMES = list(LABEL_MAP.keys())


def extract_feat(window):
    """和 ESP32 端完全相同的特征提取"""
    feats = []
    for col in range(6):
        col_data = window[:, col]
        feats.append(np.mean(col_data))
        feats.append(np.std(col_data))
        feats.append(np.max(col_data))
        feats.append(np.min(col_data))
        feats.append(np.ptp(col_data))
    return np.array(feats, dtype=np.float64)


def rf_predict_mpy(features, trees, scaler_mean, scaler_scale, n_classes):
    """模拟 ESP32 MicroPython 推理逻辑"""
    # 标准化
    feat_std = []
    for i in range(len(features)):
        feat_std.append((features[i] - scaler_mean[i]) / scaler_scale[i])

    # 投票
    votes = [0] * n_classes
    for tree in trees:
        node = 0
        f_arr = tree["feature"]
        t_arr = tree["threshold"]
        l_arr = tree["children_left"]
        r_arr = tree["children_right"]
        v_arr = tree["value"]

        while f_arr[node] != -2:
            fid = f_arr[node]
            if feat_std[fid] <= t_arr[node]:
                node = l_arr[node]
            else:
                node = r_arr[node]

        leaf_vals = v_arr[node][0]
        best_c = 0
        best_v = leaf_vals[0]
        for c in range(1, len(leaf_vals)):
            if leaf_vals[c] > best_v:
                best_v = leaf_vals[c]
                best_c = c
        votes[best_c] += 1

    winner = 0
    for c in range(1, n_classes):
        if votes[c] > votes[winner]:
            winner = c

    return winner, votes[winner]


def main():
    # 1. 加载模型
    rf = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")

    with open("rf_params.json", "r") as f:
        params = json.load(f)

    trees = params["trees"]
    scaler_mean = params["scaler_mean"]
    scaler_scale = params["scaler_scale"]
    n_classes = params["n_classes"]

    # 2. 加载所有数据
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    all_windows = []
    all_labels = []

    for fname in sorted(files):
        parts = fname.split("_")
        if len(parts) < 3 or parts[2] not in LABEL_MAP:
            continue
        label = LABEL_MAP[parts[2]]
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        data = df[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]].values

        for i in range(0, len(data) - WINDOW_SIZE + 1, STEP):
            window = data[i:i+WINDOW_SIZE]
            feats = extract_feat(window)
            all_windows.append(feats)
            all_labels.append(label)

    X = np.array(all_windows)
    y = np.array(all_labels)
    print(f"Total samples: {len(X)}")

    # 3. sklearn 预测
    X_scaled = scaler.transform(X)
    sklearn_preds = rf.predict(X_scaled)
    sklearn_acc = np.mean(sklearn_preds == y)
    print(f"\nsklearn 全量准确率: {sklearn_acc:.4f}")

    # 4. MicroPython 模拟预测
    mpy_preds = []
    for i, feats in enumerate(X):
        pred, votes = rf_predict_mpy(
            feats.tolist(), trees, scaler_mean, scaler_scale, n_classes
        )
        mpy_preds.append(pred)

    mpy_preds = np.array(mpy_preds)
    mpy_acc = np.mean(mpy_preds == y)
    print(f"MicroPython 模拟准确率: {mpy_acc:.4f}")

    # 5. 逐动作对比
    print("\n=== 各类别对比 ===")
    for cls_name in CLASS_NAMES:
        cls_id = LABEL_MAP[cls_name]
        mask = y == cls_id
        if mask.sum() > 0:
            sk_acc = np.mean(sklearn_preds[mask] == y[mask])
            mp_acc = np.mean(mpy_preds[mask] == y[mask])
            match = np.mean(sklearn_preds[mask] == mpy_preds[mask])
            print(f"  {cls_name:12s} (n={mask.sum():3d}): "
                  f"sklearn={sk_acc:.3f}  mpy_sim={mp_acc:.3f}  "
                  f"agreement={match:.3f}")

    # 6. 两端一致性
    agreement = np.mean(sklearn_preds == mpy_preds)
    print(f"\n两端预测完全一致率: {agreement:.4f}")

    # 不一致的样本
    diff_idx = np.where(sklearn_preds != mpy_preds)[0]
    if len(diff_idx) > 0:
        print(f"\n不一致样本: {len(diff_idx)} / {len(X)} "
              f"({len(diff_idx)/len(X):.2%})")
        for idx in diff_idx[:5]:
            print(f"  #{idx}: sklearn={CLASS_NAMES[sklearn_preds[idx]]}, "
                  f"mpy={CLASS_NAMES[mpy_preds[idx]]}, "
                  f"true={CLASS_NAMES[y[idx]]}")

    return sklearn_acc, mpy_acc, agreement


if __name__ == "__main__":
    main()
