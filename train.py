import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

# ===================== 参数和采集端完全一致 =====================
DATA_DIR = "./dataset"
WINDOW_SIZE = 128
STEP = 64
RANDOM_SEED = 42

# 动作标签和文件名完全匹配
LABEL_MAP = {
    "sit": 0,
    "stand": 1,
    "walk": 2,
    "upstairs": 3,
    "downstairs": 4,
    "run": 5
}
CLASS_NAMES = list(LABEL_MAP.keys())

# ===================== 1. 加载数据集 =====================
def load_data():
    X, y = [], []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
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
    print(f"加载完成：共 {len(X)} 个样本")
    return X, y

# ===================== 2. 特征提取（和ESP32端完全一致） =====================
def extract_feat(windows):
    feats = []
    for w in windows:
        feat = []
        for col in range(6):
            col_data = w[:,col]
            feat.append(np.mean(col_data))
            feat.append(np.std(col_data))
            feat.append(np.max(col_data))
            feat.append(np.min(col_data))
            feat.append(np.ptp(col_data))
        feats.append(feat)
    return np.array(feats)

# ===================== 3. 训练与导出 =====================
if __name__ == "__main__":
    X, y = load_data()
    X_feat = extract_feat(X)

    # 划分数据集 + 标准化
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练随机森林（15棵树，max_depth=8，ESP32内存极致优化）
    rf = RandomForestClassifier(n_estimators=15, max_depth=8, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    
    # 打印精度
    y_pred = rf.predict(X_test)
    print(f"\n模型准确率：{accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # 保存PC端模型
    joblib.dump(rf, "rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    # 导出ESP32用的JSON参数
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
    
    with open("rf_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)
    
    print("\n[OK] 训练完成，已生成 rf_params.json")