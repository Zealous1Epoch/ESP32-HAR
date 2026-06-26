# ESP32 推理验证测试 - 用6类动作的特征向量验证模型
import gc, math, json, time
try:
    import ujson as json
except:
    pass

N_FEATURES = 30

# ===== 6类动作的预计算特征向量 (来自PC端真实数据) =====
f_sit = [-572.375, 55.54151, -455.0, -711.0, 256.0, 5562.469, 89.6195, 5890.0, 5370.0, 520.0, 1963.688, 147.7118, 2375.0, 1507.0, 868.0, -4.234375, 62.92785, 182.0, -185.0, 367.0, 11.64844, 148.4261, 395.0, -389.0, 784.0, 5.710938, 90.43214, 238.0, -238.0, 476.0]

f_stand = [265.8125, 278.4499, 945.0, -407.0, 1352.0, 2196.844, 467.6834, 3274.0, 1138.0, 2136.0, 2091.844, 548.5088, 3095.0, -201.0, 3296.0, -29.02344, 472.7708, 1097.0, -977.0, 2074.0, 32.09375, 683.5062, 1720.0, -2883.0, 4603.0, -144.9688, 399.4188, 680.0, -1632.0, 2312.0]

f_walk = [-8333.938, 1769.492, -4955.0, -11143.0, 6188.0, 14845.13, 1749.083, 20350.0, 11262.0, 9088.0, 5604.188, 1561.86, 9775.0, 1891.0, 7884.0, -1290.688, 4007.196, 4210.0, -10681.0, 14891.0, -1289.734, 5945.732, 7951.0, -15946.0, 23897.0, -846.8047, 6162.675, 10544.0, -9108.0, 19652.0]

f_upstairs = [-3317.25, 3210.217, 5485.0, -7587.0, 13072.0, 10702.38, 2888.371, 19954.0, 5238.0, 14716.0, 6342.875, 2207.032, 11987.0, 2427.0, 9560.0, 582.1406, 3214.036, 6948.0, -5158.0, 12106.0, -924.8672, 5654.043, 8857.0, -14302.0, 23159.0, -1570.906, 6624.316, 8534.0, -13334.0, 21868.0]

f_downstairs = [-799.7188, 3676.193, 6449.0, -6271.0, 12720.0, 6400.0, 2169.986, 11970.0, 2334.0, 9636.0, 5054.844, 1939.507, 9691.0, 243.0, 9448.0, 657.125, 2790.468, 6535.0, -5899.0, 12434.0, 930.6094, 5353.237, 12719.0, -10449.0, 23168.0, 668.7969, 5844.671, 8547.0, -10552.0, 19099.0]

f_run = [-4136.281, 5230.484, 10304.0, -15684.0, 25988.0, 6771.125, 5241.238, 22263.0, -5993.0, 28256.0, 13676.19, 7666.309, 35619.0, -2505.0, 38124.0, -69.14844, 14017.19, 22410.0, -30330.0, 52740.0, 965.4219, 11488.2, 27953.0, -21650.0, 49603.0, -869.7813, 6043.05, 16926.0, -17869.0, 34795.0]

tests = [
    (0, "sit", f_sit),
    (1, "stand", f_stand),
    (2, "walk", f_walk),
    (3, "upstairs", f_upstairs),
    (4, "downstairs", f_downstairs),
    (5, "run", f_run),
]

# ===== 推理函数 (和main.py完全一致) =====
def rf_predict(features, trees, scaler_mean, scaler_scale, n_classes):
    feat_std = [0.0] * N_FEATURES
    for i in range(N_FEATURES):
        feat_std[i] = (features[i] - scaler_mean[i]) / scaler_scale[i]

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

# ===== 主测试 =====
print("=" * 40)
print("ESP32 RF Inference Test")
print("=" * 40)

# 加载模型
gc.collect()
print(f"Free mem: {gc.mem_free()}")
with open("rf_params.json", "r") as f:
    params = json.load(f)
gc.collect()
print(f"Model loaded: {params['n_estimators']} trees, {params['n_classes']} classes")
print(f"Free mem: {gc.mem_free()}")

trees = params["trees"]
scaler_mean = params["scaler_mean"]
scaler_scale = params["scaler_scale"]
n_classes = params["n_classes"]

# 逐条测试
correct = 0
print("\n--- Results ---")
for expected_id, name, feats in tests:
    pred, votes = rf_predict(feats, trees, scaler_mean, scaler_scale, n_classes)
    ok = "OK" if pred == expected_id else "FAIL"
    if pred == expected_id:
        correct += 1
    print(f"{ok}  {name:12s}  expected={expected_id}  predicted={pred}  votes={votes}")

print(f"\nAccuracy: {correct}/{len(tests)} = {correct/len(tests):.1%}")
print("=" * 40)
