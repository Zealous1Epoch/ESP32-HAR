# ESP32-S3 MPU6050 HAR - USB串口版
# 通过USB线直连电脑，串口输出实时识别结果
# 无需WiFi热点

import gc, math, time
from machine import SoftI2C, Pin
import neopixel
from array import array

try:
    import ujson as json
except:
    import json

# ===== 配置 =====
SDA_PIN = 8
SCL_PIN = 9
LED_PIN = 48
MPU_ADDR = 0x68

WINDOW_SIZE = 128
WINDOW_STEP = 64
SAMPLE_MS = 10      # 100Hz (原20ms/50Hz，提速一倍)
N_AXES = 6
N_FEATURES = 34  # 30基础 + 4方向感知
SMOOTH_FRAMES = 2   # 2帧平滑 (原3帧，减少延迟)

ACT_NAMES = ["sit", "stand", "walk", "upstairs", "downstairs", "run"]

LED_COLORS = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255),
    3: (255, 255, 0), 4: (255, 0, 255), 5: (255, 255, 255),
}

current_action = "--"
current_votes = 0
inference_ready = False
np = None
i2c = None


def init_led():
    global np
    if np is None:
        np = neopixel.NeoPixel(Pin(LED_PIN), 1)


def set_led(r, g, b):
    init_led()
    if np:
        np[0] = (r, g, b)
        np.write()


def init_i2c():
    global i2c
    if i2c is None:
        i2c = SoftI2C(sda=Pin(SDA_PIN), scl=Pin(SCL_PIN), freq=100000)


# ===== MPU6050 =====
def conv(v):
    return v - 65536 if v >= 32768 else v


def init_mpu():
    init_i2c()
    for i in range(5):
        try:
            i2c.writeto_mem(MPU_ADDR, 0x6B, b'\x00')
            time.sleep_ms(50)
            return True
        except OSError:
            time.sleep_ms(50)
    return False


def read_mpu_raw():
    for _ in range(5):
        try:
            buf = i2c.readfrom_mem(MPU_ADDR, 0x3B, 14)
            ax = conv((buf[0] << 8) | buf[1])
            ay = conv((buf[2] << 8) | buf[3])
            az = conv((buf[4] << 8) | buf[5])
            gx = conv((buf[8] << 8) | buf[9])
            gy = conv((buf[10] << 8) | buf[11])
            gz = conv((buf[12] << 8) | buf[13])
            if abs(ax) > 32000 or abs(ay) > 32000 or abs(az) > 32000:
                continue
            return ax, ay, az, gx, gy, gz
        except OSError:
            time.sleep_ms(10)
    return None


ax_off = ay_off = az_off = gx_off = gy_off = gz_off = 0


def calibrate(n=100):
    global ax_off, ay_off, az_off, gx_off, gy_off, gz_off
    s = [0] * 6
    v = 0
    t0 = time.ticks_ms()
    while v < n and time.ticks_diff(time.ticks_ms(), t0) < 10000:
        d = read_mpu_raw()
        if d is None:
            time.sleep_ms(10)
            continue
        for i in range(6):
            s[i] += d[i]
        v += 1
        time.sleep_ms(10)
    if v > 0:
        ax_off, ay_off, az_off, gx_off, gy_off, gz_off = [x // v for x in s]


def get_cal():
    d = read_mpu_raw()
    if d is None:
        return None
    return (d[0] - ax_off, d[1] - ay_off, d[2] - az_off,
            d[3] - gx_off, d[4] - gy_off, d[5] - gz_off)


def extract_features(buf):
    feats = array('f', [0.0] * N_FEATURES)
    # 1. 原始30维: 每轴 mean/std/max/min/ptp
    axis_means = [0.0] * N_AXES  # 暂存均值用于方向特征
    for axis in range(N_AXES):
        s = 0.0
        mn = 1e9
        mx = -1e9
        for i in range(WINDOW_SIZE):
            v = buf[i * N_AXES + axis]
            s += v
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        mean = s / WINDOW_SIZE
        axis_means[axis] = mean
        sq = 0.0
        for i in range(WINDOW_SIZE):
            d = buf[i * N_AXES + axis] - mean
            sq += d * d
        b = axis * 5
        feats[b] = mean
        feats[b + 1] = math.sqrt(sq / WINDOW_SIZE)
        feats[b + 2] = mx
        feats[b + 3] = mn
        feats[b + 4] = mx - mn

    # 2. 新增4维方向感知特征 (索引30-33)
    ax_m, ay_m, az_m = axis_means[0], axis_means[1], axis_means[2]
    gx_m, gy_m, gz_m = axis_means[3], axis_means[4], axis_means[5]

    # 加速度幅值统计
    acc_mag_sum = 0.0
    acc_mag_sq = 0.0
    gyro_mag_sum = 0.0
    for i in range(WINDOW_SIZE):
        axv = buf[i * N_AXES + 0]
        ayv = buf[i * N_AXES + 1]
        azv = buf[i * N_AXES + 2]
        gxv = buf[i * N_AXES + 3]
        gyv = buf[i * N_AXES + 4]
        gzv = buf[i * N_AXES + 5]
        acc_mag_sum += math.sqrt(axv * axv + ayv * ayv + azv * azv)
        acc_mag_sq += (axv * axv + ayv * ayv + azv * azv)
        gyro_mag_sum += math.sqrt(gxv * gxv + gyv * gyv + gzv * gzv)

    acc_mag_mean = acc_mag_sum / WINDOW_SIZE
    acc_mag_var = (acc_mag_sq / WINDOW_SIZE) - (acc_mag_mean * acc_mag_mean)
    if acc_mag_var < 0:
        acc_mag_var = 0.0

    feats[30] = acc_mag_mean                       # 加速度幅值均值
    feats[31] = math.sqrt(acc_mag_var)              # 加速度幅值标准差

    # 倾斜角 (核心! 坐 vs 站 的区别)
    horiz = math.sqrt(ax_m * ax_m + ay_m * ay_m) + 1e-10
    feats[32] = math.atan2(horiz, abs(az_m))        # 传感器相对垂直的倾斜角

    feats[33] = gyro_mag_sum / WINDOW_SIZE          # 陀螺仪幅值均值

    return feats


def rf_predict(feats, trees, sm, ss, nc):
    fs = array('f', [0.0] * N_FEATURES)
    for i in range(N_FEATURES):
        fs[i] = (feats[i] - sm[i]) / ss[i]
    votes = [0] * nc
    for tree in trees:
        node = 0
        fa = tree["feature"]
        ta = tree["threshold"]
        la = tree["children_left"]
        ra = tree["children_right"]
        va = tree["value"]
        while fa[node] != -2:
            if fs[fa[node]] <= ta[node]:
                node = la[node]
            else:
                node = ra[node]
        lv = va[node][0]
        bc = 0
        bv = lv[0]
        for c in range(1, len(lv)):
            if lv[c] > bv:
                bv = lv[c]
                bc = c
        votes[bc] += 1
    w = 0
    for c in range(1, nc):
        if votes[c] > votes[w]:
            w = c
    return w, votes[w]


# ===== 主循环 =====
def main():
    global current_action, current_votes, inference_ready

    print("ESP32-S3 HAR - USB Serial Mode")
    print("===============================")

    set_led(255, 100, 0)  # Orange = booting

    # 1. 初始化 MPU6050
    print("Init MPU6050...")
    if not init_mpu():
        print("ERROR: MPU6050 not found!")
        set_led(255, 0, 0)
        while True:
            time.sleep(1)
    print("MPU6050 OK")

    # 2. 加载模型
    print("Loading model...")
    gc.collect()
    try:
        with open("rf_params.json", "r") as f:
            params = json.load(f)
    except Exception as e:
        print("ERROR: rf_params.json not found!")
        print(e)
        while True:
            set_led(255, 0, 0)
            time.sleep(0.2)
            set_led(0, 0, 0)
            time.sleep(0.2)
    gc.collect()
    print("Model loaded: {} trees, {} classes".format(params["n_estimators"], params["n_classes"]))

    trees = params["trees"]
    sm = params["scaler_mean"]
    ss = params["scaler_scale"]
    nc = params["n_classes"]

    # 3. 校准
    print("Calibrating (hold still)...")
    for _ in range(5):
        set_led(255, 255, 0)
        time.sleep(0.1)
        set_led(0, 0, 0)
        time.sleep(0.1)
    calibrate(50)  # 50次校准 (原100次，加速启动)
    print("Calibration done")

    # 4. 推理循环
    BUF_SZ = WINDOW_SIZE * N_AXES
    buf = array('f', [0.0] * BUF_SZ)
    pos = 0
    sq = array('b', [0] * SMOOTH_FRAMES)
    sqp = 0
    last_t = time.ticks_ms()
    last_printed = ""

    inference_ready = True
    set_led(0, 10, 0)  # Dim green = ready
    print("READY")
    print("---")  # PC端以此为同步标记

    while True:
        now = time.ticks_ms()
        if time.ticks_diff(now, last_t) < SAMPLE_MS:
            time.sleep_ms(2)
            continue
        d = get_cal()
        if d is None:
            time.sleep_ms(5)
            continue
        last_t = now

        for v in d:
            buf[pos] = float(v)
            pos += 1

        if pos < BUF_SZ:
            continue

        # 提取特征 + 推理
        feats = extract_features(buf)
        try:
            pred, votes = rf_predict(feats, trees, sm, ss, nc)
        except:
            pred = -1
            votes = 0

        # 平滑滤波
        sq[sqp] = pred
        sqp = (sqp + 1) % SMOOTH_FRAMES
        same = True
        for i in range(1, SMOOTH_FRAMES):
            if sq[i] != sq[0]:
                same = False
                break

        if same and pred >= 0:
            current_action = ACT_NAMES[pred]
            current_votes = votes
            set_led(*LED_COLORS.get(pred, (255, 100, 0)))
            # 每次推理都输出（网页需要持续更新）
            print('{"act":"%s","pred":%d,"votes":%d}' % (current_action, pred, current_votes))

        # 滑动窗口
        shift = WINDOW_STEP * N_AXES
        keep = BUF_SZ - shift
        for i in range(keep):
            buf[i] = buf[i + shift]
        pos = keep


main()
