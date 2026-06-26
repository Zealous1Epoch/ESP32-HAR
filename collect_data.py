# ESP32 数据采集脚本 — 通过串口命令控制
# 用法: 上传到ESP32, 打开串口, 输入动作名开始采集
# 每个动作采集30秒, 自动保存为CSV

import gc, math, time
from machine import SoftI2C, Pin
from array import array

try:
    import neopixel
    HAS_LED = True
except:
    HAS_LED = False

# ===== 配置 =====
SDA_PIN = 8
SCL_PIN = 9
LED_PIN = 48
MPU_ADDR = 0x68
SAMPLE_MS = 10          # 100Hz
DURATION_S = 30         # 每段30秒
TOTAL_SAMPLES = DURATION_S * (1000 // SAMPLE_MS)  # 3000

ACT_NAMES = ["sit", "stand", "walk", "upstairs", "downstairs", "run"]

np = None
i2c = None

def init_led():
    global np
    if HAS_LED and np is None:
        np = neopixel.NeoPixel(Pin(LED_PIN), 1)

def set_led(r, g, b):
    init_led()
    if np:
        np[0] = (r, g, b)
        np.write()

def beep(n=1, ms=100):
    for _ in range(n):
        set_led(255, 255, 255)
        time.sleep_ms(ms)
        set_led(0, 0, 0)
        time.sleep_ms(ms)

def init_i2c():
    global i2c
    if i2c is None:
        i2c = SoftI2C(sda=Pin(SDA_PIN), scl=Pin(SCL_PIN), freq=400000)

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

def get_next_seq(action):
    """找到该动作的下一个序号"""
    existing = []
    try:
        for f in __import__('os').listdir():
            if f.startswith("data_p1_" + action) and f.endswith(".csv"):
                num = f.replace("data_p1_" + action + "_", "").replace(".csv", "")
                try:
                    existing.append(int(num))
                except:
                    pass
    except:
        pass
    if not existing:
        return 1
    return max(existing) + 1

def collect(action):
    """采集指定动作的数据"""
    seq = get_next_seq(action)
    fname = f"data_p1_{action}_{seq:03d}.csv"
    n = TOTAL_SAMPLES

    print(f"\n📣 准备采集: {action} (#{seq})")
    print(f"   时长: {DURATION_S}s | 采样率: {1000//SAMPLE_MS}Hz | 样本数: {n}")
    print(f"   文件名: {fname}")

    # 3-2-1 倒计时
    for count in [3, 2, 1]:
        print(f"   {count}...")
        set_led(255, 255, 0)
        time.sleep(0.5)
        set_led(0, 0, 0)
        time.sleep(0.5)

    # 开始采集
    print(f"   🔴 开始! 请做「{action}」动作...")
    set_led(255, 0, 0)  # 红色=采集中

    data = []
    dropped = 0
    t0 = time.ticks_ms()
    last_t = t0

    while len(data) < n:
        now = time.ticks_ms()
        if time.ticks_diff(now, last_t) < SAMPLE_MS:
            time.sleep_ms(1)
            continue

        d = read_mpu_raw()
        if d is None:
            dropped += 1
            time.sleep_ms(5)
            continue

        last_t = now
        ax, ay, az, gx, gy, gz = d
        data.append(f"{ax},{ay},{az},{gx},{gy},{gz}")

        # 每秒闪一次LED
        if len(data) % (1000 // SAMPLE_MS) == 0:
            elapsed = time.ticks_diff(now, t0) / 1000
            set_led(0, 0, 255)
            time.sleep_ms(20)
            set_led(255, 0, 0)
            if len(data) % (5 * (1000 // SAMPLE_MS)) == 0:
                print(f"   进度: {len(data)}/{n} ({elapsed:.0f}s)")

    elapsed = time.ticks_diff(time.ticks_ms(), t0) / 1000

    # 保存
    set_led(0, 255, 0)  # 绿色=保存中
    print(f"   采集完成! 实际耗时: {elapsed:.1f}s | 丢弃: {dropped}")

    try:
        with open(fname, "w") as f:
            f.write("acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
            f.write("\n".join(data) + "\n")
        size = __import__('os').stat(fname)[6]
        print(f"   ✅ 已保存: {fname} ({size} bytes)")
        beep(2, 50)
    except Exception as e:
        print(f"   ❌ 保存失败: {e}")
        beep(5, 100)
        return None

    set_led(0, 10, 0)  # 暗绿=就绪
    gc.collect()
    print(f"   内存剩余: {gc.mem_free()} bytes")
    return fname

# ===== 主循环 =====
def main():
    print("\n" + "=" * 45)
    print("  ESP32 MPU6050 数据采集工具")
    print("=" * 45)
    print(f"  采样率: {1000//SAMPLE_MS}Hz | 时长: {DURATION_S}s")
    print(f"  动作: {', '.join(ACT_NAMES)}")
    print("-" * 45)
    print("  输入动作名 + 回车 = 开始采集")
    print("  list = 查看文件 | exit = 退出")
    print("=" * 45)

    # 初始化
    print("\n初始化 MPU6050...")
    if not init_mpu():
        print("❌ MPU6050 未找到! 检查接线")
        while True:
            set_led(255, 0, 0)
            time.sleep(0.5)
    print("✅ MPU6050 OK")
    set_led(0, 10, 0)
    gc.collect()

    # 显示已有文件
    try:
        files = [f for f in __import__('os').listdir() if f.endswith('.csv')]
        if files:
            print(f"\n已有 {len(files)} 个CSV文件")
    except:
        pass

    print("\n就绪。输入动作名开始:\n")

    import sys
    while True:
        try:
            cmd = sys.stdin.readline().strip().lower()
        except:
            time.sleep(1)
            continue

        if cmd in ACT_NAMES:
            collect(cmd)
        elif cmd == "list":
            try:
                files = sorted([f for f in __import__('os').listdir() if f.endswith('.csv')])
                print(f"\n{len(files)} 个文件:")
                for f in files:
                    print(f"  {f}")
            except Exception as e:
                print(f"读取失败: {e}")
        elif cmd == "reboot":
            print("重启...")
            __import__('machine').reset()
        elif cmd == "exit":
            print("退出采集。上传 run main.py 推理")
            break
        elif cmd in ("help", "?"):
            print(f"命令: {', '.join(ACT_NAMES)} | list | exit | reboot")
        elif cmd:
            print(f"? 未知: {cmd} (试试: {', '.join(ACT_NAMES)})")

        print()  # 空行


# ===== 入口 =====
if __name__ == "__main__":
    main()
else:
    main()
