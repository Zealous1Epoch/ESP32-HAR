# HMC5883L 磁力计测试
from machine import SoftI2C, Pin
import time

i2c = SoftI2C(sda=Pin(8), scl=Pin(9), freq=100000)

# 初始化 HMC5883L
i2c.writeto_mem(0x1E, 0x00, b'\x70')  # Config A: 8-average, 15Hz
i2c.writeto_mem(0x1E, 0x02, b'\x00')  # Mode: continuous
time.sleep_ms(10)

def read_mag():
    data = i2c.readfrom_mem(0x1E, 0x03, 6)
    mx = (data[0]<<8) | data[1]
    if mx >= 32768: mx -= 65536
    mz = (data[2]<<8) | data[3]
    if mz >= 32768: mz -= 65536
    my = (data[4]<<8) | data[5]
    if my >= 32768: my -= 65536
    return mx, my, mz

print("HMC5883L 磁力计测试")
print("旋转传感器，观察数值变化...")
print()

for i in range(15):
    mx, my, mz = read_mag()
    mag = (mx*mx + my*my + mz*mz) ** 0.5
    # 计算航向角 (假设传感器水平)
    import math
    heading = math.atan2(my, mx) * 180 / math.pi
    if heading < 0: heading += 360
    print(f"  X={mx:6d} Y={my:6d} Z={mz:6d}  |mag|={mag:.0f}  heading={heading:.0f}°")
    time.sleep(0.5)

print()
print("磁力计正常!" if mag > 100 else "磁力计读数异常!")
