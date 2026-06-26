#!/usr/bin/env python3
"""
PC端数据采集交互终端
用法: python3 collect_pc.py
连接ESP32 → 启动采集器 → 输入动作名开始30秒采集
"""
import serial
import serial.tools.list_ports
import time
import threading
import sys
import os

def find_esp32():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        keywords = ["CP210", "CH340", "CH343", "FTDI", "Silicon Labs",
                    "Espressif", "USB Serial", "esp32", "ESP32",
                    "wch", "WCH", "10C4", "1A86", "303A"]
        desc = p.description.lower()
        vid_pid = f"{p.vid:04X}:{p.pid:04X}" if p.vid and p.pid else ""
        for kw in keywords:
            if kw.lower() in desc or kw.lower() in vid_pid.lower():
                return p.device
    return None

def main():
    # 查找串口
    port = None
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        print("🔍 查找 ESP32...")
        port = find_esp32()
        if port is None:
            ports = list(serial.tools.list_ports.comports())
            if ports:
                print("可用串口:")
                for p in ports:
                    print(f"  {p.device}  {p.description}")
            else:
                print("未检测到串口!")
            sys.exit(1)

    print(f"✅ 连接 {port}...")
    ser = serial.Serial(port, 115200, timeout=0.5)
    time.sleep(0.5)

    # 清缓冲区并启动采集器
    while ser.in_waiting:
        ser.read(ser.in_waiting)

    # 如果ESP32在REPL，发送import
    ser.write(b'\r\x03')  # Ctrl-C 中断可能的东西
    time.sleep(0.2)
    ser.write(b'\r\n')
    time.sleep(0.2)
    while ser.in_waiting:
        ser.read(ser.in_waiting)

    print("📡 启动采集器...")
    ser.write(b"import collect_data\r\n")
    time.sleep(3)

    # 打印ESP32输出
    while ser.in_waiting:
        data = ser.read(ser.in_waiting).decode('utf-8', errors='replace')
        print(data, end='', flush=True)

    # 后台线程读取ESP32输出
    stop_flag = threading.Event()

    def reader():
        buf = b""
        while not stop_flag.is_set():
            try:
                if ser.in_waiting:
                    buf += ser.read(ser.in_waiting)
                    if b'\n' in buf:
                        lines = buf.split(b'\n')
                        for line in lines[:-1]:
                            text = line.decode('utf-8', errors='replace')
                            # 不重复打印用户输入
                            if text.strip() and not text.startswith('>'):
                                print(text, flush=True)
                        buf = lines[-1]
            except:
                pass
            time.sleep(0.05)

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    print()
    print("=" * 55)
    print("  📣 ESP32 数据采集器已就绪!")
    print("  输入动作名 → 3-2-1倒计时 → 30秒采集")
    print()
    print("  命令:")
    print("    sit       — 采集静坐数据 (30s)")
    print("    stand     — 采集站立数据 (30s)")
    print("    walk      — 采集走路数据 (30s)")
    print("    upstairs  — 采集上楼数据 (30s)")
    print("    downstairs— 采集下楼数据 (30s)")
    print("    run       — 采集跑步数据 (30s)")
    print("    list      — 查看已采集的文件")
    print("    exit      — 退出采集")
    print("=" * 55)
    print()

    # 主循环: 读取用户输入
    try:
        for line in sys.stdin:
            cmd = line.strip()
            if cmd == 'exit':
                print("👋 退出采集器")
                break
            if cmd:
                ser.write((cmd + '\r\n').encode())
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n👋 退出")

    stop_flag.set()
    ser.close()

    print()
    print("下一步:")
    print("  1. 用 mpremote 把ESP32上的CSV下载到 dataset/")
    print("  2. python3 train_v2.py  重新训练 (51维)")
    print("  3. 上传 rf_params.json 到ESP32")
    print("  4. 上传 main_serial.py 或 main_wifi.py (51维推理)")


if __name__ == "__main__":
    main()
