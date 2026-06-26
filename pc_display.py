#!/usr/bin/env python3
"""
HAR PC端实时显示 - 通过USB串口接收ESP32-S3推理结果
依赖: pip install pyserial
用法: python pc_display.py
      程序会自动查找ESP32串口，也可手动指定: python pc_display.py /dev/cu.usbmodemXXX
"""

import sys
import json
import time
import os

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("请先安装 pyserial: pip install pyserial")
    sys.exit(1)


# ===== 颜色配置 =====
COLORS = {
    "sit":        "\033[48;5;196m\033[38;5;15m",  # 红底白字
    "stand":      "\033[48;5;46m\033[38;5;0m",    # 绿底黑字
    "walk":       "\033[48;5;27m\033[38;5;15m",    # 蓝底白字
    "upstairs":   "\033[48;5;220m\033[38;5;0m",    # 黄底黑字
    "downstairs": "\033[48;5;201m\033[38;5;15m",   # 紫底白字
    "run":        "\033[48;5;255m\033[38;5;0m",    # 白底黑字
    "unknown":    "\033[48;5;240m\033[38;5;15m",   # 灰底白字
    "init":       "\033[48;5;240m\033[38;5;15m",
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CLEAR = "\033[2J\033[H"

EMOJI = {
    "sit": "🪑", "stand": "🧍", "walk": "🚶",
    "upstairs": "🔼", "downstairs": "🔽", "run": "🏃",
    "--": "⏳", "init": "⏳",
}

BAR_CHARS = " ▁▂▃▄▅▆▇█"


# ===== 查找ESP32串口 =====
def find_esp32_port():
    """自动查找ESP32的USB串口"""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        # ESP32常见的USB芯片: CP210x, CH340, CH343, FTDI
        vid_pid = f"{p.vid:04X}:{p.pid:04X}" if p.vid and p.pid else ""
        keywords = ["CP210", "CH340", "CH343", "FTDI", "Silicon Labs",
                    "USB Serial", "esp32", "ESP32", "wch", "WCH",
                    "Espressif", "10C4", "1A86", "303A"]
        for kw in keywords:
            if kw.lower() in p.description.lower() or kw.lower() in vid_pid.lower():
                return p.device
    # 没找到，列出所有端口让用户选
    return None


def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("未检测到任何串口设备")
        return
    print(f"\n{BOLD}可用串口:{RESET}")
    for i, p in enumerate(ports):
        vid_pid = f"{p.vid:04X}:{p.pid:04X}" if p.vid and p.pid else "????:????"
        print(f"  [{i}] {p.device}  {p.description}  ({vid_pid})")


def draw_dashboard(act, votes, history, pred_id):
    """绘制终端仪表盘"""
    print(CLEAR, end="")

    # 标题
    print(f"{BOLD}{DIM}╭──────────────────────────────────────────╮{RESET}")

    # 当前动作 - 大字显示
    color = COLORS.get(act, COLORS["unknown"])
    emoji = EMOJI.get(act, "❓")
    bar = votes_bar(votes, 15) if votes > 0 else ""

    print(f"{DIM}│{RESET}  ESP32-S3 HAR 实时检测                    {DIM}│{RESET}")
    print(f"{DIM}│{RESET}                                           {DIM}│{RESET}")
    print(f"{DIM}│{RESET}   当前动作: {color}{BOLD} {emoji} {act:12s} {RESET}{DIM}     │{RESET}")
    if bar:
        print(f"{DIM}│{RESET}   投票置信: {color}{bar}{RESET} {votes}/15    {DIM}    │{RESET}")
    print(f"{DIM}│{RESET}                                           {DIM}│{RESET}")

    # 历史动作条
    print(f"{DIM}│{RESET}   历史: ", end="")
    for h in history[-20:]:
        h_color = COLORS.get(h, COLORS["unknown"])
        short = {"sit": "坐", "stand": "站", "walk": "走",
                 "upstairs": "上", "downstairs": "下", "run": "跑",
                 "--": "·", "init": "·"}.get(h, "?")
        print(f"{h_color} {short} {RESET}", end="")
    print(f" {DIM}  │{RESET}")

    print(f"{DIM}╰──────────────────────────────────────────╯{RESET}")
    print(f"\n{DIM}按 Ctrl+C 退出{RESET}")


def votes_bar(votes, max_votes=15):
    """置信度条"""
    ratio = min(votes / max_votes, 1.0)
    n = int(ratio * 8)
    return BAR_CHARS[n]


# ===== 主函数 =====
def main():
    # 确定串口
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        print("正在查找 ESP32...")
        port = find_esp32_port()
        if port is None:
            list_ports()
            print("\n请手动指定串口: python pc_display.py <串口路径>")
            print("例如: python pc_display.py /dev/cu.usbmodem1101")
            sys.exit(1)
        print(f"找到 ESP32: {port}")

    # 连接串口（dsrdtr=False 防止自动复位ESP32）
    print(f"连接 {port} (115200 baud)...")
    ser = serial.Serial(port, 115200, timeout=5, dsrdtr=False, rtscts=False)
    time.sleep(0.3)

    # 发送 Ctrl-C 确保在 REPL，然后手动运行
    ser.write(b'\r\x03')
    time.sleep(0.2)
    ser.write(b'\r\x03')
    time.sleep(0.2)

    # 清空残留
    while ser.in_waiting:
        ser.read(ser.in_waiting)

    # 启动推理
    print("启动推理...")
    ser.write(b'import main\r\n')

    # 等待 ESP32 启动完成
    print("等待 ESP32 就绪...")
    start = time.time()
    while time.time() - start < 30:
        line = ser.readline().decode("utf-8", "ignore").strip()
        if line:
            print(f"  ESP32: {line}")
        if line == "---":
            print("\n就绪! 开始实时检测...\n")
            time.sleep(0.3)
            break
    else:
        print("超时未收到就绪信号，尝试直接读取...")

    # 实时显示循环
    history = []
    last_act = "--"
    last_votes = 0

    # 清屏绘制
    draw_dashboard("--", 0, history, -1)

    while True:
        try:
            line = ser.readline().decode("utf-8", "ignore").strip()
            if not line:
                continue

            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    act = data.get("act", last_act)
                    votes = data.get("votes", 0)
                    pred_id = data.get("pred", -1)
                    last_act = act
                    last_votes = votes
                    history.append(act)
                    if len(history) > 200:
                        history = history[-200:]
                    draw_dashboard(act, votes, history, pred_id)
                except json.JSONDecodeError:
                    pass
            else:
                # 非JSON行 — 日志信息
                print(f"  {DIM}ESP32:{RESET} {line}")

        except KeyboardInterrupt:
            print(f"\n{CLEAR}{BOLD}退出。{RESET}")
            break
        except Exception as e:
            print(f"错误: {e}")
            time.sleep(0.5)

    ser.close()


if __name__ == "__main__":
    main()
