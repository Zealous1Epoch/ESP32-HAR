#!/usr/bin/env python3
"""
HAR 实时网页仪表盘
- 读取 ESP32 串口数据
- 启动本地 HTTP 服务器 (localhost:8080)
- 浏览器打开即可看到实时检测结果
依赖: pyserial (已安装), 无需额外 pip 安装
用法: python3 web_display.py
"""
import serial
import serial.tools.list_ports
import json
import time
import threading
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

# ===== 配置 =====
PORT_AUTO = None  # 自动查找
HTTP_PORT = 8080

# 全局状态（串口线程写入，HTTP线程读取）
state = {
    "act": "--",
    "votes": 0,
    "pred": -1,
    "history": [],
    "connected": False,
    "accuracy": "95.4%",
    "model": "RF 25trees",
}
state_lock = threading.Lock()

# ===== HTML 页面 =====
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ESP32 HAR 实时检测</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#0a0a1e;color:#e0e0e0;min-height:100vh;overflow-x:hidden}
.header{text-align:center;padding:30px 20px 10px}
.header h1{font-size:1.8em;font-weight:700;background:linear-gradient(135deg,#667eea,#764ba2);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header .sub{color:#666;font-size:0.85em;margin-top:4px}

/* 主卡片 */
.main-card{max-width:700px;margin:0 auto;padding:0 20px}
.act-display{background:rgba(255,255,255,0.03);border:2px solid rgba(255,255,255,0.08);
  border-radius:24px;padding:40px 20px;text-align:center;margin-bottom:16px;
  transition:all 0.4s ease;position:relative;overflow:hidden}
.act-display::before{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;
  background:radial-gradient(circle,transparent 60%,rgba(100,200,255,0.03) 100%);
  transition:opacity 0.4s;opacity:0}
.act-display.active::before{opacity:1}

.act-emoji{font-size:5em;line-height:1.2;transition:transform 0.3s}
.act-display.active .act-emoji{animation:bounce 0.6s ease}
@keyframes bounce{0%,100%{transform:scale(1)}50%{transform:scale(1.15)}}
.act-name{font-size:2.2em;font-weight:800;margin:8px 0;transition:color 0.3s}
.act-en{font-size:0.8em;color:#555;text-transform:uppercase;letter-spacing:3px}
.confidence{margin-top:16px;display:flex;align-items:center;justify-content:center;gap:10px}
.conf-label{font-size:0.8em;color:#666}
.conf-bar-outer{width:200px;height:8px;background:rgba(255,255,255,0.08);border-radius:4px;overflow:hidden}
.conf-bar-inner{height:100%;border-radius:4px;transition:width 0.4s,background 0.4s}
.conf-num{font-size:0.9em;font-weight:700;min-width:50px}

/* 6 个动作卡片 */
.act-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;max-width:700px;
  margin:0 auto 16px;padding:0 20px}
.act-card{background:rgba(255,255,255,0.03);border:2px solid rgba(255,255,255,0.06);
  border-radius:16px;padding:18px 10px;text-align:center;transition:all 0.3s;
  cursor:default}
.act-card.highlight{transform:scale(1.05);border-color:var(--color);box-shadow:0 0 30px var(--glow)}
.act-card .emoji{font-size:2em}
.act-card .name{font-size:0.85em;font-weight:600;margin-top:4px}
.act-card .bar-container{height:4px;background:rgba(255,255,255,0.06);border-radius:2px;
  margin-top:8px;overflow:hidden}
.act-card .bar-fill{height:100%;border-radius:2px;transition:width 0.4s;background:var(--color)}

/* 历史 */
.history-bar{max-width:700px;margin:0 auto;padding:0 20px 30px}
.history-bar h3{font-size:0.8em;color:#555;margin-bottom:8px}
.history-scroll{display:flex;gap:8px;flex-wrap:wrap}
.history-dot{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:0.7em;font-weight:700;animation:fadeIn 0.3s}
@keyframes fadeIn{from{opacity:0;transform:scale(0.5)}to{opacity:1;transform:scale(1)}}

/* 底部状态栏 */
.status{text-align:center;padding:10px;font-size:0.75em;color:#444}
.status .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.status .dot.on{background:#00ff88;box-shadow:0 0 8px #00ff88}
.status .dot.off{background:#ff4444}

/* 响应式 */
@media(max-width:500px){
  .act-grid{grid-template-columns:repeat(2,1fr)}
  .act-emoji{font-size:3.5em}
  .act-name{font-size:1.6em}
}
</style>
</head>
<body>

<div class="header">
  <h1>⚡ ESP32-S3 HAR 实时检测</h1>
  <div class="sub">MPU6050 · 随机森林 25trees · 6类活动</div>
</div>

<div class="main-card">
  <div class="act-display" id="mainDisplay">
    <div class="act-emoji" id="actEmoji">⏳</div>
    <div class="act-name" id="actName">--</div>
    <div class="act-en" id="actEn">等待数据...</div>
    <div class="confidence">
      <span class="conf-label">置信度</span>
      <div class="conf-bar-outer"><div class="conf-bar-inner" id="confBar" style="width:0%"></div></div>
      <span class="conf-num" id="confNum">--</span>
    </div>
  </div>
</div>

<div class="act-grid" id="actGrid"></div>

<div class="history-bar">
  <h3>📋 活动历史</h3>
  <div class="history-scroll" id="historyScroll"></div>
</div>

<div class="status">
  <span class="dot" id="statusDot"></span>
  <span id="statusText">连接中...</span>
  &nbsp;|&nbsp; 模型: RF 25trees &nbsp;|&nbsp; 准确率: 95.4%
</div>

<script>
// ===== 配置 =====
const ACTS = [
  {id:0, key:'sit',        name:'静坐',   emoji:'🪑', color:'#FF6B6B', glow:'rgba(255,107,107,0.3)'},
  {id:1, key:'stand',      name:'站立',   emoji:'🧍', color:'#4ECDC4', glow:'rgba(78,205,196,0.3)'},
  {id:2, key:'walk',       name:'走路',   emoji:'🚶', color:'#45B7D1', glow:'rgba(69,183,209,0.3)'},
  {id:3, key:'upstairs',   name:'上楼',   emoji:'🔼', color:'#FFEAA7', glow:'rgba(255,234,167,0.3)'},
  {id:4, key:'downstairs', name:'下楼',   emoji:'🔽', color:'#DDA0DD', glow:'rgba(221,160,221,0.3)'},
  {id:5, key:'run',        name:'跑步',   emoji:'🏃', color:'#FF8C42', glow:'rgba(255,140,66,0.4)'},
];
const UNKNOWN = {id:-1, key:'--', name:'--', emoji:'⏳', color:'#555', glow:'rgba(85,85,85,0.2)'};

const TOTAL_TREES = 25;  // 模型树数量
let currentAct = '--';
let historyData = [];

// ===== 构建网格 =====
const grid = document.getElementById('actGrid');
ACTS.forEach(a => {
  const card = document.createElement('div');
  card.className = 'act-card';
  card.id = 'card-' + a.key;
  card.style.setProperty('--color', a.color);
  card.style.setProperty('--glow', a.glow);
  card.innerHTML = `<div class="emoji">${a.emoji}</div>
    <div class="name">${a.name}</div>
    <div class="bar-container"><div class="bar-fill" id="bar-${a.key}" style="width:0%"></div></div>`;
  grid.appendChild(card);
});

// ===== 获取数据 =====
async function poll() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();
    update(d);
    document.getElementById('statusDot').className = 'dot on';
    document.getElementById('statusText').textContent = 'ESP32 已连接';
  } catch(e) {
    document.getElementById('statusDot').className = 'dot off';
    document.getElementById('statusText').textContent = '等待 ESP32...';
  }
}

// ===== 更新 UI =====
function update(d) {
  if (!d || d.act === undefined) return;

  const act = d.act;
  const votes = d.votes || 0;
  const pred = d.pred !== undefined ? d.pred : -1;
  const info = ACTS.find(a => a.key === act) || UNKNOWN;

  // 主显示区
  const display = document.getElementById('mainDisplay');
  const changed = act !== currentAct;
  currentAct = act;

  if (changed) {
    display.className = 'act-display active';
    setTimeout(() => display.className = 'act-display', 600);
  }

  document.getElementById('actEmoji').textContent = info.emoji;
  document.getElementById('actName').textContent = info.name;
  document.getElementById('actName').style.color = info.color;
  document.getElementById('actEn').textContent = act;

  // 置信度条
  const pct = votes / TOTAL_TREES * 100;
  const bar = document.getElementById('confBar');
  bar.style.width = pct + '%';
  bar.style.background = pct > 60 ? info.color : '#ff4444';
  document.getElementById('confNum').textContent = votes + '/' + TOTAL_TREES;

  // 动作卡片高亮
  ACTS.forEach(a => {
    const card = document.getElementById('card-' + a.key);
    const barEl = document.getElementById('bar-' + a.key);
    if (a.key === act) {
      card.classList.add('highlight');
    } else {
      card.classList.remove('highlight');
    }
    // 投票分布条
    if (d.all_votes && d.all_votes[a.id] !== undefined) {
      barEl.style.width = (d.all_votes[a.id] / TOTAL_TREES * 100) + '%';
    } else if (a.key === act) {
      barEl.style.width = pct + '%';
    } else {
      barEl.style.width = '0%';
    }
  });

  // 历史
  if (changed && act !== '--') {
    historyData.unshift({act, votes, time: new Date().toLocaleTimeString()});
    if (historyData.length > 50) historyData.length = 50;
    renderHistory();
  }
}

function renderHistory() {
  const container = document.getElementById('historyScroll');
  container.innerHTML = historyData.slice(0, 20).map(h => {
    const info = ACTS.find(a => a.key === h.act) || UNKNOWN;
    return `<div class="history-dot" style="background:${info.color};color:#fff"
      title="${info.name} ${h.votes}/${TOTAL_TREES} @ ${h.time}">${info.emoji}</div>`;
  }).join('');
}

// ===== 轮询 =====
setInterval(poll, 300);
poll();
</script>
</body>
</html>"""


# ===== HTTP 服务器 =====
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # 静默日志

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/api/state":
            with state_lock:
                data = json.dumps(state, ensure_ascii=False)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


# ===== 串口读取线程 =====
def serial_reader(port):
    global state
    while True:
        ser = None
        try:
            ser = serial.Serial(port, 115200, timeout=1, dsrdtr=False, rtscts=False)
            time.sleep(0.3)

            with state_lock:
                state["connected"] = True
            print("✅ 串口已连接")

            while True:
                try:
                    raw = ser.readline()
                    if not raw:
                        continue
                    # 尝试多种方式解码（ESP32 串口偶尔有乱码）
                    try:
                        line = raw.decode("utf-8").strip()
                    except:
                        line = raw.decode("utf-8", "ignore").strip()

                    # 找到 JSON 部分
                    if "{" in line and "}" in line:
                        start = line.index("{")
                        end = line.rindex("}") + 1
                        json_str = line[start:end]
                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError:
                            continue  # 损坏的JSON，跳过

                        with state_lock:
                            state["act"] = data.get("act", state["act"])
                            state["votes"] = data.get("votes", 0)
                            state["pred"] = data.get("pred", -1)
                            if state["act"] != "--":
                                hist = state["history"]
                                if not hist or hist[-1]["act"] != state["act"]:
                                    hist.append({
                                        "act": state["act"],
                                        "votes": state["votes"],
                                        "time": time.strftime("%H:%M:%S"),
                                    })
                                if len(hist) > 200:
                                    state["history"] = hist[-200:]
                except (serial.SerialException, OSError, UnicodeDecodeError):
                    break

        except Exception as e:
            print(f"串口错误: {e}")
        finally:
            if ser:
                try: ser.close()
                except: pass

        with state_lock:
            state["connected"] = False
        print("🔄 串口断开，3秒后重连...")
        time.sleep(3)


# ===== 主入口 =====
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
    # 确定串口
    port = PORT_AUTO
    if port is None:
        print("🔍 查找 ESP32...")
        port = find_esp32()
        if port is None:
            ports = list(serial.tools.list_ports.comports())
            if ports:
                print("可用串口:")
                for p in ports:
                    print(f"  {p.device}  {p.description}")
                print(f"\n请指定: python3 web_display.py <串口>")
            else:
                print("未检测到串口")
            sys.exit(1)
    elif len(sys.argv) > 1:
        port = sys.argv[1]

    print(f"✅ 串口: {port}")

    # 启动串口线程
    print("📡 启动 ESP32 推理...")
    t = threading.Thread(target=serial_reader, args=(port,), daemon=True)
    t.start()

    # 等待就绪
    print("⏳ 等待 ESP32 就绪...")
    for _ in range(80):  # 最多等 8 秒
        with state_lock:
            if state["connected"]:
                break
        time.sleep(0.1)

    # 启动 HTTP 服务器
    print(f"\n{'='*50}")
    print(f"🌐 网页已启动!")
    print(f"   打开浏览器访问: http://localhost:{HTTP_PORT}")
    print(f"   按 Ctrl+C 退出")
    print(f"{'='*50}\n")

    server = HTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 退出")
        server.shutdown()


if __name__ == "__main__":
    main()
