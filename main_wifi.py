# ESP32-S3 MPU6050 HAR - WiFi热点版
# ESP32自建热点(ESP32-HAR) → 手机连上 → 浏览器打开192.168.4.1
# 无需USB数据线

import gc, math, time, network, socket
from machine import SoftI2C, Pin
from array import array

try:
    import ujson as json
except:
    import json

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

WINDOW_SIZE = 128
WINDOW_STEP = 64
SAMPLE_MS = 10
N_AXES = 6
N_FEATURES = 34
SMOOTH_FRAMES = 2

ACT_NAMES = ["sit", "stand", "walk", "upstairs", "downstairs", "run"]

LED_COLORS = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255),
    3: (255, 255, 0), 4: (255, 0, 255), 5: (255, 255, 255),
}

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

ax_off = ay_off = az_off = gx_off = gy_off = gz_off = 0

def calibrate(n=50):
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
    axis_means = [0.0] * N_AXES
    for axis in range(N_AXES):
        s = 0.0
        mn = 1e9
        mx = -1e9
        for i in range(WINDOW_SIZE):
            v = buf[i * N_AXES + axis]
            s += v
            if v < mn: mn = v
            if v > mx: mx = v
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

    ax_m, ay_m, az_m = axis_means[0], axis_means[1], axis_means[2]
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
        m = axv * axv + ayv * ayv + azv * azv
        acc_mag_sum += math.sqrt(m)
        acc_mag_sq += m
        gyro_mag_sum += math.sqrt(gxv * gxv + gyv * gyv + gzv * gzv)

    acc_mag_mean = acc_mag_sum / WINDOW_SIZE
    acc_mag_var = (acc_mag_sq / WINDOW_SIZE) - (acc_mag_mean * acc_mag_mean)
    if acc_mag_var < 0: acc_mag_var = 0.0
    feats[30] = acc_mag_mean
    feats[31] = math.sqrt(acc_mag_var)
    horiz = math.sqrt(ax_m * ax_m + ay_m * ay_m) + 1e-10
    feats[32] = math.atan2(horiz, abs(az_m))
    feats[33] = gyro_mag_sum / WINDOW_SIZE
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

# ===== 网页 (优化版仪表盘) =====
HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0,user-scalable=no">
<title>ESP32 HAR</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,sans-serif;background:#0a0a1e;color:#e0e0e0;min-height:100vh;padding:10px}
h1{text-align:center;font-size:1.3em;padding:15px 0 5px;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{text-align:center;color:#555;font-size:.7em;margin-bottom:12px}

.main{background:rgba(255,255,255,.03);border:2px solid rgba(255,255,255,.08);border-radius:20px;padding:25px 15px;text-align:center;margin-bottom:10px;transition:all .4s}
.main.active{border-color:#00ff88;box-shadow:0 0 30px rgba(0,255,136,.12)}
.emoji{font-size:4em;line-height:1.2}
.name{font-size:1.8em;font-weight:800;margin:5px 0}
.name_cn{font-size:.7em;color:#555;text-transform:uppercase;letter-spacing:2px}
.bar-outer{width:80%;max-width:250px;height:6px;background:rgba(255,255,255,.08);border-radius:3px;margin:10px auto;overflow:hidden}
.bar-inner{height:100%;border-radius:3px;transition:width .4s}
.votes{font-size:.8em;color:#666}

.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px}
.card{background:rgba(255,255,255,.03);border:2px solid rgba(255,255,255,.06);border-radius:14px;padding:14px 8px;text-align:center;transition:all .3s}
.card.hl{transform:scale(1.05)}
.card .e{font-size:1.6em}
.card .n{font-size:.75em;font-weight:600;margin-top:2px}
.card .b{height:3px;background:rgba(255,255,255,.06);border-radius:2px;margin-top:5px;overflow:hidden}
.card .bf{height:100%;border-radius:2px;transition:width .4s}

.hist{padding:5px 0}
.hist h3{font-size:.7em;color:#555;margin-bottom:6px}
.hist-dots{display:flex;gap:6px;flex-wrap:wrap}
.dot{width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.6em;font-weight:700}

.ft{text-align:center;padding:8px;font-size:.65em;color:#444}
.ft .d{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:5px;background:#00ff88;box-shadow:0 0 6px #00ff88}
</style></head><body>

<h1>⚡ ESP32 HAR 实时检测</h1>
<div class="sub">MPU6050 · RF 15trees · 6类活动</div>

<div class="main" id="main">
  <div class="emoji" id="em">⏳</div>
  <div class="name" id="nm">--</div>
  <div class="name_cn" id="en">等待数据</div>
  <div class="bar-outer"><div class="bar-inner" id="bar" style="width:0;background:#555"></div></div>
  <div class="votes" id="vt">--</div>
</div>

<div class="grid" id="grid"></div>

<div class="hist">
  <h3>📋 活动历史</h3>
  <div class="hist-dots" id="hist"></div>
</div>

<div class="ft"><span class="d"></span> ESP32 已连接 | 模型: RF 15trees</div>

<script>
var TOTAL=15;
var ACTS=[
  {k:'sit',n:'静坐',e:'🪑',c:'#FF6B6B',g:'rgba(255,107,107,.3)'},
  {k:'stand',n:'站立',e:'🧍',c:'#4ECDC4',g:'rgba(78,205,196,.3)'},
  {k:'walk',n:'走路',e:'🚶',c:'#45B7D1',g:'rgba(69,183,209,.3)'},
  {k:'upstairs',n:'上楼',e:'🔼',c:'#FFEAA7',g:'rgba(255,234,167,.3)'},
  {k:'downstairs',n:'下楼',e:'🔽',c:'#DDA0DD',g:'rgba(221,160,221,.3)'},
  {k:'run',n:'跑步',e:'🏃',c:'#FF8C42',g:'rgba(255,140,66,.4)'}
];

var grid=document.getElementById('grid');
ACTS.forEach(function(a){
  var d=document.createElement('div');
  d.className='card';d.id='c'+a.k;
  d.style.setProperty('--c',a.c);d.style.setProperty('--g',a.g);
  d.innerHTML='<div class="e">'+a.e+'</div><div class="n">'+a.n+'</div><div class="b"><div class="bf" id="b'+a.k+'" style="width:0;background:'+a.c+'"></div></div>';
  grid.appendChild(d);
});

var last='',hist=[];
function poll(){
  fetch('/api').then(function(r){return r.json()}).then(function(d){
    var a=d.a, v=d.v, info=ACTS.find(function(x){return x.k===a})||{c:'#555'};
    var ch=a!==last;last=a;
    if(ch){
      document.getElementById('main').className='main active';
      setTimeout(function(){document.getElementById('main').className='main'},600);
    }
    document.getElementById('em').textContent=info.e||'?';
    document.getElementById('nm').textContent=info.n||a;
    document.getElementById('nm').style.color=info.c||'#555';
    document.getElementById('en').textContent=a;
    var p=v/TOTAL*100;
    var bar=document.getElementById('bar');
    bar.style.width=p+'%';bar.style.background=p>50?info.c:'#ff4444';
    document.getElementById('vt').textContent=v+'/'+TOTAL;
    ACTS.forEach(function(x){
      document.getElementById('c'+x.k).style.borderColor=x.k===a?x.c:'rgba(255,255,255,.06)';
      document.getElementById('c'+x.k).style.boxShadow=x.k===a?'0 0 20px '+x.g:'none';
      document.getElementById('b'+x.k).style.width=x.k===a?p+'%':'0%';
    });
    if(ch&&a!=='--'){
      hist.unshift({a:a,v:v});
      if(hist.length>30)hist.length=30;
      var hd=document.getElementById('hist');
      hd.innerHTML=hist.slice(0,18).map(function(h){
        var inf=ACTS.find(function(x){return x.k===h.a})||{c:'#555',e:'?'};
        return '<div class="dot" style="background:'+inf.c+';color:#fff" title="'+inf.n+' '+h.v+'/'+TOTAL+'">'+inf.e+'</div>';
      }).join('');
    }
  }).catch(function(){});
}
setInterval(poll,400);poll();
</script></body></html>"""

# ===== 推理数据 (共享) =====
current_action = "--"
current_votes = 0
current_pred = -1
inference_ready = False

def inference_loop():
    global current_action, current_votes, current_pred, inference_ready

    set_led(0, 0, 255)
    time.sleep(0.3)

    gc.collect()
    try:
        with open("rf_params.json", "r") as f:
            params = json.load(f)
    except:
        while True:
            set_led(255, 0, 0)
            time.sleep(0.2)
            set_led(0, 0, 0)
            time.sleep(0.2)

    gc.collect()
    trees = params["trees"]
    sm = params["scaler_mean"]
    ss = params["scaler_scale"]
    nc = params["n_classes"]

    for _ in range(3):
        set_led(255, 255, 0)
        time.sleep(0.1)
        set_led(0, 0, 0)
        time.sleep(0.1)

    calibrate(50)

    BUF_SZ = WINDOW_SIZE * N_AXES
    buf = array('f', [0.0] * BUF_SZ)
    pos = 0
    sq = array('b', [0] * SMOOTH_FRAMES)
    sqp = 0
    last_t = time.ticks_ms()

    inference_ready = True
    set_led(0, 10, 0)
    print("READY")

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

        feats = extract_features(buf)
        try:
            pred, votes = rf_predict(feats, trees, sm, ss, nc)
        except:
            pred = -1
            votes = 0

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
            current_pred = pred
            set_led(*LED_COLORS.get(pred, (255, 100, 0)))

        shift = WINDOW_STEP * N_AXES
        keep = BUF_SZ - shift
        for i in range(keep):
            buf[i] = buf[i + shift]
        pos = keep


# ===== 主入口 =====
def main():
    global current_action, current_votes

    print("ESP32-S3 HAR - WiFi Mode")
    print("========================")

    set_led(255, 100, 0)

    # 1. MPU6050
    print("Init MPU6050...")
    if not init_mpu():
        print("ERROR: MPU6050 not found!")
        set_led(255, 0, 0)
        while True:
            time.sleep(1)
    print("MPU6050 OK")

    # 2. WiFi AP模式 - ESP32自建热点
    print("Starting WiFi AP...")
    gc.collect()

    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    time.sleep(0.5)
    ap.config(essid='ESP32-HAR', security=0, channel=6)
    try:
        ap.config(txpower=20.5)
    except:
        pass
    time.sleep(1)

    if ap.active():
        web_ip = ap.ifconfig()[0]
        print(f"WiFi AP OK: {web_ip}")
        set_led(0, 255, 0)
    else:
        web_ip = ""
        for _ in range(5):
            set_led(255, 0, 0)
            time.sleep(0.2)
            set_led(0, 0, 0)
            time.sleep(0.2)
        print("WiFi AP FAILED!")
        while True:
            time.sleep(1)

    # 3. 启动推理线程
    print("Starting inference...")
    import _thread
    _thread.start_new_thread(inference_loop, ())
    for _ in range(50):
        if inference_ready:
            break
        time.sleep_ms(100)

    if not inference_ready:
        print("Inference init timeout!")
        while True:
            time.sleep(1)

    # 4. Web服务器
    print("Starting web server...")
    gc.collect()

    addr = socket.getaddrinfo('0.0.0.0', 80)[0][-1]
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(addr)
    s.listen(3)
    s.settimeout(5)

    print(f"\n{'='*40}")
    print(f"  WiFi: ESP32-HAR (无密码)")
    print(f"  网页: http://{web_ip}")
    print(f"  LED:  当前动作颜色指示")
    print(f"{'='*40}\n")

    while True:
        try:
            cl, addr = s.accept()
        except OSError:
            gc.collect()
            continue

        try:
            req = cl.recv(1024).decode('utf-8', 'ignore')

            if '/api' in req:
                # JSON API
                resp = '{"a":"%s","v":%d,"p":%d}' % (
                    current_action.replace('"', '\\"'),
                    current_votes,
                    current_pred
                )
                cl.send('HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n')
                cl.send(resp)
            else:
                # HTML页面
                cl.send('HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\n\r\n')
                cl.send(HTML)
        except Exception as e:
            pass
        finally:
            try:
                cl.close()
            except:
                pass


main()
