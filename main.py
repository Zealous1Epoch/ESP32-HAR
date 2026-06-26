# ESP32-S3 MPU6050 HAR - WiFi网页版
# 连接手机热点(vivo) → 手机浏览器直接查看
# LED + 网页双显示

import gc, math, time, _thread, network, socket
from machine import SoftI2C, Pin
import neopixel
from array import array

try:
    import ujson as json
except:
    import json

# ===== 配置 =====
HOTSPOT_SSID = "handsomelai"
HOTSPOT_PASSWORD = "12345678"

SDA_PIN = 8
SCL_PIN = 9
LED_PIN = 48
MPU_ADDR = 0x68

WINDOW_SIZE = 128
WINDOW_STEP = 64
SAMPLE_MS = 20
N_AXES = 6
N_FEATURES = 30
SMOOTH_FRAMES = 3

ACT_NAMES = ["sit", "stand", "walk", "upstairs", "downstairs", "run"]

LED_COLORS = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255),
    3: (255, 255, 0), 4: (255, 0, 255), 5: (255, 255, 255),
}

current_action = "init"; current_votes = 0
inference_ready = False; data_lock = None
np = None; i2c = None; web_ip = "0.0.0.0"

def init_led():
    global np
    if np is None: np = neopixel.NeoPixel(Pin(LED_PIN), 1)
def set_led(r,g,b):
    init_led()
    if np: np[0]=(r,g,b); np.write()

def init_i2c():
    global i2c
    if i2c is None: i2c = SoftI2C(sda=Pin(SDA_PIN), scl=Pin(SCL_PIN), freq=100000)

# ===== MPU6050 =====
def conv(v): return v-65536 if v>=32768 else v

def init_mpu():
    init_i2c()
    for i in range(5):
        try:
            i2c.writeto_mem(MPU_ADDR,0x6B,b'\x00')
            time.sleep_ms(50); return True
        except OSError: time.sleep_ms(50)
    return False

def read_mpu_raw():
    for _ in range(5):
        try:
            buf=i2c.readfrom_mem(MPU_ADDR,0x3B,14)
            ax=conv((buf[0]<<8)|buf[1]); ay=conv((buf[2]<<8)|buf[3])
            az=conv((buf[4]<<8)|buf[5]); gx=conv((buf[8]<<8)|buf[9])
            gy=conv((buf[10]<<8)|buf[11]); gz=conv((buf[12]<<8)|buf[13])
            if abs(ax)>32000 or abs(ay)>32000 or abs(az)>32000: continue
            return ax,ay,az,gx,gy,gz
        except OSError: time.sleep_ms(10)
    return None

ax_off=ay_off=az_off=gx_off=gy_off=gz_off=0
def calibrate(n=100):
    global ax_off,ay_off,az_off,gx_off,gy_off,gz_off
    s=[0]*6; v=0; t0=time.ticks_ms()
    while v<n and time.ticks_diff(time.ticks_ms(),t0)<10000:
        d=read_mpu_raw()
        if d is None: time.sleep_ms(10); continue
        for i in range(6): s[i]+=d[i]
        v+=1; time.sleep_ms(10)
    if v>0: ax_off,ay_off,az_off,gx_off,gy_off,gz_off=[x//v for x in s]

def get_cal():
    d=read_mpu_raw()
    if d is None: return None
    return (d[0]-ax_off,d[1]-ay_off,d[2]-az_off,
            d[3]-gx_off,d[4]-gy_off,d[5]-gz_off)

def extract_features(buf):
    feats=array('f',[0.0]*N_FEATURES)
    for axis in range(N_AXES):
        s=0.0;mn=1e9;mx=-1e9
        for i in range(WINDOW_SIZE):
            v=buf[i*N_AXES+axis];s+=v
            if v<mn:mn=v
            if v>mx:mx=v
        mean=s/WINDOW_SIZE;sq=0.0
        for i in range(WINDOW_SIZE):
            d=buf[i*N_AXES+axis]-mean;sq+=d*d
        b=axis*5
        feats[b]=mean;feats[b+1]=math.sqrt(sq/WINDOW_SIZE)
        feats[b+2]=mx;feats[b+3]=mn;feats[b+4]=mx-mn
    return feats

def rf_predict(feats,trees,sm,ss,nc):
    fs=array('f',[0.0]*N_FEATURES)
    for i in range(N_FEATURES): fs[i]=(feats[i]-sm[i])/ss[i]
    votes=[0]*nc
    for tree in trees:
        node=0
        fa=tree["feature"];ta=tree["threshold"]
        la=tree["children_left"];ra=tree["children_right"]
        va=tree["value"]
        while fa[node]!=-2:
            if fs[fa[node]]<=ta[node]: node=la[node]
            else: node=ra[node]
        lv=va[node][0];bc=0;bv=lv[0]
        for c in range(1,len(lv)):
            if lv[c]>bv: bv=lv[c];bc=c
        votes[bc]+=1
    w=0
    for c in range(1,nc):
        if votes[c]>votes[w]: w=c
    return w,votes[w]

# ===== 网页 =====
HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>HAR</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,sans-serif;background:#0a0a1a;color:#fff;
display:flex;align-items:center;justify-content:center;min-height:100vh}
.card{background:rgba(255,255,255,0.05);border:2px solid rgba(100,100,255,0.3);
border-radius:24px;padding:50px 30px;text-align:center;max-width:360px;width:90%}
.card.active{border-color:#00ff88;box-shadow:0 0 40px rgba(0,255,136,0.15)}
h2{color:#8888cc;font-size:1em;margin-bottom:5px}
.desc{color:#555;font-size:0.7em;margin-bottom:30px}
.act{font-size:4em;font-weight:800;color:#ccc;transition:color 0.3s}
.act.live{color:#00ff88}
.votes{color:#666;font-size:0.85em;margin-top:12px}
</style></head><body>
<div class="card" id="c"><h2>ESP32 HAR</h2>
<div class="desc">MPU6050 + Random Forest</div>
<div class="act" id="a">--</div>
<div class="votes" id="v"></div></div>
<script>
var last='';
setInterval(function(){
 fetch('/api').then(function(r){return r.json()}).then(function(d){
  var a=document.getElementById('a');
  a.textContent=d.a;
  if(d.a!==last){
   a.className='act '+(d.a!=='--'?'live':'');
   document.getElementById('c').className='card '+(d.a!=='--'?'active':'');
   last=d.a;
  }
  if(d.v>0) document.getElementById('v').textContent='votes: '+d.v;
 }).catch(function(){});
},1000);
</script></body></html>"""

# ===== 推理线程 =====
def inference_loop():
    global current_action, current_votes, inference_ready
    set_led(0,0,255); time.sleep(0.5)

    gc.collect()
    try:
        with open("rf_params.json","r") as f: params=json.load(f)
    except:
        while True: set_led(255,0,0); time.sleep(0.2); set_led(0,0,0); time.sleep(0.2)
    gc.collect()

    trees=params["trees"]; sm=params["scaler_mean"]
    ss=params["scaler_scale"]; nc=params["n_classes"]

    for _ in range(5): set_led(255,255,0); time.sleep(0.1); set_led(0,0,0); time.sleep(0.1)
    calibrate(100)

    BUF_SZ=WINDOW_SIZE*N_AXES; buf=array('f',[0.0]*BUF_SZ); pos=0
    sq=array('b',[0]*SMOOTH_FRAMES); sqp=0; last_t=time.ticks_ms()

    inference_ready=True; set_led(0,10,0)

    while True:
        now=time.ticks_ms()
        if time.ticks_diff(now,last_t)<SAMPLE_MS: time.sleep_ms(2); continue
        d=get_cal()
        if d is None: time.sleep_ms(5); continue
        last_t=now
        for v in d: buf[pos]=float(v); pos+=1
        if pos<BUF_SZ: continue

        feats=extract_features(buf)
        try: pred,votes=rf_predict(feats,trees,sm,ss,nc)
        except: pred=-1;votes=0

        sq[sqp]=pred; sqp=(sqp+1)%SMOOTH_FRAMES
        same=True
        for i in range(1,SMOOTH_FRAMES):
            if sq[i]!=sq[0]: same=False; break
        if same and pred>=0:
            current_action=ACT_NAMES[pred]; current_votes=votes
            set_led(*LED_COLORS.get(pred,(255,100,0)))

        shift=WINDOW_STEP*N_AXES; keep=BUF_SZ-shift
        for i in range(keep): buf[i]=buf[i+shift]
        pos=keep

# ===== 主入口 =====
def main():
    global data_lock, web_ip

    set_led(255,100,0)

    # 1. MPU6050
    if not init_mpu():
        set_led(255,0,0)
        while True: time.sleep(1)

    # 2. WiFi AP模式 - ESP32自己发射热点
    gc.collect()
    sta=network.WLAN(network.STA_IF); sta.active(False)
    ap=network.WLAN(network.AP_IF); ap.active(False)
    time.sleep(0.5)

    web_ip = ""
    ap.active(True)
    time.sleep(0.3)
    ap.config(essid='ESP32-HAR', security=0, channel=6)
    try:
        ap.config(txpower=20.5)
    except:
        pass
    time.sleep(1)

    if ap.active():
        web_ip = ap.ifconfig()[0]  # 192.168.4.1
        set_led(0, 255, 0)  # Green = OK
        with open("wifi.log","w") as f: f.write(f"ap_ok:{web_ip}")
    else:
        web_ip = ""
        for _ in range(3): set_led(255,0,0); time.sleep(0.3); set_led(0,0,0); time.sleep(0.3)

    # 3. 启动socket
    data_lock = _thread.allocate_lock()
    gc.collect()

    if web_ip:
        try:
            addr = socket.getaddrinfo('0.0.0.0',80)[0][-1]
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(addr)
            s.listen(3)
            s.settimeout(3)
        except Exception as e:
            web_ip = ""
            for _ in range(3): set_led(255,0,0); time.sleep(0.2); set_led(255,100,0); time.sleep(0.2)

    # 4. 启动推理
    _thread.start_new_thread(inference_loop,())
    for _ in range(40):
        if inference_ready: break
        time.sleep_ms(100)

    # 5. Web服务器主循环
    if web_ip:
        set_led(0,255,0)
        while True:
            try:
                cl,addr=s.accept()
            except OSError:
                continue
            try:
                req=cl.recv(1024).decode('utf-8','ignore')
                if '/api' in req:
                    resp='{"a":"%s","v":%d}'%(current_action.replace('"','\\"'),current_votes)
                    cl.send('HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n')
                    cl.send(resp)
                else:
                    cl.send('HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\n\r\n')
                    cl.send(HTML)
            except:
                pass
            finally:
                cl.close()
    else:
        # WiFi失败, 主线程空闲
        while True:
            time.sleep(30)
            gc.collect()

main()
