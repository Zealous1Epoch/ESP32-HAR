#!/usr/bin/env python3
"""用真实API数据生成答辩PPT用的仪表盘截图"""
import json, os, urllib.request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = "./plots"
BG = '#0a0f1e'
FG = '#c8d0e0'
GRAY = '#555'

ACTS = [
    ('sit', '静坐', '#FF6B6B'),
    ('stand', '站立', '#4ECDC4'),
    ('walk', '走路', '#45B7D1'),
    ('upstairs', '上楼', '#FFEAA7'),
    ('downstairs', '下楼', '#DDA0DD'),
    ('run', '跑步', '#FF8C42'),
]
ACT_MAP = {a[0]: a for a in ACTS}  # key -> (key, name, color)

# 从运行中的服务器获取真实数据
try:
    with urllib.request.urlopen('http://localhost:8080/api/state', timeout=3) as r:
        state = json.loads(r.read())
    print(f"获取实时数据: {state['act']} ({state['votes']}/15票)")
except Exception as e:
    print(f"无法连接服务器: {e}")
    print("使用模拟数据")
    state = {
        "act": "sit", "votes": 8, "pred": 0,
        "all_votes": [8, 3, 0, 1, 3, 0],
        "history": [
            {"act": "sit", "votes": 7, "time": "15:35:32"},
            {"act": "stand", "votes": 6, "time": "15:35:38"},
            {"act": "walk", "votes": 7, "time": "15:35:44"},
            {"act": "upstairs", "votes": 6, "time": "15:35:50"},
            {"act": "walk", "votes": 5, "time": "15:35:56"},
            {"act": "downstairs", "votes": 4, "time": "15:36:02"},
            {"act": "sit", "votes": 8, "time": "15:36:08"},
        ],
        "connected": True
    }

all_votes = state.get('all_votes', [0]*6)
current_act = state['act']
current_votes = state['votes']
history = state.get('history', [])
pct = current_votes / 15 * 100

# 统计各类出现次数
act_counts = {}
act_seq = []
for h in reversed(history):
    a = h['act']
    act_counts[a] = act_counts.get(a, 0) + 1
    act_seq.append(a)

total_detections = len(history)
changes = sum(1 for i in range(1, len(act_seq)) if act_seq[i] != act_seq[i-1])
top_act = max(act_counts, key=act_counts.get) if act_counts else '--'

def setup_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=GRAY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#1a1a3a')

# ===== 1. 投票分布柱状图 =====
def chart_votes():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(BG)
    setup_ax(ax)

    colors = [a[2] for a in ACTS]
    names = [a[1] for a in ACTS]
    x = range(6)

    bars = ax.bar(x, all_votes, color=colors, alpha=0.85, width=0.55, edgecolor=BG, linewidth=1)
    # 当前动作高亮边框
    cur_idx = [a[0] for a in ACTS].index(current_act) if current_act in [a[0] for a in ACTS] else 0
    bars[cur_idx].set_edgecolor(colors[cur_idx])
    bars[cur_idx].set_linewidth(3)

    for i, v in enumerate(all_votes):
        if v > 0:
            ax.text(i, v + 0.3, str(v), ha='center', va='bottom', color='white', fontsize=13, fontweight='bold')

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=11, color=FG)
    ax.set_ylim(0, 16)
    ax.set_ylabel('票数 / 15', color=GRAY, fontsize=10)
    ax.set_title(f'投票分布  (当前: {ACT_MAP.get(current_act, current_act)})', color=FG, fontsize=13, fontweight='bold', pad=10)
    ax.set_yticks([0, 3, 6, 9, 12, 15])
    ax.yaxis.grid(True, color='#1a1a3a', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(f'{OUT_DIR}/chart_votes.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

# ===== 2. 置信度仪表盘 =====
def chart_gauge():
    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    color = ACT_MAP.get(current_act, (None, None, GRAY))[2] if current_act in ACT_MAP else GRAY

    theta = np.linspace(np.pi * 0.75, np.pi * 2.25, 100)
    ax.fill_between(theta, 0.55, 1.0, color='#1a1a2a', alpha=0.8)
    val_angle = np.pi * 0.75 + (np.pi * 1.5) * min(pct / 100, 1)
    theta_val = np.linspace(np.pi * 0.75, val_angle, 100)
    ax.fill_between(theta_val, 0.6, 0.95, color=color, alpha=0.9)

    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)

    ax.text(0, 0, f'{pct:.0f}%', ha='center', va='center', fontsize=36, fontweight='bold', color='white')
    ax.text(0, -0.3, f'{current_votes} / 15 票', ha='center', va='center', fontsize=12, color=GRAY)
    ax.set_title('置信度', color=FG, fontsize=13, fontweight='bold', pad=18)

    plt.tight_layout()
    fig.savefig(f'{OUT_DIR}/chart_gauge.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

# ===== 3. 活动统计饼图 =====
def chart_donut():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    act_info = {a[0]: a for a in ACTS}
    labels = []
    sizes = []
    colors = []
    for a in ACTS:
        cnt = act_counts.get(a[0], 0)
        if cnt > 0:
            labels.append(f'{a[1]}  {cnt}次')
            sizes.append(cnt)
            colors.append(a[2])

    if not sizes:
        sizes = [1]
        colors = ['#333']
        labels = ['无数据']

    wedges, _ = ax.pie(sizes, labels=None, colors=colors, startangle=90,
                        wedgeprops={'width': 0.4, 'edgecolor': BG, 'linewidth': 2})

    total = sum(sizes)
    ax.text(0, 0, str(total), ha='center', va='center', fontsize=28, fontweight='bold', color='white')
    ax.text(0, -0.15, '次切换', ha='center', va='center', fontsize=10, color=GRAY)

    legend_labels = [f'{l}' for l in labels]
    ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=10, labelcolor=FG, frameon=False)

    ax.set_title('活动分布统计', color=FG, fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    fig.savefig(f'{OUT_DIR}/chart_donut.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

# ===== 4. 活动时间线 =====
def chart_timeline():
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)
    ax.axis('off')

    act_info = {a[0]: a for a in ACTS}
    seq = act_seq[-25:]  # 最近25次

    for i, act in enumerate(seq):
        info = act_info.get(act, (None, '?', GRAY))
        circle = plt.Circle((i + 0.3, 0.5), 0.4, color=info[2], ec=BG, linewidth=1)
        ax.add_patch(circle)
        ax.text(i + 0.3, 0.5, info[1][0], ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax.set_title(f'活动时间线 (最近{len(seq)}次)', color=FG, fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    fig.savefig(f'{OUT_DIR}/chart_timeline.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

# ===== 5. 仪表盘总览 =====
def chart_overview():
    from matplotlib.image import imread

    fig = plt.figure(figsize=(16, 10), facecolor=BG)

    # 标题
    fig.text(0.5, 0.97, 'ESP32-S3 HAR 实时检测仪表盘', ha='center', fontsize=22, fontweight='bold', color=FG)
    status_text = '已连接' if state.get('connected') else '未连接'
    fig.text(0.5, 0.94, f'MPU6050 + HMC5883L | 15trees | 51维 | 6类活动 | 准确率 88.4% | 状态: {status_text}',
             ha='center', fontsize=10, color=GRAY)

    # 投票图
    ax1 = fig.add_axes([0.02, 0.35, 0.46, 0.58])
    ax1.imshow(imread(f'{OUT_DIR}/chart_votes.png'))
    ax1.axis('off')

    # 仪表
    ax2 = fig.add_axes([0.50, 0.35, 0.20, 0.58])
    ax2.imshow(imread(f'{OUT_DIR}/chart_gauge.png'))
    ax2.axis('off')

    # 统计数字
    ax3 = fig.add_axes([0.72, 0.35, 0.27, 0.58])
    ax3.set_facecolor('#111125')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    cur_act_name = ACT_MAP.get(current_act, (None, current_act, GRAY))[1]
    cur_color = ACT_MAP.get(current_act, (None, None, GRAY))[2]
    top_act_name = ACT_MAP.get(top_act, (None, top_act, GRAY))[1]

    stats = [
        ('总检测次数', str(total_detections)),
        ('动作切换次数', str(changes)),
        ('最多动作', top_act_name),
        ('当前动作', cur_act_name),
        ('置信度', f'{pct:.0f}% ({current_votes}/15)'),
    ]
    for i, (label, val) in enumerate(stats):
        y = 0.88 - i * 0.17
        val_color = cur_color if label == '当前动作' else 'white'
        ax3.text(0.5, y + 0.05, val, ha='center', va='center', fontsize=16 if len(val) < 6 else 13,
                 fontweight='bold', color=val_color, transform=ax3.transAxes)
        ax3.text(0.5, y - 0.03, label, ha='center', va='center', fontsize=9, color=GRAY, transform=ax3.transAxes)
        if i < 4:
            ax3.plot([0.12, 0.88], [y - 0.09, y - 0.09], color='#1a1a3a', linewidth=0.5, transform=ax3.transAxes)
    ax3.set_title('实时统计', color=FG, fontsize=13, fontweight='bold', pad=10)

    # 时间线
    ax4 = fig.add_axes([0.02, 0.03, 0.46, 0.30])
    ax4.imshow(imread(f'{OUT_DIR}/chart_timeline.png'))
    ax4.axis('off')

    # 饼图
    ax5 = fig.add_axes([0.50, 0.03, 0.49, 0.30])
    ax5.imshow(imread(f'{OUT_DIR}/chart_donut.png'))
    ax5.axis('off')

    fig.savefig(f'{OUT_DIR}/dashboard_overview.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('生成完成')

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    chart_votes()
    chart_gauge()
    chart_donut()
    chart_timeline()
    chart_overview()
    print(f'所有图片保存到 {OUT_DIR}/')
    print(f'当前动作: {current_act} | 票数: {current_votes}/15 | 历史: {total_detections}条')
