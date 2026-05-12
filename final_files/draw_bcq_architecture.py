import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

def draw_box(x, y, w, h, color, text, text_color='white', fontsize=10, lw=2):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='black', lw=lw)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold')

def draw_arrow(x1, y1, x2, y2, color='black', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

box_w = 3.2
box_h = 0.7

draw_box(3.4, 8.5, box_w, box_h, '#4A90D9', 'State (9 features)', fontsize=11)
ax.text(5.0, 8.2, 'CPC, CTR, CPC_change, Conversions,\nPosition, Day, DeviceType, IsWeekend, BudgetRemaining',
        ha='center', va='top', fontsize=8, color='gray', fontstyle='italic')

draw_arrow(5.0, 8.5, 5.0, 7.3, lw=2.5)

draw_box(1.0, 6.0, box_w, box_h, '#50C878', 'Hidden Layer 1\n(128 units, ReLU)', fontsize=10)
draw_box(5.8, 6.0, box_w, box_h, '#50C878', 'Hidden Layer 1\n(128 units, ReLU)', fontsize=10)

ax.text(0.5, 6.35, 'Q-Network', fontsize=9, color='#2E7D32', fontweight='bold')
ax.text(7.1, 6.35, 'Behavior Clone', fontsize=9, color='#2E7D32', fontweight='bold')

draw_arrow(5.0, 7.3, 2.6, 6.7, lw=2)
draw_arrow(5.0, 7.3, 7.4, 6.7, lw=2)

draw_arrow(2.6, 6.0, 2.6, 4.7, lw=2)
draw_arrow(7.4, 6.0, 7.4, 4.7, lw=2)

draw_box(1.0, 4.0, box_w, box_h, '#F5A623', 'Q-Values Q(s,a)\n[5 actions]', fontsize=10)
draw_box(5.8, 4.0, box_w, box_h, '#F5A623', 'Behavior Policy π_β(a|s)\n[5 action probs]', fontsize=9)

ax.text(2.6, 3.7, 'Q(s,0)  Q(s,1)  ...  Q(s,4)', ha='center', va='top', fontsize=7.5, color='#B8860B')
ax.text(7.4, 3.7, 'π_β(0|s)  π_β(1|s)  ...  π_β(4|s)', ha='center', va='top', fontsize=7.5, color='#B8860B')

draw_arrow(2.6, 4.0, 3.8, 2.7, lw=2.5)
draw_arrow(7.4, 4.0, 6.2, 2.7, lw=2.5)

draw_box(2.2, 1.8, 5.6, 0.9, '#D64545', 'Action Selection\nargmaxₐ [Q(s,a) + β · log π_β(a|s)]',
         fontsize=10, text_color='white')

ax.text(5.0, 1.55, 'β = imitation weight,  action_flexibility = λ',
        ha='center', va='top', fontsize=8, color='gray', fontstyle='italic')

draw_arrow(5.0, 1.8, 5.0, 0.6, lw=3)

draw_box(3.4, 0.0, box_w, box_h, '#9B59B6', 'Selected Action (0-4)', fontsize=11)

ax.text(5.0, 0.3, '0: Decrease  1: Slight↓  2: Keep  3: Slight↑  4: Increase',
        ha='center', va='top', fontsize=7.5, color='#7D3C98')

ax.set_title('DiscreteBCQ Neural Network Architecture', fontsize=14, fontweight='bold', pad=15)

import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bcq_architecture.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved to {out_path}")
plt.close()
