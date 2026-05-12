"""
Generate UML-style DiscreteBCQ architecture diagram for thesis.
Uses matplotlib with clean academic styling.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors (thesis-friendly palette)
COLOR_STATE   = '#D6E8F7'
COLOR_Q      = '#C6EFCE'
COLOR_B      = '#E2D1F2'
COLOR_OUT    = '#FFEB9C'
COLOR_FUSE   = '#F8D7DA'
COLOR_ACTION = '#D6E8F7'
BORDER_COLOR = '#333333'

def draw_box(x, y, w, h, color, text, fontsize=9, align='center', border_color=BORDER_COLOR, subtitle=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor=border_color, linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha=align, va='center',
            fontsize=fontsize, fontfamily='serif')
    if subtitle:
        ax.text(x + w/2, y + h/2 - 0.25, subtitle, ha=align, va='top',
                fontsize=7, fontfamily='serif', color='#555555', fontstyle='italic')

def draw_arrow(x1, y1, x2, y2, label='', lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=lw))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.12, label, ha='center', va='bottom',
                fontsize=7, fontfamily='serif', color='#555555', fontstyle='italic')

def package_box(x, y, w, h, color, label):
    """Draw a UML-style package (folder tab)."""
    tab_h = 0.25
    tab_w = 1.8
    # Main body
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor=BORDER_COLOR, linewidth=1.0,
                         linestyle='--')
    ax.add_patch(box)
    # Label at top-left
    ax.text(x + 0.15, y + h - 0.02, label, ha='left', va='top',
            fontsize=8, fontfamily='serif', fontweight='bold')

# ============================================================
# Layout coordinates
# ============================================================
# State input
draw_box(3.5, 8.2, 3.0, 1.0, COLOR_STATE,
         'State s (9 features)', fontsize=10)
ax.text(5.0, 8.5, 'CPC, CTR, CPC_change, Conversions,\nPosition, Day, DeviceType, IsWeekend, BudgetRemaining',
        ha='center', va='top', fontsize=7, fontfamily='serif', color='#555555')

draw_arrow(5.0, 8.2, 5.0, 7.0)

# Package: Q-Network
package_box(0.5, 4.0, 4.0, 3.0, '#E8F5E9', '«component» Q-Network')
q_ys = [5.8, 4.8]
for i, (y, label) in enumerate(zip(q_ys, ['Hidden Layer 1', 'Hidden Layer 2'])):
    draw_box(0.8, y, 3.4, 0.55, COLOR_Q,
             f'{label}  128 units, ReLU', fontsize=9)
    if i < len(q_ys) - 1:
        draw_arrow(2.5, y, 2.5, y - 0.55)

draw_box(0.8, 4.05, 3.4, 0.55, COLOR_OUT,
         'Q(s, a₀)  Q(s, a₁)  Q(s, a₂)  Q(s, a₃)  Q(s, a₄)', fontsize=8)
draw_arrow(2.5, 4.05, 2.5, 4.0-0.55)

# Package: Behavioral Clone
package_box(5.5, 4.0, 4.0, 3.0, '#F3E5F5', '«component» Behavioral Clone')
b_ys = [5.8, 4.8]
for i, (y, label) in enumerate(zip(b_ys, ['Hidden Layer 1', 'Hidden Layer 2'])):
    draw_box(5.8, y, 3.4, 0.55, COLOR_B,
             f'{label}  128 units, ReLU', fontsize=9)
    if i < len(b_ys) - 1:
        draw_arrow(7.5, y, 7.5, y - 0.55)

draw_box(5.8, 4.05, 3.4, 0.55, COLOR_OUT,
         'π_β(a₀|s)  π_β(a₁|s)  π_β(a₂|s)  π_β(a₃|s)  π_β(a₄|s)', fontsize=8)
draw_arrow(7.5, 4.05, 7.5, 4.0-0.55)

# Shared input arrows
draw_arrow(5.0, 7.0, 2.5, 6.35, label='shared input')
draw_arrow(5.0, 7.0, 7.5, 6.35, label='shared input')

# Action Selection
draw_arrow(2.5, 4.0, 4.2, 2.5)
draw_arrow(7.5, 4.0, 5.8, 2.5)

draw_box(2.8, 1.5, 4.4, 1.0, COLOR_FUSE,
         'Action Selection', fontsize=10)
ax.text(5.0, 1.8, r'π(s) = argmaxₐ [ Q(s,a) + β · log π_β(a|s) ]',
        ha='center', va='center', fontsize=9, fontfamily='serif',
        fontstyle='italic')

draw_arrow(5.0, 1.5, 5.0, 0.4, lw=1.5)

# Output
draw_box(3.2, 0.0, 3.6, 0.4, COLOR_ACTION,
         'Selected Action a ∈ {0, 1, 2, 3, 4}', fontsize=9)

# Labels for actions
actions = ['0: Decrease', '1: Slight ↓', '2: Keep', '3: Slight ↑', '4: Increase']
for i, act in enumerate(actions):
    x = 1.0 + i * 1.8
    ax.text(x, -0.25, act, ha='center', va='top',
            fontsize=6.5, fontfamily='serif', color='#555555')

# Title
ax.set_title('DiscreteBCQ — Architecture Diagram', fontsize=13,
             fontweight='bold', fontfamily='serif', pad=8, y=0.98)

# Save
out_path = r"D:\VisualCodeProjects\Python\Diploma_ISU\pre_vkr\test_ppo\bcq_uml_architecture.png"
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")
plt.close()
