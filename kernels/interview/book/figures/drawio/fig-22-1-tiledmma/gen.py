#!/usr/bin/env python3
# gen_fig22_1.py — FIG-22-1 TiledMMA partition 三栏图（QK 64x64 + PV 64x16）
# 铁律：CJK 无 bold、无 &#10;、下标纯文本
import sys

W, H = 1050, 640
parts = []
def P(s):
    parts.append(s)

def box(x, y, w, h, text, fill="#ffffff", stroke="#9aa5b1", fs=13, bold=False,
        fc="#1f2933", dashed=0, align="center"):
    st = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
          f"fontSize={fs};fontColor={fc};align={align};verticalAlign=middle;fontFamily=Helvetica")
    if bold: st += ";fontStyle=1"
    if dashed: st += ";dashed=1"
    P(f'<mxCell value="{text}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def text(x, y, w, h, t, fs=13, fc="#102a43", bold=False):
    st = f"text;html=1;align=left;verticalAlign=middle;fontSize={fs};fontColor={fc};fontFamily=Helvetica"
    if bold: st += ";fontStyle=1"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

BLUE, GREEN, ORANGE, GRAY = "#2171b5", "#2e8540", "#d97706", "#e7ecf3"

# 标题（纯 ASCII bold + 汉字不 bold 分开）
text(40, 16, 700, 26, "TiledMMA partition — QK (64,64,64) &amp; PV (64,16,16) on 128 threads (4 warps)", fs=17, bold=True)
text(40, 42, 700, 20, "同一组线程承载两个 tile：QK 大 tile 吃满吞吐，PV 小 tile 压住 acc_O 寄存器", fs=13, fc="#48586a")

# ---- 上栏：线程划分 ----
text(40, 78, 400, 20, "① 线程划分：QK tile (64×64)，4 个 warp 各覆盖 16 行", fs=14, fc=BLUE, bold=False)
BX, BY, BW, BH = 40, 102, 800, 24
for w in range(4):
    y = BY + w * (BH + 6)
    box(BX, y, 120, BH, f"warp {w}: rows [{16*w}, {16*w+16})", fill=GRAY, stroke="#627d98", fs=12)
    for j in range(8):
        box(BX + 140 + j * 101, y, 94, BH, f"n{j}: cols [{8*j}, {8*j+8})", fill="#f0f4f8", stroke="#9aa5b1", fs=12)
text(BX, BY + 4 * (BH + 6) + 2, 900, 18, "warp 内沿 N 方向重复 8 个 m16n8 原子（MMA_N = 8）", fs=12, fc="#48586a")

# ---- 中栏：fragment 账目 ----
text(40, 248, 500, 20, "② fragment 账目：每线程形状 (MMA, MMA_M, MMA_K)，实测值", fs=14, fc=GREEN)
FY, FH = 272, 84
cards = [
    ("A (Q)", BLUE, "(8, 1, 4)", "32 half / thread", "双射（无重复）", "128 thr × 32 = 4096 = 64×64"),
    ("B (K)", GREEN, "(4, 8, 4)", "128 half / thread", "×4 重复（4 个 M-warp 共享）", "128 thr × 128 = 4× 4096"),
    ("C (S)", ORANGE, "(4, 1, 8)", "32 float / thread", "双射（无重复）", "128 thr × 32 = 4096 = 64×64"),
]
for i, (name, color, shape, cnt, dup, cons) in enumerate(cards):
    x = 40 + i * 273
    box(x, FY, 250, FH, "", fill="#ffffff", stroke=color)
    box(x, FY, 250, 22, name, fill=color, stroke=color, fs=13, bold=True, fc="#ffffff")
    text(x + 10, FY + 26, 230, 18, f"fragment = {shape}", fs=13, bold=True)
    text(x + 10, FY + 46, 230, 16, cnt, fs=12)
    text(x + 10, FY + 64, 230, 16, dup, fs=12, fc="#627d98")
text(40, FY + FH + 6, 900, 18, "守恒式：A 与 C 双射覆盖整个 tile；B 按 M-warp 数重复（式 22-dup），这是 warp 排布的代价所在", fs=12, fc="#48586a")

# ---- 下栏：PV 寄存器账 ----
text(40, 396, 600, 20, "③ PV 寄存器账：acc_O 按堆叠条形图（d-chunk 0..7）", fs=14, fc=ORANGE)
SY, SH = 424, 40
for c in range(8):
    x = 40 + c * 96
    box(x, SY, 88, SH, f"chunk {c}", fill="#fde9d0", stroke="#d97706", fs=11)
    text(x, SY + SH + 2, 88, 16, "32 float", fs=11, fc="#48586a")
box(40, SY, 8 * 96 - 8, SH, "", fill="none", stroke="#d97706")
# 总量 + 警戒线
text(846, SY + 10, 190, 18, "合计 = 256 寄存器", fs=13, bold=True, fc="#b45309")
box(660, SY - 10, 3, SH + 20, "", fill="none", stroke="#dc2626")
text(40, 502, 970, 18, "警戒：PV 的 N tile 拉到 64 → 每 chunk 128 float → 8× 超支，寄存器不可能常驻（红线上即不可行）", fs=11.5, fc="#dc2626")

# 底注
text(40, 526, 970, 18, "PV tile (64,16,16)：MMA_N = 2（16/8），N 方向只重复 2 个原子——用吞吐换 acc_O 可控，这是 FFPA 的核心权衡", fs=12, fc="#48586a")
text(40, 548, 970, 18, "实测：本章 case C–F 对上述全部形状与覆盖逐点核对（static_assert + 打印）", fs=12, fc="#627d98")

xml = f'''<mxfile host="app.diagrams.net">
  <diagram id="fig22-1" name="FIG-22-1">
    <mxGraphModel dx="{W}" dy="{H}" grid="0" page="1" pageWidth="{W}" pageHeight="600" math="0" shadow="0">
      <root><mxCell id="0"/><mxCell id="1" parent="0"/>
        {''.join(parts)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
out = sys.argv[1] if len(sys.argv) > 1 else 'fig-22-1.drawio'
open(out, 'w').write(xml)
print(f'wrote {out}')
