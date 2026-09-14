#!/usr/bin/env python3
# gen_fig26_1.py — FIG-26-1 双 TiledMma 数据流（QK chunk 循环 | Phase 2 | PV chunk 循环）
import sys

W, H = 900, 500
parts = []
def P(s): parts.append(s)

BLUE, GREEN, ORANGE = "#2171b5", "#2e8540", "#d97706"
LBLUE, LGREEN, LORANGE = "#bfd7ea", "#c9e4c8", "#fde9d0"

def box(x, y, w, h, text, fill="#ffffff", stroke="#9aa5b1", fs=12.5, bold=False,
        fc="#1f2933", rounded=0, dashed=0):
    st = (f"rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
          f"fontSize={fs};fontColor={fc};align=center;verticalAlign=middle;fontFamily=Helvetica")
    if bold: st += ";fontStyle=1"
    if dashed: st += ";dashed=1"
    P(f'<mxCell value="{text}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def text(x, y, w, h, t, fs=13, fc="#102a43", bold=False):
    st = f"text;html=1;align=left;verticalAlign=middle;fontSize={fs};fontColor={fc};fontFamily=Helvetica"
    if bold: st += ";fontStyle=1"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def arrow(x1, y1, x2, y2, color="#48586a", dashed=0, wy=None):
    st = (f"endArrow=blockThin;endFill=1;html=1;strokeColor={color};strokeWidth=2;rounded=0;")
    if dashed: st += "dashed=1;"
    geo = f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/>'
    if wy is not None:
        geo += f'<Array as="points"><mxPoint x="{(x1+x2)//2}" y="{wy}"/></Array>'
    geo += '</mxGeometry>'
    P(f'<mxCell value="" style="{st}" edge="1" parent="1">{geo}</mxCell>')

# 标题
text(30, 12, 840, 24, "ffpa_split_d_cute (128 threads): dual TiledMma dataflow", fs=17, bold=True)
text(30, 38, 840, 20, "QK Tile&lt;64,64,16&gt; EURepeat&lt;1,8,1&gt; ｜ PV Tile&lt;64,16,16&gt; EURepeat&lt;1,2,1&gt; —— S 到 P 到 O 一条寄存器流水", fs=12.5, fc="#48586a")

# ---- 左栏：QK chunk 循环 ----
LX, LY, LW = 40, 80, 240
box(LX, LY, LW, 30, "QK chunk 循环 (d = 0..C-1)", fill=BLUE, stroke=BLUE, fs=13.5, bold=True, fc="#ffffff", rounded=1)
for i, t in enumerate([
    "Q[d], K[d] --cp.async--&gt; smem",
    "S += Q[d] @ K[d]^T",
    "gemm_ss, 4 个 k16 原子",
]):
    box(LX, LY + 40 + i * 42, LW, 34, t, fill=LBLUE, stroke=BLUE, fs=12)
box(LX, LY + 170, LW, 36, "S: 32 个 f32/线程 = ((2,2),1,8)", fill="#ffffff", stroke=BLUE, fs=12.5, bold=True, fc=BLUE)
arrow(LX + 120, LY + 166, LX + 120, LY + 170, BLUE)
# C 个 chunk 汇入
text(LX, LY + 212, LW, 16, "C 个 [64,64] chunk 顺序累加，S 跨 chunk 存活", fs=11, fc="#48586a")

# ---- 中栏：Phase 2 ----
MX, MY, MW = 350, 80, 200
box(MX, MY, MW, 30, "Phase 2: rowcol 视图", fill=ORANGE, stroke=ORANGE, fs=13.5, bold=True, fc="#ffffff", rounded=1)
for i, t in enumerate([
    "row max / sum (每 lane 2 行)",
    "P = exp2(S·scale - m)  就地覆盖",
    "P: convert_type + Aregs 重解释",
]):
    box(MX, MY + 40 + i * 42, MW, 34, t, fill=LORANGE, stroke=ORANGE, fs=11.5)
text(MX - 10, MY + 172, MW + 20, 30, "row_max[2], row_sum[2]&#160;&#160;&#160;row_scale[2] = exp2(m_old - m_new)", fs=10.5, fc="#b45309")

# ---- 右栏：PV chunk 循环 ----
RX, RY, RW = 620, 80, 240
box(RX, RY, RW, 30, "PV chunk 循环 (v = 0..C-1)", fill=GREEN, stroke=GREEN, fs=13.5, bold=True, fc="#ffffff", rounded=1)
for i, t in enumerate([
    "O[v] *= row_scale  (rescale)",
    "O[v] += P @ V[v]  (ns 视图)",
]):
    box(RX, RY + 40 + i * 42, RW, 34, t, fill=LGREEN, stroke=GREEN, fs=12)
box(RX, RY + 128, RW, 36, "O: C × [64,16] 分量独立累加", fill="#ffffff", stroke=GREEN, fs=12.5, bold=True, fc=GREEN)

# 主管线箭头 S -> Phase2 -> P -> PV
arrow(LX + LW, MY + 100, MX, MY + 100, ORANGE)          # S -> Phase2
text(285, MY + 78, 60, 16, "S (32)", fs=11.5, fc=ORANGE)
arrow(MX + MW, MY + 100, RX, RY + 56, GREEN)            # P -> PV 第一格
text(542, RY + 6, 80, 16, "P (Aregs)", fs=11.5, fc=GREEN)

# row_scale 虚线：Phase2 -> PV rescale
arrow(MX + MW - 30, MY + 196, RX + 30, RY + 50, "#b45309", dashed=1)
text(396, 268, 160, 16, "row_scale (虚线)", fs=11, fc="#b45309")

# ---- 底部寄存器账本 ----
BY = 330
text(40, BY, 300, 18, "寄存器账本（每线程，D=512 → C=8）", fs=13.5, fc="#102a43")
ledger = [
    ("S / P", "32 f32", BLUE, "Phase 2 就地覆盖，不新增"),
    ("O[C][32]", "256 f32", GREEN, "8 chunk × 32，acc 主体"),
    ("V frag", "dup × M-warp", ORANGE, "ns 视图按需取，不常驻全量"),
]
for i, (name, cnt, color, note) in enumerate(ledger):
    x = 40 + i * 285
    box(x, BY + 26, 120, 40, name, fill="#ffffff", stroke=color, fs=13, bold=True, fc=color)
    box(x + 126, BY + 26, 150, 40, cnt, fill="#f0f4f8", stroke="#bcccdc", fs=12.5)
    text(x, BY + 70, 276, 16, note, fs=11, fc="#48586a")
text(40, BY + 96, 820, 18, "S、P、O 全程在寄存器：smem 只承载 Q/K/V 的 chunk 流水——这就是「split-D」把 softmax 从两遍访存变成一遍寄存器更新的全部代价结构", fs=12, fc="#48586a")

xml = f'''<mxfile host="app.diagrams.net">
  <diagram id="fig26-1" name="FIG-26-1">
    <mxGraphModel dx="{W}" dy="{H}" grid="0" page="1" pageWidth="{W}" pageHeight="470" math="0" shadow="0">
      <root><mxCell id="0"/><mxCell id="1" parent="0"/>
        {''.join(parts)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
out = sys.argv[1] if len(sys.argv) > 1 else 'fig-26-1.drawio'
open(out, 'w').write(xml)
print(f'wrote {out}')
