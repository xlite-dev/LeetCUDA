#!/usr/bin/env python3
# gen_fig12_1.py — FIG-12-1 smem XOR swizzle 前后 quad 分布（8x16 fp16, 32B rows）
# 上：before/after 两表对照；下：ldmatrix.x4 phase0 的 quad 序列带对比（2-way vs 1-way）
import sys

W, H = 1080, 800
RH = 34                     # 数据行高
CW = 78                     # 数据格宽（两位数 quad 号）
LX, RX = 80, 600            # 左右表原点
TY = 148                    # 表头行 y
QY = 520                    # quad 带起点
QW, QH = 56, 40             # quad 带格

parts = []
def P(s): parts.append(s)

def box(x, y, w, h, text, fill="#ffffff", stroke="#bcccdc", fs=14, bold=False, fc="#1f2933"):
    st = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
          f"fontSize={fs};fontColor={fc};align=center;verticalAlign=middle;fontFamily=Helvetica")
    if bold: st += ";fontStyle=1"
    P(f'<mxCell value="{text}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def text(x, y, w, h, t, fs=14, fc="#102a43", bold=False, align="left"):
    st = f"text;html=1;align={align};verticalAlign=middle;fontSize={fs};fontColor={fc};fontFamily=Helvetica"
    if bold: st += ";fontStyle=1"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def arrow(pts, color="#2171b5", dashed=0, width=2):
    st = (f"endArrow=blockThin;endFill=1;html=1;strokeColor={color};strokeWidth={width};"
          f"rounded=0;dashed={dashed}")
    P(f'<mxCell value="" style="{st}" edge="1" parent="1">'
      f'<mxGeometry relative="1" as="geometry"><Array as="points">'
      + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
      + f'</Array><mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
        f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/></mxGeometry></mxCell>')

RED, GREEN, BLUE, ORANGE = "#dc2626", "#2e8540", "#2171b5", "#d97706"

# 标题
text(30, 14, 900, 26, "smem XOR swizzle 前后：8x16 fp16 tile（32 B 行宽），quad = (8i+4c) mod 32", fs=18, bold=True)
text(30, 44, 1000, 20, "quad = 16 B 四联组（4 个 bank）；ldmatrix.x4 phase0 由 lanes 0..7 按行 0..7 读 chunk0", fs=13.5, fc="#48586a")

# 表标题
text(LX, TY - 34, 400, 24, "[before] 直接排布", fs=15, bold=True, fc=RED)
text(LX + 210, TY - 34, 320, 24, "行 4-7 重复行 0-3 的 quad", fs=12.5, fc=RED)
text(RX, TY - 34, 400, 24, "[after] c' = c XOR ((i&gt;&gt;2)&amp;1)", fs=15, bold=True, fc=GREEN)
text(RX + 250, TY - 34, 260, 24, "行 4-7 两 chunk 对调", fs=12.5, fc=GREEN)

def table(x, data, hl_rows, hl_fill, note_row=None, note_text=""):
    # 列头
    box(x, TY, 64, RH, "row i", fill="#e7ecf3", stroke="#9aa5b1", fs=13, bold=True)
    box(x + 68, TY, CW, RH, "chunk0", fill="#e7ecf3", stroke="#9aa5b1", fs=13, bold=True)
    box(x + 72 + CW, TY, CW, RH, "chunk1", fill="#e7ecf3", stroke="#9aa5b1", fs=13, bold=True)
    for i, (q0, q1) in enumerate(data):
        y = TY + (i + 1) * RH
        hl = i in hl_rows
        f = hl_fill if hl else ("#ffffff" if i % 2 == 0 else "#f5f7fa")
        s = (RED if x == LX else GREEN) if hl else "#bcccdc"
        box(x, y, 64, RH, str(i), fill="#e7ecf3", stroke="#9aa5b1", fs=13, bold=True)
        box(x + 68, y, CW, RH, str(q0), fill=f, stroke=s, fs=15, bold=hl, fc=s if hl else "#1f2933")
        box(x + 72 + CW, y, CW, RH, str(q1), fill=f, stroke=s, fs=15, bold=hl, fc=s if hl else "#1f2933")
    if note_row is not None:
        y = TY + (note_row + 1) * RH + 6
        text(x + 148 + CW, y, 140, 22, note_text, fs=12, fc=RED)

before = [(0,4),(8,12),(16,20),(24,28),(0,4),(8,12),(16,20),(24,28)]
after  = [(0,4),(8,12),(16,20),(24,28),(4,0),(12,8),(20,16),(28,24)]
table(LX, before, hl_rows={4,5,6,7}, hl_fill="#fde9d0", note_row=4, note_text="&lt;- 与行 0 重复：2-way")
table(RX, after,  hl_rows={4,5,6,7}, hl_fill="#c9e4c8", note_row=4, note_text="&lt;- flipped：8 行铺满 32 bank")

# 中间 XOR 箭头：左表行 4-7 块 -> 右表行 4-7 块
mid_x1, mid_x2 = LX + 148 + 2 * CW, RX
arrow([(mid_x1, TY + 5 * RH - 20), (520, TY + 5 * RH - 20), (520, TY + 4.5 * RH), (mid_x2, TY + 4.5 * RH)], color=ORANGE, width=3)
text(452, TY + 4.5 * RH - 26, 140, 20, "XOR (i&gt;&gt;2)&amp;1", fs=12.5, fc=ORANGE, bold=True)

# quad 带对比
text(30, QY - 30, 1000, 22, "ldmatrix.x4 phase0 读到的 quad 序列（lanes 0..7 = rows 0..7, chunk0）：", fs=14.5, bold=True)
qb = [0,8,16,24,0,8,16,24]
qa = [0,8,16,24,4,12,20,28]
label = ["row0","row1","row2","row3","row4","row5","row6","row7"]
for k in range(8):
    x = LX + k * (QW + 4)
    rep = k >= 4
    box(x, QY, QW, QH, str(qb[k]), fill="#fde9d0" if rep else "#ffffff",
        stroke=RED if rep else "#bcccdc", fs=14, bold=rep, fc=RED if rep else "#1f2933")
    box(x, QY + QH + 64, QW, QH, str(qa[k]), fill="#c9e4c8" if k >= 4 else "#ffffff",
        stroke=GREEN if k >= 4 else "#bcccdc", fs=14, bold=k >= 4, fc=GREEN if k >= 4 else "#1f2933")
    text(x, QY + QH + 4, QW, 18, label[k], fs=10.5, fc="#627d98", align="center")
# 重复标注：before 带重复段连线
def bracket(x1, x2, y, color, label, ly):
    arrow([(x1, y), (x1, y + 10), (x2, y + 10), (x2, y)], color=color, width=1.5)
    text((x1 + x2) // 2 - 70, ly, 200, 18, label, fs=11.5, fc=color, align="center")
bx0, bx1 = LX, LX + 3 * (QW + 4)
by0, by1 = LX + 4 * (QW + 4), LX + 7 * (QW + 4)
text(LX, QY - 2, 500, 18, "before", fs=12.5, fc=RED, bold=True)
text(LX, QY + QH + 62, 500, 18, "after", fs=12.5, fc=GREEN, bold=True)

# 右侧结论卡
CY = QY - 6
box(700, CY, 350, 60, "before: quads {0,8,16,24,0,8,16,24}", fill="#fde9d0", stroke=RED, fs=13.5, fc="#8c2f0f")
box(700, CY + 66, 350, 46, "每个 quad 服务 2 行 -> 2-way bank conflict", fill="#ffffff", stroke=RED, fs=13, fc=RED)
box(700, CY + 138, 350, 60, "after: quads {0,8,16,24,4,12,20,28}", fill="#c9e4c8", stroke=GREEN, fs=13.5, fc="#1e5631")
box(700, CY + 204, 350, 46, "8 个互异 quad 铺满 32 bank -> 1-way", fill="#ffffff", stroke=GREEN, fs=13, fc=GREEN)

# 底注
text(30, H - 96, 1020, 18, "chunk1（lanes 16..31）镜像同一模式。", fs=12.5, fc="#48586a")
text(30, H - 76, 1020, 18, "BK=64 统一 tile 中只有 slice 内翻转存活 -> 每 phase 4-way，见 FIG-12-3", fs=12.5, fc="#48586a")

xml = f'''<mxfile host="app.diagrams.net">
  <diagram id="fig12-1" name="FIG-12-1">
    <mxGraphModel dx="{W}" dy="{H}" grid="0" page="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">
      <root><mxCell id="0"/><mxCell id="1" parent="0"/>
        {''.join(parts)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
out = sys.argv[1] if len(sys.argv) > 1 else 'fig-12-1.drawio'
open(out, 'w').write(xml)
print(f'wrote {out}')
