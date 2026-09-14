#!/usr/bin/env python3
# gen_fig20_1.py — FIG-20-1 colex 线性化表（L=(4,8):(1,4)）
# 左：8x4 偏移表（行=c1，列=c0，格内 c0+4*c1）；右：展平 1D 数带；高亮 f(1,3)=13 / f(3,1)=7
import sys

CELL_W, CELL_H = 54, 34
GX, GY = 60, 116          # 左表原点（列头顶与副标题行留净空）
RX, RY = 520, 116         # 右数带原点（8 列 x 4 行）
W, H = 1010, 470

def cell(x, y, w, h, text, fill="#ffffff", stroke="#9aa5b1", fontsize=13, bold=False, fontcolor="#1f2933"):
    style = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"fontSize={fontsize};fontColor={fontcolor};align=center;verticalAlign=middle;"
             f"fontFamily=Helvetica")
    if bold:
        style += ";fontStyle=1"
    return (f'<mxCell id="c{cell.n}" value="{text}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def arrow(x1, y1, x2, y2, color="#2171b5", dashed=0):
    style = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor={color};strokeWidth=2;"
             f"endArrow=blockThin;endFill=1;dashed={dashed}")
    return (f'<mxCell id="c{cell.n}" style="{style}" edge="1" parent="1" '
            f'source="c{cell.src}" target="c{cell.dst}"><mxGeometry relative="1" as="geometry"/></mxCell>')

cell.n = 0

parts = []
def P(s):
    cell.n += 1
    parts.append(s)

# 标题
P(f'<mxCell id="title" value="Layout L = (4,8):(1,4)：" '
  f'style="text;html=1;align=left;verticalAlign=middle;fontSize=18;fontStyle=1;fontColor=#102a43;" vertex="1" parent="1">'
  f'<mxGeometry x="50" y="20" width="220" height="26" as="geometry"/></mxCell>')
P(f'<mxCell id="title1b" value="行 = c1（慢），列 = c0（快）" '
  f'style="text;html=1;align=left;verticalAlign=middle;fontSize=16;fontColor=#102a43;" vertex="1" parent="1">'
  f'<mxGeometry x="280" y="22" width="300" height="24" as="geometry"/></mxCell>')
P(f'<mxCell id="title2" value="f(c0, c1) = c0 + 4·c1&#160;&#160;（colex 读法：c0 变化最快）" '
  f'style="text;html=1;align=left;verticalAlign=middle;fontSize=15;fontColor=#48586a;" vertex="1" parent="1">'
  f'<mxGeometry x="50" y="48" width="640" height="24" as="geometry"/></mxCell>')

# 左表：列头 c0
P(cell(GX, GY - CELL_H, 44, CELL_H, "c1 ╲ c0", fill="#e7ecf3", bold=True))
for j in range(4):
    P(cell(GX + 44 + j * CELL_W, GY - CELL_H, CELL_W, CELL_H, str(j), fill="#e7ecf3", bold=True))
# 行头 + 格
hl = {(1, 3): ("#bfd7ea", "#2171b5"), (3, 1): ("#c9e4c8", "#2e8540")}  # (c0,c1): (fill, stroke)
left_ids = {}
for i in range(8):
    P(cell(GX, GY + i * CELL_H, 44, CELL_H, str(i), fill="#e7ecf3", bold=True))
    for j in range(4):
        v = j + 4 * i
        if (j, i) in hl:
            f, s = hl[(j, i)]
            left_ids[(j, i)] = cell.n
            P(cell(GX + 44 + j * CELL_W, GY + i * CELL_H, CELL_W, CELL_H, str(v), fill=f, stroke=s, bold=True, fontcolor=s))
        else:
            P(cell(GX + 44 + j * CELL_W, GY + i * CELL_H, CELL_W, CELL_H, str(v)))

# 右数带：offset 0..31 按 8 列排（col-major 展平：offset k 在 (k%8, k/8) 位置显示原表数值顺序）
P(f'<mxCell id="rt" value="展平后 1D 数带（offset = 线性下标）" '
  f'style="text;html=1;align=left;verticalAlign=middle;fontSize=13;fontStyle=1;fontColor=#102a43;" vertex="1" parent="1">'
  f'<mxGeometry x="{RX}" y="{GY - CELL_H - 4}" width="300" height="{CELL_H}" as="geometry"/></mxCell>')
right_ids = {}
for k in range(32):
    r, c = k % 8, k // 8
    x = RX + c * (CELL_W + 4)
    y = RY + r * CELL_H
    if k in (13, 7):
        f, s = hl[(1, 3)] if k == 13 else hl[(3, 1)]
        right_ids[k] = cell.n
        P(cell(x, y, CELL_W, CELL_H, str(k), fill=f, stroke=s, bold=True, fontcolor=s))
    else:
        P(cell(x, y, CELL_W, CELL_H, str(k)))

# 高亮连线：左(1,3)->右13（蓝），左(3,1)->右7（绿）
def edge(src_id, dst_id, color, wy1, wy2):
    cell.n += 1
    return (f'<mxCell id="c{cell.n}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;'
            f'strokeColor={color};strokeWidth=2;endArrow=blockThin;endFill=1;exitX=1;exitY=0.5;'
            f'exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" '
            f'source="c{src_id}" target="c{dst_id}"><mxGeometry relative="1" as="geometry">'
            f'<Array as="points"><mxPoint x="440" y="{wy1}"/><mxPoint x="440" y="{wy2}"/></Array>'
            f'</mxGeometry></mxCell>')

P(edge(left_ids[(1, 3)], right_ids[13], "#2171b5", 195, 255))
P(edge(left_ids[(3, 1)], right_ids[7], "#2e8540", 135, 315))

# 读法标注
P(f'<mxCell id="note1" value="f(1,3) = 1·1 + 3·4 = 13" style="text;html=1;align=left;fontSize=14;fontColor=#2171b5;fontStyle=1;" '
  f'vertex="1" parent="1"><mxGeometry x="775" y="130" width="220" height="24" as="geometry"/></mxCell>')
P(f'<mxCell id="note2" value="f(3,1) = 3·1 + 1·4 = 7" style="text;html=1;align=left;fontSize=14;fontColor=#2e8540;fontStyle=1;" '
  f'vertex="1" parent="1"><mxGeometry x="775" y="160" width="220" height="24" as="geometry"/></mxCell>')
P(f'<mxCell id="note3" value="整数输入时 f(x)=x" style="text;html=1;align=left;fontSize=13;fontColor=#48586a;" '
  f'vertex="1" parent="1"><mxGeometry x="775" y="194" width="225" height="20" as="geometry"/></mxCell>')
P(f'<mxCell id="note3b" value="（compact + col-major）" style="text;html=1;align=left;fontSize=13;fontColor=#48586a;" '
  f'vertex="1" parent="1"><mxGeometry x="775" y="212" width="225" height="20" as="geometry"/></mxCell>')

# 底部对照
P(f'<mxCell id="foot" value="对照：row-major 版 (4,8):(8,1) 的 f(1,3) = 1·8 + 3·1 = 11 —— 同一 shape，stride 决定读序" '
  f'style="text;html=1;align=left;fontSize=13;fontColor=#48586a;" vertex="1" parent="1">'
  f'<mxGeometry x="60" y="430" width="920" height="24" as="geometry"/></mxCell>')

xml = f'''<mxfile host="app.diagrams.net">
  <diagram id="fig20-1" name="FIG-20-1">
    <mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {''.join(parts)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
out = sys.argv[1] if len(sys.argv) > 1 else 'fig-20-1-colex/fig-20-1.drawio'
open(out, 'w').write(xml)
print(f'wrote {out}: {cell.n} cells')
