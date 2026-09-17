#!/usr/bin/env python3
# gen_fig23_1.py — FIG-23-1 SW128 原子物理排布（8x8 XOR 置换网格）
import sys

W, H = 1080, 470
CW, CH = 52, 38          # 格
GX, GY = 130, 96         # 网格原点（列头顶与说明行留 8px 净空）
parts = []
def P(s): parts.append(s)

def box(x, y, w, h, text, fill="#ffffff", stroke="#bcccdc", fs=13, bold=False, fc="#1f2933"):
    st = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
          f"fontSize={fs};fontColor={fc};align=center;verticalAlign=middle;fontFamily=Helvetica")
    if bold: st += ";fontStyle=1"
    P(f'<mxCell value="{text}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def text(x, y, w, h, t, fs=13, fc="#102a43", bold=False, align="left"):
    st = f"text;html=1;align={align};verticalAlign=middle;fontSize={fs};fontColor={fc};fontFamily=Helvetica"
    if bold: st += ";fontStyle=1"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

BLUE = "#2171b5"
# 标题
text(30, 14, 840, 24, "SW128 atom: 8x64 half = 8 rows x 128 B, 8 chunks of 16 B per row", fs=17, bold=True)
text(30, 40, 840, 20, "物理 chunk 位置 = j XOR (r mod 8)：行长 128B 刚好翻 3 个地址位，第 12 章的「对角线置换」在这里是位级精确的", fs=13, fc="#48586a")

# 列头 c0..c7（逻辑 chunk）
for j in range(8):
    box(GX + j * CW, GY - 30, CW, 30, f"c{j}", fill="#e7ecf3", stroke="#9aa5b1", fs=13, bold=True)
text(GX - 96, GY - 30, 90, 30, "logical chunk", fs=12, fc="#627d98", align="right")
# 行头 + 网格
note_r = {0: "identity", 1: "swap pairs", 4: "halves swap", 7: "reverse"}
for r in range(8):
    y = GY + r * CH
    box(GX - 46, y, 44, CH, f"r={r}", fill="#e7ecf3", stroke="#9aa5b1", fs=12, bold=True)
    if note_r.get(r):
        text(GX + 8 * CW + 12, y + 8, 110, 22, note_r[r], fs=11, fc="#627d98")
    for j in range(8):
        pj = j ^ r
        hl = "#bfd7ea" if (r in (1, 2, 4)) else ("#ffffff" if r % 2 == 0 else "#f5f7fa")
        box(GX + j * CW, y, CW, CH, f"{pj}", fill=hl, stroke="#bcccdc", fs=13,
            bold=(r in (1, 2, 4)), fc=BLUE if (r in (1, 2, 4)) else "#1f2933")

# 迁移示例线：r=1 行内 c0->c1（chunk0 从逻辑位置0迁到物理位置1）
# 用显式坐标折线（不绑 cell，避免锚点漂移）
def arrow(pts, color=BLUE, label=None, lx=0, ly=0):
    P(f'<mxCell value="" style="endArrow=blockThin;endFill=1;html=1;strokeColor={color};strokeWidth=2;rounded=0;" edge="1" parent="1">'
      f'<mxGeometry relative="1" as="geometry"><Array as="points">'
      + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
      + f'</Array><mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
        f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/></mxGeometry></mxCell>')
    if label:
        text(lx, ly, 130, 18, label, fs=11.5, fc=color)

# r=1: 逻辑 c0 -> 物理 c1；逻辑 c1 -> 物理 c0（交叉）
y1c = GY + 1 * CH + CH // 2
arrow([(GX + 8, y1c - 8), (GX + CW * 1.5, y1c - 26), (GX + CW * 1 + CW - 8, y1c - 8)])
# r=2: 逻辑 c0 -> 物理 c2
y2c = GY + 2 * CH + CH // 2
arrow([(GX + 8, y2c + 6), (GX + CW * 1.5, y2c + 30), (GX + CW * 2 + CW - 8, y2c + 6)])
# r=4: 逻辑 c0 -> 物理 c4（走行下方绕）
y4c = GY + 4 * CH + CH // 2
arrow([(GX + 8, y4c - 4), (GX + CW * 2, y4c - 52), (GX + CW * 4 + CW - 10, y4c - 52), (GX + CW * 4 + CW - 10, y4c - 4)])

# 公式侧栏
text(690, GY + 30, 200, 20, "物理偏移(r, c)，half 单位：", fs=12.5, fc="#102a43")
box(690, GY + 54, 200, 44, "smem = 64·r + 8·(j XOR r) + (c mod 8)", fill="#f0f4f8", stroke=BLUE, fs=12, bold=True, fc=BLUE)
text(690, GY + 106, 370, 18, "与 swizzle&lt;64&gt;（第 12 章）同构；", fs=11, fc="#48586a")
text(690, GY + 126, 370, 18, "TMA SWIZZLE_128B 写 64×64 half tile 实测 4096/4096 逐点一致", fs=11, fc="#48586a")

text(30, 414, 840, 20, "蓝格行 r=1/2/4 的箭头示例：逻辑 chunk c0 的物理落点分别翻 1/2/4 对应的地址位——同一张表就是 Swizzle&lt;3,4,3&gt; 在 128B 行上的全部行为", fs=12, fc="#48586a")

xml = f'''<mxfile host="app.diagrams.net">
  <diagram id="fig23-1" name="FIG-23-1">
    <mxGraphModel dx="{W}" dy="{H}" grid="0" page="1" pageWidth="{W}" pageHeight="440" math="0" shadow="0">
      <root><mxCell id="0"/><mxCell id="1" parent="0"/>
        {''.join(parts)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
out = sys.argv[1] if len(sys.argv) > 1 else 'fig-23-1.drawio'
open(out, 'w').write(xml)
print(f'wrote {out}')
