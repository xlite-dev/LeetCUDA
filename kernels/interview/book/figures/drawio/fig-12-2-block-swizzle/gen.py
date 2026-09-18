#!/usr/bin/env python3
# gen_fig12_2.py — FIG-12-2 block swizzle: C-tile grid 光栅化前后（L2 footprint 对比）
# 左：default 2D grid 宽条带 resident；右：3D swizzle 近方块 resident；底部 footprint/L2 实测
import sys

W, H = 1120, 660
LGX, LGY = 80, 160        # 左网格原点
RGX, RGY = 656, 160       # 右网格原点
CW, CH = 19, 19           # 缩略格（纯填色无文字）
parts = []
def P(s): parts.append(s)

def box(x, y, w, h, text="", fill="#ffffff", stroke="#bcccdc", fs=13, bold=False, fc="#1f2933"):
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

def arrow(pts, color="#d97706", width=3, dashed=0):
    st = (f"endArrow=blockThin;endFill=1;html=1;strokeColor={color};strokeWidth={width};"
          f"rounded=0;dashed={dashed}")
    P(f'<mxCell value="" style="{st}" edge="1" parent="1">'
      f'<mxGeometry relative="1" as="geometry"><Array as="points">'
      + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
      + f'</Array><mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
        f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/></mxGeometry></mxCell>')

ORANGE, GRAY, BLUE, RED = "#d97706", "#e7ecf3", "#2171b5", "#dc2626"

# 标题
text(30, 12, 900, 26, "block swizzle：C-tile 光栅化前后（L2 footprint 对比）", fs=18, bold=True)
text(30, 42, 1000, 20, "T_m=16 x T_n=128，常驻集 R ≈ 110 blocks（110 SM × 1 block/SM），组宽 S=16；resident set = 一个 wave 里同时驻留 SM 的 C-tile 集合", fs=13.5, fc="#48586a")

# 左：default 2D grid
text(30, LGY - 64, 460, 22, "[default] 2D grid：x 最快，行优先光栅", fs=15, bold=True, fc=RED)
text(30, LGY - 42, 460, 18, "y=0 一行即被 110 blocks 填满", fs=12, fc=RED)
LC, LR = 24, 8
for r in range(LR):
    for c in range(LC):
        f = "#fde9d0" if r < 1 else GRAY
        s = ORANGE if r < 1 else "#d9e2ec"
        box(LGX + c * CW, LGY + r * CH, CW - 2, CH - 2, fill=f, stroke=s)
text(LGX, LGY + LR * CH + 8, 360, 18, "x: 0 ......... 109（W=110 块，网格 T_n=128 列）", fs=12, fc="#627d98")
text(LGX, LGY + LR * CH + 28, 340, 20, "resident = W=110 x H=1 宽条带", fs=13, fc=RED, bold=True)
brace_y = LGY + 1 * CH
arrow([(LGX - 14, LGY), (LGX - 14, brace_y)], color=RED, width=2)
text(LGX - 70, LGY + CH - 8, 46, 20, "H=1 行", fs=12, fc=RED, align="right")

# 中间转换箭头
arrow([(LGX + LC * CW + 10, LGY + LR * CH // 2 + 4), (RGX - 16, LGY + LR * CH // 2 + 4)], color=BLUE, width=3)
text(LGX + LC * CW + 4, LGY + LR * CH // 2 - 44, 100, 20, "bx = z*gridDim.x", fs=11, fc=BLUE, bold=True)
text(LGX + LC * CW + 4, LGY + LR * CH // 2 - 22, 100, 20, "(z 组折叠进 bx)", fs=11, fc=BLUE)

# 右：3D swizzle
text(RGX, LGY - 64, 460, 22, "[swizzle] 3D grid：z 组内 16 列，近似方块", fs=15, bold=True, fc="#2e8540")
text(RGX, LGY - 42, 460, 18, "z=0: bx=0..15；共 7 组 x 16 tiles（112 槽驻留 110 块）", fs=12, fc="#2e8540")
RC, RR = 16, 7
for r in range(RR + 4):
    for c in range(RC + 4):
        inr = 2 <= r < 2 + RR and 2 <= c < 2 + RC
        f = "#c9e4c8" if inr else GRAY
        s = "#2e8540" if inr else "#d9e2ec"
        box(RGX + c * CW, LGY + r * CH, CW - 2, CH - 2, fill=f, stroke=s)
text(RGX, LGY + (RR + 4) * CH + 8, 400, 20, "resident = 16 x 7 近方块（同 110 tiles）", fs=13, fc="#2e8540", bold=True)
text(RGX, LGY + (RR + 4) * CH + 30, 420, 18, "launch 顺序 x -> y -> z 不变，bx 重排实现", fs=12, fc="#627d98")

# 底部 footprint 公式 + 实测三卡
FY = 520
text(30, FY - 30, 900, 22, "L2 footprint F（每个 K-slab 的 unique A+B tiles）：F = (H*BM + W*BN) * K * 2B", fs=14.5, bold=True)
box(40, FY, 330, 70, "", fill="#fde9d0", stroke=RED)
text(56, FY + 8, 310, 24, "default：(1*128 + 110*128) * K * 2B", fs=12.5, fc="#8c2f0f")
text(56, FY + 36, 310, 24, "= 14208 * K * 2B", fs=14, fc="#8c2f0f", bold=True)
box(400, FY, 330, 70, "", fill="#c9e4c8", stroke="#2e8540")
text(416, FY + 8, 310, 24, "swizzle：(7*128 + 16*128) * K * 2B", fs=12.5, fc="#1e5631")
text(416, FY + 36, 310, 24, "= 2944 * K * 2B", fs=14, fc="#1e5631", bold=True)
box(760, FY, 330, 70, "footprint 缩小 4.8x", fill="#e7ecf3", stroke=BLUE, fs=16, bold=True, fc=BLUE)
text(40, FY + 78, 1000, 20, "实测 L2 sector hit rate：64.45% -> 96.15%，端到端 117.9 -> 122.9 TFLOPS（+4.2%，PRO 5000，2048x16384x4096 HGEMM）", fs=13, fc="#48586a")

xml = f'''<mxfile host="app.diagrams.net">
  <diagram id="fig12-2" name="FIG-12-2">
    <mxGraphModel dx="{W}" dy="{H}" grid="0" page="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">
      <root><mxCell id="0"/><mxCell id="1" parent="0"/>
        {''.join(parts)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
out = sys.argv[1] if len(sys.argv) > 1 else 'fig-12-2.drawio'
open(out, 'w').write(xml)
print(f'wrote {out}')
