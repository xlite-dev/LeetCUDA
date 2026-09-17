#!/usr/bin/env python3
# gen_fig25_1.py — FIG-25-1 flash_attn.cuh CuTe 块结构对照（四层：共享层 -> 三实现 -> smoke）
import sys

W, H = 1150, 900
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

def arrow(pts, color="#627d98", dashed=0, width=2):
    st = (f"endArrow=blockThin;endFill=1;html=1;strokeColor={color};strokeWidth={width};"
          f"rounded=0;dashed={dashed}")
    P(f'<mxCell value="" style="{st}" edge="1" parent="1">'
      f'<mxGeometry relative="1" as="geometry"><Array as="points">'
      + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
      + f'</Array><mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
        f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/></mxGeometry></mxCell>')

BLUE, GREEN, ORANGE, GRAY = "#2171b5", "#2e8540", "#d97706", "#e7ecf3"

text(30, 12, 900, 26, "flash_attn.cuh CuTe 块（L2196-3488）结构对照", fs=18, bold=True)
text(30, 42, 1050, 20, "共享层（traits + 五个工具函数）被三种调度结构复用；41 行实现 4 是 TMA 搬运自检的最小闭环", fs=13.5, fc="#48586a")

# 第一层：共享层（两个 traits 卡 + 工具函数横条）
TY = 86
text(30, TY, 300, 20, "共享层（L2197-2490）", fs=15, bold=True, fc=BLUE)
t1 = [("FlashAttn2CuTeTraits&lt;D&gt;  Br=128", "SmemLayoutQ (128,D) SW128", "SmemLayoutKV (64,D) SW128", "SmemLayoutVt = comp(KV)", "TiledMma (8,1,1)x(128,16,16)")]
t2 = [("FlashAttn3CuTeTraits&lt;D&gt;  Br=64", "SmemLayoutQKV (64,D) SW128", "SmemLayoutVt = comp(QKV)", "TiledMma (4,1,1)x(64,16,16)", "")]
for i, (title, *lines) in enumerate((t1[0], t2[0])):
    x = 60 + i * 520
    box(x, TY + 24, 500, 130, "", fill="#f0f6fb", stroke=BLUE)
    text(x + 14, TY + 30, 470, 20, title, fs=13.5, bold=True, fc=BLUE)
    ls = [l for l in (lines if i == 0 else (t2[0][1], t2[0][2], t2[0][3], t2[0][4])) if l]
    for j, l in enumerate(ls):
        text(x + 14, TY + 56 + j * 24, 470, 22, l, fs=12.5, fc="#243b53")
box(60, TY + 164, 1020, 44, "", fill="#f0f6fb", stroke=BLUE)
text(74, TY + 172, 1000, 20, "convert_layout_acc_rowcol (4,MM,NN)->((2,MM),(2,NN))   |   convert_layout_acc_Aregs (零拷贝)   |   convert_type / gemm_ss / gemm_rs", fs=12.5, fc="#243b53")

# 第二层：三个实现卡
IY = TY + 250
impl = [
    ("实现1 cp.async", "256 threads", "sQ(128,D) / sK[Sk](64,D) / sV(64,D)", "cp.async 组栈 wait&lt;N&gt;", "全员搬运 + 计算", BLUE),
    ("实现2 TMA + WS", "384 = 128P + 256C", "mbarrier 事务计数", "V-first, K-after", "早释放 K", ORANGE),
    ("实现3 FA3 双 consumer", "384 = 128P + 2x128C", "cid = tile XOR 1 路由", "每 C 独立 bank 屏障", "收尾 smem scratch + 归并 (max pivot)", GREEN),
]
for i, (name, th, l1, l2, l3, c) in enumerate(impl):
    x = 60 + i * 350
    box(x, IY, 330, 150, "", fill="#ffffff", stroke=c)
    box(x, IY, 330, 30, name, fill=GRAY, stroke=c, fs=13.5, bold=True, fc=c)
    text(x + 14, IY + 36, 300, 20, th, fs=12.5, fc="#243b53", bold=True)
    text(x + 14, IY + 60, 300, 20, l1, fs=12, fc="#48586a")
    text(x + 14, IY + 82, 300, 20, l2, fs=12, fc="#48586a")
    text(x + 14, IY + 104, 300, 40, l3, fs=12, fc="#48586a")
    # 实线箭头：共享层 -> 实现
    arrow([(x + 165, TY + 196), (x + 165, IY - 4)], color=c, width=2)

# 第三层：实现4 smoke 横贯卡
SY = IY + 200
box(60, SY, 1020, 66, "", fill="#f7f3ec", stroke="#b45309")
text(74, SY + 8, 1000, 22, "实现4 smoke（41 行，L3454-3495）：TMA descriptor + SW128 布局 + mbarrier 事务屏障的最小闭环", fs=13.5, bold=True, fc="#b45309")
text(74, SY + 34, 1000, 20, "不计算 attention——只验证「搬运 + 布局 + 屏障」三件套本身；三个实现的公共底座照此自检", fs=12.5, fc="#7c5a1e")
# 虚线箭头：三实现 -> 实现4
for i in range(3):
    x = 60 + i * 350 + 165
    arrow([(x, IY + 154), (x, SY - 4)], color="#b45309", width=2, dashed=1)

# 底部图例
LY = SY + 92
box(60, LY, 14, 14, "", fill="#f0f6fb", stroke=BLUE)
text(82, LY - 2, 240, 20, "布局类型（共享层）", fs=12.5, fc="#48586a")
box(320, LY, 14, 14, "", fill=GRAY, stroke=ORANGE)
text(342, LY - 2, 260, 20, "调度结构（三实现各异）", fs=12.5, fc="#48586a")
text(560, LY - 2, 520, 20, "实线 = 复用同一份布局类型；虚线 = 公共底座自检", fs=12.5, fc="#48586a")

xml = f'''<mxfile host="app.diagrams.net">
  <diagram id="fig25-1" name="FIG-25-1">
    <mxGraphModel dx="{W}" dy="{H}" grid="0" page="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">
      <root><mxCell id="0"/><mxCell id="1" parent="0"/>
        {''.join(parts)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
out = sys.argv[1] if len(sys.argv) > 1 else 'fig-25-1.drawio'
open(out, 'w').write(xml)
print(f'wrote {out}')
