#!/usr/bin/env python3
# gen_fig22_2.py — FIG-22-2：知乎 @竹熙佳处《写给大家看的 CuTe 教程：tiled mma》fig-1 重建
# 来源：figures/zhihu/zhuxijiachu-tiled-mma/fig-1.jpg（5822x3306，原图本地归档）
# 重建原则：忠实原图主网格（16x8、thread 成对着色、红虚线高亮 thread-0、紫色虚线引出
# 2x2 data callout、N8/M16 尺寸箭头、省略行压缩），配色映射为书内系列色；
# 右侧补 A/B/C fragment 大小卡片与「atom = 最小可复制单元」卡片（任务主题）。
import os, sys

parts = []
def P(s): parts.append(s)

def box(x, y, w, h, t="", fill="#ffffff", stroke="#bcccdc", fs=13, fc="#1f2933",
        dashed=0, sw=1):
    st = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
          f"fontSize={fs};fontColor={fc};align=center;verticalAlign=middle;fontFamily=Helvetica")
    if dashed: st += ";dashed=1"
    if sw != 1: st += f";strokeWidth={sw}"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def text(x, y, w, h, t, fs=13, fc="#102a43", align="left"):
    st = (f"text;html=1;align={align};verticalAlign=middle;fontSize={fs};"
          f"fontColor={fc};fontFamily=Helvetica")
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def arrow(pts, color="#48586a", width=1.5, dashed=0, both=0):
    st = "endArrow=blockThin;endFill=1;html=1;strokeColor=%s;strokeWidth=%s;rounded=0;" % (color, width)
    if both: st = "startArrow=blockThin;startFill=1;" + st
    if dashed: st += "dashed=1;"
    P(f'<mxCell value="" style="{st}" edge="1" parent="1">'
      f'<mxGeometry relative="1" as="geometry"><Array as="points">'
      + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
      + f'</Array><mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
        f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/></mxGeometry></mxCell>')

BLUE, GREEN, ORANGE, RED, PURPLE = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#6a51a3"
TEAL, ROSE, OLIVE, GRAYTX = "#0e7490", "#be185d", "#4d7c0f", "#48586a"
LBF, LGF, LOF, LRF, LPF, LTF, LSF, LVF = ("#deebf7", "#dff0e2", "#fdeadb", "#fde4e4",
                                          "#e8e1f3", "#d9eef3", "#f8e1eb", "#e9f0da")
GRAYBG = "#e7ecf3"
# thread 8 色循环：红/蓝/绿/橙/紫/青/玫/橄榄（深色描边+字，浅色填充）
THR = [(RED, LRF), (BLUE, LBF), (GREEN, LGF), (ORANGE, LOF),
       (PURPLE, LPF), (TEAL, LTF), (ROSE, LSF), (OLIVE, LVF)]

def fig_22_2():
    parts.clear()
    text(20, 8, 1120, 26, "MMA atom 的 thread-value 布局：32 threads 分食 16×8 tile（SM80 m16n8k16 f16）", fs=16)
    text(20, 36, 1120, 18, "每 thread 持相邻 2 列 × 2 行（m 与 m+8 行）共 4 格；T = thread 编号，颜色按 8 色循环", fs=12, fc=GRAYTX)

    GX, GY, CW, CH = 150, 110, 46, 46
    # N/M 尺寸箭头（对应原图 N8 / M16）
    arrow([(GX, 72), (GX + 8 * CW, 72)], color=GRAYTX, width=1.5, both=1)
    text(300, 76, 90, 16, "N = 8", fs=12, fc=GRAYTX)
    arrow([(86, GY), (86, 400)], color=GRAYTX, width=1.5, both=1)
    text(16, 240, 62, 16, "M = 16", fs=12, fc=GRAYTX)
    # 列头
    for c in range(8):
        text(GX + c * CW, 90, CW - 2, 16, f"n{c}", fs=11.5, fc="#627d98", align="center")
    # 行：m=0,1,⋯,7,8,⋯,15（省略行压缩，同原图红点省略号）
    rows = [(0, GY), (1, GY + CH), (7, GY + 2 * CH + 26), (8, GY + 3 * CH + 26),
            (15, GY + 4 * CH + 60)]
    for m, y in rows:
        text(100, y + 13, 44, 20, f"m={m}", fs=11.5, fc="#627d98", align="right")
        for c in range(8):
            t = 4 * m + c // 2
            stroke, fill = THR[t % 8]
            box(GX + c * CW, y, CW - 2, CH - 2, f"T{t}", fill=fill, stroke=stroke,
                fs=13, fc=stroke)
    # 省略号（3 个灰点 × 两处）
    dotx = GX + 4 * CW - 3
    for dy in (204, 212, 220, 326, 333, 340):
        box(dotx, dy, 6, 6, fill="#8aa2b8", stroke="none")
    # thread-0 红虚线高亮（m=0 与 m=8 两处，各 2 格）
    box(GX, GY, 2 * CW - 2, CH - 2, fill="none", stroke=RED, dashed=1, sw=2.5)
    box(GX, rows[3][1], 2 * CW - 2, CH - 2, fill="none", stroke=RED, dashed=1, sw=2.5)
    # 网格下注释
    text(GX, 406, 380, 18, "行 m 与行 m+8 由同一组 threads 持有（T 编号重复出现）", fs=12, fc=GRAYTX)
    text(GX, 426, 400, 18, "thread t → 行 t/4，列 2*(t%4) 与 2*(t%4)+1", fs=12, fc=GRAYTX)
    text(GX, 446, 400, 18, "本图即 C/D（M×N）的 fragment 视图；k8 atom 的 A 同构", fs=12, fc=GRAYTX)

    # 紫色虚线引线：两处 thread-0 → 右上 callout（绕网格外走线，避开表头/标签/正文）
    arrow([(155, GY - 2), (155, 60), (545, 60), (545, 122), (576, 122)],
          color=PURPLE, width=1.5, dashed=1)
    arrow([(196, rows[3][1] + CH - 2), (196, 350), (556, 350), (556, 190), (576, 190)],
          color=PURPLE, width=1.5, dashed=1)

    # ── 右上：thread-0 callout（原图 2x2 data 面板）──
    box(580, 100, 570, 156, fill="#ffffff", stroke="#bcccdc")
    text(596, 110, 440, 20, "thread == 0 拿到 4 个 data（两处红虚线框）", fs=13, fc=PURPLE)
    for (cx, cy, v) in [(596, 136, "data (0, 0)"), (734, 136, "data (0, 1)"),
                        (596, 174, "data (8, 0)"), (734, 174, "data (8, 1)")]:
        box(cx, cy, 132, 32, v, fill=GRAYBG, stroke="#8aa2b8", fs=12.5)
    text(890, 134, 250, 18, "相邻 2 列 × 2 行（m 与 m+8）", fs=12, fc=GRAYTX)
    text(890, 154, 250, 18, "4 值 = 该 thread 的 fragment", fs=12, fc=GRAYTX)
    text(890, 174, 250, 18, "fragment 即 mma 指令的寄存器输入", fs=12, fc=GRAYTX)

    # ── 右中：A/B/C fragment 大小卡片 ──
    box(580, 272, 570, 220, fill="#ffffff", stroke="#bcccdc")
    text(596, 282, 540, 20, "m16n8k16 f16 atom：每 thread 的 fragment 大小", fs=13.5)
    for (chipy, name, c1, c2, l1, l2) in [
        (312, "A", BLUE, LBF, "A（16×16，M×K）：每 thread 8 值",
         "m 行: a0 a1 a4 a5；m+8 行: a2 a3 a6 a7"),
        (360, "B", GREEN, LGF, "B（16×8，K×N）：每 thread 4 值",
         "n 列固定；k 行: b0 b1，k+8 行: b2 b3"),
        (408, "C", ORANGE, LOF, "C/D（16×8，M×N）：每 thread 4 值",
         "即左图布局：m 行: c0 c1，m+8 行: c2 c3")]:
        box(596, chipy, 12, 42, fill=c2, stroke=c1)
        text(618, chipy - 2, 520, 18, l1, fs=12.5, fc=c1)
        text(618, chipy + 18, 530, 18, l2, fs=12, fc=GRAYTX)
    text(596, 462, 545, 18, "守恒：32×(8+4+4) = 512 值 = 256+128+128，恰为 A+B+C 全部元素", fs=12)

    # ── 右下：atom 最小可复制单元卡片 ──
    box(580, 508, 570, 188, fill=LGF, stroke=GREEN)
    text(596, 518, 540, 20, "atom = 最小可复制的 MMA 单元", fs=14, fc="#1e5631")
    text(596, 546, 545, 18, "一条 mma.sync（m16n8k16 f16）由 32 threads 各出一个 fragment，协作算出 16×8 输出", fs=12, fc="#1e5631")
    text(596, 568, 545, 18, "每 thread：A 8 值 / B 4 值 / C 4 值；D 累加回 C fragment", fs=12, fc="#1e5631")
    text(596, 590, 545, 18, "tile 更大 = atom 沿 M/N 复制堆叠（见左下），映射规律不变", fs=12, fc="#1e5631")
    text(596, 612, 545, 18, "CuTe：make_tiled_mma(atom, m, n) 即按复制数生成 TiledMMA", fs=11.5, fc="#1e5631")

    # ── 左下：atom 复制示意（任务主题：最小可复制单元）──
    text(60, 474, 380, 20, "atom 复制（tiling）→ 更大的 GEMM tile", fs=13)
    box(96, 498, 300, 124, fill="none", stroke="#8aa2b8", dashed=1)
    for (bx, by) in [(108, 512), (252, 512), (108, 566), (252, 566)]:
        box(bx, by, 132, 48, "atom 16×8", fill=LGF, stroke=GREEN, fs=11.5, fc="#1e5631")
    text(412, 512, 128, 18, "2×2 复制 →", fs=12, fc=GRAYTX)
    text(412, 532, 128, 18, "M32×N16 tile", fs=12, fc=GRAYTX)
    text(412, 552, 128, 18, "64 threads", fs=12, fc=GRAYTX)
    text(412, 572, 128, 18, "映射不变", fs=12, fc=GRAYTX)
    text(60, 640, 500, 18, "TiledMMA 的 atom 数 = (M/16)×(N/8)，每个 atom 内的映射与上图相同", fs=12, fc=GRAYTX)
    return mk(parts, 1160, 700, "fig-22-2")

def mk(parts, W, H, did):
    xml = ('<mxfile host="app.diagrams.net"><diagram id="%s" name="Page-1">'
           '<mxGraphModel dx="800" dy="600" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
           'arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d" math="0" shadow="0">'
           '<root><mxCell id="0"/><mxCell id="1" parent="0"/>%s</root></mxGraphModel></diagram></mxfile>'
           ) % (did, W, H, ''.join(parts))
    outdir = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "figures/drawio",
                          "fig-22-2-mma-atom-fragments")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, did + ".drawio")
    open(path, "w").write(xml)
    return path

if __name__ == "__main__":
    print("wrote", fig_22_2(), "OK")
