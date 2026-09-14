#!/usr/bin/env python3
# gen_batch_d.py — 批次 D：FIG-20-2 / 20-3 / 21-1 / 21-2 / 21-3 / 24-2
import os, sys
import xml.etree.ElementTree as ET

def mk(parts, W, H, did):
    return (f'<mxfile host="app.diagrams.net"><diagram id="{did}" name="{did}">'
            f'<mxGraphModel dx="800" dy="600" grid="0" page="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
            f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{"".join(parts)}'
            f'</root></mxGraphModel></diagram></mxfile>')

class F:
    def __init__(self):
        self.p = []
    def box(self, x, y, w, h, t="", fill="#ffffff", stroke="#bcccdc", fs=13, bold=False, fc="#1f2933"):
        st = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
              f"fontSize={fs};fontColor={fc};align=center;verticalAlign=middle;fontFamily=Helvetica")
        if bold: st += ";fontStyle=1"
        self.p.append(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
                      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    def text(self, x, y, w, h, t, fs=13, fc="#102a43", bold=False, align="left"):
        st = f"text;html=1;align={align};verticalAlign=middle;fontSize={fs};fontColor={fc};fontFamily=Helvetica"
        if bold: st += ";fontStyle=1"
        self.p.append(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
                      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    def line(self, pts, color="#2171b5", width=2, dashed=0, arrow_end=1):
        ae = "blockThin;endFill=1" if arrow_end else "none"
        st = f"endArrow={ae};html=1;strokeColor={color};strokeWidth={width};rounded=0;dashed={dashed}"
        self.p.append(f'<mxCell value="" style="{st}" edge="1" parent="1">'
                      f'<mxGeometry relative="1" as="geometry"><Array as="points">'
                      + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
                      + f'</Array><mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
                        f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/></mxGeometry></mxCell>')

BLUE, GREEN, ORANGE, RED, GRAY = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#e7ecf3"
LBF, LGF, LOF = "#f0f6fb", "#c9e4c8", "#fde9d0"

def fig_20_2():
    f = F()
    f.text(24, 12, 1112, 28, "mode 树与四种 mode 操作（Lg = ((2,4),(3,2)):((1,2),(8,24))）", fs=16, bold=True)
    f.box(60, 60, 300, 44, "Lg 根节点（rank-2 嵌套布局）", fill="#d9e2ec", stroke="#486581", fs=13, bold=True)
    f.box(24, 140, 180, 44, "mode0 = (2,4):(1,2)", fill=LBF, stroke=BLUE, fs=12.5, bold=True, fc=BLUE)
    f.box(216, 140, 180, 44, "mode1 = (3,2):(8,24)", fill=LBF, stroke=BLUE, fs=12.5, bold=True, fc=BLUE)
    f.line([(150, 104), (114, 140)], color="#486581", width=2, arrow_end=0)
    f.line([(270, 104), (306, 140)], color="#486581", width=2, arrow_end=0)
    f.text(24, 192, 380, 20, "mode0：2 元素 stride 1，4 元素 stride 2", fs=12, fc="#48586a")
    f.text(24, 214, 380, 20, "mode1：3 元素 stride 8，2 元素 stride 24", fs=12, fc="#48586a")
    f.box(470, 56, 666, 32, "四种 mode 操作（结果 shape:stride）", fill="#d9e2ec", stroke="#486581", fs=13.5, bold=True)
    f.box(470, 96, 666, 40, "projection：layout&lt;0&gt;(Lg) = (2,4):(1,2)；layout&lt;1&gt;(Lg) = (3,2):(8,24)", fill="#ffffff", stroke=BLUE, fs=12.5)
    f.box(470, 144, 666, 40, "merge：group&lt;0,2&gt;(Lg) = ((2,4),(3,2)):((1,2),(8,24))（rank-1）", fill="#ffffff", stroke=GREEN, fs=12.5)
    f.box(470, 192, 666, 40, "flatten：Lg = (2,4,3,2):(1,2,8,24)", fill="#ffffff", stroke=ORANGE, fs=12.5)
    f.box(470, 240, 666, 44, "coalesce：Lg = (48):(1) —— 一段连续 48 格", fill=LGF, stroke=GREEN, fs=13, bold=True, fc="#1e5631")
    f.text(470, 292, 666, 22, "strides 链式衔接：1 -&gt; 2 -&gt; 8 -&gt; 24（2 = 1*2，8 = 2*4，24 = 8*3），编译期消除冗余 mode", fs=12.5, fc="#48586a")
    f.box(24, 330, 1112, 42, "kernel 里「fragment mode 与 rest mode 分离」就是一次投影；合并/展平把嵌套结构变成可直接索引的线性结构", fill=LOF, stroke=ORANGE, fs=13, bold=True, fc="#7c3a06")
    return mk(f.p, 1160, 384, "fig-20-2")

def fig_20_3():
    f = F()
    f.text(24, 12, 1112, 28, "FA2 P 矩阵布局转换：两条路线，一个结果（zero-copy）", fs=16, bold=True)
    f.box(24, 56, 300, 56, "S（C fragment）(4, MMA_M, MMA_N)", fill=LBF, stroke=BLUE, fs=13.5, bold=True, fc=BLUE)
    f.box(800, 56, 336, 56, "P（A fragment）((4,2), MMA_M, MMA_N/2)", fill=LBF, stroke=BLUE, fs=13.5, bold=True, fc=BLUE)
    f.box(24, 152, 1112, 32, "路线 a（tri-dao，几何重排）—— 本书 convert_layout_acc_Aregs 采用（flash_attn.cuh，被 ffpa_attn.cuh 复用）", fill="#d9e2ec", stroke=BLUE, fs=13, bold=True, fc=BLUE)
    f.box(40, 192, 520, 40, "logical_divide：拆 MMA_N -&gt; (2, MMA_N/2)", fill="#ffffff", stroke=BLUE, fs=12.5)
    f.box(596, 192, 524, 40, "regroup：mode0 + mode2.0，mode1，mode2.1（四碎片重组）", fill="#ffffff", stroke=BLUE, fs=12.5)
    f.line([(560, 212), (596, 212)], color=BLUE, width=2)
    f.box(24, 260, 1112, 32, "路线 b（reed，代数复合）", fill="#d9e2ec", stroke=GREEN, fs=13, bold=True, fc=GREEN)
    f.box(40, 300, 340, 40, "inv = left_inverse(Layout_C)", fill="#ffffff", stroke=GREEN, fs=12.5)
    f.box(400, 300, 340, 40, "a = inv . compose(Layout_A)", fill="#ffffff", stroke=GREEN, fs=12.5)
    f.box(760, 300, 360, 40, "out = Layout_C . compose(a)", fill="#ffffff", stroke=GREEN, fs=12.5)
    f.line([(380, 320), (400, 320)], color=GREEN, width=2)
    f.line([(740, 320), (760, 320)], color=GREEN, width=2)
    f.text(40, 348, 1080, 22, "映射语言：(x_a, y_a) --Layout_A--&gt; offset0 --inv--&gt; (x_c, y_c) --Layout_C--&gt; offset1（compose 1/2 对应 eq. 20-fa2-reed2/reed3）", fs=12.5, fc="#48586a")
    f.line([(174, 112), (174, 192)], color=BLUE, width=2)
    f.line([(174, 232), (174, 260)], color=BLUE, width=2)
    f.line([(968, 112), (968, 192)], color=GREEN, width=2, dashed=1)
    f.line([(968, 232), (968, 260)], color=GREEN, width=2, dashed=1)
    f.box(24, 386, 1112, 44, "两条路线结果等价：zero-copy —— 同一批寄存器，只是索引语言不同", fill=LGF, stroke=GREEN, fs=14, bold=True, fc="#1e5631")
    return mk(f.p, 1160, 440, "fig-20-3")

def fig_21_1():
    f = F()
    f.text(24, 12, 1112, 28, "Tensor = Engine + Layout（2D 例子，每元素 4B）", fs=16, bold=True)
    f.text(24, 56, 240, 24, "逻辑坐标 (m, n)", fs=13.5, bold=True, fc="#102a43")
    for m in range(4):
        for n in range(4):
            hi = (m, n) == (2, 1)
            f.box(24 + n * 56, 88 + m * 44, 50, 38, f"({m},{n})",
                  fill=LOF if hi else "#ffffff", stroke=ORANGE if hi else "#bcccdc",
                  fs=12, bold=hi, fc="#7c3a06" if hi else "#627d98")
    f.text(24, 268, 240, 20, "(2,1) 高亮", fs=12, fc="#7c3a06")
    f.box(300, 130, 280, 60, "layout：f(m, n) = m + 4n", fill=LGF, stroke=GREEN, fs=14, bold=True, fc=GREEN)
    f.text(300, 194, 280, 22, "f(2,1) = 2 + 4 = 6", fs=13.5, bold=True, fc="#1e5631")
    f.line([(252, 158), (300, 158)], color=GREEN, width=2)
    f.box(620, 130, 200, 60, "offset = 6", fill="#ffffff", stroke="#486581", fs=14, bold=True)
    f.line([(580, 158), (620, 158)], color=GREEN, width=2)
    f.box(860, 122, 276, 76, "engine：base_ptr + offset * 4B（C++ iterator：gmem / smem / rmem 指针）", fill=LBF, stroke=BLUE, fs=12.5, fc=BLUE)
    f.line([(820, 158), (860, 158)], color=BLUE, width=2)
    f.text(860, 212, 300, 20, "gmem 数带（base + 6*4B 处的元素）", fs=12, fc="#48586a")
    vals = ["..", "4", "5", "6", "7", ".."]
    for i, v in enumerate(vals):
        hi = v == "6"
        f.box(860 + i * 46, 236, 42, 40, v, fill=LOF if hi else "#ffffff",
              stroke=ORANGE if hi else "#bcccdc", fs=14, bold=hi, fc="#7c3a06" if hi else "#627d98")
    f.box(24, 330, 340, 56, "layout：坐标 -&gt; offset 的函数（上一章全部内容）", fill=LGF, stroke=GREEN, fs=12.5, fc="#1e5631")
    f.box(388, 330, 340, 56, "engine：offset -&gt; 数据（可加偏移的迭代器）", fill=LBF, stroke=BLUE, fs=12.5, fc=BLUE)
    f.box(752, 330, 384, 56, "tensor = (engine, layout)：两次映射串联成可索引对象", fill=LOF, stroke=ORANGE, fs=13, bold=True, fc="#7c3a06")
    f.line([(194, 386), (752, 352)], color=GREEN, width=2, dashed=1)
    f.line([(558, 386), (752, 366)], color=BLUE, width=2, dashed=1)
    return mk(f.p, 1160, 400, "fig-21-1")

def fig_21_2():
    f = F()
    f.text(24, 12, 1112, 28, "4 线程 x (4,4) 小例：手算表 vs CuTe 打印（v4.6.1 实测）", fs=16, bold=True)
    f.text(24, 52, 500, 24, "hand table（朴素 CUDA 手算，一轮覆盖 2x4）", fs=13.5, bold=True, fc="#102a43")
    f.text(84, 86, 60, 24, "t \\ v", fs=12.5, bold=True, fc="#486581")
    f.text(150, 86, 100, 24, "v = 0", fs=12.5, bold=True, fc="#486581")
    f.text(258, 86, 100, 24, "v = 1", fs=12.5, bold=True, fc="#486581")
    cells = [["(0,0)", "(0,1)"], ["(0,2)", "(0,3)"], ["(1,0)", "(1,1)"], ["(1,2)", "(1,3)"]]
    for t, row in enumerate(cells):
        y = 112 + t * 46
        f.text(84, y + 8, 50, 24, str(t), fs=13, bold=True, fc="#486581")
        f.box(146, y, 100, 40, row[0], fill=LBF if t < 2 else "#ffffff", stroke=BLUE, fs=13)
        f.box(254, y, 100, 40, row[1], fill=LBF if t < 2 else "#ffffff", stroke=BLUE, fs=13)
    f.text(24, 300, 500, 22, "蓝色两行 = 第 1 轮（rows 0..1）；第 2 轮覆盖 rows 2..3，两轮拼满 (4,4)", fs=12, fc="#48586a")
    f.box(580, 52, 556, 250, "", fill="#ffffff", stroke="#486581")
    f.text(596, 60, 520, 22, "CuTe printout（make_tiled_copy 实测）", fs=13, bold=True, fc="#102a43")
    lines = [
        "ThrLayout : (_2,_2):(_2,_1)",
        "ValLayout : (_1,_2)",
        "layout_mn : ((_1,_2),(_2,_2)):((_0,_2),(_4,_1))",
        "layout_tv : ((_2,_2),_2):((_4,_1),_2)",
        "tiler     : (_2,_4) -&gt; 2 rounds",
        "tvS(t,v)  : m = t/2；n = 2*(t%2) + v",
    ]
    y = 88
    for ln in lines:
        f.text(600, y, 520, 26, ln, fs=13, fc="#243b53")
        y += 32
    f.text(596, 280, 520, 20, "tvS 公式与左表逐格一致（8 个 (t,v) 逐点核对）", fs=12, fc="#48586a")
    f.box(24, 330, 542, 56, "MN-layout（数据视角）：t = 2*m + (n/2)，v = n%2", fill=LBF, stroke=BLUE, fs=13, bold=True, fc=BLUE)
    f.box(618, 330, 542, 56, "TV-layout（任务视角）：m = t/2，n = 2*(t%2) + v", fill=LGF, stroke=GREEN, fs=13, bold=True, fc=GREEN)
    f.text(566, 344, 52, 28, "&lt;-&gt;", fs=14, bold=True, fc="#486581")
    f.text(24, 396, 1112, 22, "两种语言互为逆映射，分别服务「哪个线程拥有这个数据」与「这个线程该搬哪些数据」", fs=12.5, fc="#48586a")
    return mk(f.p, 1160, 428, "fig-21-2")

def fig_21_3():
    f = F()
    f.text(24, 12, 1112, 28, "thread x value 网格：G-&gt;S 拷贝（ThrLayout (32,4):(4,1)，ValLayout (1,8)，32x32 tile）", fs=15.5, bold=True)
    f.text(84, 52, 60, 22, "m / n", fs=12.5, bold=True, fc="#486581")
    heads = ["n: 0-7", "n: 8-15", "n: 16-23", "n: 24-31"]
    for j, hd in enumerate(heads):
        f.text(146 + j * 216, 52, 210, 22, hd, fs=12.5, bold=True, fc="#486581")
        f.box(146 + j * 216, 78, 210, 40, f"t = 4m + {j}", fill=GRAY, stroke="#486581", fs=12.5, bold=True, fc="#243b53")
    for m in range(4):
        y = 124 + m * 46
        f.text(84, y + 8, 50, 24, str(m), fs=13, bold=True, fc="#486581")
        for j in range(4):
            f.box(146 + j * 216, y, 210, 40, f"t = {4*m + j}（v = 0..7）",
                  fill=LBF if j % 2 == 0 else "#ffffff", stroke=BLUE, fs=12)
    f.text(24, 312, 1000, 22, "m = 4..31 同构（t = 4m + j）；每线程 8 个连续元素 = 一条 128-bit cp.async", fs=12.5, fc="#48586a")
    f.box(24, 344, 542, 44, "正：t = 4m + j（j = 哪个 8 宽列组）；v = n mod 8", fill="#ffffff", stroke=BLUE, fs=12.5)
    f.box(618, 344, 518, 44, "逆：m = t/4；n = 8*(t mod 4) + v（eq. 21-g2s-tv）", fill="#ffffff", stroke=GREEN, fs=12.5)
    f.box(24, 398, 542, 44, "Tiler = product_each(shape(Layout_MN)) = (32,32)", fill="#ffffff", stroke="#486581", fs=12.5)
    f.box(618, 398, 518, 44, "守恒：NumThr * NumVal = 128 x 8 = 1024 = |Tiler|", fill=LGF, stroke=GREEN, fs=12.5, bold=True, fc="#1e5631")
    f.box(24, 452, 1112, 48, "对照 S2R 拷贝：Tiler = (32,16) 无全覆盖，P*V = 1024 -&gt; 2x 重复（N 方向 warp 重读同一 A，eq. 21-dup）", fill="#fbe3e3", stroke=RED, fs=13, fc="#7f1d1d")
    return mk(f.p, 1160, 512, "fig-21-3")

def fig_24_2():
    f = F()
    f.text(24, 12, 1212, 28, "CuTe HGEMM 的 kStage = 2 流水时序（单 CTA，每个 BK tile 2 个 k-step，含寄存器双缓冲）", fs=15.5, bold=True)
    cols = [("prefetch", 150), ("k_tile=0 ks=0", 180), ("k_tile=0 ks=1", 180), ("k_tile=1 ks=0", 180), ("k_tile=1 ks=1", 180), ("epilogue", 280)]
    x = 130
    xs = []
    for name, w in cols:
        f.box(x, 52, w - 6, 30, name, fill="#d9e2ec", stroke="#486581", fs=12, bold=True)
        xs.append((x, w))
        x += w
    lanes = [("G-&gt;S (LDGSTS)", BLUE), ("S-&gt;R (LDSM)", ORANGE), ("MMA", GREEN)]
    y = 96
    ys = []
    for name, c in lanes:
        f.text(24, y + 10, 104, 40, name, fs=12, bold=True, fc=c)
        ys.append(y)
        y += 74
    def cell(lane, col, txt, fill, stroke, fs=11):
        cx, cw = xs[col]
        f.box(cx, ys[lane], cw - 6, 62, txt, fill=fill, stroke=stroke, fs=fs)
    cell(0, 0, "G-&gt;S stage0（1 tile）+ cp_async_fence + wait&lt;0&gt;", LBF, BLUE, 11)
    cell(0, 1, "G-&gt;S next tile -&gt; s1", LBF, BLUE, 11.5)
    cell(0, 2, "wait&lt;0&gt; + __syncthreads，翻转 s0 -&gt; s1", "#ffffff", BLUE, 11)
    cell(0, 3, "G-&gt;S +1 tile -&gt; s0", LBF, BLUE, 11.5)
    cell(0, 4, "交替往复 ...", "#ffffff", BLUE, 11.5)
    cell(1, 1, "S2R load (s0, k1)", LOF, ORANGE, 11.5)
    cell(1, 3, "S2R load (s1, k1)", LOF, ORANGE, 11.5)
    cell(2, 1, "MMA(tCrA/B(s0, k0))", LGF, GREEN, 11.5)
    cell(2, 2, "MMA(tCrA/B(s0, k1))", LGF, GREEN, 11.5)
    cell(2, 3, "MMA(s1, k0)", LGF, GREEN, 11.5)
    cell(2, 4, "MMA(s1, k1)", LGF, GREEN, 11.5)
    epi = ["R2R cast（f32 -&gt; f16）", "R2S 写入 sC（复用 pipe j）", "__syncthreads", "S2G -&gt; gmem"]
    ye = 96
    for t in epi:
        f.box(1150, ye, 274, 30, t, fill=GRAY, stroke="#486581", fs=11, fc="#243b53")
        ye += 36
    f.text(1150, 244, 274, 40, "sC 与一个 A stage 别名：32x32x4 = 128x32 = 4096 half", fs=11, fc="#48586a")
    f.box(24, 330, 1212, 48, "三重重叠：MMA 消费 stage s_k 的 k_step，LDSM 预取同一 stage 的 k_step+1，cp.async 同时填另一个 stage —— s0/s1 交替贯穿", fill=LOF, stroke=ORANGE, fs=13.5, bold=True, fc="#7c3a06")
    return mk(f.p, 1480, 392, "fig-24-2")

FIGS = {
    "fig-20-2": (fig_20_2, "fig-20-2-mode-ops"),
    "fig-20-3": (fig_20_3, "fig-20-3-layout-convert"),
    "fig-21-1": (fig_21_1, "fig-21-1-engine-layout"),
    "fig-21-2": (fig_21_2, "fig-21-2-tv-table"),
    "fig-21-3": (fig_21_3, "fig-21-3-g2s-grid"),
    "fig-24-2": (fig_24_2, "fig-24-2-kstage-pipeline"),
}

if __name__ == "__main__":
    root = sys.argv[1]
    for stem, (fn, d) in FIGS.items():
        xml = fn()
        ET.fromstring(xml)
        dp = os.path.join(root, d)
        os.makedirs(dp, exist_ok=True)
        open(os.path.join(dp, f"{stem}.drawio"), "w").write(xml)
        print(f"wrote {stem} OK cells={xml.count('<mxCell') - 2}")
