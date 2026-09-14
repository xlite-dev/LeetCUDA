#!/usr/bin/env python3
# gen_batch_a.py — 批次 A：FIG-1-1 / 1-2 / 4-1 / 5-1 / 7-1 / 8-1 六张 drawio 图
# 产物直接写 figures/drawio/<id>/<stem>.drawio
import os, sys

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
        st = (f"endArrow={ae};html=1;strokeColor={color};strokeWidth={width};rounded=0;dashed={dashed}")
        self.p.append(f'<mxCell value="" style="{st}" edge="1" parent="1">'
                      f'<mxGeometry relative="1" as="geometry"><Array as="points">'
                      + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
                      + f'</Array><mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
                        f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/></mxGeometry></mxCell>')

BLUE, GREEN, ORANGE, RED, GRAY = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#e7ecf3"

def fig_1_1():
    f = F()
    f.text(40, 16, 700, 26, "速率/容量金字塔（H100 数量级参考，base.cuh L16-21）", fs=17, bold=True)
    rows = [
        ("Register", "~100+ TB/s", "256 KB/SM", "thread", 190, "#bfd7ea"),
        ("SMEM / L1", "~19 TB/s", "~228 KB/SM", "block", 300, "#a6cce4"),
        ("L2 Cache", "~12 TB/s", "50 MB", "device", 450, "#8db9d2"),
        ("HBM3", "~3.35 TB/s", "80 GB", "device", 640, "#74a4bf"),
    ]
    y = 70
    for name, bw, cap, scope, w, fill in rows:
        f.box(80, y, w, 54, name, fill=fill, stroke=BLUE, fs=15, bold=True, fc="#102a43")
        f.text(80 + w + 20, y + 6, 170, 20, bw, fs=13.5, fc="#243b53")
        f.text(80 + w + 20, y + 28, 170, 20, cap, fs=13, fc="#48586a")
        f.text(820, y + 16, 100, 20, f"({scope})", fs=12.5, fc="#627d98")
        y += 66
    f.line([(48, 64), (48, y - 10)], color="#627d98", width=2, arrow_end=1)
    f.text(280, y + 16, 420, 22, "离 SM 越远：带宽越低、容量越大", fs=13, fc="#48586a", align="center")
    return mk(f.p, 950, y + 50, "fig1-1")

def fig_1_2():
    f = F()
    f.text(40, 14, 700, 26, "Roofline 模型：P = min(P-peak, AI x BW)", fs=17, bold=True)
    OX, OY, PW, PH = 120, 60, 760, 380   # 原点与轴长
    R_X = 330                            # ridge point x
    P_Y = OY                             # P-peak 线 y
    f.line([(OX, OY + PH), (OX + PW, OY + PH)], color="#627d98", width=2, arrow_end=1)  # x
    f.line([(OX, OY + PH), (OX, OY)], color="#627d98", width=2, arrow_end=1)           # y
    f.text(OX + PW - 60, OY + PH + 8, 140, 20, "AI (FLOP/Byte)", fs=13, fc="#48586a")
    f.text(OX - 40, OY - 4, 200, 20, "P (TFLOPS)", fs=13, fc="#48586a")
    # 水平段 + 斜线段
    f.line([(R_X, P_Y), (OX + PW - 30, P_Y)], color=RED, width=3, arrow_end=0)
    f.line([(OX, OY + PH), (R_X, P_Y)], color=RED, width=3, arrow_end=0)
    f.text(R_X + 60, P_Y - 26, 260, 20, "compute-bound: P = P-peak", fs=13.5, fc=RED, bold=True)
    f.text(OX + 60, OY + PH - 120, 260, 20, "memory-bound: P = AI x BW", fs=13.5, fc=ORANGE, bold=True)
    f.text(OX + 66, OY + PH - 96, 200, 18, "(斜率 = BW)", fs=12, fc=ORANGE)
    f.line([(R_X, P_Y), (R_X, OY + PH)], color="#9aa5b1", width=1.5, dashed=1, arrow_end=0)
    f.text(R_X - 44, OY + PH + 8, 120, 20, "AI* = P-peak / BW", fs=12.5, fc="#48586a")
    # 实例点
    f.box(660, 84, 12, 12, "", fill=BLUE, stroke=BLUE)
    f.text(500, 108, 330, 40, "GEMM K=4096: AI=683 高于 AI*=295 -> compute-bound", fs=12.5, fc=BLUE)
    f.box(196, 396, 12, 12, "", fill=GREEN, stroke=GREEN)
    f.text(214, 412, 330, 40, "softmax AI=0.625 远低于 AI*=20 -> 严重 memory-bound", fs=12.5, fc=GREEN)
    f.text(40, 458, 900, 20, "例：H100 SXM（base.cuh L84-86 口径）FP16 TC dense P-peak 约 989 TFLOPS，BW=3.35 TB/s -> AI* 约 295；", fs=12.5, fc="#48586a")
    f.text(40, 478, 900, 20, "FP32 CUDA core 约 67 TFLOPS -> AI* 约 20", fs=12.5, fc="#48586a")
    return mk(f.p, 980, 512, "fig1-2")

def fig_4_1():
    f = F()
    f.text(30, 14, 900, 26, "Softmax 三级递进：3-pass -> 2-pass -> 1-pass（online）", fs=17, bold=True)
    cols = [
        ("naive / textbook", "3-pass", [
            "gmem x 第 1 遍: m = max(x)",
            "gmem x 第 2 遍: s = SUM e^x",
            "  e^x 可能上溢 inf",
            "gmem x 第 3 遍: y = e^x / s",
            "3 次 gmem 全量读"], RED, "#fde9d0"),
        ("safe / Level 2", "2-pass", [
            "gmem x -> regs（第 1 次读）",
            "block reduce max -> m",
            "gmem x 第 2 次读（L1 命中）",
            "e^(x-m) 不超 1，无溢出",
            "reduce sum -> s；y = e^(x-m)/s",
            "2 次 block sync"], ORANGE, "#fde9d0"),
        ("online / Level 3", "1-pass", [
            "gmem x -> regs（唯一一次读）",
            "init (m, d) = (x, 1)",
            "MD 蝶形归约：",
            "  d = d1 + d2 * e^(m2-m1)",
            "  m, d 同步重标定",
            "y = e^(x-m) / d，1 次 sync"], GREEN, "#c9e4c8"),
    ]
    for i, (name, tag, lines, c, fill) in enumerate(cols):
        x = 50 + i * 350
        f.box(x, 60, 320, 40, f"{name}（{tag}）", fill=GRAY, stroke=c, fs=14.5, bold=True, fc=c)
        f.box(x, 104, 320, 246, "", fill=fill if i == 2 else "#ffffff", stroke=c)
        for j, l in enumerate(lines):
            f.text(x + 14, 112 + j * 34, 296, 30, l, fs=12.5, fc="#243b53")
        if i < 2:
            f.line([(x + 330, 180), (x + 352, 180)], color="#627d98", width=3)
    f.text(30, 372, 1000, 20, "箭头方向 = 演进：读 gmem 次数 3 -> 2 -> 1；溢出风险 -> 无；block sync 3 -> 2 -> 1（fast math：expf / __fdividef）", fs=12.5, fc="#48586a")
    return mk(f.p, 1100, 410, "fig4-1")

def fig_5_1():
    f = F()
    f.text(30, 14, 950, 26, "两个分块 attention 状态的合并 == 整体 softmax 加权（定理 5.1）", fs=17, bold=True)
    # 左右状态卡
    def state(x, tag, rng, c):
        f.box(x, 60, 420, 172, "", fill="#f0f6fb", stroke=c)
        f.box(x, 60, 420, 32, f"KV chunk {tag}: s{rng}", fill=GRAY, stroke=c, fs=14, bold=True, fc=c)
        lines = [f"m({tag}) = max s{rng}", f"l({tag}) = SUM e^(s - m({tag}))",
                 f"O({tag}) = SUM e^(s-m({tag})) * V / l({tag})", f"LSE({tag}) = m({tag}) + log l({tag})"]
        for j, l in enumerate(lines):
            f.text(x + 16, 100 + j * 32, 400, 28, l, fs=12.5, fc="#243b53")
    state(50, "A", "(1..n)", BLUE)
    state(610, "B", "(n+1..n+m)", GREEN)
    # 汇聚箭头
    f.line([(260, 236), (430, 300), (500, 300)], color="#627d98", width=2)
    f.line([(820, 236), (650, 300), (580, 300)], color="#627d98", width=2)
    # 合并卡
    f.box(240, 306, 600, 158, "", fill="#f7f3ec", stroke="#b45309")
    f.box(240, 306, 600, 30, "merge：以 L = max(LSE(A), LSE(B)) 为 shift 基准", fill=GRAY, stroke="#b45309", fs=13.5, bold=True, fc="#b45309")
    ml = ["w(A) = e^(LSE(A)-L)      w(B) = e^(LSE(B)-L)",
          "alpha = w(A)/(w(A)+w(B))    beta = w(B)/(w(A)+w(B))",
          "O = alpha * O(A) + beta * O(B)"]
    for j, l in enumerate(ml):
        f.text(256, 344 + j * 34, 580, 30, l, fs=13.5, fc="#7c5a1e")
    # 结论条
    f.line([(540, 470), (540, 496)], color="#b45309", width=2)
    f.box(120, 500, 840, 46, "O == softmax([s(1..n); s(n+1..n+m)]) 作用于 [V(A); V(B)] —— 与一次性 attention 逐元素相等", fill="#c9e4c8", stroke=GREEN, fs=13, bold=True, fc="#1e5631")
    return mk(f.p, 1080, 570, "fig5-1")

def fig_7_1():
    f = F()
    f.text(30, 14, 1000, 26, "转置读的 bank 分布：PAD=1 使步长 64B 变 68B，tx 项在 mod 32 下重新出现", fs=16.5, bold=True)
    f.text(60, 52, 420, 22, "[无 PAD] tile[64][16]：addr = 64*tx + ty", fs=14, bold=True, fc=RED)
    f.text(620, 52, 440, 22, "[PAD=1] tile[64][17]：addr = 68*tx + 17*j + ty", fs=14, bold=True, fc=GREEN)
    # 左：2 bank x 16 线程
    f.text(60, 84, 420, 20, "ty=0 列 16 线程与 ty=1 列 16 线程落在同 2 个 bank", fs=12.5, fc="#48586a")
    for b in range(2):
        f.box(90 + b * 150, 112, 130, 40, f"bank {b}", fill="#fde9d0", stroke=RED, fs=13, bold=True, fc="#8c2f0f")
        f.text(90 + b * 150, 156, 130, 22, "16 线程", fs=12.5, fc=RED)
    f.box(60, 196, 420, 34, "=> 16-way conflict，读 smem 吞吐 1/16", fill="#ffffff", stroke=RED, fs=13.5, bold=True, fc=RED)
    # 右：8 bank x 2 线程
    f.text(620, 84, 440, 20, "tx 步进 +4：8 个 bank 各 2 线程（tx=0/8 同 bank）", fs=12.5, fc="#48586a")
    for b in range(8):
        x = 640 + b * 54
        f.box(x, 112, 48, 40, f"{b*4}", fill="#c9e4c8", stroke=GREEN, fs=13, bold=True, fc="#1e5631")
        f.text(x, 156, 48, 22, "2 线程", fs=11.5, fc=GREEN)
    f.text(620, 180, 440, 18, "bank 编号 = 4*tx mod 32（tx=0..7 示意）", fs=11.5, fc="#627d98")
    f.box(620, 196, 440, 34, "=> 2-way conflict，读 smem 吞吐 1/2", fill="#ffffff", stroke=GREEN, fs=13.5, bold=True, fc=GREEN)
    # 中间转化
    f.line([(500, 150), (600, 150)], color=ORANGE, width=3)
    f.text(480, 118, 140, 20, "+PAD=1", fs=13, fc=ORANGE, bold=True)
    f.text(30, 258, 1040, 40, "原因：64 mod 32 = 0（tx 项被吞），68 mod 32 = 4（tx 项以 +4 步进重新出现）；warp0: ty={0,1}, tx=0..15, j=0（base.cuh 转置读 Step2）", fs=12.5, fc="#48586a")
    return mk(f.p, 1100, 310, "fig7-1")

def fig_8_1():
    f = F()
    f.text(30, 14, 950, 26, "warp-per-row：block(32,4) -> 每 block 4 行，1 warp = 1 行", fs=17, bold=True)
    # 左上：grid 分块
    for b in range(3):
        x = 60 + b * 150
        f.box(x, 60, 140, 44, f"block{b}", fill="#f0f6fb", stroke=BLUE, fs=13.5, bold=True, fc=BLUE)
        if b < 2:
            f.text(x + 140, 72, 10, 20, "", fs=10)
    f.text(60, 108, 460, 18, "grid.x = ceil(M/4)", fs=12, fc="#627d98")
    # 右上：warp 映射
    f.box(560, 44, 500, 76, "", fill="#ffffff", stroke="#bcccdc")
    for w in range(4):
        f.text(576, 50 + w * 18, 200, 16, f"warp {w} (ty={w})", fs=11.5, fc="#243b53")
        f.text(740, 50 + w * 18, 240, 16, f"-> row m+{w}", fs=11.5, fc=BLUE, bold=True)
    f.text(560, 124, 500, 18, "block(32,4) = 4 warps，one block covers 4 rows", fs=12, fc="#627d98")
    # 中部：k32 pass 条带
    f.text(30, 158, 600, 20, "k32（K = 32 * NUM-ITERS）：lane l 取 k = 32*w + l", fs=13.5, bold=True)
    for w in range(2):
        x = 60 + w * 470
        f.box(x, 184, 440, 34, f"pass w={w}:  lane0 lane1 ... lane31 -> k = 32*{w}+l", fill="#f0f6fb" if w == 0 else "#e9eff6", stroke=BLUE, fs=12.5, fc="#243b53")
        f.text(x + 10, 220, 430, 16, f"x[{32*w}] ... x[{32*w+31}]，每 lane 串行累加 sum += a[m][k] * x[k]", fs=11.5, fc="#48586a")
    f.text(60, 244, 900, 18, "累加完成后 5 步 XOR 蝶形归约 -> lane 0 写 y[m]", fs=12.5, fc="#48586a")
    # 下部：k16 双行
    f.text(30, 286, 600, 20, "k16（K=16，kRowPerWarp=2）：段宽 16，两行各 16 lane", fs=13.5, bold=True)
    f.box(60, 312, 440, 30, "lane 0..15 -> row m   (k = lane mod 16)", fill="#c9e4c8", stroke=GREEN, fs=12.5, fc="#1e5631")
    f.box(510, 312, 440, 30, "lane 16..31 -> row m+1 (k = lane mod 16)", fill="#c9e4c8", stroke=GREEN, fs=12.5, fc="#1e5631")
    f.text(60, 350, 950, 36, "reduce 小于 16 的 mask 序列 8,4,2,1；row-major A 中 m 与 m+1 行连续 -> 32 lane 仍是 1 个 128B 合并事务；k128 变体：单行 float4 加宽（4 lane 一组）", fs=12.5, fc="#48586a")
    return mk(f.p, 1080, 400, "fig8-1")

FIGS = {
    "fig-1-1-mem-hierarchy": ("fig-1-1", fig_1_1),
    "fig-1-2-roofline": ("fig-1-2", fig_1_2),
    "fig-4-1-softmax-flow": ("fig-4-1", fig_4_1),
    "fig-5-1-merge-states": ("fig-5-1", fig_5_1),
    "fig-7-1-bank-pad": ("fig-7-1", fig_7_1),
    "fig-8-1-warp-per-row": ("fig-8-1", fig_8_1),
}

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "figures/drawio"
    for d, (stem, fn) in FIGS.items():
        path = os.path.join(outdir, d)
        os.makedirs(path, exist_ok=True)
        xml = fn()
        open(os.path.join(path, f"{stem}.drawio"), "w").write(xml)
        # 自检：XML well-formed + 无裸泛型尖括号
        import xml.etree.ElementTree as ET
        ET.fromstring(xml)
        print(f"wrote {path}/{stem}.drawio  cells={xml.count(chr(60)+chr(109)+chr(120))} OK")
