#!/usr/bin/env python3
# gen_batch_b1.py — 批次 B1：FIG-9-1 / 10-1 / 11-1 / 11-2a / 12-3
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

def fig_9_1():
    f = F()
    f.text(30, 14, 950, 26, "grid -> block -> warp -> thread 四级 tiling（C 矩阵视角）", fs=17, bold=True)
    # grid 级：C 分块
    f.text(50, 56, 400, 20, "grid 级：C[M,N] 切成 (M/BM) x (N/BN) 个 block tile", fs=13.5, bold=True, fc=BLUE)
    for r in range(2):
        for c in range(3):
            f.box(80 + c * 130, 82 + r * 70, 120, 60, f"tile {r},{c}" if (r, c) in ((0,0),(0,1),(1,0)) else "",
                  fill="#f0f6fb" if (r, c) == (0, 0) else "#ffffff", stroke=BLUE if (r, c) == (0, 0) else "#9aa5b1",
                  fs=12, bold=(r, c) == (0, 0), fc=BLUE)
    f.text(80, 232, 420, 60, "BM=BN=128, BK=32：每 block 1024 线程；smem = sA[128][32]+sB[32][128] = 32KB；沿 K 循环 K/32 轮，每轮两道栅栏夹计算窗口", fs=12, fc="#48586a")
    # warp 级
    f.text(560, 56, 500, 20, "warp 级（隐式）：一个 warp 承包 4 x 128 C 横带", fs=13.5, bold=True, fc=ORANGE)
    for i in range(4):
        f.box(580, 84 + i * 24, 460, 20, f"row band (tid/32)*{4-i}  x  128 cols", fill="#fde9d0" if i % 2 == 0 else "#ffffff", stroke=ORANGE, fs=10.5, fc="#8c2f0f")
    # thread 级
    f.text(50, 300, 500, 20, "thread 级：每线程一个 4 x 4 寄存器 tile（16 输出 sum[4][4]）", fs=13.5, bold=True, fc=GREEN)
    f.text(80, 328, 200, 18, "n: n0..n0+3  n0=(tid%32)*4", fs=11.5, fc="#627d98")
    for i in range(4):
        for j in range(4):
            f.box(80 + j * 36, 352 + i * 30, 34, 28, "tt", fill="#c9e4c8", stroke=GREEN, fs=10.5, fc="#1e5631")
    f.text(250, 352, 460, 90, "每 k 步：16 FMAs、8 次 smem 读（4 A + 4 B）；AI(smem) = 1.0 FLOP/B（naive 0.25）", fs=12.5, fc="#243b53")
    f.box(50, 480, 1000, 34, "AI 阶梯：naive 0.25 -> block 32x32: 8 -> block 128x128: 32 FLOP/B（乘 smem 复用逐级放大）", fill=GRAY, stroke=BLUE, fs=13, bold=True, fc="#102a43")
    return mk(f.p, 1100, 530, "fig9-1")

def fig_10_1():
    f = F()
    f.text(30, 14, 950, 26, "2-stage double buffering：load(k) 与 compute(k-1) 重叠", fs=17, bold=True)
    # 时间轴
    f.line([(80, 60), (1060, 60)], color="#9aa5b1", width=1.5, arrow_end=1)
    f.text(1000, 36, 80, 20, "time", fs=12.5, fc="#627d98")
    # LD 泳道
    f.text(30, 86, 60, 20, "LD unit", fs=13.5, bold=True, fc=ORANGE)
    for k in range(4):
        x = 100 + k * 220
        f.box(x, 80, 190, 40, f"ld {k}", fill="#fde9d0", stroke=ORANGE, fs=13, bold=True, fc="#8c2f0f")
    f.text(100, 124, 700, 18, "cp.async 分组：...wait group 0 at first barrier...", fs=11.5, fc="#627d98")
    # TC 泳道
    f.text(30, 176, 60, 20, "TC unit", fs=13.5, bold=True, fc=GREEN)
    for k in range(4):
        x = 210 + k * 220
        f.box(x, 170, 190, 40, f"cp {k}", fill="#c9e4c8", stroke=GREEN, fs=13, bold=True, fc="#1e5631")
    # 对齐标注
    f.line([(205, 130), (205, 168)], color="#9aa5b1", width=1.5, dashed=1, arrow_end=0)
    f.text(214, 132, 300, 18, "首迭代 k=1：load 1 写 S1，compute 0 读 S0", fs=11.5, fc="#48586a")
    # smem 条
    f.text(30, 240, 800, 20, "smem 双缓冲：S0 承载偶数 tile，S1 承载奇数 tile", fs=13.5, bold=True)
    f.box(100, 268, 420, 34, "S0 : tile 0 | tile 2 | tile 4 ...  (even k)", fill="#f0f6fb", stroke=BLUE, fs=12.5, fc="#243b53")
    f.box(560, 268, 420, 34, "S1 : tile 1 | tile 3 | tile 5 ...  (odd k)", fill="#f0f6fb", stroke=BLUE, fs=12.5, fc="#243b53")
    f.box(50, 322, 1000, 34, "steady state：Tstep = max(Tload, Tcompute)，短的一侧被完全隐藏；compute 读 (k-1)%2，load 写 k%2", fill=GRAY, stroke=BLUE, fs=13, bold=True, fc="#102a43")
    return mk(f.p, 1100, 380, "fig10-1")

def fig_11_1():
    f = F()
    f.text(30, 14, 1050, 26, "ldmatrix 寻址：A x4 一次装载 m16k16 四象限；B x2 装载 k16n8", fs=16.5, bold=True)
    # 左：A 四象限
    f.text(50, 52, 500, 20, "A m16k16 tile -> four 8x8 matrices（x4，每 lane 一个地址）", fs=13, bold=True, fc=BLUE)
    quads = [("matrix0  m 0-7, k 0-7", 60, 90, "RA[i][0]", "#f0f6fb"),
             ("matrix1  m 8-15, k 0-7", 230, 90, "RA[i][1]", "#e9eff6"),
             ("matrix2  m 0-7, k 8-15", 60, 190, "RA[i][2]", "#e9eff6"),
             ("matrix3  m 8-15, k 8-15", 230, 190, "RA[i][3]", "#f0f6fb")]
    for name, x, y, reg, fill in quads:
        f.box(x, y, 150, 84, "", fill=fill, stroke=BLUE)
        f.text(x + 8, y + 6, 140, 18, name, fs=11.5, bold=True, fc=BLUE)
        f.text(x + 8, y + 30, 140, 44, "rows = t%16\ncols = (t/16)*8", fs=11, fc="#48586a")
    f.text(60, 288, 380, 20, "t=0-7 -> m0; t=8-15 -> m1; t=16-23 -> m2; t=24-31 -> m3", fs=11.5, fc="#48586a")
    f.box(60, 312, 380, 40, "象限序 [UL, LL, UR, LR] == mma 期望序（Eq 11.1），开箱即用", fill="#c9e4c8", stroke=GREEN, fs=12, bold=True, fc="#1e5631")
    # 右：B 两矩阵
    f.text(560, 52, 500, 20, "B k16n8（存成 B^T n16k16 行主序）-> two 8x8（x2）", fs=13, bold=True, fc=GREEN)
    for j, (name, x, y, reg, half) in enumerate([("matrix0  n 0-7, k 0-7", 600, 90, "RB[j][0]", "upper k half"),
                                                  ("matrix1  n 0-7, k 8-15", 770, 90, "RB[j][1]", "lower k half")]):
        f.box(x, y, 150, 84, "", fill="#f0faf5", stroke=GREEN)
        f.text(x + 8, y + 6, 140, 18, name, fs=11.5, bold=True, fc=GREEN)
        f.text(x + 8, y + 30, 140, 44, "rows = t%8\ncols = ((t/8)%2)*8", fs=11, fc="#48586a")
    f.text(600, 188, 400, 20, "t=0-7 -> k 低半；t=8-15 -> k 高半（只用 t 小于 16）", fs=11.5, fc="#48586a")
    f.box(600, 312, 470, 40, "装载 B^T 的行 == 装载 B 的列：无需 .trans", fill="#c9e4c8", stroke=GREEN, fs=12.5, bold=True, fc="#1e5631")
    f.text(50, 372, 1030, 20, "地址公式：A row = warpM*64 + i*16 + (t%16)，col k = (t/16)*8；B row n = warpN*32 + j*8 + (t%8)", fs=12, fc="#48586a")
    return mk(f.p, 1100, 410, "fig11-1")

def fig_11_2a():
    f = F()
    f.text(30, 12, 1050, 24, "A fragment 寄存器布局（m16 x k16 row-major）：g = lane 右移 2，tig = lane mod 4", fs=16, bold=True)
    f.text(30, 40, 900, 18, "Tn = lane n 持有该元素；每格 2 个相邻 f16 装一条 .b32（依据 PTX ISA Figure 79 行/列公式）", fs=12, fc="#48586a")
    CW, CH = 50, 32
    GX, GY = 110, 92
    # 列头 k
    for k in range(16):
        f.box(GX + k * CW, GY - CH, CW, CH, str(k), fill=GRAY, stroke="#9aa5b1", fs=12, bold=True)
    # 行 m 0..15（省略 9-14 中间行，画 0-8 与 15 关键行）: 画 m 0,1,7,8,15 五行示意
    rows = [0, 1, 7, 8, 15]
    for ri, m in enumerate(rows):
        y = GY + ri * (CH + 6)
        f.box(GX - 56, y, 52, CH, f"m {m}", fill=GRAY, stroke="#9aa5b1", fs=12, bold=True)
        for k in range(16):
            g = m % 8
            tig = k % 4
            lane = g * 4 + tig
            if k >= 8:
                lane += (m // 8) * 0  # upper half cols same lanes
            # R 上下半：m 0-7 -> R0/R2（cols 0-7 -> R0, 8-15 -> R2）；m 8-15 -> R1/R3
            quad = ("R0" if m < 8 else "R1") if k < 8 else ("R2" if m < 8 else "R3")
            fill = {"R0": "#f0f6fb", "R1": "#e9eff6", "R2": "#fde9d0", "R3": "#fdf0e0"}[quad]
            f.box(GX + k * CW, y, CW, CH, f"T{lane}", fill=fill, stroke="#bcccdc", fs=11.5)
    for ri in range(len(rows) - 1):
        if rows[ri + 1] - rows[ri] > 1:
            f.text(GX - 52, GY + ri * (CH + 6) + CH + 2, 300, 16, f"... (m {rows[ri]+1}-{rows[ri+1]-1} 同构)", fs=10.5, fc="#9aa5b1")
    # 象限注释卡
    f.box(980, GY - 10, 190, 30, "R0 = a0,a1 (m 0-7, k 0-7)", fill="#f0f6fb", stroke=BLUE, fs=11, bold=True, fc=BLUE)
    f.box(980, GY + 26, 190, 30, "R1 = a2,a3 (m 8-15, k 0-7)", fill="#e9eff6", stroke=BLUE, fs=11, bold=True, fc=BLUE)
    f.box(980, GY + 62, 190, 30, "R2 = a4,a5 (m 0-7, k 8-15)", fill="#fde9d0", stroke=ORANGE, fs=11, bold=True, fc="#8c2f0f")
    f.box(980, GY + 98, 190, 30, "R3 = a6,a7 (m 8-15, k 8-15)", fill="#fdf0e0", stroke=ORANGE, fs=11, bold=True, fc="#8c2f0f")
    f.text(30, GY + 5 * (CH + 6) + 12, 1000, 40, "每 lane 4 条 .b32：R0 行 g 列 (2*tig, 2*tig+1)；R1 行 g+8 同列；R2/R3 列 +8。象限序 R0=UL, R1=LL, R2=UR, R3=LR == mma 期望", fs=12.5, fc="#48586a")
    return mk(f.p, 1200, 420, "fig11-2a")

def fig_12_3():
    f = F()
    f.text(30, 14, 1000, 26, "XOR swizzle 位运算：Swizzle=B,M,S（fp16 tile 字节偏移）", fs=17, bold=True)
    # bit 布局条（BK=16, a = 32i + 16c + w）
    f.text(50, 56, 700, 20, "BK=16 tile 字节偏移 a = 32*i + 16*c + w 的 bit 布局：", fs=13.5, bold=True)
    bits = [("bit 9-7: i（行）", 240, BLUE, "#f0f6fb"), ("bit 6-5: ...", 180, "#9aa5b1", "#ffffff"),
            ("bit 4: c（chunk）", 200, ORANGE, "#fde9d0"), ("bit 3-0: w（chunk 内）", 260, "#9aa5b1", "#ffffff")]
    x = 60
    for name, w, c, fill in bits:
        f.box(x, 84, w, 40, name, fill=fill, stroke=c, fs=12, bold=True, fc=c)
        x += w + 4
    f.text(60, 132, 900, 20, "行号从 bit5 起 -> bit7 = (i 右移 2 且 mod 2) 是 XOR 源位；chunk 位 bit4 是 XOR 目标位", fs=12.5, fc="#48586a")
    # apply 公式卡
    f.box(60, 164, 460, 60, "", fill="#f0f6fb", stroke=BLUE)
    f.text(74, 170, 440, 22, "apply(a) = a ^ ((a 和 0x80) 右移 3)", fs=14, bold=True, fc=BLUE)
    f.text(74, 194, 440, 22, "= a ^ (bit7(a) 左移 4)   // bit4 异或 bit7（B=1,M=4,S=3）", fs=12, fc="#243b53")
    # 行宽决定冲突度
    f.text(50, 244, 700, 20, "行宽决定「多少行号位能挤进 bank 位 [2,7)」-> 最小冲突度：", fs=13.5, bold=True)
    f.box(60, 272, 980, 34, "32B 行（BK=16）：bit4=c，bit7=row 右移 2 -> 可达 1-way", fill="#c9e4c8", stroke=GREEN, fs=12.5, bold=True, fc="#1e5631")
    f.box(60, 312, 980, 34, "128B 行（BK=64）：a = 128i+32s+16c+w，bits [i:7..][s:5-6][c:4][w:0-3]", fill="#f0f6fb", stroke=BLUE, fs=12.5, fc="#243b53")
    f.box(60, 352, 480, 34, "per-slice swizzle 小于 16：仅 bit4 异或 bit7 -> 2 quad/phase = 4-way", fill="#ffffff", stroke=BLUE, fs=12, fc="#243b53")
    f.box(560, 352, 480, 34, "全宽 swizzle 小于 64（B=3）：bits 4-6 异或 bits 7-9 -> 8 quad = 1-way", fill="#c9e4c8", stroke=GREEN, fs=12, bold=True, fc="#1e5631")
    f.box(60, 398, 980, 34, "源位域 [M+S, M+S+B) 与目标位域 [M, M+B) 不相交（S 不小于 B）-> 两次应用复原：对合（引理 12.1）", fill=GRAY, stroke=BLUE, fs=12.5, bold=True, fc="#102a43")
    return mk(f.p, 1100, 450, "fig12-3")

FIGS = {
    "fig-9-1-gemm-tiling": ("fig-9-1", fig_9_1),
    "fig-10-1-double-buffer": ("fig-10-1", fig_10_1),
    "fig-11-1-ldmatrix": ("fig-11-1", fig_11_1),
    "fig-11-2a-fragment": ("fig-11-2a", fig_11_2a),
    "fig-12-3-xor-bits": ("fig-12-3", fig_12_3),
}

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "figures/drawio"
    for d, (stem, fn) in FIGS.items():
        path = os.path.join(outdir, d)
        os.makedirs(path, exist_ok=True)
        xml = fn()
        ET.fromstring(xml)
        open(os.path.join(path, f"{stem}.drawio"), "w").write(xml)
        print(f"wrote {stem} OK cells={xml.count('<mxCell')}")
