#!/usr/bin/env python3
# gen_batch_e.py — 批次 E（收尾）：FIG-2-1 / FIG-3-1
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
        st = f"endArrow={ae};html=1;strokeColor={color};strokeWidth={width};rounded=0;dashed={dashed};startArrow=blockThin;startFill=1"
        self.p.append(f'<mxCell value="" style="{st}" edge="1" parent="1">'
                      f'<mxGeometry relative="1" as="geometry"><Array as="points">'
                      + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
                      + f'</Array><mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
                        f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/></mxGeometry></mxCell>')

BLUE, GREEN, ORANGE, RED, GRAY = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#e7ecf3"
LBF, LGF, LOF = "#f0f6fb", "#c9e4c8", "#fde9d0"

def fig_2_1():
    f = F()
    f.text(24, 12, 1112, 28, "warp shuffle 蝶形归约（W=8，R=3 轮；5 步同理，base.cuh L111-148）", fs=16, bold=True)
    rounds = [
        ("round 1（mask = 4）", [(0, 4), (1, 5), (2, 6), (3, 7)], BLUE),
        ("round 2（mask = 2）", [(0, 2), (1, 3), (4, 6), (5, 7)], GREEN),
        ("round 3（mask = 1）", [(0, 1), (2, 3), (4, 5), (6, 7)], ORANGE),
    ]
    notes = [
        "lane i 加上 lane i^4：v = [0+4] [1+5] [2+6] [3+7] [4+0] [5+1] [6+2] [7+3]",
        "lane i 加上 lane i^2：lane 0..3 各含 {0,2,4,6} 族的和，lane 4..7 各含 {1,3,5,7} 族的和",
        "lane i 加上 lane i^1：全和 x0+...+x7 出现在每一个 lane",
    ]
    x0 = 40
    for (title, pairs, c), note in zip(rounds, notes):
        f.text(x0, 52, 260, 24, title, fs=13.5, bold=True, fc=c)
        for i in range(8):
            f.box(x0, 84 + i * 40, 130, 34, f"lane {i}", fill=LBF if c == BLUE else (LGF if c == GREEN else LOF), stroke=c, fs=12.5)
        for j, (a, b) in enumerate(pairs):
            lx = x0 + 134 + j * 16
            f.line([(lx, 101 + a * 40), (lx, 101 + b * 40)], color=c, width=1.6)
        x0 += 372
    f.text(40, 428, 1112, 22, "round1：lane i 加上 lane i^4 —— v = [0+4] [1+5] [2+6] [3+7] [4+0] [5+1] [6+2] [7+3]", fs=12.5, fc=BLUE)
    f.text(40, 452, 1112, 22, "round2：lane i 加上 lane i^2 —— lane 0..3 各含 {0,2,4,6} 族的和，lane 4..7 各含 {1,3,5,7} 族的和", fs=12.5, fc="#1e5631")
    f.text(40, 476, 1112, 22, "round3：lane i 加上 lane i^1 —— 全和 x0+...+x7 出现在每一个 lane", fs=12.5, fc="#7c3a06")
    f.box(24, 512, 1112, 42, "配对跨度 4 -&gt; 2 -&gt; 1：每轮 XOR 掩码减半，log2(W) 轮后全 lane 持全和", fill=LGF, stroke=GREEN, fs=14, bold=True, fc="#1e5631")
    return mk(f.p, 1160, 566, "fig-2-1")

def fig_3_1():
    f = F()
    f.text(24, 12, 1112, 28, "一条 warp 访存指令的合并过程（fp32，每格 4B，一个 sector = 8 格 = 32B）", fs=15.5, bold=True)
    def sector(x, y, nvalid, tag, c):
        for k in range(8):
            v = k < nvalid
            f.box(x + k * 13, y, 11, 30, "", fill=LGF if v else "#f0f1f3", stroke=c if v else "#d1d5db")
        f.text(x, y + 32, 104, 18, tag, fs=10.5, fc="#48586a")
    f.text(40, 56, 1000, 24, "coalesced：lane0..31 访问连续 128B", fs=13.5, bold=True, fc=GREEN)
    for j in range(4):
        sector(40 + j * 110, 84, 8, f"sector{j}", GREEN)
    f.box(500, 84, 240, 30, "S = 4 个 sector", fill=LGF, stroke=GREEN, fs=12.5, bold=True, fc="#1e5631")
    f.box(756, 84, 180, 30, "有效率 100%", fill=LGF, stroke=GREEN, fs=12.5, bold=True, fc="#1e5631")
    f.text(40, 156, 1000, 24, "stride = 2：访问 0, 2, 4, ..., 62 号元素，跨度 256B", fs=13.5, bold=True, fc=ORANGE)
    for j in range(6):
        sector(40 + j * 110, 184, 4, f"sector{j}", ORANGE)
    f.text(690, 186, 40, 26, "...", fs=14, fc="#48586a")
    f.box(756, 184, 180, 30, "S = 8，有效率 50%", fill=LOF, stroke=ORANGE, fs=12.5, bold=True, fc="#7c3a06")
    f.text(40, 256, 1000, 24, "大 stride：每线程各落一个 sector", fs=13.5, bold=True, fc=RED)
    for j in range(6):
        sector(40 + j * 110, 284, 1, f"sector{j}", RED)
    f.text(690, 286, 60, 26, "... x32", fs=14, fc="#48586a")
    f.box(756, 284, 180, 30, "S = 32，有效率 12.5%", fill="#fbe3e3", stroke=RED, fs=12.5, bold=True, fc="#7f1d1d")
    f.text(40, 330, 896, 22, "绿格 = 本次指令真正用到的字节；灰格 = 同 sector 内被一起搬上来的无用字节", fs=12.5, fc="#48586a")
    f.box(24, 362, 1112, 42, "事务粒度 = 32B sector：无用字节照样从 HBM 搬上来 —— 有效率 = 有用字节 / 搬运字节", fill=LOF, stroke=ORANGE, fs=14, bold=True, fc="#7c3a06")
    return mk(f.p, 1160, 416, "fig-3-1")

FIGS = {
    "fig-2-1": (fig_2_1, "fig-2-1-shuffle-butterfly"),
    "fig-3-1": (fig_3_1, "fig-3-1-coalescing"),
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
