#!/usr/bin/env python3
# fig-31-2-fp4-pipeline — fp4 前处理链数据流（km → qm → quantize → delta_s → 主 kernel）
def mk(parts, W, H, did):
    return ('<mxfile host="app.diagrams.net"><diagram id="%s" name="%s">'
            '<mxGraphModel dx="800" dy="600" grid="0" page="1" pageWidth="%d" pageHeight="%d" math="0" shadow="0">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>%s'
            '</root></mxGraphModel></diagram></mxfile>') % (did, did, W, H, ''.join(parts))

class F:
    def __init__(self):
        self.p = []
    def box(self, x, y, w, h, t="", fill="#ffffff", stroke="#bcccdc", fs=13, bold=False, fc="#1f2933"):
        st = ('rounded=0;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=%s;'
              'fontSize=%d;fontColor=%s;align=center;verticalAlign=middle;fontFamily=Helvetica') % (fill, stroke, fs, fc)
        if bold: st += ';fontStyle=1'
        self.p.append('<mxCell value="%s" style="%s" vertex="1" parent="1">'
                      '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>' % (t, st, x, y, w, h))
    def text(self, x, y, w, h, t, fs=13, fc="#102a43", bold=False, align="left"):
        st = 'text;html=1;align=%s;verticalAlign=middle;fontSize=%d;fontColor=%s;fontFamily=Helvetica' % (align, fs, fc)
        if bold: st += ';fontStyle=1'
        self.p.append('<mxCell value="%s" style="%s" vertex="1" parent="1">'
                      '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>' % (t, st, x, y, w, h))
    def line(self, pts, color="#2171b5", width=2, dashed=0, arrow_end=1):
        ae = "blockThin;endFill=1" if arrow_end else "none"
        st = 'endArrow=%s;html=1;strokeColor=%s;strokeWidth=%d;rounded=0;dashed=%d' % (ae, color, width, dashed)
        self.p.append('<mxCell value="" style="%s" edge="1" parent="1">'
                      '<mxGeometry relative="1" as="geometry"><Array as="points">'
                      % st + ''.join('<mxPoint x="%d" y="%d"/>' % (x, y) for x, y in pts[1:-1])
                      + '</Array><mxPoint x="%d" y="%d" as="sourcePoint"/>'
                        '<mxPoint x="%d" y="%d" as="targetPoint"/></mxGeometry></mxCell>' % (pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]))

BLUE, GREEN, ORANGE, RED, PURPLE = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#6a51a3"
LBF, LGF, LOF, LRF, LPF = "#f0f6fb", "#c9e4c8", "#fde9d0", "#fbe5e5", "#f1eaf8"
INK, GRAY = "#102a43", "#9fb3c8"

W, H = 1160, 800
f = F()
f.text(24, 8, 1110, 30, "fp4 前处理链数据流：均值先行，量化居中，rank-1 修正收尾", fs=17, bold=True)

# ===== 顶：输入 =====
f.box(24, 48, 190, 52, "Q (B,S,H,D)&#10;原生 stride 读", fill="#ffffff", stroke=INK, fs=12.5, bold=True)
f.box(250, 48, 190, 52, "K (B,S,H,D)&#10;原生 stride 读", fill="#ffffff", stroke=INK, fs=12.5, bold=True)
f.box(476, 48, 190, 52, "V (B,S,H,D)&#10;原生 stride 读", fill="#ffffff", stroke=INK, fs=12.5, bold=True)

# ===== 第一行：均值先行 =====
f.text(24, 112, 1110, 20, "① 均值先行（是量化 kernel 的输入 bias，必须先算）", fs=13.5, bold=True, fc=INK)
f.box(24, 138, 254, 58, "q_m = mean(Q, 128 行块)&#10;fp32 + dtype 双份（fused 内产出）", fill=LPF, stroke=PURPLE, fs=11.5, fc=PURPLE)
f.box(318, 138, 190, 58, "k&#772; = mean(K, 序列)&#10;外部小 kernel", fill=LPF, stroke=PURPLE, fs=11.5, fc=PURPLE)
f.box(558, 138, 254, 58, "q_km = q_m · k&#772;（每行常数）&#10;与 v_m = mean(V, 列)（可选）", fill=LPF, stroke=PURPLE, fs=11.5, fc=PURPLE)
f.line([(120, 100), (120, 138)], color=PURPLE, width=2)
f.line([(345, 100), (345, 138)], color=PURPLE, width=2)
f.line([(560, 100), (560, 138)], color=PURPLE, width=2, dashed=1)

# ===== 第二行：量化 kernel =====
f.text(24, 210, 1110, 20, "② 量化 kernel：减 bias → (WHT fused) → 1×16 块 absmax → cvt e2m1/e4m3", fs=13.5, bold=True, fc=INK)
y2 = 236
f.box(24, y2, 268, 64, "Q 量化&#10;sub q_m ·（WHT fused：蝶形 + 2×shfl_xor）", fill=LGF, stroke=GREEN, fs=11.5, fc=GREEN)
f.box(318, y2, 268, 64, "K 量化（perm32 写序）&#10;sub k&#772; · 列 j ← token π(j)", fill=LGF, stroke=GREEN, fs=11.5, fc=GREEN)
f.box(612, y2, 268, 64, "V&#7488; 量化（smem 转置）&#10;（可选 sub v_m，恒等加回）", fill=LGF, stroke=GREEN, fs=11.5, fc=GREEN)
f.line([(120, 196), (120, y2)], color=PURPLE, width=2)
f.line([(345, 196), (345, y2)], color=PURPLE, width=2)
f.line([(560, 196), (560, y2)], color=PURPLE, width=2)
f.line([(120, 100), (120, 138)], color=GRAY, width=0)
# 输入到量化
f.line([(70, 100), (70, y2)], color=INK, width=2)
f.line([(280, 100), (280, y2)], color=INK, width=2)
f.line([(510, 100), (510, y2)], color=INK, width=2)
f.line([(700, 100), (700, y2)], color=INK, width=2)

# D<=128 fused 虚线框
f.box(12, y2 - 22, 890, 100, "", fill="none", stroke=RED)
f.text(16, y2 - 20, 400, 16, "D ≤ 128：三段合一个 launch（[Q|K|V] grid 段）", fs=11, bold=True, fc=RED)

# ===== 第三行：输出 workspace =====
y3 = 340
f.text(24, y3 - 6, 1110, 20, "③ 输出 workspace（e2m1 数据 + ue4m3 SF，SF 按 atom 块序写）", fs=13.5, bold=True, fc=INK)
f.box(24, y3 + 18, 268, 52, "Q&#770; + s_Q&#10;(B,H,N,D/2)+(B,H,N,D/16)", fill=LBF, stroke=BLUE, fs=11.5, fc=BLUE)
f.box(318, y3 + 18, 268, 52, "K&#770;&#7488; + s_K&#10;perm32 存储列序", fill=LBF, stroke=BLUE, fs=11.5, fc=BLUE)
f.box(612, y3 + 18, 268, 52, "V&#770;&#7488; + s_V&#10;(B,H,D,N/2) 转置布局", fill=LBF, stroke=BLUE, fs=11.5, fc=BLUE)
f.line([(120, y2 + 64), (120, y3 + 18)], color=INK, width=2)
f.line([(345, y2 + 64), (345, y3 + 18)], color=INK, width=2)
f.line([(680, y2 + 64), (680, y3 + 18)], color=INK, width=2)

# ===== 第四行：delta_s =====
y4 = 448
f.text(24, y4 - 6, 1110, 20, "④ ΔS rank-1 修正：q_m K&#7488; − q_km（免物化 K−k&#772;）", fs=13.5, bold=True, fc=INK)
f.box(24, y4 + 20, 300, 66, "delta_s kernel（128×128 tile）&#10;wmma 版 / CuTe+TMA 版 按 shape dispatch&#10;大 N：torch 链 1.4ms → ~0.3ms", fill=LOF, stroke=ORANGE, fs=11.5, fc="#8a4b08")
f.box(360, y4 + 20, 240, 66, "ΔS (B,H,M_b,N_kv) f32&#10;pad 列零填（主 kernel -inf）", fill=LRF, stroke=RED, fs=11.5, fc=RED)
f.line([(324, y4 + 53), (360, y4 + 53)], color=INK, width=2)
f.line([(100, 196), (100, y2)], color=PURPLE, width=0)
f.line([(140, 196), (140, y2 + 60), (140, y4 + 20)], color=PURPLE, width=2, dashed=1)
f.line([(330, y2 + 64), (330, y3 + 70), (330, y4 + 20)], color=PURPLE, width=2, dashed=1)

# ===== 底：主 kernel =====
y5 = 580
f.box(24, y5, 1110, 78, "", fill="#fbfcfd", stroke=GRAY)
f.text(40, y5 + 6, 1070, 20, "⑤ fp4 persist-D 主 kernel（下一章）：S = Q&#770;K&#770;&#7488; + ΔS → 两级 P 量化 → PV blockscale → lse/epilogue 加回", fs=13, bold=True, fc=INK)
f.box(60, y5 + 34, 200, 36, "lse 修正：&#10;scale · dot(q_row, k&#772;)", fill="#ffffff", stroke=PURPLE, fs=11, fc=PURPLE)
f.box(300, y5 + 34, 220, 36, "O epilogue：+ v_m（可选，恒等）", fill="#ffffff", stroke=PURPLE, fs=11, fc=PURPLE)
f.box(560, y5 + 34, 250, 36, "O pad 后切回 d_og 宽度输出", fill="#ffffff", stroke=BLUE, fs=11, fc=BLUE)
for x in (120, 345, 680):
    f.line([(x, y3 + 70), (x, y5)], color=BLUE, width=2)
f.line([(480, y4 + 86), (480, y5)], color=RED, width=2)

# 右侧竖注（check_text_fit 按整串估宽，右侧文本须极短）
f.text(912, y2 + 4, 220, 48, "三不变性见正文&#10;（旁路 / WHT / pad）", fs=11, fc="#48586a")
f.text(912, y4 + 24, 220, 60, "mask/bias 必须&#10;perm-aware（π&#8315;&#185;）", fs=11.5, bold=True, fc=RED)

print(mk(f.p, W, H, "fig31-2"))
