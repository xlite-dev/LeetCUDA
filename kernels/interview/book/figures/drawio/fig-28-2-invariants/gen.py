#!/usr/bin/env python3
# fig-28-2-invariants — 三条不变变换：改数值、保输出（smooth-K / smooth-V / WHT）
import xml.etree.ElementTree as ET
import sys

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
    def bar(self, x, y, w, h, color):
        # 简易柱条（用于 outlier 示意）
        self.p.append('<mxCell value="" style="rounded=0;fillColor=%s;strokeColor=none;" vertex="1" parent="1">'
                      '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>' % (color, x, y, w, h))

BLUE, GREEN, ORANGE, RED, PURPLE = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#6a51a3"
LBF, LGF, LOF, LRF, LPF = "#f0f6fb", "#c9e4c8", "#fde9d0", "#fbe5e5", "#f1eaf8"
INK = "#102a43"
GRAY = "#9fb3c8"

W, H = 1160, 720

f = F()
f.text(24, 8, 1112, 30, "三条不变变换：改数值、保输出 —— outlier 处理的代数基础", fs=17, bold=True)

# ================= 泳道 1：smooth-K =================
y1 = 60
f.box(24, y1, 1112, 168, "", fill="#f7f9fb", stroke=GRAY)
f.box(36, y1 + 12, 176, 34, "smooth-K（列均值平移）", fill=LPF, stroke=PURPLE, fs=13.5, bold=True, fc=PURPLE)
f.text(36, y1 + 54, 380, 20, "K&#39; = K − 1·k̄^T    （k̄ = 每列均值，逐行减同一向量）", fs=13, fc=INK, bold=True)
f.text(36, y1 + 78, 380, 20, "S&#39; = QK&#39;^T = S − q_s·1^T   行内常数平移", fs=13, fc=INK)
f.text(36, y1 + 102, 380, 20, "softmax(S&#39;) = softmax(S)：分子分母同乘 e^{q_s}", fs=13, fc=INK)
f.text(36, y1 + 126, 380, 20, "⇒ P、O 逐 bit 不变", fs=13.5, fc="#1e5631", bold=True)
# 右侧：不变量来源 + 修正
f.box(440, y1 + 50, 300, 96, "", fill=LBF, stroke=BLUE)
f.text(452, y1 + 56, 276, 20, "不变量来源：softmax 行平移不变性", fs=12.5, bold=True, fc=BLUE)
f.text(452, y1 + 78, 276, 20, "唯一代价：lse(S&#39;) = lse(S) − q_s", fs=12.5, fc="#48586a")
f.text(452, y1 + 100, 276, 20, "epilogue 回加 δ_Q·δ_K·(Q·k̄)", fs=12.5, fc="#48586a")
f.text(452, y1 + 122, 276, 20, "（smooth_k_qk_dot，quad 内归约）", fs=11.5, fc="#627d98")
# logits 平移示意（迷你坐标）
f.box(770, y1 + 42, 350, 110, "", fill="#ffffff", stroke=GRAY)
f.text(780, y1 + 46, 330, 18, "logits 平移示意（同一条曲线整体左右移）", fs=11.5, fc="#627d98")
f.line([(800, y1 + 96), (1100, y1 + 96)], color=GRAY, width=1.5, arrow_end=0)
f.line([(950, y1 + 58), (950, y1 + 132)], color=GRAY, width=1.5, arrow_end=0)
f.line([(950, y1 + 96), (1020, y1 + 66), (1090, y1 + 90), (1100, y1 + 122)], color=RED, width=2, arrow_end=0)
f.line([(890, y1 + 90), (950, y1 + 96)], color=RED, width=2, arrow_end=0)

# ================= 泳道 2：smooth-V =================
y2 = 244
f.box(24, y2, 1112, 168, "", fill="#f7f9fb", stroke=GRAY)
f.box(36, y2 + 12, 176, 34, "smooth-V（均值剥离）", fill=LPF, stroke=PURPLE, fs=13.5, bold=True, fc=PURPLE)
f.text(36, y2 + 54, 380, 20, "V&#39; = V − 1·v̄^T    （同 smooth-K 的列均值形态）", fs=13, fc=INK, bold=True)
f.text(36, y2 + 78, 380, 20, "O&#39; = P·V&#39; = PV − (P·1)·v̄^T = O − v̄^T", fs=13, fc=INK)
f.text(36, y2 + 102, 380, 20, "P·1 = 1（softmax 行和恒为 1）", fs=13, fc=INK)
f.text(36, y2 + 126, 380, 20, "⇒ 输出每行减同一均值，最后加回 v̄ 即精确还原", fs=13.5, fc="#1e5631", bold=True)
# 右侧：残差对称化
f.box(440, y2 + 50, 300, 96, "", fill=LBF, stroke=BLUE)
f.text(452, y2 + 56, 276, 20, "残差对称化（sage recipe）", fs=12.5, bold=True, fc=BLUE)
f.text(452, y2 + 78, 276, 20, "amax = max(|max−μ|, |min−μ|)", fs=12.5, fc="#48586a")
f.text(452, y2 + 100, 276, 20, "中心化后对称 → e4m3 域 [−448,448] 两侧同时铺满", fs=12, fc="#48586a")
f.text(452, y2 + 122, 276, 20, "f16 PV 累加路径：v_scale_max 压缩留裕量", fs=11.5, fc="#627d98")
# 对称化示意（迷你柱图：左偏分布 vs 对称分布）
f.box(770, y2 + 42, 350, 110, "", fill="#ffffff", stroke=GRAY)
f.text(780, y2 + 46, 330, 18, "减均值前后残差分布（示意）", fs=11.5, fc="#627d98")
f.line([(800, y2 + 132), (1100, y2 + 132)], color=GRAY, width=1.5, arrow_end=0)
for i, hh in enumerate([16, 22, 18, 12, 8, 5, 3, 2, 1, 1, 24, 6]):
    f.bar(810 + i * 22, y2 + 130 - hh, 16, hh, ORANGE)
f.text(780, y2 + 134, 340, 16, "橙色=原始（右偏 outlier），剥离均值后分布趋对称、scale 收敛", fs=10.5, fc="#627d98")

# ================= 泳道 3：WHT =================
y3 = 428
f.box(24, y3, 1112, 168, "", fill="#f7f9fb", stroke=GRAY)
f.box(36, y3 + 12, 176, 34, "Hadamard 旋转", fill=LPF, stroke=PURPLE, fs=13.5, bold=True, fc=PURPLE)
f.text(36, y3 + 54, 380, 20, "Ĥ = H_n/√n 正交（ĤĤ^T = I）", fs=13, fc=INK, bold=True)
f.text(36, y3 + 78, 380, 20, "(Q·Ĥ)(K·Ĥ)^T = Q·(ĤĤ^T)·K^T = QK^T", fs=13, fc=INK)
f.text(36, y3 + 102, 380, 20, "逐元素精确、零修正项（连 lse 都不用动）", fs=13, fc=INK)
f.text(36, y3 + 126, 380, 20, "radix-2 蝶形：(a,b)→(a+b, a−b)，每 bit 一轮", fs=13.5, fc="#1e5631", bold=True)
# 右侧：whitening
f.box(440, y3 + 50, 300, 96, "", fill=LBF, stroke=BLUE)
f.text(452, y3 + 56, 276, 20, "whitening / Jackson 性质", fs=12.5, bold=True, fc=BLUE)
f.text(452, y3 + 78, 276, 20, "每个输出坐标 = 全部输入的带符号和", fs=12.5, fc="#48586a")
f.text(452, y3 + 100, 276, 20, "能量摊到所有坐标 → 块 scale 收敛", fs=12.5, fc="#48586a")
f.text(452, y3 + 122, 276, 20, "FA-3 incoherent proc.：FP8 RMSE ↓2.6×", fs=11.5, fc="#627d98")
# outlier 摊平示意：旋转前一根高柱，旋转后均匀
f.box(770, y3 + 42, 350, 110, "", fill="#ffffff", stroke=GRAY)
f.text(780, y3 + 46, 330, 18, "行幅度：旋转前（左）vs 旋转后（右）", fs=11.5, fc="#627d98")
f.line([(800, y3 + 132), (1100, y3 + 132)], color=GRAY, width=1.5, arrow_end=0)
for i, hh in enumerate([62, 10, 6, 4, 3, 2, 2, 1, 1, 1, 1, 1]):
    f.bar(810 + i * 16, y3 + 130 - hh, 12, hh, ORANGE)
for i in range(12):
    f.bar(1030 + i * 6, y3 + 130 - 24, 4, 24, GREEN)
f.text(780, y3 + 134, 340, 16, "橙色=outlier 主导块 scale，绿色=旋转后幅度均匀（各自独立坐标轴示意）", fs=10.5, fc="#627d98")

# ================= 底部结论 =================
f.box(24, 616, 1112, 66, "", fill=LGF, stroke=GREEN)
f.text(44, 622, 1072, 22, "共同目的：在量化之前把 outlier 能量摊平 → block scale 收敛 → 量化步长利用率最大化", fs=14, bold=True, fc="#1e5631")
f.text(44, 648, 1072, 22, "共同约束：变换必须保持输出精确不变（或仅需行级标量修正）——这是代数定理，不是近似", fs=13.5, fc=INK)
f.text(24, 692, 1112, 18, "对照 ffpa-attn(861d75e)：cute/fp8/smooth_k.cuh · smooth_v.cuh · cute/hadamard.cuh（宽度规则：pow2 D≤512 全宽，否则 blockdiag H_64）", fs=12, fc="#627d98")

xml = mk(f.p, W, H, "fig-28-2")
ET.fromstring(xml)
out = sys.argv[1] if len(sys.argv) > 1 else "fig-28-2-invariants.drawio"
with open(out, "w") as fh:
    fh.write(xml)
print("written", out)
