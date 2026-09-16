#!/usr/bin/env python3
# fig-32-1-two-level-p — 两级 P 量化：域拉伸与 exp2 折叠
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

W, H = 1160, 780
f = F()
f.text(24, 8, 1110, 30, "两级 P 量化：online softmax 下的域拉伸与 exp2 折叠", fs=17, bold=True)

# ===== 上：问题（C2 挑战）与方案 =====
f.text(24, 46, 540, 20, "① 问题：直接块量化的 SF 窄域浪费（SA3 C2）", fs=14, bold=True, fc=RED)
f.box(24, 72, 500, 90, "P ∈ [0,1]，行内分布剧变：&#10;causal 早行近 one-hot / dense 行小而均匀&#10;→ s = amax(块)/6 ∈ [0, 0.167]&#10;ue4m3 在窄域的有效位数被浪费（SF 自身先损）", fill=LRF, stroke=RED, fs=12, fc=RED)
f.text(610, 46, 530, 20, "② 方案：两级拉伸（SA3 §3.2）", fs=14, bold=True, fc=GREEN)
f.box(610, 72, 530, 90, "第一级：整行拉伸到满量程（s_P1 = rowmax/2688）&#10;第二级：per-16 组 absmax 归一（s_P2, ue4m3）&#10;→ 组 amax 恰好铺满 ue4m3 域 [0, 448]&#10;2688 = 448（ue4m3 满）× 6（e2m1 满）", fill=LGF, stroke=GREEN, fs=12, fc=GREEN)

# ===== 中：三级域拉伸数轴 =====
y2 = 190
f.text(24, y2 - 4, 1110, 20, "③ 域拉伸链（三个域的端点）", fs=14, bold=True, fc=INK)
# P 域
f.box(24, y2 + 22, 300, 46, "softmax 输出（已减行 max）&#10;值域 [0, 1]", fill="#ffffff", stroke=GRAY, fs=12.5, fc=INK)
# P2 域
f.box(400, y2 + 22, 320, 46, "P₂ = P̃ × 2688&#10;值域 [0, 2688]（全局常数拉伸）", fill=LPF, stroke=PURPLE, fs=12.5, bold=True, fc=PURPLE)
# e2m1 域
f.box(800, y2 + 22, 340, 46, "P̂₂ ∈ (0, 6]（e2m1 数据）&#10;s_P2 ∈ (0, 448]（ue4m3 组 SF）", fill=LGF, stroke=GREEN, fs=12.5, bold=True, fc=GREEN)
f.line([(324, y2 + 45), (400, y2 + 45)], color=PURPLE, width=2.5)
f.line([(720, y2 + 45), (800, y2 + 45)], color=GREEN, width=2.5)
f.text(322, y2 + 50, 76, 34, "×2688", fs=10.5, fc=PURPLE, bold=True)
f.text(726, y2 + 50, 72, 34, "÷s_P2", fs=10.5, fc=GREEN, bold=True)
f.text(24, y2 + 76, 1110, 20, "还原：O = FP4MM(P̂₂, s_P2, V̂, s_V) —— 2688 在 O 与 row_sum 同现，finalize 相除精确消去", fs=12, fc="#48586a")

# ===== 下左：online softmax 退化 =====
y3 = 300
f.box(24, y3, 540, 220, "", fill="#fbfcfd", stroke=GRAY)
f.text(40, y3 + 8, 510, 20, "④ 实现的红利：第一级退化成常数", fs=13.5, bold=True, fc=INK)
f.text(40, y3 + 34, 510, 40, "online softmax 先减运行行 max ⇒&#10;rowmax(P̃) ≡ 1 ⇒ s_P1 ≡ 1/(448×6)", fs=12.5, bold=True, fc=PURPLE)
f.box(40, y3 + 84, 510, 60, "P₂ = exp2(S·L − m·L + log2(1/2688)) ∈ [0, 2688]&#10;s_P2 = exp2(a·L − m_sc + log2(1/6)) ∈ (0, 448]&#10;（L = scale·log2e；编译期常数 −11.392 / −2.585）", fill="#fbfcfd", stroke="#d9e2ec", fs=11.5, fc="#48586a")
f.text(40, y3 + 152, 510, 20, "逐行的 s_P1 存储与乘回全部消失", fs=12.5, bold=True, fc=GREEN)
f.text(40, y3 + 176, 510, 20, "组归一也折进 exp2：q = exp2((s−a)·L + log2 6)", fs=12, fc=INK)

# ===== 下中：shuffle 链 =====
f.box(592, y3, 250, 220, "", fill="#fbfcfd", stroke=GRAY)
f.text(608, y3 + 8, 220, 20, "⑤ 同链双归约", fs=13.5, bold=True, fc=INK)
f.box(608, y3 + 36, 218, 40, "lane 持 8 元素&#10;组 = 2 lane × 8", fill="#ffffff", stroke=GRAY, fs=11.5, fc=INK)
f.box(608, y3 + 88, 218, 40, "shfl_xor(1)&#10;→ 16 元素组 absmax", fill=LBF, stroke=BLUE, fs=11.5, bold=True, fc=BLUE)
f.box(608, y3 + 140, 218, 40, "shfl_xor(2)&#10;→ quad 内行 max", fill=LOF, stroke=ORANGE, fs=11.5, bold=True, fc="#8a4b08")
f.line([(717, y3 + 76), (717, y3 + 88)], color=INK, width=2)
f.line([(717, y3 + 128), (717, y3 + 140)], color=INK, width=2)
f.text(608, y3 + 188, 220, 20, "组尺度+行 max 冗余减半", fs=11.5, fc="#48586a")

# ===== 下右：守卫 =====
f.box(870, y3, 270, 220, "", fill="#fbfcfd", stroke=GRAY)
f.text(886, y3 + 8, 240, 20, "⑥ 三个守卫", fs=13.5, bold=True, fc=INK)
f.text(886, y3 + 34, 240, 20, "全 masked 组：clamp 防 NaN", fs=11.5, fc=RED)
f.text(886, y3 + 84, 240, 20, "InfCheck：全 masked 行守卫", fs=11.5, fc=RED)
f.text(886, y3 + 134, 240, 20, "FirstTile：−∞ 重启防泄漏", fs=11.5, fc=RED)
f.text(886, y3 + 184, 240, 20, "ex2.approx.ftz.f32 单 MUFU", fs=11.5, bold=True, fc=GREEN)

# ===== 底：lazy rescale 与 mxfp8 =====
y4 = 546
f.box(24, y4, 540, 100, "", fill="#fbfcfd", stroke=GRAY)
f.text(40, y4 + 6, 510, 20, "⑦ lazy rescale（PC-11 per-row 守卫）", fs=13, bold=True, fc=INK)
f.box(40, y4 + 30, 510, 60, "跨 tile 行 max 更新不即时重乘 O；下一个 PV 累加前&#10;scores_scale(mi) &lt; 1.0f 的行才补乘（dense 96.5% 命中零 FMUL），&#10;FMUL 永不无守卫进入 MMA 依赖链", fill="#fbfcfd", stroke="#d9e2ec", fs=11.5, fc="#48586a")
f.box(592, y4, 548, 100, "", fill="#fbfcfd", stroke=GRAY)
f.text(608, y4 + 6, 520, 20, "⑧ MXFP8 PV 变体的域常量切换", fs=13, bold=True, fc=INK)
f.box(608, y4 + 30, 520, 60, "Vᵀ 换 e4m3 + ue8m0（32 组）时：P 侧同换 32 粒度，&#10;域常量 2688 → 448（无 e2m1 因子）、SF 取 2 的幂；&#10;历史 latent bug：lse 无条件沿用 2688（差 ln 6）", fill="#fbfcfd", stroke="#d9e2ec", fs=11.5, fc="#48586a")

f.text(24, y4 + 116, 1110, 20, "口诀：两级量化在实现里只剩一组组 absmax——第一级被 online softmax 折成常数，第二级的除法被 exp2 参数吸收。", fs=12.5, bold=True, fc=PURPLE)

print(mk(f.p, W, H, "fig32-1"))
