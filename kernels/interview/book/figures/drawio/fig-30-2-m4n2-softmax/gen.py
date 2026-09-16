#!/usr/bin/env python3
# fig-30-2-m4n2-softmax — (4,2,1) atom 布局与跨 N-warp softmax 单 barrier 协议
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

W, H = 1180, 824
f = F()
f.text(24, 8, 1132, 30, "M4N2 (4,2,1) 布局与跨 N-warp softmax 单 barrier 协议", fs=17, bold=True)

# ===== 左上：warp 网格 =====
f.text(24, 46, 560, 20, "① 8 warps = 4 M-warp × 2 N-warp（kBr=64, kBc=64）", fs=14, bold=True)
gx, gy = 24, 70
cw, ch = 92, 46
for r in range(2):
    for c in range(4):
        wid = r * 4 + c
        fill = LGF if r == 0 else LOF
        stroke = GREEN if r == 0 else ORANGE
        f.box(gx + c * (cw + 8), gy + r * (ch + 8), cw, ch,
              "W%d" % wid, fill=fill, stroke=stroke, fs=14, bold=True, fc=stroke)
f.text(gx, gy + 2 * ch + 20, 416, 18, "N-warp 0（列 0-31）", fs=12, bold=True, fc=GREEN)
f.text(gx, gy + 2 * ch + 40, 416, 18, "N-warp 1（列 32-63）：peer = warp_id ⊕ 4", fs=12, bold=True, fc=ORANGE)
f.text(gx, gy + 2 * ch + 64, 430, 40, "每 warp：16 行 × 32 列的 S 片段；&#10;行统计只持半行 Bc/2 列 → 必须跨 warp 合并", fs=11.5, fc="#48586a")
# peer 弧线（W0<->W4 示意）
f.line([(gx + cw / 2, gy + ch), (gx + cw / 2 + 30, gy + ch + 4), (gx + cw / 2 + 30, gy + ch + 8)], color=PURPLE, width=2, dashed=1)

# ===== 右上：P roundtrip =====
f.text(500, 46, 656, 20, "② P 的 SMEM roundtrip（每 N-warp 无法重建完整 A-operand）", fs=14, bold=True)
f.box(500, 74, 150, 52, "P e4m3x2&#10;（寄存器）", fill=LBF, stroke=BLUE, fs=12.5, bold=True, fc=BLUE)
f.box(690, 74, 190, 52, "SMEM P staging&#10;4KB · SW64 · DefaultCopy", fill="#ffffff", stroke=GREEN, fs=12, fc=INK)
f.box(920, 74, 150, 52, "LDSM_N 读回&#10;A-fragment", fill=LOF, stroke=ORANGE, fs=12.5, bold=True, fc="#8a4b08")
f.line([(650, 100), (690, 100)], color=INK, width=2)
f.line([(880, 100), (920, 100)], color=INK, width=2)
f.text(500, 132, 570, 18, "写侧 make_tiled_copy_C(QK mma)，读侧 make_tiled_copy_A(PV mma)", fs=11.5, fc="#627d98")
f.text(500, 150, 570, 18, "stmatrix 是 b16 操作写不了 1B e4m3 → DefaultCopy 向量化 store", fs=11.5, fc=RED)
f.text(500, 172, 570, 18, "syncthreads 顺带发布 sum 段写入（协议③的便车）", fs=12, bold=True, fc=PURPLE)

# ===== 下半：smem_exchange 协议 =====
y2 = 276
f.box(24, y2, 1132, 400, "", fill="#fbfcfd", stroke=GRAY)
f.text(40, y2 + 8, 1100, 22, "③ smem_exchange 单 barrier 协议（1KB = max 段 [8][16] ‖ sum 段 [8][16]）", fs=14, bold=True, fc=INK)

# 左列：分区框 + RAW + 一致性契约
f.box(40, y2 + 40, 330, 56, "max 段 [8 warps][16 rows]", fill=LBF, stroke=BLUE, fs=12.5, bold=True, fc=BLUE)
f.box(40, y2 + 104, 330, 56, "sum 段 [8 warps][16 rows]", fill=LOF, stroke=ORANGE, fs=12.5, bold=True, fc="#8a4b08")
f.text(40, y2 + 166, 336, 34, "两段分置：sum 写与 peer max 读无同步，&#10;合一块 = 偶发 RAW 覆写", fs=11.5, fc=RED)
f.text(40, y2 + 216, 340, 20, "一致性契约（违反即 O 列段错位）：", fs=12.5, bold=True, fc=RED)
f.text(40, y2 + 238, 340, 20, "max 合并：m = max(m_C0, m_C1)（顺序无关）", fs=11.5, fc=INK)
f.text(40, y2 + 258, 340, 20, "sum 合并：ℓ = ℓ_C0 + ℓ_C1（舍入级差异）", fs=11.5, fc=INK)
f.text(40, y2 + 278, 340, 20, "lse 只由 n_warp==0 写（共享 Q 行，L970）", fs=11.5, fc="#48586a")
f.text(40, y2 + 298, 340, 20, "writer = lane%4==0；row_local = lane/4+8r", fs=11.5, fc="#627d98")
f.text(40, y2 + 318, 340, 20, "fp8 变体：offset=log2(v_s·448) 折 exp2", fs=11.5, fc=BLUE)
f.text(40, y2 + 338, 340, 20, "tile sum 先 ×1/(v_s·448) 再入 sum 段", fs=11.5, fc=BLUE)

# 右列：三步横条
f.box(420, y2 + 40, 712, 96, "", fill=LBF, stroke=BLUE)
f.text(432, y2 + 48, 688, 18, "step 1 · 唯一显式 barrier（syncthreads）", fs=12.5, bold=True, fc=BLUE)
f.text(432, y2 + 70, 688, 18, "各 warp：tile max（shfl_xor 1/2 归约）→ 写 max 段", fs=12, fc=INK)
f.text(432, y2 + 92, 688, 18, "随后全 CTA syncthreads——整个协议只有这一次", fs=12, fc=INK)

f.box(420, y2 + 148, 712, 120, "", fill=LOF, stroke=ORANGE)
f.text(432, y2 + 156, 688, 18, "step 2 · 无 barrier", fs=12.5, bold=True, fc="#8a4b08")
f.text(432, y2 + 178, 688, 18, "读 max(自己, peer) → 全局 m；更新 (m, ℓ, scale)", fs=12, fc=INK)
f.text(432, y2 + 200, 688, 18, "发射 P = exp2(s·γ − (m − offset))——P 直接落在量化域", fs=12, fc=INK)
f.text(432, y2 + 222, 688, 18, "本 warp tile sum × 1/(v_s·448) 写 sum 段（发布留给步骤③）", fs=12, fc=INK)

f.box(420, y2 + 280, 712, 96, "", fill=LGF, stroke=GREEN)
f.text(432, y2 + 288, 688, 18, "step 3 · 搭便车（P roundtrip barrier）", fs=12.5, bold=True, fc=GREEN)
f.text(432, y2 + 310, 688, 18, "P 写读 SMEM roundtrip 自带的 syncthreads 顺带发布 sum 写", fs=12, fc=INK)
f.text(432, y2 + 332, 688, 18, "finalize_row_sum_m4n2：ℓ = ℓ·scale + Σ(自己, peer)——省 2 次 barrier", fs=12, fc=INK)

xml = mk(f.p, W, H, "fig-30-2")
open("/workspace/dev/vipshop/LeetCUDA/kernels/interview/book/figures/drawio/fig-30-2-m4n2-softmax/fig-30-2-m4n2-softmax.drawio", "w").write(xml)
print("written", len(xml))
