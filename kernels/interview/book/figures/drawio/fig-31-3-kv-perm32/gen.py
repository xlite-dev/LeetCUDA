#!/usr/bin/env python3
# fig-31-3-kv-perm32 — 32 列窗口双射置换与 perm-aware masking
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
f.text(24, 8, 1110, 30, "kv_perm32：32 列窗口双射置换与 perm-aware masking", fs=17, bold=True)

# ===== 上：置换动机 =====
f.text(24, 46, 700, 20, "① 为什么置换：对齐 C/A fragment 布局", fs=14, bold=True)
f.box(24, 74, 190, 56, "QK MMA&#10;C 累加器布局", fill=LBF, stroke=BLUE, fs=12.5, bold=True, fc=BLUE)
f.box(250, 74, 190, 56, "PV MMA&#10;A 操作数布局", fill=LOF, stroke=ORANGE, fs=12.5, bold=True, fc="#8a4b08")
f.box(466, 74, 260, 56, "列错位 → 不搬 P&#10;改搬 K（SA3 §3.3）", fill=LRF, stroke=RED, fs=12.5, bold=True, fc=RED)
f.line([(214, 102), (250, 102)], color=INK, width=2)
f.line([(440, 102), (466, 102)], color=INK, width=2)
f.text(770, 76, 360, 20, "fp8 的对照解法：16 SHFL + 32 PRMT 搬 P", fs=12, fc="#48586a")
f.text(770, 98, 360, 20, "（reorg-free：第 29 章）；fp4 换成搬 K 列", fs=12, fc="#48586a")
f.text(24, 140, 700, 20, "置换 fused 进 K/V&#7488; 量化 kernel 写侧：存储列 j ← 原 token π(j)", fs=12, fc="#48586a")

# ===== 中：置换表 =====
f.text(24, 174, 1110, 20, "② 32 列窗口内的双射 π（跨窗口恒等；闭式 π(j) = (j&~31) + (loc/8)·2 + ((loc%8)/2)·8 + loc%8）", fs=14, bold=True)
ty = 200
f.text(24, ty + 6, 130, 24, "存储列 j：", fs=12, bold=True, fc=INK, align="right")
f.text(24, ty + 40, 130, 24, "原 token：", fs=12, bold=True, fc=PURPLE, align="right")
perm = [0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27,
        4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31]
gx = 168
cw = 28
for j, tok in enumerate(perm):
    hot = j in (0, 2, 4, 6) or tok in (0, 8, 16, 24)
    f.box(gx + j * (cw + 3), ty, cw, 30, str(j), fill="#f5f8fa", stroke=GRAY, fs=10.5, fc=INK)
    f.box(gx + j * (cw + 3), ty + 34, cw, 30, str(tok), fill=(LPF if tok % 8 in (0, 1) else "#ffffff"),
          stroke=(PURPLE if tok % 8 in (0, 1) else GRAY), fs=10.5, bold=(tok % 8 in (0, 1)),
          fc=(PURPLE if tok % 8 in (0, 1) else "#48586a"))
f.text(gx, ty + 72, 990, 20, "紫色 = 每对 (8k, 8k+1) 的 token 被提前到窗口头部：取数步长呈 0,1,8,9,16,17,24,25,… 的双指数节奏", fs=11.5, fc="#48586a")

# ===== 下：perm-aware masking 对照 =====
y3 = 330
f.box(24, y3, 1110, 440, "", fill="#fbfcfd", stroke=GRAY)
f.text(40, y3 + 8, 1070, 22, "③ masking / bias 必须经 π 回到原始 token 位置判定", fs=14, bold=True, fc=INK)

# 左：正确路径
f.text(48, y3 + 42, 500, 20, "正确（perm-aware）:", fs=13, bold=True, fc=GREEN)
f.box(48, y3 + 66, 130, 44, "smem 列 j", fill="#ffffff", stroke=GRAY, fs=12)
f.box(218, y3 + 66, 170, 44, "pos = kv_tile·Bc&#10;+ kv_perm32(j)", fill=LPF, stroke=PURPLE, fs=11.5, bold=True, fc=PURPLE)
f.box(422, y3 + 66, 130, 44, "causal 判定&#10;pos ≤ q_row", fill=LGF, stroke=GREEN, fs=11.5, bold=True, fc=GREEN)
f.line([(178, y3 + 88), (218, y3 + 88)], color=INK, width=2)
f.line([(388, y3 + 88), (422, y3 + 88)], color=INK, width=2)
f.text(48, y3 + 118, 520, 60, "S 的逻辑列 j 与 K 的存储列同步置换，语义列 = π(j)。&#10;attn_bias 同款：bias 列索引进 gmem/smem 读时过 π(j)。&#10;tile 级跳过（mask_start_tile / Tc_eff）不受影响：π 是窗口内双射。", fs=11.5, fc="#48586a")

# 右：错误路径
f.text(620, y3 + 42, 500, 20, "错误（上游 SA3 的真实 bug）:", fs=13, bold=True, fc=RED)
f.box(620, y3 + 66, 130, 44, "smem 列 j", fill="#ffffff", stroke=GRAY, fs=12)
f.box(790, y3 + 66, 170, 44, "直接用 j 当 token&#10;位置做判定", fill=LRF, stroke=RED, fs=11.5, bold=True, fc=RED)
f.box(994, y3 + 66, 118, 44, "causal&#10;算错", fill=LRF, stroke=RED, fs=12, bold=True, fc=RED)
f.line([(750, y3 + 88), (790, y3 + 88)], color=INK, width=2)
f.line([(960, y3 + 88), (994, y3 + 88)], color=INK, width=2)
f.text(620, y3 + 118, 500, 60, "置换只改存储位置，&#10;causal 边界在语义域：&#10;N=512 max_abs 3.3（量级错误）", fs=11.5, fc=RED)

# 底部：fragment 视角小结
f.text(48, y3 + 200, 1070, 20, "fragment 视角（为什么这张表能对齐布局）:", fs=13, bold=True, fc=INK)
f.box(48, y3 + 226, 330, 66, "QK 后 S fragment&#10;线程 t 持逻辑列 (t%4)·2, (t%4)·2+1 …", fill=LBF, stroke=BLUE, fs=11.5, fc=BLUE)
f.box(424, y3 + 226, 330, 66, "PV 的 A 操作数&#10;线程 t 需要 K 行 (t%4)·8 …（步长 8）", fill=LOF, stroke=ORANGE, fs=11.5, fc="#8a4b08")
f.box(800, y3 + 226, 234, 66, "π 把步长-8 的取数&#10;变成连续存储列", fill=LGF, stroke=GREEN, fs=11.5, bold=True, fc=GREEN)
f.line([(378, y3 + 259), (424, y3 + 259)], color=INK, width=2)
f.line([(754, y3 + 259), (800, y3 + 259)], color=INK, width=2)
f.text(48, y3 + 300, 1070, 20, "同一张 π 表贯穿三处：量化写侧（load_token_id）、masking、attn_bias——单侧改动即静默错位。", fs=12, bold=True, fc=RED)

print(mk(f.p, W, H, "fig31-3"))
