#!/usr/bin/env python3
# fig-30-1-split-d-walls — 大 D 的两堵墙与 dispatch 三分路由
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

W, H = 1180, 754
f = F()
f.text(24, 8, 1132, 30, "大 D 的两堵墙与 dispatch 三分路由（dispatch/cute_fp8.cuh L60-64）", fs=17, bold=True)

# ===== D 轴三分条 =====
y0 = 64
f.text(24, y0 - 4, 300, 20, "① D 轴三分（fp8 家族，fp16/fp4 同构）", fs=14, bold=True)
seg = [(24, 250, "persist-D  D≤224", LBF, BLUE, "WS 128P+256C · O=D/2"),
       (274, 420, "split-D M8N1  224<D<768", LGF, GREEN, "kBr/kBc=128/128 · non-WS 256T · O=D/2"),
       (644, 480, "split-D M4N2  D≥768", LOF, ORANGE, "kBr/kBc=64/64 · non-WS 256T · O=D/4")]
for x, w, t, fill, stroke, sub in seg:
    f.box(x, y0 + 22, w, 66, "", fill=fill, stroke=stroke)
    f.text(x + 6, y0 + 26, w - 12, 22, t, fs=13.5, bold=True, fc=stroke, align="center")
    f.text(x + 6, y0 + 52, w - 12, 30, sub, fs=11.5, fc="#48586a", align="center")
f.line([(644, y0 + 10), (644, y0 + 96)], color=INK, width=2, dashed=1, arrow_end=0)
f.text(644, y0 - 8, 160, 18, "交叉点 D=768", fs=12.5, bold=True, fc=INK, align="center")
f.line([(274, y0 + 10), (274, y0 + 96)], color=GRAY, width=1.5, dashed=1, arrow_end=0)

# ===== 墙一：smem =====
y1 = 190
f.box(24, y1, 540, 170, "", fill=LRF, stroke=RED)
f.text(40, y1 + 8, 508, 22, "墙一 smem：K tile 字节 = Bc×D×1B，O(D)", fs=14, bold=True, fc=RED)
f.text(40, y1 + 36, 508, 20, "D=512、Bc=128 → K tile 64KB；加 V stage + Q 即超 99KB 预算", fs=12.5, fc=INK)
f.text(40, y1 + 60, 508, 20, "persist-D 的 K stage-0 复用只在 D≤224 挤得过", fs=12.5, fc=INK)
f.box(40, y1 + 88, 240, 62, "整块 K tile 一次进 smem&#10;（persist-D：D≤224 专属）", fill="#ffffff", stroke=RED, fs=12, fc=INK)
f.box(300, y1 + 88, 240, 62, "split-D：D 切 chunk 流水&#10;每 stage 与总 D 无关", fill="#ffffff", stroke=GREEN, fs=12, bold=True, fc="#1e5631")
f.line([(280, y1 + 119), (300, y1 + 119)], color=GREEN, width=2)
f.text(40, y1 + 154, 508, 14, "chunk_index = kv_tile·kDChunks + c，stage/phase 环形推进", fs=11.5, fc="#627d98")

# ===== 墙二：寄存器 =====
f.box(596, y1, 560, 170, "", fill=LPF, stroke=PURPLE)
f.text(612, y1 + 8, 528, 22, "墙二 寄存器：O 累加器 = D/(2N_w) f32/线程", fs=14, bold=True, fc=PURPLE)
f.text(612, y1 + 36, 528, 20, "M8N1 单 warp 独占整 D 列：D=512 → 256 regs 压线（上限 255）", fs=12.5, fc=INK)
f.text(612, y1 + 60, 528, 20, "D≥896 大量 spill 崩塌（D=1024 实测 ~100T）", fs=12.5, fc=INK)
f.box(612, y1 + 88, 250, 62, "M8N1 (8,1,1)：O=D/2&#10;D=1024 → 512 regs 溢出", fill="#ffffff", stroke=PURPLE, fs=12, fc=INK)
f.box(882, y1 + 88, 250, 62, "M4N2 (4,2,1)：O=D/4&#10;D=1024 → 256 regs 边缘 spill 154T", fill="#ffffff", stroke=ORANGE, fs=12, bold=True, fc="#8a4b08")
f.line([(862, y1 + 119), (882, y1 + 119)], color=ORANGE, width=2)

# ===== 公式带 =====
y2 = 396
f.box(24, y2, 1132, 150, "", fill="#fbfcfd", stroke=GRAY)
f.text(40, y2 + 8, 1100, 22, "② 寄存器压力模型：N_w 消去，只有减 M_w 有效", fs=14, bold=True, fc=INK)
f.text(40, y2 + 40, 640, 20, "per-thread：O_acc = 4·(kBr/16M_w)·(D/8N_w)/(kBr/16M_w) = D/(2N_w)", fs=13, fc=INK)
f.text(40, y2 + 66, 640, 20, "per-SM 池占比：[D/(2N_w)] / [65536/(32M_wN_w)] = D·M_w/4096", fs=13, bold=True, fc=PURPLE)
f.text(40, y2 + 92, 640, 20, "N_w 精确消去：M4N4 ≡ M4N2（同为 D/1024）；M2N4 → D/2048 是 D>1024 正解", fs=12.5, fc="#48586a")
f.line([(696, y2 + 40), (696, y2 + 130)], color=GRAY, width=1.5, arrow_end=0)
f.text(712, y2 + 56, 428, 18, "被否决的捷径①：QK M8N1 + PV M4N2 混合布局", fs=12.5, bold=True, fc=RED)
f.text(712, y2 + 74, 428, 50, "QK/PV 共享 Q tile，M 维须一致：&#10;kBr=128 → PV 仍 D/2&#10;kBr=64 → M8N1 空 4 warp", fs=11, fc=INK)
f.text(712, y2 + 104, 428, 18, "被否决的捷径②：WS 版 split-D", fs=12.5, bold=True, fc=RED)
f.text(712, y2 + 122, 428, 16, "setmaxnreg 上限 232 装不下 D=512 的 256-reg O", fs=11.5, fc=INK)

# ===== 数据带 =====
y3 = 576
f.box(24, y3, 1132, 150, "", fill=LGF, stroke=GREEN)
f.text(40, y3 + 8, 1100, 22, "③ 实测交叉点（5090，fp16 同构；PRO 5000 数据见正文 30.6）", fs=14, bold=True, fc="#1e5631")
f.text(40, y3 + 36, 540, 20, "D≤640：M8N1 快 +2~16%（无 P roundtrip、kBr 大一倍）", fs=12.5, fc=INK)
f.text(40, y3 + 60, 540, 20, "D≥768：M4N2 快 +7%@768、+11%@896、+55%@1024", fs=12.5, bold=True, fc=INK)
f.text(40, y3 + 84, 540, 20, "FFPA_FP8_FORCE_KERNEL=split_d|m4n2（224<D≤1024）可强制 A/B", fs=12, fc="#627d98")
f.text(40, y3 + 108, 540, 20, "PRO 5000 实测（本机 headdim 止于 512）：D=512 self 203T / 2.98x", fs=12, fc="#627d98")
f.line([(604, y3 + 36), (604, y3 + 128)], color=GRAY, width=1.5, arrow_end=0)
f.text(620, y3 + 36, 520, 20, "split-D 是精确变换：", fs=13, bold=True, fc=GREEN)
f.text(620, y3 + 60, 520, 20, "QK 侧 D 是归约维：Σ_c Q_c K_c^T 累加进同一 S（无新舍入点）", fs=12, fc=INK)
f.text(620, y3 + 84, 520, 20, "PV 侧 D 是输出列维：各 chunk 的 O 段独立累加", fs=12, fc=INK)
f.text(620, y3 + 108, 520, 20, "softmax 行统计 (m,ℓ) 是 KV 行维的量——与 D 切分无关", fs=12, bold=True, fc=INK)

xml = mk(f.p, W, H, "fig-30-1")
open("/workspace/dev/vipshop/LeetCUDA/kernels/interview/book/figures/drawio/fig-30-1-split-d-walls/fig-30-1-split-d-walls.drawio", "w").write(xml)
print("written", len(xml))
