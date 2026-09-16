#!/usr/bin/env python3
# fig-29-2-persist-d-ws — persist-D WS 结构：128P+256C、setmaxnreg、Q s2r 常驻与 smem 复用
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
        self.p.append('<mxCell value="" style="rounded=0;fillColor=%s;strokeColor=none;" vertex="1" parent="1">'
                      '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>' % (color, x, y, w, h))

BLUE, GREEN, ORANGE, RED, PURPLE = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#6a51a3"
LBF, LGF, LOF, LRF, LPF = "#f0f6fb", "#c9e4c8", "#fde9d0", "#fbe5e5", "#f1eaf8"
INK = "#102a43"
GRAY = "#9fb3c8"

W, H = 1160, 720

f = F()
f.text(24, 8, 1112, 30, "persist-D WS 结构：128 producer + 256 consumer、Q s2r 常驻与 K stage-0 槽位复用", fs=17, bold=True)

# ===== 左：线程/寄存器结构 =====
f.box(24, 60, 380, 250, "", fill="#f7f9fb", stroke=GRAY)
f.text(40, 66, 348, 22, "384 线程 __launch_bounds__(384,1)", fs=14, bold=True, fc=INK)
f.box(52, 96, 324, 52, "producer WG · 128T · TMA 装载&#10;setmaxnreg dealloc → 32 regs", fill=LBF, stroke=BLUE, fs=12.5, bold=True, fc=BLUE)
f.box(52, 162, 324, 52, "consumer 8 warps · 256T · 全计算&#10;setmaxnreg alloc → 232 regs（fp8 无条件）", fill=LGF, stroke=GREEN, fs=12.5, bold=True, fc="#1e5631")
f.text(52, 224, 324, 20, "寄存器从 170 上限抬到 232：P 量化 + rowsum +", fs=11.5, fc="#627d98")
f.text(52, 244, 324, 20, "rescale 状态；静态 170 会 spill ~60 regs", fs=11.5, fc="#627d98")
f.text(52, 268, 324, 18, "CTA 驻留寄存器受限恒 1/SM", fs=11.5, fc="#627d98")

# ===== 中：smem 布局（复用前 vs 后）=====
f.box(440, 60, 380, 250, "", fill="#f7f9fb", stroke=GRAY)
f.text(456, 66, 348, 22, "SMEM：[Q | K stages | V stages] → Q 区复用", fs=14, bold=True, fc=INK)
# before: 80KB
f.text(456, 96, 348, 18, "朴素（stages 2, D=128）：80KB", fs=12, fc="#48586a")
f.bar(456, 118, 160, 34, "#d5e5f7")   # Q 16KB
f.bar(616, 118, 130, 34, "#c9e4c8")   # K
f.bar(746, 118, 60, 34, "#c9e4c8")    # V
f.text(456, 118, 160, 34, "Q 16KB（死区）", fs=11, fc=BLUE)
f.text(616, 118, 130, 34, "K×2", fs=11, fc="#1e5631")
f.text(746, 118, 60, 34, "V×2", fs=11, fc="#1e5631")
# after: 64KB
f.text(456, 170, 348, 18, "kPersistQs2r：Q 进寄存器，slot0 承接最后 K stage → 64KB", fs=12, bold=True, fc=INK)
f.bar(456, 192, 221, 34, "#c9e4c8")
f.bar(677, 192, 65, 34, "#c9e4c8")
f.bar(742, 192, 64, 34, "#a8d5a8")
f.text(456, 192, 221, 34, "K stages", fs=11, fc="#1e5631")
f.text(677, 192, 65, 34, "V", fs=11, fc="#1e5631")
f.text(742, 192, 64, 34, "K last", fs=11, fc="#14532d")
f.text(456, 236, 348, 20, "省的 16KB 买 L1（共池 ~4-5μs/16KB），不买 occupancy", fs=11.5, fc="#627d98")
f.text(456, 258, 348, 20, "槽位轮转 kKSlot(s)=(s+1)%S：复用区首写/首消费都在稳态，", fs=11.5, fc="#627d98")
f.text(456, 278, 348, 20, "q_consumed 等待藏进整整一 tile 计算", fs=11.5, fc="#627d98")

# ===== 右：装载流水 =====
f.box(860, 60, 276, 250, "", fill="#f7f9fb", stroke=GRAY)
f.text(876, 66, 244, 22, "producer 装载序（时基↓）", fs=14, bold=True, fc=INK)
for i, (t, c) in enumerate([
    ("Q TMA（一次性）", BLUE), ("V(0..S-2) 先于 K 发射", GREEN),
    ("K(0..S-2)（slot 1..，无 Q 依赖）", GREEN), ("稳态循环：V(t+S-1) → K(t+S-1)", GREEN),
    ("K stage0（复用区）等 q_consumed", ORANGE)]):
    f.box(876, 96 + i * 40, 244, 34, t, fill="#ffffff", stroke=c, fs=11.5, fc=INK)
f.text(876, 288, 244, 16, "consumer 先 arrive 全部初始 empty 再等 Q", fs=11, fc="#627d98")

# ===== 下：consumer 主循环五 Phase =====
f.box(24, 340, 1112, 120, "", fill=LGF, stroke=GREEN)
f.text(40, 346, 1072, 22, "consumer kv 循环（#pragma unroll 1，每 tile 五个 Phase）", fs=14, bold=True, fc="#1e5631")
phases = ["① QK GEMM&#10;gemm_rs（Q 常驻寄存器）&#10;int8: s32 原位 cast f32",
          "② log2 域 softmax&#10;exp_offset=log2(v_s·448)&#10;发射 P̃（含 mask/两级剪枝）",
          "③ P 打包&#10;perm pack 4 PRMT&#10;零 SHFL（reorg-free）",
          "④ rowsum MMA + PV&#10;f16 acc: FFMA 吸收 rescale&#10;inst_buf 摘出反馈链",
          "⑤ epilogue&#10;O=(1/448)/rowsum·o_acc&#10;STSM→smem→TMA"]
for i, ph in enumerate(phases):
    f.box(44 + i * 214, 374, 196, 76, ph, fill="#ffffff", stroke=GREEN, fs=10.5, fc=INK)
    if i < 4:
        f.line([(240 + i * 214, 412), (258 + i * 214, 412)], color=GREEN, width=2)

# ===== 底部要点 =====
f.box(24, 490, 1112, 86, "", fill="#f7f9fb", stroke=GRAY)
f.text(44, 496, 1072, 20, "Q s2r 常驻：D=128 实测 -8μs（1028.5→1020.5）、零 spill（NCU local ld/st = 0）；smooth-K 的 qkm 点积必须在循环前 hoist（Q 区会被覆盖）", fs=12.5, fc=INK)
f.text(44, 520, 1072, 20, "输出 epilogue：STSM 进已释放 smem（Q/K/V 全消费完）→ 单次合并 TMA store；尾 tile 退化为行守卫直接 R→G", fs=12.5, fc=INK)
f.text(44, 544, 1072, 20, "V^T per-head (D,N) 平面 TMA；descriptor dim1 必须是 Nkv（列坐标才能正确 tile）", fs=12.5, fc=INK)

f.text(24, 600, 1112, 20, "证伪清单（§5.8 摘录）：WS 双 consumer 负优化 · persistent work loop 零收益 · stages 加深负 · 128P+128C +31.6% · V 驻留寄存器 +48%", fs=12.5, fc=RED)
f.text(24, 626, 1112, 20, "已落地：rescale+absorb FFMA 融合 · max-pass 延迟 scale · per-row rescale gating · Q quant 2 线程/行", fs=12.5, fc="#1e5631")
f.text(24, 668, 1112, 18, "对照 ffpa-attn(861d75e)：cute/fp8/sm_120/persist_d.cuh L193-214（smem 复用）· L256-263/L406-409（reg 重分配）· L560-955（主循环）", fs=12, fc="#627d98")

xml = mk(f.p, W, H, "fig-29-2")
ET.fromstring(xml)
out = sys.argv[1] if len(sys.argv) > 1 else "fig-29-2-persist-d-ws.drawio"
with open(out, "w") as fh:
    fh.write(xml)
print("written", out)
