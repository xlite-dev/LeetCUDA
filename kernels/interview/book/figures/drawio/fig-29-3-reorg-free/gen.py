#!/usr/bin/env python3
# fig-29-3-reorg-free — 跨 lane 重排 vs 就地打包 + V^T 列置换（归约轴双射不变性）
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
    def cell(self, x, y, s, color, fs=10.5):
        # 单字节格子
        self.p.append('<mxCell value="%s" style="rounded=0;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=#ffffff;fontSize=%s;fontColor=#102a43;fontFamily=Helvetica;" vertex="1" parent="1">'
                      '<mxGeometry x="%d" y="%d" width="22" height="22" as="geometry"/></mxCell>' % (s, color, fs, x, y))

BLUE, GREEN, ORANGE, RED, PURPLE = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#6a51a3"
LBF, LGF, LOF, LRF, LPF = "#f0f6fb", "#c9e4c8", "#fde9d0", "#fbe5e5", "#f1eaf8"
INK = "#102a43"
GRAY = "#9fb3c8"
CQ, CA = "#d5e5f7", "#fde9d0"   # C-fragment 底色 / A-fragment 底色

W, H = 1160, 740

f = F()
f.text(24, 8, 1112, 30, "reorg-free 打包：跨 lane 重排（16 SHFL）→ 就地打包 + V^T 列置换（4 PRMT、零 SHFL）", fs=17, bold=True)

# ===== 顶部：为什么必须重排（fragment 布局）=====
f.box(24, 52, 1112, 176, "", fill="#f7f9fb", stroke=GRAY)
f.text(40, 58, 1072, 20, "m16n8k32 的两个布局（g = lane>>2 组行，t = lane&amp;3 组内）：C 持列对 {2t,2t+1}，A 要列 {4t..4t+3} —— 跨 lane", fs=13, bold=True, fc=INK)
# C fragment 示意（4 quad lanes 的一行）
f.text(60, 92, 500, 18, "C fragment（QK 累加器，fp32×4）：lane t 持列对 {2t, 2t+1}", fs=12, fc=BLUE)
for t in range(4):
    f.cell(60 + t * 56, 114, "2t", CQ)
    f.cell(60 + t * 56 + 22, 114, "+1", CQ)
    f.text(60 + t * 56, 140, 44, 14, "lane t", fs=10, fc="#627d98", align="center")
# A fragment 示意
f.text(600, 92, 500, 18, "A fragment（PV 的 P 操作数，8bit×4/寄存器）：lane t 要列 {4t..4t+3}", fs=12, fc="#8a4b08")
for t in range(4):
    for j in range(4):
        f.cell(600 + t * 78, 114 + 0, "%d" % (4 * t + j), CA)
    f.text(600 + t * 78, 140, 66, 14, "需要 lane 2t、2t+1 的字节", fs=9.5, fc=RED, align="center")
f.line([(560, 114), (596, 114)], color=RED, width=2)
f.text(40, 160, 1072, 18, "结论：纯线程内置换无法拼出自然 A 布局（pycute 数值验证）——要么跨 lane 搬字节，要么改数学", fs=12.5, bold=True, fc=INK)

# ===== 左：方案 A 跨 lane 重排 =====
f.box(24, 248, 540, 236, "", fill=LRF, stroke=RED)
f.text(40, 254, 508, 22, "方案 A：ReorgC8bitToA8bit（跨 lane，历史默认）", fs=13.5, bold=True, fc=RED)
f.text(40, 280, 508, 20, "每线程每 128 列 tile：16× __shfl_sync + 32× __byte_perm", fs=13, fc=INK)
f.text(40, 304, 508, 20, "选择性 selector 按 lane%4 分叉（构造器状态 + 分支）", fs=13, fc=INK)
f.text(40, 328, 508, 20, "对 peer lane 的字节做 quad 内交换（peer map 2-bit 压缩）", fs=13, fc=INK)
f.text(40, 358, 508, 22, "代价：全部落在 QK→softmax→PV 关键路径上", fs=13, bold=True, fc=RED)
f.text(40, 384, 508, 20, "split-D 家族仍在使用（kBc 结构不同，未迁移）", fs=12.5, fc="#627d98")
f.text(40, 420, 508, 20, "V^T 存储无需置换（自然列序）", fs=12.5, fc="#627d98")

# ===== 右：方案 B 就地打包 + 列置换 =====
f.box(600, 248, 536, 236, "", fill=LGF, stroke=GREEN)
f.text(616, 254, 504, 22, "方案 B：PackC8bitToA8bitPermVT（reorg-free，默认）", fs=13.5, bold=True, fc="#1e5631")
f.text(616, 280, 504, 20, "数学：Σ_k P[m,k]V[k,n] 对任意双射 π 不变 —— P 不动，V 搬家", fs=13, fc=INK)
f.text(616, 304, 504, 20, "就地打包：4× __byte_perm（0x5410/0x7632/0xDC98/0xFEBA）", fs=13, fc=INK)
f.text(616, 328, 504, 20, "selector 与 lane 无关：无构造器状态、无分支", fs=13, fc=INK)
f.text(616, 358, 504, 22, "A 槽位 s 携带置换索引 π(s) = 8(r>>1)+2t+(r&amp;1)（32 列周期）", fs=13, bold=True, fc="#1e5631")
f.text(616, 384, 504, 20, "V^T 按 π⁻¹(j)=4*((j>>1)&amp;3)+2*((j>>3)&amp;1)+(j&amp;1) 预置换列写", fs=12.5, fc=INK)
f.text(616, 420, 504, 20, "置换不跨 32 列组 → V^T 写仍同一 32B sector（quantize +0.5%）", fs=12.5, fc="#627d98")

f.line([(564, 366), (600, 366)], color=INK, width=2.5)

# ===== 底部：配对契约 =====
f.box(24, 508, 1112, 96, "", fill=LPF, stroke=PURPLE)
f.text(44, 514, 1072, 20, "配对契约（CONTRACT）：pack 与 V^T 置换必须同开同关 —— 由 launcher 单一编译期常量 reorg_free 强制，两侧永不发散", fs=13, bold=True, fc=PURPLE)
f.text(44, 538, 1072, 20, "注释原话加感叹号：this pack must NEVER run against an unpermuted V^T —— 错配 = 静默错值（allclose 才能抓到）", fs=12.5, fc=INK)
f.text(44, 562, 1072, 20, "rowsum MMA（B 全 1）与 PV 在双射下同样精确；等效性已 pycute 数值验证（.tmp/fp8_persist_d/verify_layouts.py 同族）", fs=12.5, fc=INK)

f.text(24, 624, 1112, 20, "出处对照：SA2 的 per-warp token 重排思想（arXiv:2411.10958）；ffpa 的数学化重构 = 把跨 lane 搬运转嫁给前处理链的免费列置换", fs=12.5, fc="#48586a")
f.text(24, 650, 1112, 20, "拆分规则：K=32 组 = 4 个 n8 C-tile（w0..w3）→ a0={0,1,4,5} a1={2,3,6,7} a2={8,9,12,13} a3={10,11,14,15}（字节序号）", fs=12.5, fc="#48586a")
f.text(24, 692, 1112, 18, "对照 ffpa-attn(861d75e)：cute/fp8/reg2reg_8b.cuh L8-50（背景+跨 lane 版）· L96-141（perm pack）· quantize_fp8.cuh L36-53（VTPermInv32）", fs=12, fc="#627d98")

xml = mk(f.p, W, H, "fig-29-3")
ET.fromstring(xml)
out = sys.argv[1] if len(sys.argv) > 1 else "fig-29-3-reorg-free.drawio"
with open(out, "w") as fh:
    fh.write(xml)
print("written", out)
