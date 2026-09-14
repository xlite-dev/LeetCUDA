#!/usr/bin/env python3
# gen_batch_c1.py — 批次 C1：FIG-15-1 / 15-2 / 16-1 / 17-1
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
LBF, LGF, LOF = "#f0f6fb", "#c9e4c8", "#fde9d0"

def fig_15_1():
    f = F()
    f.text(24, 12, 1040, 28, "FlashAttention online softmax：单个 Q 块（Br 行）流式扫过全部 Tc 个 KV 块", fs=16, bold=True)
    f.box(24, 50, 250, 46, "m [Br] 运行行最大", fill=LBF, stroke=BLUE, fs=13, bold=True, fc=BLUE)
    f.box(286, 50, 180, 46, "l [Br] 运行行和", fill=LGF, stroke=GREEN, fs=13, bold=True, fc=GREEN)
    f.box(478, 50, 260, 46, "O [Br,d] 未归一化累加器", fill=LOF, stroke=ORANGE, fs=13, bold=True, fc=ORANGE)
    f.box(750, 50, 310, 46, "Q tile [Br,d]：Step1 一次性进 smem，外循环不再读 HBM", fill=GRAY, stroke="#486581", fs=12)
    f.text(24, 98, 500, 20, "三状态全程驻留寄存器，跨整个 KV 循环存活", fs=12.5, fc="#48586a")
    f.box(24, 126, 1036, 34, "for j = 1 .. Tc   （外层循环流式扫 KV，每块一遍）", fill="#d9e2ec", stroke="#486581", fs=14, bold=True)
    steps = [
        ("K^[j] [Bc,d]  --(cp.async / TMA)--&gt;  smem", "(3a) load", BLUE),
        ("S^[j] = Q K^[j]^T / sqrt(d)     [Br x Bc]", "(3b) QK^T -&gt; R_S", BLUE),
        ("m_new = max(m, rowmax(S^[j]))", "(3c) online", GREEN),
        ("P^[j] = exp(S^[j] - m_new)     [Br x Bc]", "softmax：P 原地写回 R_S", GREEN),
        ("l_new = exp(m - m_new) * l + rowsum(P^[j])", "", GREEN),
        ("PV = P^[j] V^[j]     [Br x d]", "(3d) PV", ORANGE),
        ("O &lt;- diag(exp(m - m_new)) * O + PV", "(3e) rescale", ORANGE),
    ]
    y = 170
    for txt, note, c in steps:
        f.box(40, y, 600, 42, txt, fill="#ffffff", stroke=c, fs=13, fc="#1f2933")
        if note:
            f.text(660, y, 400, 42, note, fs=12.5, fc=c, bold=True)
        y += 52
    f.line([(340, 172), (340, 526)], color="#9aa5b1", width=1.5, arrow_end=0)
    f.box(24, 548, 1036, 42, "循环结束后：O &lt;- O / l    （全程只做一次最终除法）(8) epilogue", fill=LGF, stroke=GREEN, fs=14, bold=True, fc="#1e5631")
    f.box(24, 598, 1036, 42, "N x N 矩阵 S 从不存在：任意时刻只有 [Br x Bc] tile 活在 R_S 寄存器里", fill=LOF, stroke=ORANGE, fs=14, bold=True, fc="#7c3a06")
    return mk(f.p, 1084, 656, "fig-15-1")

def fig_15_2():
    f = F()
    f.text(24, 12, 1072, 28, "FA1 -&gt; FA2 -&gt; FA3：时间去哪了（时间结构示意）", fs=16, bold=True)
    f.box(24, 48, 1072, 32, "FA1 (A100)：warp 沿 K/V 切分，softmax 串行夹在两次 GEMM 之间", fill=LBF, stroke=BLUE, fs=14, bold=True, fc=BLUE)
    labels = ["GEMM1", "softmax", "GEMM2", "GEMM1", "softmax", "GEMM2"]
    x = 24
    for i, lb in enumerate(labels):
        c = RED if lb == "softmax" else BLUE
        fill = "#fbe3e3" if lb == "softmax" else LBF
        f.box(x, 88, 165, 36, lb, fill=fill, stroke=c, fs=12.5, bold=True, fc=c)
        x += 170
    f.text(24, 128, 1072, 20, "... 串行重复；partial (m, l, O) per warp，跨 warp 归约必须走 smem（merge_attn_states）", fs=12.5, fc="#48586a")
    f.box(24, 162, 1072, 32, "FA2 (A100/H100)：warp 沿 Q 行切分，消灭跨 warp 归约", fill=LGF, stroke=GREEN, fs=14, bold=True, fc=GREEN)
    for w in range(4):
        yw = 202 + w * 34
        f.text(24, yw, 80, 28, f"warp{w}", fs=12, fc="#243b53", bold=True)
        xx = 108
        for lb in ["GEMM1", "softmax", "GEMM2"]:
            c = RED if lb == "softmax" else GREEN
            fill = "#fbe3e3" if lb == "softmax" else LGF
            f.box(xx, yw, 150, 28, lb, fill=fill, stroke=c, fs=11.5, bold=True, fc=c)
            xx += 155
    f.text(620, 208, 476, 24, "每 warp 独占 16 行 Q：私有 (m, l, O)，K/V 全 warp 共享", fs=12.5, fc="#48586a")
    f.text(620, 236, 476, 24, "另加 CTA 级并行：grid.x = N/Br，沿 seqlen(Q) 喂饱 SM", fs=12.5, fc="#48586a")
    f.text(620, 264, 476, 24, "O 保持未归一化 -&gt; 尾部一次 diag(1/l)", fs=12.5, fc="#48586a")
    f.box(24, 346, 1072, 32, "FA3 (H100)：warp specialization + pingpong，softmax 与 GEMM 在不同运算单元上重叠", fill=LOF, stroke=ORANGE, fs=14, bold=True, fc=ORANGE)
    f.box(24, 386, 300, 30, "producer warp：TMA 异步提交（单线程）", fill=GRAY, stroke="#486581", fs=12)
    f.text(24, 422, 100, 22, "WG1 tensor", fs=11.5, fc=ORANGE, bold=True)
    for i in range(3):
        f.box(108 + i * 190, 420, 175, 26, "GEMM", fill=LOF, stroke=ORANGE, fs=11.5, bold=True, fc=ORANGE)
    f.text(24, 456, 100, 22, "WG2 CUDA", fs=11.5, fc=RED, bold=True)
    for i in range(3):
        f.box(198 + i * 190, 454, 175, 26, "softmax", fill="#fbe3e3", stroke=RED, fs=11.5, bold=True, fc=RED)
    f.text(24, 490, 1072, 20, "pingpong：WG1 跑 GEMM 时 WG2 跑 softmax（反之亦然）；再加 FP8 GEMM（block scaling + incoherent processing）", fs=12.5, fc="#48586a")
    return mk(f.p, 1120, 522, "fig-15-2")

def fig_16_1():
    f = F()
    f.text(24, 12, 776, 28, "Split-Q warp 划分（Br=128, Bc=64, D=64, 8 warps, kStagesK=2）", fs=15, bold=True)
    f.text(24, 40, 776, 20, "（升级自 LeetCUDA README 的 4-warp ASCII 底稿）", fs=12, fc="#48586a")
    f.text(24, 68, 420, 24, "Q tile [Br=128, D]（blockIdx.x）", fs=13, bold=True, fc=BLUE)
    for w in range(8):
        yw = 96 + w * 33
        f.box(24, yw, 420, 29, f"warp_QP {w}   rows {w*16} - {w*16+15}", fill=LBF if w % 2 == 0 else "#ffffff", stroke=BLUE, fs=12, fc="#243b53")
    f.text(24, 362, 420, 20, "Q 仅 Step1 加载一次，整个外循环驻留 smem", fs=12, fc="#48586a")
    f.text(500, 68, 300, 24, "K/V tiles [Bc=64, D] x Tc", fs=13, bold=True, fc=GREEN)
    f.box(500, 96, 300, 76, "K：2-stage 环形缓冲（t, t+1）", fill=LGF, stroke=GREEN, fs=12.5, fc="#1e5631")
    f.box(500, 180, 300, 76, "V：1-stage，在 3b+3c 期间在途", fill=LGF, stroke=GREEN, fs=12.5, fc="#1e5631")
    f.text(500, 264, 300, 40, "全部 8 warp 读同一 K_t / V_t（warp_KV = 0，smem 广播）", fs=12, fc="#48586a")
    f.line([(498, 134), (450, 218)], color=GREEN, width=2)
    f.text(24, 314, 776, 20, "per-warp 私有状态（纯寄存器，零跨 warp 通信）：", fs=13, bold=True, fc="#102a43")
    f.box(24, 338, 380, 40, "m, l in R^16   lane_block_row_max / sum_old (fp32)", fill=LBF, stroke=BLUE, fs=12, fc="#243b53")
    f.box(420, 338, 380, 40, "O in R^16xD   R_D 累加器 fragment（f16/f32 acc）", fill=LOF, stroke=ORANGE, fs=12, fc="#243b53")
    f.text(24, 388, 776, 22, "per KV tile t，每个 warp 执行三段（全部落在寄存器 fragment）：", fs=13, bold=True, fc="#102a43")
    f.box(24, 414, 776, 38, "S_w = Q_w @ K_t^T  -&gt; R_S     （8x m16n8，32 条 HMMA）", fill="#ffffff", stroke=BLUE, fs=12.5)
    f.box(24, 458, 776, 38, "P_w = exp(S_w*s - m) -&gt; R_S 原地覆盖   （softmax pass 2）", fill="#ffffff", stroke=GREEN, fs=12.5)
    f.box(24, 502, 776, 38, "O_w += P_w @ V_t  -&gt; R_O -&gt; R_D   （A 操作数 = R_S 直接复用）", fill="#ffffff", stroke=ORANGE, fs=12.5)
    f.box(24, 558, 380, 66, "Split-KV（FA1 式）：warp 各持 16 列，行状态撕裂到多 warp，每 tile 走 smem 归并（merge_attn_states）", fill="#fbe3e3", stroke=RED, fs=11.5, fc="#7f1d1d")
    f.box(420, 558, 380, 66, "Split-Q：一个 warp 端到端独占 16 行 Q —— 零归并、零通信", fill=LGF, stroke=GREEN, fs=12, bold=True, fc="#1e5631")
    return mk(f.p, 824, 640, "fig-16-1")

def fig_17_1():
    f = F()
    f.text(24, 12, 1132, 28, "producer/consumer 双流水时序（Sk=2, Sv=1, D=64）", fs=16, bold=True)
    f.box(24, 50, 540, 32, "producer WG0（1 个活跃线程，TMA 提交）", fill=LBF, stroke=BLUE, fs=14, bold=True, fc=BLUE)
    f.box(616, 50, 540, 32, "consumer WG1（8 warps, 256 线程）", fill=LGF, stroke=GREEN, fs=14, bold=True, fc=GREEN)
    f.box(24, 92, 540, 38, "P0：TMA Q --expect_tx 16KB--&gt; full_Q", fill="#ffffff", stroke=BLUE, fs=12.5)
    f.box(24, 138, 540, 38, "P1：TMA K[0]-&gt;st0；TMA K[1]-&gt;st1（预取 Sk-1=1 个 tile）", fill="#ffffff", stroke=BLUE, fs=12.5)
    f.box(24, 184, 540, 60, "P2（每轮 k）：TMA V[k]-&gt;v0，随后 TMA K[k+2]-&gt;st（覆盖已被 consumer 释放的空槽）", fill="#ffffff", stroke=BLUE, fs=12.5)
    f.box(24, 254, 540, 44, "producer 领先 consumer 至多 Sk-1 个 K tile / Sv 个 V tile", fill=GRAY, stroke=BLUE, fs=12, fc="#243b53")
    f.box(616, 92, 540, 38, "C0：256x arrive empty_K[0..1]；256x arrive empty_V[0]", fill="#ffffff", stroke=GREEN, fs=12.5)
    f.box(616, 138, 540, 34, "wait full_Q（一次，Q 就位）", fill="#ffffff", stroke=GREEN, fs=12.5)
    f.text(616, 178, 540, 22, "---- iter k = 0 ----", fs=12.5, bold=True, fc="#1e5631")
    rows = [
        ("wait full_K[0]：QK 用 K[0]", "#ffffff", GREEN),
        ("3b QK^T：ldmatrix + HMMA ——&gt; arrive empty_K[0]（用完即提前释放）", "#ffffff", GREEN),
        ("3c softmax（纯寄存器，不碰 smem）", "#ffffff", GREEN),
        ("wait full_V[0]（恰在 3d 前）—— 3d PV：R_S 复用 + X2_T(V) —— arrive empty_V[0]", "#ffffff", GREEN),
        ("3e rescale（m, l, O）", "#ffffff", GREEN),
    ]
    y = 202
    for t, fill, c in rows:
        f.box(626, y, 530, 34, t, fill=fill, stroke=c, fs=11.5)
        y += 40
    f.box(616, 406, 540, 38, "iter k = 1：同构（wait full_K[1] -&gt; QK^T -&gt; softmax -&gt; PV -&gt; rescale）", fill=GRAY, stroke=GREEN, fs=11.5, fc="#243b53")
    f.line([(564, 214), (616, 214)], color=ORANGE, width=2, dashed=1)
    f.line([(564, 300), (616, 300)], color=ORANGE, width=2, dashed=1)
    f.box(24, 462, 1132, 44, "overlap：计算 tile k 与搬运 tile k+1 完全重叠 —— 搬运延迟被计算掩盖", fill=LOF, stroke=ORANGE, fs=14, bold=True, fc="#7c3a06")
    f.box(24, 514, 1132, 42, "K[2] 装载窗口 = QK[0] + softmax[0] + PV[0]（empty_K[0] 在 QK[0] 结束即释放）", fill="#ffffff", stroke=BLUE, fs=13)
    f.box(24, 562, 1132, 42, "V[1] 装载窗口 = QK[0] + softmax[0]（empty_V[0] 在 PV[0] 结束才释放，Sv=1 窗口更窄）", fill="#ffffff", stroke=GREEN, fs=13)
    return mk(f.p, 1180, 616, "fig-17-1")

FIGS = {
    "fig-15-1": (fig_15_1, "fig-15-1-online-softmax"),
    "fig-15-2": (fig_15_2, "fig-15-2-fa1-fa2-fa3"),
    "fig-16-1": (fig_16_1, "fig-16-1-split-q-warp"),
    "fig-17-1": (fig_17_1, "fig-17-1-dual-pipeline"),
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
