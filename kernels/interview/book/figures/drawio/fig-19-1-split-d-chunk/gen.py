#!/usr/bin/env python3
# gen_batch_c2.py — 批次 C2：FIG-17-2 / 18-1 / 19-1 / 19-2
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

def fig_17_2():
    f = F()
    f.text(24, 12, 1012, 28, "mbarrier 阵列拓扑：谁 arrive / 谁 wait（init count = 257）", fs=16, bold=True)
    f.text(24, 48, 240, 24, "Q：一对一单屏障", fs=13.5, bold=True, fc=BLUE)
    f.box(300, 44, 240, 44, "full_Q", fill=LBF, stroke=BLUE, fs=15, bold=True, fc=BLUE)
    f.text(24, 96, 500, 20, "producer：arrive_tx(kQTileBytes = 16KB, D=64)", fs=12.5, fc="#48586a")
    f.line([(280, 66), (300, 66)], color=BLUE, width=2)
    f.text(300, 92, 500, 20, "consumer：256x arrive + wait（一次性，无 empty_Q）", fs=12.5, fc="#48586a")
    f.line([(420, 90), (420, 68)], color=GREEN, width=2)
    f.text(580, 40, 460, 20, "D=128：kTmaChunks 次 load 共享同一 full 屏障", fs=12.5, fc="#48586a")
    f.text(580, 60, 460, 20, "多次 complete-tx 记账，一次 expect_tx 声明总量", fs=12.5, fc="#48586a")
    f.box(24, 150, 1012, 32, "stage s in [0, Sk)：K 屏障对（V 同形，Sv 份；V 在 PV 后释放）", fill="#d9e2ec", stroke="#486581", fs=14, bold=True)
    f.box(60, 200, 210, 48, "full_K[s]", fill=LBF, stroke=BLUE, fs=15, bold=True, fc=BLUE)
    f.box(60, 320, 210, 48, "empty_K[s]", fill=GRAY, stroke=GREEN, fs=15, bold=True, fc="#1e5631")
    f.text(300, 196, 700, 22, "producer：arrive_tx(8/16KB)", fs=12.5, fc="#48586a")
    f.line([(296, 207), (274, 207)], color=BLUE, width=2)
    f.text(300, 226, 700, 22, "consumer：256x arrive（随后 wait —— QK^T 前阻塞在相位）", fs=12.5, fc="#48586a")
    f.line([(296, 237), (274, 237)], color=GREEN, width=2)
    f.text(300, 316, 700, 22, "consumer（QK^T 用完 K 即刻）：256x arrive(free)", fs=12.5, fc="#48586a")
    f.line([(296, 327), (274, 327)], color=GREEN, width=2)
    f.text(300, 346, 700, 22, "producer 写覆盖前：wait（等自己那一份 arrive 凑齐）", fs=12.5, fc="#48586a")
    f.line([(296, 357), (274, 357)], color=BLUE, width=2)
    f.text(24, 400, 1012, 24, "per-phase 记账（每轮预算恰好 257/round，相位循环永不错位）：", fs=13.5, bold=True, fc="#102a43")
    f.box(24, 428, 496, 48, "full[s] 每轮 = 256（consumer arrive）+ 1（producer arrive_tx）+ tx 落地", fill=LGF, stroke=GREEN, fs=12.5, fc="#1e5631")
    f.box(540, 428, 496, 48, "empty[s] 每轮 = 256（consumer free）+ 1（producer 写前 arrive）", fill=LGF, stroke=GREEN, fs=12.5, fc="#1e5631")
    f.box(24, 488, 1012, 42, "D = 128 时同一 tile 的多次 TMA 共享一个 full 屏障：多次 complete-tx 记账、一次 expect_tx 声明总量", fill=LOF, stroke=ORANGE, fs=13.5, bold=True, fc="#7c3a06")
    return mk(f.p, 1060, 546, "fig-17-2")

def fig_18_1():
    f = F()
    f.text(24, 12, 1112, 28, "FA3 双 consumer 角色图（384 线程，Br = Bc = 64）", fs=16, bold=True)
    f.box(24, 48, 1112, 32, "WG0 [0,127] producer：单线程提交 TMA", fill=LBF, stroke=BLUE, fs=14, bold=True, fc=BLUE)
    f.box(40, 88, 340, 44, "P0：TMA Q -&gt; full_Q（257 = 2 WG x 128 + 1）", fill="#ffffff", stroke=BLUE, fs=12.5)
    f.box(396, 88, 724, 44, "P2：for tile：cid = tile &amp; 1；K[tile]（Sk=1）/ K[tile+2]（Sk&gt;=2）-&gt; K_smem[cid][stg]；V[tile] -&gt; V_smem[cid]（wait empty -&gt; full）", fill="#ffffff", stroke=BLUE, fs=12)
    f.box(24, 152, 542, 32, "WG1 [128,255] consumer_id = 0：tiles 0,2,4,...（ceil Tc/2）", fill=LBF, stroke=BLUE, fs=13.5, bold=True, fc=BLUE)
    f.box(594, 152, 542, 32, "WG2 [256,383] consumer_id = 1：tiles 1,3,5,...（floor Tc/2）", fill=LGF, stroke=GREEN, fs=13.5, bold=True, fc=GREEN)
    steps = [
        "C0：128x arrive empty_K[cid][s] + empty_V[cid]",
        "wait full_Q（两 WG 共享同一屏障，257）",
        "loop it：wait full_K[cid][it % Sk] -&gt; QK^T（ldmatrix+HMMA）- arrive empty_K[cid][..]",
        "softmax（m, l；纯寄存器）",
        "wait full_V[cid] -&gt; PV（R_S 复用 + X2_T）- arrive empty_V[cid] -&gt; rescale R_D",
    ]
    y = 194
    for t in steps:
        f.box(24, y, 542, 36, t, fill="#ffffff", stroke=BLUE, fs=11.5)
        f.box(594, y, 542, 36, t.replace("cid", "cid"), fill="#ffffff", stroke=GREEN, fs=11.5)
        y += 42
    f.box(24, 414, 1112, 34, "收尾：__syncthreads（全部 TMA 落地，K/V smem 生命周期结束）", fill="#d9e2ec", stroke="#486581", fs=13, bold=True)
    f.box(24, 460, 542, 48, "WG2：(R_D, m1, l1) -&gt; smem scratch（按 wg_tid 索引）后 __syncthreads", fill="#ffffff", stroke=GREEN, fs=12)
    f.line([(570, 484), (594, 484)], color=GREEN, width=2)
    f.box(594, 460, 542, 48, "WG1：读 scratch 归并后写出 O -&gt; gmem", fill="#ffffff", stroke=BLUE, fs=12)
    f.box(24, 524, 1112, 64, "m = max(m0, m1)；a = e^(m0-m)，b = e^(m1-m)；l = a*l0 + b*l1；Oacc = a*Oacc0 + b*Oacc1；O = Oacc / l", fill=LGF, stroke=GREEN, fs=13.5, bold=True, fc="#1e5631")
    return mk(f.p, 1160, 606, "fig-18-1")

def fig_19_1():
    f = F()
    f.text(24, 12, 1072, 28, "Split-D：D 维切分（D = 512 -&gt; C = 8 个 64 宽 chunk）", fs=16, bold=True)
    rows = [("Q [Br=64, D=512]", BLUE, LBF), ("K [Bc=64, D=512]", GREEN, LGF), ("V [Bc=64, D=512]", ORANGE, LOF)]
    y = 48
    for name, c, fill in rows:
        f.text(24, y + 8, 170, 24, name, fs=13, bold=True, fc=c)
        for i in range(8):
            f.box(200 + i * 112, y, 106, 36, f"c{i}", fill=fill, stroke=c, fs=12.5, bold=True, fc="#243b53")
        y += 44
    f.text(24, 180, 1072, 20, "chunk c = 64 个 f16 宽 = 恰好 1 个 SW128 swizzle atom", fs=12.5, fc="#48586a")
    f.box(24, 212, 1072, 46, "Phase 1  QK^T：S[64,64] += Q(c) @ K(c)^T，c = 0..7 —— 点积内部精确求和，无权重（共享同一个 S acc）", fill=LBF, stroke=BLUE, fs=13, bold=True, fc=BLUE)
    f.box(24, 266, 1072, 46, "Phase 2  softmax(S)：m，l，p = exp2(S*scale - m) —— 与 D 无关", fill=LGF, stroke=GREEN, fs=13, bold=True, fc=GREEN)
    f.box(24, 320, 1072, 46, "Phase 3  PV：O_c[64,64] *= rho；O_c += P @ V(c)，c = 0..7 —— C 个独立 O acc，同一 rho 广播到全部", fill=LOF, stroke=ORANGE, fs=13, bold=True, fc=ORANGE)
    f.box(24, 374, 1072, 42, "Epi：O_c /= l（c = 0..7）-&gt; 以 (q_tile, c) 为坐标 store [64,64] tile", fill=GRAY, stroke="#486581", fs=13, fc="#243b53")
    f.text(24, 428, 1072, 24, "流水槽位（TMA 版，Sk = Sv = 2）：", fs=13.5, bold=True, fc="#102a43")
    f.box(24, 456, 700, 46, "idx = kv_tile * C + c；stage = idx % Sk；phase = (idx / Sk) &amp; 1", fill="#ffffff", stroke="#486581", fs=14, bold=True)
    f.box(748, 456, 348, 46, "D=128（C=2）：phase 周期 1 个 kv_tile", fill="#ffffff", stroke=BLUE, fs=12)
    f.box(748, 510, 348, 46, "D=192（C=3）：周期 2（gcd(C,Sk)=1）", fill="#ffffff", stroke=GREEN, fs=12)
    return mk(f.p, 1120, 572, "fig-19-1")

def fig_19_2():
    f = F()
    f.text(24, 12, 1072, 28, "跨 chunk 两阶段合并（C = 3，D = 192 示例）", fs=16, bold=True)
    f.box(24, 48, 1072, 32, "stage 1：单个点积内部的精确归约（QK^T，无权重）", fill=LBF, stroke=BLUE, fs=14, bold=True, fc=BLUE)
    f.box(40, 92, 300, 40, "S^(0) = Q0 K0^T", fill="#ffffff", stroke=BLUE, fs=13)
    f.box(386, 92, 300, 40, "S^(1) = Q1 K1^T", fill="#ffffff", stroke=BLUE, fs=13)
    f.box(732, 92, 300, 40, "S^(2) = Q2 K2^T", fill="#ffffff", stroke=BLUE, fs=13)
    f.line([(190, 134), (420, 168)], color=BLUE, width=2)
    f.line([(536, 134), (520, 168)], color=BLUE, width=2)
    f.line([(882, 134), (620, 168)], color=BLUE, width=2)
    f.box(320, 170, 480, 40, "acc += （mma.sync 链）", fill=GRAY, stroke=BLUE, fs=13, fc="#243b53")
    f.box(320, 222, 480, 44, "S[64,64] 完整 score tile（D-free）", fill=LBF, stroke=BLUE, fs=14, bold=True, fc=BLUE)
    f.box(24, 290, 1072, 32, "stage 2：跨 KV tile 的 online softmax —— rho 必须广播到全部 C 个 acc", fill=LGF, stroke=GREEN, fs=14, bold=True, fc=GREEN)
    f.box(24, 336, 260, 60, "rho_t = exp(m_{t-1} - m_t)", fill=LGF, stroke=GREEN, fs=14, bold=True, fc="#1e5631")
    cols = [
        "O_0 *= rho_t；O_0 += exp(S-m_t) V_0（列 0..63）",
        "O_1 *= rho_t；O_1 += exp(S-m_t) V_1（列 64..127）",
        "O_2 *= rho_t；O_2 += exp(S-m_t) V_2（列 128..191）",
    ]
    x = 316
    for t in cols:
        f.box(x, 336, 252, 60, t, fill=LOF, stroke=ORANGE, fs=11.5)
        f.line([(286, 366), (x, 366)], color=GREEN, width=2, dashed=1)
        x += 264
    f.box(24, 412, 740, 42, "l_t = rho_t * l_{t-1} + rowsum(exp(S - m_t)) —— 每行全程只有一份 l", fill="#ffffff", stroke=GREEN, fs=13)
    f.box(24, 466, 1072, 42, "最后一个 tile 后：O_c /= l_T（全部 c）-&gt; concat = 完整 softmax(QK^T/sqrt(D)) V", fill=GRAY, stroke="#486581", fs=13, fc="#243b53")
    f.box(24, 524, 526, 56, "对比 split-KV（ch18）：需要第二次 max —— a = exp(m0-m)，b = exp(m1-m) 两套归并权重", fill="#fbe3e3", stroke=RED, fs=12, fc="#7f1d1d")
    f.box(570, 524, 526, 56, "对比 split-D：无第二次 max —— rho 构造上被全部 chunk 共享（定理 eq:19-invariant 的不变量）", fill=LGF, stroke=GREEN, fs=12, bold=True, fc="#1e5631")
    return mk(f.p, 1120, 596, "fig-19-2")

FIGS = {
    "fig-17-2": (fig_17_2, "fig-17-2-mbarrier-topo"),
    "fig-18-1": (fig_18_1, "fig-18-1-fa3-dual-consumer"),
    "fig-19-1": (fig_19_1, "fig-19-1-split-d-chunk"),
    "fig-19-2": (fig_19_2, "fig-19-2-two-stage-merge"),
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
