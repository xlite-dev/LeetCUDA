#!/usr/bin/env python3
# fig-32-2-persist-d-loop — fp4 persist-D 主循环数据流：producer/consumer 与五步
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
f.text(24, 8, 1110, 30, "fp4 persist-D 主循环：128T producer + 256T consumer 与每 tile 五步", fs=17, bold=True)

# ===== 上：work loop 与 grid 契约 =====
f.box(24, 46, 1110, 66, "", fill="#fbfcfd", stroke=GRAY)
f.text(40, 52, 1070, 20, "work loop：for (work_id = blockIdx.x; work_id &lt; total_work; work_id += gridDim.x)   // 单 kernel 单路径，runtime grid 定风格", fs=12.5, bold=True, fc=INK)
f.text(40, 76, 520, 28, "dense：grid = min(total_work, SMs)，persistent，流水摊一次、跨 work 预取", fs=11.5, fc="#48586a")
f.text(600, 76, 520, 28, "causal：grid = total_work，block-per-work，短 work 早完早释放（多数 Tc_eff ≪ Tc）", fs=11.5, fc="#48586a")

# ===== 左：producer =====
y2 = 130
f.box(24, y2, 250, 300, "", fill="#f7faf7", stroke=GREEN)
f.text(40, y2 + 8, 220, 20, "producer（128T）", fs=14, bold=True, fc=GREEN)
f.box(40, y2 + 36, 218, 44, "TMA 装载&#10;K/SFK · Vᵀ/SFVt · ΔS · bias", fill=LGF, stroke=GREEN, fs=11.5, fc=GREEN)
f.box(40, y2 + 92, 218, 44, "Q/SFQ TMA（每 work）&#10;epilogue_done 后才发", fill="#ffffff", stroke=GREEN, fs=11.5, fc=INK)
f.box(40, y2 + 148, 218, 44, "等待 k_empty / v_empty&#10;（stage 释放信号）", fill="#ffffff", stroke=GREEN, fs=11.5, fc=INK)
f.box(40, y2 + 202, 218, 84, "反向 barrier 全集：&#10;k_empty[s] · v_empty[s]（256 arrivals）&#10;q_consumed · epilogue_done&#10;mbarrier 永不 re-init：&#10;全局 kv-tile 计数器驱动 phase", fill="#f7faf7", stroke="#d9e2ec", fs=10.5, fc="#48586a")

# ===== 中：smem stages =====
f.box(296, y2, 190, 300, "", fill="#f6f8fa", stroke=GRAY)
f.text(312, y2 + 8, 160, 20, "smem stages", fs=13, bold=True, fc=INK)
f.box(312, y2 + 36, 158, 52, "K + SFK&#10;kStages 级", fill=LBF, stroke=BLUE, fs=11.5, fc=BLUE)
f.box(312, y2 + 100, 158, 52, "Vᵀ + SFVt&#10;（slot 0 可别名 Q 区）", fill=LBF, stroke=BLUE, fs=11.5, fc=BLUE)
f.box(312, y2 + 164, 158, 52, "Q + SFQ&#10;（消费后释放）", fill=LPF, stroke=PURPLE, fs=11.5, fc=PURPLE)
f.box(312, y2 + 228, 158, 52, "O staging&#10;SW128（叠 Q/K 区）", fill=LOF, stroke=ORANGE, fs=11.5, fc="#8a4b08")
f.line([(274, y2 + 60), (296, y2 + 60)], color=GREEN, width=2)
f.line([(274, y2 + 150), (296, y2 + 126)], color=GREEN, width=2)

# ===== 右：consumer 五步 =====
f.box(508, y2, 626, 300, "", fill="#fbf9f4", stroke=ORANGE)
f.text(524, y2 + 8, 590, 20, "consumer（256T = 8 warp）：每 kv tile 五步", fs=14, bold=True, fc="#8a4b08")
steps = [
    ("① add_delta_s", "rank-1 项预载 C 累加器（写 tSrS 寄存器）", LPF, PURPLE),
    ("② gemm_ss_fp4", "S += (Q̂·s_Q)(K̂·s_K)：A 寄存器驻留 / B 流入 / zip 对", LBF, BLUE),
    ("③ masking", "kv-tail + causal：pos = kv_tile·Bc + π(j)（perm-aware）；bias 注入同过 π", LRF, RED),
    ("④ softmax_with_quant", "P₂ 域融合：组 absmax + 行 max 同链；出寄存器即 P̂₂/s_P2", LGF, GREEN),
    ("⑤ gemm_rs_fp4", "O += (P̂₂·s_P2)(V̂·s_V)：P 现场打包不过 smem；v_full 在④后才等", LOF, ORANGE),
]
sy = y2 + 34
for title, desc, fill, stroke in steps:
    f.box(524, sy, 590, 44, "", fill=fill, stroke=stroke)
    f.text(532, sy + 2, 170, 20, title, fs=12, bold=True, fc=stroke)
    f.text(700, sy + 2, 410, 40, desc, fs=10.5, fc=INK)
    if sy < y2 + 200:
        f.line([(819, sy + 44), (819, sy + 50)], color=INK, width=1.5, arrow_end=1)
    sy += 50
f.text(524, y2 + 286, 590, 12, "kv_tile > 0 时 ⑤ 前插 rescale_acc（per-row 守卫）", fs=10.5, fc="#48586a")

f.line([(486, y2 + 62), (508, y2 + 62)], color=BLUE, width=2)
f.line([(486, y2 + 190), (508, y2 + 234)], color=BLUE, width=2)
f.line([(508, y2 + 100), (486, y2 + 130)], color=ORANGE, width=2, dashed=1)

# ===== 下：Q 驻留与 epilogue =====
y3 = 456
f.box(24, y3, 540, 190, "", fill="#fbfcfd", stroke=GRAY)
f.text(40, y3 + 8, 510, 20, "Q 寄存器驻留（per-work 常量）", fs=13, bold=True, fc=INK)
f.text(40, y3 + 32, 510, 40, "Q/SFQ s2r 一次进寄存器，跨整个 kv 循环；&#10;gemm_ss_fp4 当 A 侧寄存器操作数消费（恒开）", fs=11.5, fc="#48586a")
f.text(40, y3 + 80, 510, 20, "lse dot 提前（kQSmemReuse 下 Q 区会被 V 复用）", fs=11.5, fc="#48586a")
f.box(40, y3 + 106, 510, 74, "历史 bug：不驻留时 A/SFA asm 操作数未初始化，&#10;cicc 折成 0 → QK 退化为 ΔS（rank-1 均值 attention），&#10;probe 容差曾掩盖——要用「输出对 K 敏感性」探针", fill="#fdf6f6", stroke="#f3c1c1", fs=11, fc=RED)

f.box(592, y3, 548, 190, "", fill="#fbfcfd", stroke=GRAY)
f.text(608, y3 + 8, 520, 20, "epilogue（kv 循环后）", fs=13, bold=True, fc=INK)
f.box(608, y3 + 32, 160, 40, "finalize&#10;÷ row_sum（0 行守卫）", fill=LGF, stroke=GREEN, fs=11, fc=GREEN)
f.box(790, y3 + 32, 160, 40, "+ v_m 加回&#10;（smooth-V 恒等）", fill=LPF, stroke=PURPLE, fs=11, fc=PURPLE)
f.box(972, y3 + 32, 152, 40, "lse 写出&#10;（式 32-lse）", fill=LBF, stroke=BLUE, fs=11, fc=BLUE)
f.box(608, y3 + 82, 516, 40, "named barrier（WAR）→ SM90_U32x2_STSM_N 进 SW128&#10;（叠 Q/K 释放区；blockscale C-fragment 用 U32x2 非 U32x4）", fill="#fbfcfd", stroke="#d9e2ec", fs=11, fc="#48586a")
f.box(608, y3 + 128, 516, 40, "→ 单条 TMA store（BHND / NHD 双布局，运行时选坐标）&#10;tail Q tile：R→G 带行守卫", fill="#fbfcfd", stroke="#d9e2ec", fs=11, fc="#48586a")

f.text(24, y4 := 668, 1110, 20, "五步环环相扣：③ 的 π 与 ② 的 C 布局、⑤ 的 A 布局三方锁定（SA3 adapter 逐字拷贝的自洽体系）。", fs=12.5, bold=True, fc=RED)
f.text(24, 692, 1110, 20, "吞吐参考：D=192 self 474T（fp4 基线）；MXFP8 PV 变体 357T（精度保险丝，D≤192）。", fs=12, fc="#48586a")

print(mk(f.p, W, H, "fig32-2"))
