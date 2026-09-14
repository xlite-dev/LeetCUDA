#!/usr/bin/env python3
# gen_batch_b2.py — 批次 B2：FIG-13-1 / 13-2 / 13-3 / 14-1 / 14-2
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

def fig_13_1():
    f = F()
    f.text(30, 14, 1050, 26, "WGMMA smem descriptor：64-bit 位域（enc(x) = (x 和 0x3FFFF) 右移 4，18-bit 窗口 16B 对齐）", fs=15.5, bold=True)
    # 位域条（按位宽比例）
    fields = [
        ("63-62\ntype\n2b = 1", 90, BLUE, "#f0f6fb"),
        ("62-52\nunused\n10b", 130, "#9aa5b1", "#ffffff"),
        ("51-49\nbase off\n3b = 0", 110, ORANGE, "#fde9d0"),
        ("48-46\nu\n3b", 70, "#9aa5b1", "#ffffff"),
        ("45-32\nSBO enc\n14b\n1024 右移 4", 190, GREEN, "#c9e4c8"),
        ("31-30\nu\n2b", 60, "#9aa5b1", "#ffffff"),
        ("30-16* LBO enc\n14b\n16 右移 4 = 1", 200, GREEN, "#c9e4c8"),
        ("15-14\nu\n2b", 60, "#9aa5b1", "#ffffff"),
        ("13-0\naddr enc\n14b\nsmem 右移 4", 190, BLUE, "#f0f6fb"),
    ]
    x = 40
    for name, w, c, fill in fields:
        f.box(x, 66, w - 4, 84, name.replace("\n", " "), fill=fill, stroke=c, fs=10.5, bold=True, fc=c)
        x += w
    f.text(40, 158, 700, 20, "type=1（矩阵描述符）；u = unused；数值列为本章 kernel BK=64 half + 128B swizzle 取值", fs=12.5, fc="#48586a")
    # 字段语义卡
    f.box(40, 196, 500, 34, "swizzle 模式：0=none  1=128B  2=64B  3=32B", fill=GRAY, stroke=BLUE, fs=12.5, fc="#243b53")
    f.box(560, 196, 500, 34, "phase = (addr 右移 7) mod 8（1024B 边界）", fill=GRAY, stroke=BLUE, fs=12.5, fc="#243b53")
    f.box(40, 238, 500, 34, "SBO：8 行 stripe 间隔（K-major swizzle 布局）", fill=GRAY, stroke=GREEN, fs=12.5, fc="#243b53")
    f.box(560, 238, 500, 34, "LBO：K 方向 stride；base addr：绝对 smem 偏移", fill=GRAY, stroke=GREEN, fs=12.5, fc="#243b53")
    f.text(30, 292, 1060, 40, "对照 common.cuh L532-565 注释与 PTX ISA 9.7.15.5.1.2.2 逐字段核校；enc 取 16B 单位使 14b 编码覆盖 256KB smem 空间（18-bit 窗口 右移 4）", fs=12.5, fc="#48586a")
    return mk(f.p, 1100, 350, "fig13-1")

def fig_13_2():
    f = F()
    f.text(30, 14, 1000, 26, "Warp specialization 数据流（kStages = 3）", fs=17, bold=True)
    # 三列
    f.box(50, 66, 220, 40, "HBM (global)", fill=GRAY, stroke="#9aa5b1", fs=14, bold=True)
    f.box(430, 66, 260, 40, "smem (per CTA)", fill=GRAY, stroke="#9aa5b1", fs=14, bold=True)
    f.box(860, 66, 220, 40, "C (global)", fill=GRAY, stroke="#9aa5b1", fs=14, bold=True)
    f.box(50, 116, 220, 30, "A tile 16KB", fill="#f0f6fb", stroke=BLUE, fs=12.5)
    f.box(50, 152, 220, 30, "B tile 16KB", fill="#f0f6fb", stroke=BLUE, fs=12.5)
    for s in range(3):
        f.box(430, 116 + s * 46, 260, 38, f"s{s}: A + B   full[{s}]", fill="#c9e4c8" if s == 0 else "#ffffff", stroke=GREEN, fs=12)
    f.box(860, 116, 220, 60, "C tile 128 x 128", fill="#fdf0e0", stroke=ORANGE, fs=12.5)
    f.box(860, 184, 220, 56, "d[2][8][4] = 64 x u32/thread", fill="#fdf0e0", stroke=ORANGE, fs=11.5)
    # 箭头
    f.line([(270, 131), (428, 131)], color=BLUE, width=3)
    f.text(280, 96, 140, 20, "TMA（1 线程）", fs=12, fc=BLUE, bold=True)
    f.text(280, 148, 140, 20, "cp.async.bulk.tensor.2d", fs=10.5, fc=BLUE)
    f.line([(690, 135), (858, 135)], color=GREEN, width=3)
    f.text(700, 100, 150, 20, "WGMMA（128 线程）", fs=12, fc=GREEN, bold=True)
    f.line([(970, 178), (970, 240), (790, 240)], color=ORANGE, width=2)
    # 双 WG 卡
    f.box(50, 268, 480, 180, "", fill="#f0f6fb", stroke=BLUE)
    f.box(50, 268, 480, 30, "WG0 producer（128 线程，1 活跃）", fill=GRAY, stroke=BLUE, fs=13, bold=True, fc=BLUE)
    f.text(64, 304, 460, 20, "P1  wait(empty[s])   等待 stage 被读空", fs=12, fc="#243b53")
    f.text(64, 328, 460, 20, "P2  TMA(A), TMA(B)   单线程提交两笔搬运", fs=12, fc="#243b53")
    f.text(64, 352, 460, 20, "P3  arrive + expect-tx(full[s], 32KB)", fs=12, fc="#243b53")
    f.text(64, 380, 460, 56, "相位翻转条件：129 次 arrive（128 consumer + 1 producer）且 32768B 落盘 —— 二者同时成立", fs=12, fc="#48586a")
    f.box(560, 268, 480, 180, "", fill="#f0faf5", stroke=GREEN)
    f.box(560, 268, 480, 30, "WG1 consumer（128 线程 warpgroup）", fill=GRAY, stroke=GREEN, fs=13, bold=True, fc=GREEN)
    f.text(574, 304, 460, 20, "C1  wait(full[s])", fs=12, fc="#243b53")
    f.text(574, 328, 460, 20, "C2  8 x wgmma  m64n128k16（异步）", fs=12, fc="#243b53")
    f.text(574, 352, 460, 20, "C3  commit + wait(0)", fs=12, fc="#243b53")
    f.text(574, 376, 460, 20, "C4  arrive(empty[s])  x128（读空信号）", fs=12, fc="#243b53")
    f.text(574, 400, 460, 36, "epilogue：累加器 d[][] 写回 C tile", fs=12, fc="#48586a")
    # 回环箭头 C4 -> P1
    f.line([(560, 392), (300, 392)], color=GREEN, width=2, dashed=1)
    f.text(330, 396, 220, 20, "C4 arrive 喂 P1 的 wait", fs=11, fc=GREEN)
    f.text(30, 462, 1060, 20, "kStages=3：计算与搬运最多重叠 2 个 tile；full[]/empty[] 双屏障阵列按 stage 解耦两支执行流", fs=12.5, fc="#48586a")
    return mk(f.p, 1100, 500, "fig13-2")

def fig_13_3():
    f = F()
    f.text(30, 14, 1000, 26, "mbarrier 状态机（单个 barrier 对象，如 full[s]）", fs=17, bold=True)
    # 状态框
    f.box(340, 60, 340, 50, "init(bar, 129)  ->  phase = p", fill=GRAY, stroke="#9aa5b1", fs=13.5, bold=True)
    f.box(340, 150, 340, 64, "pending = P, tx = T", fill="#f0f6fb", stroke=BLUE, fs=14, bold=True, fc=BLUE)
    f.line([(500, 110), (500, 148)], color="#627d98", width=2)
    # arrive / expect-tx 侧入口
    f.box(60, 156, 220, 34, "arrive（计数 1）：P 减 1", fill="#fde9d0", stroke=ORANGE, fs=12, fc="#8c2f0f")
    f.line([(280, 173), (338, 173)], color=ORANGE, width=2)
    f.box(60, 200, 220, 34, "expect-tx（dT）：T 加 dT", fill="#fde9d0", stroke=ORANGE, fs=12, fc="#8c2f0f")
    f.line([(280, 217), (338, 196)], color=ORANGE, width=2)
    # 判定
    f.box(760, 150, 300, 40, "P 大于 0 或 T 大于 0 ?", fill="#ffffff", stroke="#9aa5b1", fs=13)
    f.line([(680, 182), (758, 172)], color="#627d98", width=2)
    f.text(690, 140, 80, 20, "未完成", fs=12, fc="#627d98")
    f.line([(1060, 150), (1060, 120), (682, 120)], color="#9aa5b1", width=1.5, dashed=1)
    f.line([(682, 120), (682, 178)], color="#9aa5b1", width=1.5, dashed=1, arrow_end=1)
    # 翻转
    f.line([(760, 176), (760, 236), (500, 236), (500, 268)], color=GREEN, width=3)
    f.text(540, 238, 220, 20, "P == 0 且 T == 0（原子翻转）", fs=12.5, fc=GREEN, bold=True)
    f.box(340, 268, 340, 56, "phase = p + 1，pending 重置为 expected", fill="#c9e4c8", stroke=GREEN, fs=13, bold=True, fc="#1e5631")
    # 唤醒
    f.line([(340, 296), (150, 296), (150, 250)], color=GREEN, width=2, dashed=1)
    f.text(60, 302, 280, 20, "waiter 按 parity 唤醒", fs=12.5, fc=GREEN)
    # 本章条件
    f.box(60, 350, 480, 76, "", fill="#f7f3ec", stroke="#b45309")
    f.text(74, 356, 460, 22, "full[s] 翻转 = 128 consumer arrive + 1 producer arrive.expect-tx(32768B) + 32768B complete-tx", fs=11.5, fc="#7c5a1e")
    f.text(74, 384, 460, 36, "-> 翻转后 consumer 读数据（complete-tx 自动：TMA 字节落盘时 T 减 dC）", fs=11.5, fc="#7c5a1e")
    f.box(560, 350, 500, 76, "", fill="#f7f3ec", stroke="#b45309")
    f.text(574, 356, 480, 22, "empty[s] 翻转 = 128 consumer arrive（C4 / C0 warmup）+ 1 producer arrive（P1）", fs=11.5, fc="#7c5a1e")
    f.text(574, 384, 480, 36, "-> 翻转后 producer 覆写该 stage（C0 预热补 128 次 arrive）", fs=11.5, fc="#7c5a1e")
    return mk(f.p, 1100, 450, "fig13-3")

def fig_14_1():
    f = F()
    f.text(30, 14, 1050, 26, "TMA box：global 张量 -> smem 128B-swizzled tile", fs=17, bold=True)
    # 左：gmem 图
    f.text(50, 52, 500, 20, "A in gmem（row-major [M,K]，minor = K）", fs=13.5, bold=True, fc=BLUE)
    f.box(80, 80, 380, 230, "", fill="#ffffff", stroke="#9aa5b1")
    f.box(200, 140, 120, 120, "box 64x128 = 16KB", fill="#f0f6fb", stroke=BLUE, fs=12.5, bold=True, fc=BLUE)
    f.text(96, 92, 100, 18, "k0 ->", fs=11.5, fc="#627d98")
    f.text(96, 186, 100, 18, "m0", fs=11.5, fc="#627d98")
    f.text(50, 322, 460, 20, "boxDim = (BK=64, BM=128)，坐标 minor-first：A box (k*64, by*128)", fs=12, fc="#48586a")
    # 右：smem XOR 网格
    f.text(560, 52, 500, 20, "smem tile（BM x BK，chunk = 16B = 8 half）：chunk 异或 行低 3 位", fs=13.5, bold=True, fc=ORANGE)
    CW, CH = 44, 32
    GX, GY = 620, 84
    for c in range(8):
        f.box(GX + c * CW, GY - CH, CW, CH, str(c), fill=GRAY, stroke="#9aa5b1", fs=11.5, bold=True)
    f.text(GX - 60, GY - CH, 56, CH, "chunk", fs=11, fc="#627d98", align="right")
    for i in range(8):
        f.box(GX - 56, GY + i * CH, 52, CH, f"i={i}", fill=GRAY, stroke="#9aa5b1", fs=11, bold=True)
        for c in range(8):
            v = c ^ (i & 7)
            hl = (i in (1, 2)) and c < 4
            f.box(GX + c * CW, GY + i * CH, CW, CH, str(v), fill="#fde9d0" if hl else "#ffffff",
                  stroke=ORANGE if hl else "#bcccdc", fs=12, bold=hl, fc="#8c2f0f" if hl else "#1f2933")
    f.text(560, GY + 8 * CH + 10, 520, 20, "chunk 撇 = chunk 异或 (i mod 8)：i=0 恒等，i=1 成对交换，i=7 全反", fs=12, fc="#48586a")
    # 中间箭头
    f.line([(470, 195), (555, 195)], color=ORANGE, width=3)
    f.text(452, 160, 130, 20, "SWIZZLE\n128B", fs=12, fc=ORANGE, bold=True)
    # 底部结论
    f.box(50, 402, 1030, 40, "smem 基址 1024B 对齐 -> 硬件从相位 0 开始重排；consumer 用同一公式（chunk 异或 (i mod 8)）逆推零相位地址", fill="#c9e4c8", stroke=GREEN, fs=13, bold=True, fc="#1e5631")
    return mk(f.p, 1120, 460, "fig14-1")

def fig_14_2():
    f = F()
    f.text(30, 14, 1050, 26, "producer/consumer 时序（kStages = 2，单 CTA）", fs=17, bold=True)
    f.line([(80, 56), (1080, 56)], color="#9aa5b1", width=1.5, arrow_end=1)
    f.text(1010, 34, 70, 20, "time", fs=12.5, fc="#627d98")
    # producer 泳道
    f.text(30, 84, 120, 20, "WG0 producer", fs=13, bold=True, fc=BLUE)
    f.text(30, 104, 120, 36, "（1 线程活跃）", fs=11, fc="#627d98")
    prod = [("P1(s0) P2 P3", 140), ("P1(s1) P2 P3", 400), ("P1(s0) ...", 660)]
    for t, x in prod:
        f.box(x, 82, 230, 40, t, fill="#f0f6fb", stroke=BLUE, fs=12.5, fc=BLUE, bold=True)
    # consumer 泳道
    f.text(30, 164, 120, 20, "WG1 consumers", fs=13, bold=True, fc=GREEN)
    f.text(30, 184, 120, 36, "（128 线程 = 4 warp）", fs=11, fc="#627d98")
    f.box(140, 158, 80, 44, "C0", fill=GRAY, stroke="#9aa5b1", fs=13, bold=True)
    f.box(240, 158, 300, 44, "C1(s0) C2 mma x128", fill="#c9e4c8", stroke=GREEN, fs=12.5, fc=GREEN, bold=True)
    f.box(560, 158, 240, 44, "C1(s1) C2", fill="#c9e4c8", stroke=GREEN, fs=12.5, fc=GREEN, bold=True)
    f.box(820, 158, 220, 44, "C1(s0) ...", fill="#c9e4c8", stroke=GREEN, fs=12.5, fc=GREEN, bold=True)
    # 重叠标注
    f.text(250, 210, 280, 18, "在 s0 上计算（ldmatrix + mma）", fs=11.5, fc=GREEN)
    f.line([(400, 122), (400, 156)], color=ORANGE, width=2, dashed=1)
    f.line([(560, 156), (620, 122)], color=ORANGE, width=2, dashed=1)
    f.text(430, 132, 240, 18, "与 producer 灌 s1 重叠", fs=11.5, fc=ORANGE)
    # 右侧屏障注释栏
    f.box(760, 252, 330, 190, "", fill="#f7f3ec", stroke="#b45309")
    f.text(774, 258, 300, 20, "屏障翻转条件", fs=13, bold=True, fc="#b45309")
    f.text(774, 282, 300, 60, "full[s]：128 consumer arrive + 1 producer arrive + 32768B 落盘", fs=11.5, fc="#7c5a1e")
    f.text(774, 346, 300, 40, "empty[s]：128 consumer arrive（C4 或 C0 预热）+ 1 producer arrive", fs=11.5, fc="#7c5a1e")
    f.text(774, 390, 300, 44, "C0：启动时对 empty[0..S-1] 各预存 128 次 arrive，点燃流水线", fs=11.5, fc="#7c5a1e")
    # 等待标注
    f.line([(140, 122), (100, 122), (100, 250), (250, 250), (250, 226)], color="#9aa5b1", width=1.5, dashed=1)
    f.text(60, 258, 190, 20, "P1 等 empty[s0] 翻转", fs=11, fc="#627d98")
    # 底部结论
    f.box(50, 252, 660, 40, "overlap：TMA(s1) 在飞时 mma(s0) 在算 -> 搬运延迟被掩盖；稳态 producer 领先至多 kStages-1 个 tile", fill=GRAY, stroke=BLUE, fs=12.5, bold=True, fc="#102a43")
    return mk(f.p, 1120, 470, "fig14-2")

FIGS = {
    "fig-13-1-wgmma-desc": ("fig-13-1", fig_13_1),
    "fig-13-2-ws-dataflow": ("fig-13-2", fig_13_2),
    "fig-13-3-mbarrier": ("fig-13-3", fig_13_3),
    "fig-14-1-tma-box": ("fig-14-1", fig_14_1),
    "fig-14-2-pc-timeline": ("fig-14-2", fig_14_2),
}

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "figures/drawio"
    for d, (stem, fn) in FIGS.items():
        path = os.path.join(outdir, d)
        os.makedirs(path, exist_ok=True)
        xml = fn()
        ET.fromstring(xml)
        open(os.path.join(path, f"{stem}.drawio"), "w").write(xml)
        print(f"wrote {stem} OK cells={xml.count('<mxCell')}")
