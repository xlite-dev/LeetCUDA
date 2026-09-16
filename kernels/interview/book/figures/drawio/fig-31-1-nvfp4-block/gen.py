#!/usr/bin/env python3
# fig-31-1-nvfp4-block — NVFP4 1×16 microscaling 块：e2m1 数据 + ue4m3 块尺度
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

W, H = 1150, 780
f = F()
f.text(24, 8, 1100, 30, "NVFP4 的 1×16 microscaling 块：e2m1 数据 + ue4m3 块尺度", fs=17, bold=True)

# ===== 左上：16 元素块 + amax → s =====
f.text(24, 46, 540, 20, "① 一个 1×16 量化块", fs=14, bold=True)
vals = ["1.2", "0.3", "2.8", "0.9", "6.0", "3.1", "0.2", "1.5",
        "0.7", "4.2", "0.4", "2.0", "0.8", "5.5", "0.3", "1.1"]
bx, by, cw, ch = 24, 72, 30, 34
for i, v in enumerate(vals):
    hot = v == "6.0"
    f.box(bx + i * (cw + 4), by, cw, ch, v, fill=(LRF if hot else "#ffffff"),
          stroke=(RED if hot else GRAY), fs=10.5, bold=hot, fc=(RED if hot else "#48586a"))
f.text(bx, by + 42, 545, 20, "amax(16 元素) = 6.0（红色格）→ s = amax / 6 = 1.0", fs=12.5, bold=True, fc=RED)
f.box(24, 146, 168, 46, "x&#770; = round(x / s)", fill=LBF, stroke=BLUE, fs=13, bold=True, fc=BLUE)
f.box(216, 146, 172, 46, "∈ [-6, +6] 编码为 e2m1", fill=LGF, stroke=GREEN, fs=12.5, bold=True, fc=GREEN)
f.line([(192, 169), (216, 169)], color=INK, width=2)
f.box(408, 146, 156, 46, "s 编码为 ue4m3", fill=LPF, stroke=PURPLE, fs=12.5, bold=True, fc=PURPLE)
f.line([(388, 169), (408, 169)], color=INK, width=2)
f.text(24, 200, 545, 20, "16 个数据共用 1 个 SF 字节：每元素均摊 4 + 8/16 = 4.5 bit", fs=12, fc="#48586a")
f.text(24, 222, 545, 20, "全零块：s = 0 → 倒数取 0（0/0 守卫），data=0 且 SF=0", fs=12, fc=RED)

# ===== 右上：两种格式位域 =====
f.text(610, 46, 520, 20, "② 数据与尺度的位域", fs=14, bold=True)
# e2m1: s e e m
f.text(610, 72, 520, 18, "e2m1 数据（4 bit）：1 符号 + 2 指数 + 1 尾数", fs=12.5, bold=True, fc=GREEN)
seg = [("s", 34, LRF, RED), ("e", 34, LGF, GREEN), ("e", 34, LGF, GREEN), ("m", 34, LBF, BLUE)]
sx, sy = 700, 96
for t, w, fill, stroke in seg:
    f.box(sx, sy, w, 32, t, fill=fill, stroke=stroke, fs=13, bold=True, fc=stroke)
    sx += w + 3
f.text(610, 134, 520, 20, "值域 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}×2^e，满量程 6", fs=12, fc="#48586a")
f.text(610, 154, 520, 20, "相对步长 ≈ 25%（fp8 e4m3 为 6.25%）", fs=12, bold=True, fc=RED)
# ue4m3: eeee mmm
f.text(610, 184, 520, 18, "ue4m3 尺度（8 bit）：4 指数 + 3 尾数（无符号）", fs=12.5, bold=True, fc=PURPLE)
seg2 = [("e", 30, LPF, PURPLE), ("e", 30, LPF, PURPLE), ("e", 30, LPF, PURPLE), ("e", 30, LPF, PURPLE),
        ("m", 30, LBF, BLUE), ("m", 30, LBF, BLUE), ("m", 30, LBF, BLUE)]
sx, sy = 700, 208
for t, w, fill, stroke in seg2:
    f.box(sx, sy, w, 32, t, fill=fill, stroke=stroke, fs=13, bold=True, fc=stroke)
    sx += w + 3
f.text(610, 246, 520, 20, "值域 [0, 448]；SF 自身舍入 6.25% 整体作用于 16 个元素", fs=12, fc="#48586a")
f.text(610, 268, 520, 20, "MXFP4 对照：ue8m0/32 块，SF 步长 100%（未采用）", fs=12, fc=GRAY)

# ===== 下半：blockscale MMA 语义 =====
y2 = 306
f.box(24, y2, 1106, 440, "", fill="#fbfcfd", stroke=GRAY)
f.text(40, y2 + 8, 1070, 22, "③ blockscale MMA：dequant 逐块隐含在乘法里（SM120 m16n32k64 mxf4nvf4）", fs=14, bold=True, fc=INK)

# 输入张量示意
f.box(48, y2 + 56, 200, 60, "A&#770;（e2m1 数据）&#10;+ s_A（ue4m3，per-16）", fill=LGF, stroke=GREEN, fs=12.5, bold=True, fc=GREEN)
f.box(48, y2 + 156, 200, 60, "B&#770;（e2m1 数据）&#10;+ s_B（ue4m3，per-16）", fill=LOF, stroke=ORANGE, fs=12.5, bold=True, fc="#8a4b08")
f.box(48, y2 + 256, 200, 60, "C（f32 累加器）&#10;S = QK 或 O = PV", fill=LBF, stroke=BLUE, fs=12.5, bold=True, fc=BLUE)

# MMA 盒
f.box(330, y2 + 120, 220, 130, "blockscale MMA&#10;&#10;逐块 (A&#770;·s_A)(B&#770;·s_B)&#10;K=64 对齐 SF 旁路", fill="#ffffff", stroke=RED, fs=13, bold=True, fc=RED)
f.line([(248, y2 + 86), (330, y2 + 150)], color=GREEN, width=2)
f.line([(248, y2 + 186), (330, y2 + 210)], color=ORANGE, width=2)
f.line([(550, y2 + 185), (248, y2 + 286)], color=BLUE, width=2)

# 右侧：块级展开
f.text(600, y2 + 52, 500, 20, "块内视角（K 归约的每 16 一段）：", fs=12.5, bold=True, fc=INK)
f.box(600, y2 + 76, 480, 46, "acc += Σ&#8321;&#8326; (a&#770;&#7522;·s_A[blk]) · (b&#770;&#7522;·s_B[blk])", fill="#ffffff", stroke=BLUE, fs=13.5, bold=True, fc=BLUE)
f.text(600, y2 + 130, 500, 20, "· 数据通路 4 bit，SF 旁路 8 bit，硬件在累加处乘回尺度", fs=12, fc="#48586a")
f.text(600, y2 + 152, 500, 20, "· K 维必须 64 对齐（SF atom 64 宽）→ D pad 到 64 倍数", fs=12, fc="#48586a")
f.text(600, y2 + 174, 500, 20, "· pad 列 data=0 且 SF=0 → 贡献恒 0（不产垃圾）", fs=12, fc="#48586a")
f.text(600, y2 + 196, 500, 20, "· SF 编码误差 = 尺度级错误：写序必须 match SF atom", fs=12, bold=True, fc=RED)

# 尺度工程三件套
f.text(600, y2 + 230, 500, 20, "让 e2m1 可用的尺度工程（本章内容）：", fs=12.5, bold=True, fc=INK)
f.box(600, y2 + 254, 150, 46, "microscaling&#10;1×16 块 + ue4m3", fill=LPF, stroke=PURPLE, fs=11.5, bold=True, fc=PURPLE)
f.box(770, y2 + 254, 150, 46, "smoothing&#10;q_m / k&#772; 强制减除", fill=LGF, stroke=GREEN, fs=11.5, bold=True, fc=GREEN)
f.box(940, y2 + 254, 140, 46, "hadamard&#10;通道摊平（fused）", fill=LOF, stroke=ORANGE, fs=11.5, bold=True, fc="#8a4b08")
f.text(600, y2 + 306, 500, 20, "均值走 fp32 精确旁路（ΔS / lse / epilogue 加回），", fs=12, fc="#48586a")
f.text(600, y2 + 326, 500, 20, "残差交给 e2m1 —— 量化预算全部留给有效信息", fs=12, fc="#48586a")

f.text(40, y2 + 340, 540, 20, "P 的困难（online、值域 [0,1]、行内分布剧变）→ 两级 P 量化，见下一章", fs=12, bold=True, fc=PURPLE)

print(mk(f.p, W, H, "fig31-1"))
