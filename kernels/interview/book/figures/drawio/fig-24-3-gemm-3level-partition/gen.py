#!/usr/bin/env python3
# gen_fig24_3.py — FIG-24-3：知乎 @reed《cute 之 简单 GEMM 实现》fig-3 重建
# 来源：figures/zhihu/reed-simple-gemm/fig-3.jpg（CuTe GEMM 三层 Partition）
# 语义：local_tile 从 gmem 切 CTA tile (128,128) -> local_partition 按 warp 分 warp tile (64,64)
#       -> partition_fragment_C 按 thread 取累加片段（16x8 MMA 内 2x2 四值）
import os, sys

parts = []
def P(s): parts.append(s)

def box(x, y, w, h, t="", fill="#ffffff", stroke="#bcccdc", fs=13, bold=False, fc="#1f2933", dashed=0):
    st = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
          f"fontSize={fs};fontColor={fc};align=center;verticalAlign=middle;fontFamily=Helvetica")
    if bold: st += ";fontStyle=1"
    if dashed: st += ";dashed=1"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def rbox(x, y, w, h, t, fill="#ffffff", stroke="#2171b5", fs=12.5, fc=None, bold=True):
    st = (f"rounded=1;arcSize=20;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
          f"fontSize={fs};fontColor={fc or stroke};align=center;verticalAlign=middle;fontFamily=Helvetica")
    if bold: st += ";fontStyle=1"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def text(x, y, w, h, t, fs=13, fc="#102a43", bold=False):
    st = f"text;html=1;align=left;verticalAlign=middle;fontSize={fs};fontColor={fc};fontFamily=Helvetica"
    if bold: st += ";fontStyle=1"
    P(f'<mxCell value="{t}" style="{st}" vertex="1" parent="1">'
      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def arrow(pts, color="#48586a", width=2, dashed=0, both=0):
    st = (f"endArrow=blockThin;endFill=1;html=1;strokeColor={color};strokeWidth={width};rounded=0;")
    if both: st = "startArrow=blockThin;startFill=1;" + st
    if dashed: st += "dashed=1;"
    P(f'<mxCell value="" style="{st}" edge="1" parent="1">'
      f'<mxGeometry relative="1" as="geometry"><Array as="points">'
      + ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
      + f'</Array><mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
        f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/></mxGeometry></mxCell>')

BLUE, GREEN, ORANGE, RED, PURPLE = "#2171b5", "#2e8540", "#d97706", "#dc2626", "#6a51a3"
LBF, LGF, LOF, LRF, GRAY = "#bfd7ea", "#c9e4c8", "#fde9d0", "#fdeaea", "#e7ecf3"
DKO, DKG = "#b45309", "#1e5631"

def fig_24_3():
  parts.clear()
  text(30, 12, 1120, 26, "CuTe GEMM 三层 partition：gmem → CTA tile → warp tile → thread value（嵌套网格逐层细化）", fs=16)
  text(30, 44, 1120, 20, "local_tile 从 gmem 切 CTA tile；local_partition 按 warp 分 warp tile；partition_fragment_C 按 thread 取累加器片段", fs=12.5, fc="#48586a")

  # ---- Panel A：gmem 矩阵 C ----
  text(60, 120, 260, 22, "gmem 矩阵 C (M, N)", fs=14.5)
  arrow([(60, 144), (324, 144)], color="#627d98", width=1)
  text(330, 136, 16, 16, "N", fs=12, fc="#627d98")
  arrow([(50, 150), (50, 448)], color="#627d98", width=1)
  text(30, 292, 16, 16, "M", fs=12, fc="#627d98")
  for r in range(5):
    for c in range(4):
      if r in (1, 2) and c in (1, 2):
        continue  # CTA tile 方块覆盖区不画浅格
      box(60 + c * 66, 150 + r * 60, 62, 56, fill="none", stroke="#d9e2ec")
  box(60, 150, 262, 296, fill="none", stroke="#8aa2b8")
  box(126, 210, 130, 118, fill=LBF, stroke=BLUE)
  text(130, 252, 110, 18, "CTA tile", fs=13, fc=BLUE, bold=True)
  text(130, 272, 110, 16, "128 × 128", fs=12, fc=ORANGE)
  text(70, 458, 250, 16, "CTA tile 坐标 = (blockIdx.y, blockIdx.x)", fs=11, fc="#627d98")
  arrow([(262, 270), (436, 270)], color=BLUE, width=2)
  text(330, 280, 100, 16, "切出 128×128", fs=11, fc="#48586a")

  # ---- Panel B：CTA tile = 2x2 warp tile ----
  rbox(440, 88, 96, 26, "local_tile")
  text(440, 120, 300, 22, "CTA tile (128, 128)", fs=14.5)
  for r in range(2):
    for c in range(2):
      hi = (r, c) == (0, 1)
      box(440 + c * 132, 158 + r * 120, 128, 118,
          fill=(LOF if hi else "none"), stroke=(DKO if hi else "#8aa2b8"))
      text(440 + c * 132 + 34, 158 + r * 120 + 40, 96, 18, "warp tile", fs=12.5, fc=(DKO if hi else "#48586a"))
      text(440 + c * 132 + 34, 158 + r * 120 + 60, 96, 16, "64 × 64", fs=12, fc=(DKO if hi else ORANGE))
  text(440, 402, 300, 18, "warp 布局 (2, 2)：4 个 warp 平分 CTA tile", fs=11.5, fc="#48586a")
  arrow([(704, 218), (782, 218)], color=DKO, width=2, dashed=1)
  text(710, 228, 74, 16, "分 4 个 warp", fs=11, fc="#48586a")

  # ---- Panel C：warp tile = 4x8 个 16x8 MMA tile + thread value 放大 ----
  rbox(786, 88, 136, 26, "local_partition")
  text(786, 120, 300, 22, "warp tile (64, 64)", fs=14.5)
  text(786, 140, 320, 16, "64×64 = 4×8 个 16×8 MMA 输出 tile", fs=11.5, fc="#48586a")
  for r in range(4):
    for c in range(8):
      hi = (r, c) == (3, 0)
      box(786 + c * 40, 162 + r * 46, 38, 44,
          fill=(LGF if hi else "none"), stroke=(GREEN if hi else "#bcccdc"))
  arrow([(802, 348), (814, 364)], color=GREEN, width=2, dashed=1)
  # thread value 泡（2x2 四值）
  for (mr, mc, v) in [(0, 0, "0"), (0, 1, "1"), (1, 0, "2"), (1, 1, "3")]:
    box(786 + mc * 56, 368 + mr * 56, 54, 54, v, fill=LGF, stroke=GREEN, fs=15, bold=True, fc=DKG)
  text(920, 368, 150, 20, "thread value", fs=13.5, fc=GREEN, bold=True)
  text(920, 392, 200, 18, "每 thread 4 个累加值 v0..v3", fs=11.5, fc="#48586a")
  text(920, 414, 230, 18, "v0 v1 在第 i 行，v2 v3 在第 i+8 行", fs=11.5, fc="#48586a")
  rbox(920, 442, 190, 30, "partition_fragment_C", fs=12)
  arrow([(918, 457), (902, 428)], color=BLUE, width=2)

  # ---- 底部层级链 ----
  box(80, 512, 110, 34, "gmem 矩阵", fill=GRAY, stroke="#8aa2b8", fs=12.5)
  arrow([(194, 529), (242, 529)], color="#627d98", width=1.5)
  box(246, 512, 160, 34, "CTA tile (128,128)", fill=LBF, stroke=BLUE, fs=12.5, fc=BLUE)
  arrow([(410, 529), (458, 529)], color="#627d98", width=1.5)
  box(462, 512, 160, 34, "warp tile (64,64)", fill=LOF, stroke=DKO, fs=12.5, fc=DKO)
  arrow([(626, 529), (674, 529)], color="#627d98", width=1.5)
  box(678, 512, 190, 34, "thread value (v0..v3)", fill=LGF, stroke=GREEN, fs=12.5, fc=DKG)
  text(170, 548, 100, 16, "local_tile", fs=11, fc=BLUE)
  text(382, 548, 110, 16, "local_partition", fs=11, fc=BLUE)
  text(588, 548, 130, 16, "partition_fragment_C", fs=11, fc=BLUE)
  text(884, 516, 262, 18, "逐层细化：128×128 = 4 × (64×64)", fs=12.5, fc="#48586a")
  text(884, 538, 262, 18, "= 128 threads × 128 值/thread", fs=12.5, fc="#48586a")
  return mk(parts, 1160, 590, "fig-24-3")

def mk(parts, W, H, did):
  xml = ('<mxfile host="app.diagrams.net"><diagram id="%s" name="Page-1">'
         '<mxGraphModel dx="800" dy="600" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
         'arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d" math="0" shadow="0">'
         '<root><mxCell id="0"/><mxCell id="1" parent="0"/>%s</root></mxGraphModel></diagram></mxfile>'
         ) % (did, W, H, ''.join(parts))
  outdir = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "figures/drawio",
                        {"fig-24-3": "fig-24-3-gemm-3level-partition"}[did])
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, did + ".drawio")
  open(path, "w").write(xml)
  return path

if __name__ == "__main__":
  print("wrote", fig_24_3(), "OK")
