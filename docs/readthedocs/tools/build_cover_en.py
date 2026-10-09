"""英文封面生成工具（一次性人工跑，产物入库）。

英文模式下封面也得是英文。封面是书稿的 ``figures/misc/cover-tikz.tex`` 编译成 PDF、再由
``convert._cover_image()`` 光栅化进页面的——字是矢量轮廓，Google 只改 HTML 文本节点，
碰不到它（和 227 张插图同一个原因）。本工具把 tex 里那几处中文换成英文、去掉 xeCJK
（英文版不需要中文字体），xelatex 出 PDF、``pdftocairo`` 出 PNG，落到
``_static/figures-en/cover.png``：``translate.js`` 的换图逻辑在英文模式下会把
``_images/cover.png`` 换成它，缺失或加载失败则回退中文封面。

    conda activate cdit
    cd docs/readthedocs
    python -m tools.build_cover_en               # 默认 240 dpi
    python -m tools.build_cover_en --dpi 120     # 与中文封面同档（体积更小）
    python -m tools.build_cover_en --keep        # 留住中间 tex/pdf 供检查

产物要入库（``.gitignore`` 里 ``_static/figures-en/*`` 只放行了这一个文件）。封面文案改了
就得重跑：替换按原文片段定位，对不上会直接报错停下——宁可停下来，也不要出一张半中半英
的封面。

字体：Humor Sans、Comic Neue（本机装在 ``~/.local/share/fonts/``，两者都是 100 KB 上下，
所以没有连进仓库）。书稿封面还用了 LXGW WenKai（25 MB，纯中文），英文版用不到。
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK_COVER = ROOT / "../../kernels/interview/book/figures/misc/cover-tikz.tex"
DEST = ROOT / "_static" / "figures-en" / "cover.png"
FONTS = ("Humor Sans", "Comic Neue")

#: 中文 → 英文，逐处替换；英文措辞与站点英文模式下的标题/副标题一致。
REPLACEMENTS: tuple[tuple[str, str], ...] = (
  ("{\\CJKfamily{zhsans}\\sffamily CUDA Kernel 优化之路};",
   "{\\sffamily The Path to CUDA Kernel Optimization};"),
  ("{从编程模型到 CuTe 与 FlashAttention};",
   "{From programming models to CuTe and FlashAttention};"),
  ("{\\CJKfamily{zhsans}\\bfseries 郑重声明\\\\\n"
   "   \\mdseries\\CJKfamily{zhsong} 本书仅用于开源学习与技术交流，\\\\未经作者本人同意，"
   "不得用于任何商业用途。};",
   "{\\bfseries Disclaimer\\\\\n"
   "   \\mdseries This book is for open-source learning and technical exchange only.\\\\\n"
   "   Any commercial use without the author's permission is prohibited.};"),
  ("{LeetCUDA Project (Own by DefTruth) $\\cdot$ 2026 年 9 月};",
   "{LeetCUDA Project (Own by DefTruth) $\\cdot$ September 2026};"),
)

#: 英文版不加载 xeCJK，只留 fontspec（``\\setmainfont`` 由它提供）+ 三个拉丁字体。
PREAMBLE_ZH = """\\usepackage{xeCJK}
\\usepackage{amsmath}
\\setmainfont{Humor Sans}
\\setsansfont{Humor Sans}
\\setmonofont{Comic Neue}
\\setCJKmainfont{LXGW WenKai}
\\setCJKsansfont{LXGW WenKai}"""

PREAMBLE_EN = """\\usepackage{fontspec}
\\usepackage{amsmath}
\\setmainfont{Humor Sans}
\\setsansfont{Humor Sans}
\\setmonofont{Comic Neue}"""


def uncommented(text: str) -> str:
  """去掉整行 ``%`` 注释（tex 里的中文注释不算正文，不用翻）。

  :param text: tex 文本。
  :returns: 去注释后的文本。
  """
  return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%"))


def translate(text: str) -> str:
  """把封面 tex 改成英文版（每处替换必须恰好命中一次）。

  :param text: 书稿封面 tex 原文。
  :returns: 英文版 tex。
  """
  if text.count(PREAMBLE_ZH) != 1:
    raise SystemExit("封面 tex 的宏包/字体声明段与预期不符，需要人工确认后更新 PREAMBLE_ZH")
  text = text.replace(PREAMBLE_ZH, PREAMBLE_EN)
  for source, target in REPLACEMENTS:
    hit = text.count(source)
    if hit != 1:
      raise SystemExit(f"替换片段命中 {hit} 次（期望 1 次），封面可能改过了：{source[:40]}…")
    text = text.replace(source, target)
  leftover = re.findall(r"[\u4e00-\u9fff]+", uncommented(text))
  if leftover:
    raise SystemExit(f"英文版里还有中文，先补进 REPLACEMENTS：{leftover[:5]}")
  return text


def check_fonts() -> None:
  """确认本机装了封面要用的两个字体。"""
  listing = subprocess.run(["fc-list", ":family"], capture_output=True, text=True, check=False).stdout
  missing = [font for font in FONTS if font.lower() not in listing.lower()]
  if missing:
    raise SystemExit(f"缺少字体 {missing}；封面用的字体在 ~/.local/share/fonts/ 下，先装好再跑")
  if shutil.which("pdftocairo") is None:
    raise SystemExit("缺少 pdftocairo（poppler-utils）")


def build(work_dir: Path, dpi: int, keep: bool) -> Path:
  """编译英文封面并光栅化成 PNG。

  :param work_dir: 中间产物目录（tex / pdf / log）。
  :param dpi: 光栅化分辨率。
  :param keep: 是否保留中间产物。
  :returns: 产物路径。
  """
  work_dir.mkdir(parents=True, exist_ok=True)
  tex = work_dir / "cover-en.tex"
  tex.write_text(translate(BOOK_COVER.read_text(encoding="utf-8")), encoding="utf-8")

  # 两遍：standalone 的紧致裁剪依赖上一遍写下的边界信息。
  for _ in range(2):
    proc = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                          cwd=work_dir, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
      log = work_dir / "cover-en.log"
      tail = log.read_text(encoding="utf-8", errors="replace")[-800:] if log.is_file() else ""
      raise SystemExit(f"xelatex 失败（{proc.returncode}）：\n{tail or proc.stdout[-800:]}")

  pdf = work_dir / "cover-en.pdf"
  proc = subprocess.run(["pdftocairo", "-png", "-r", str(dpi), "-singlefile",
                         str(pdf), str(pdf.with_suffix(""))],
                        capture_output=True, text=True, check=False)
  png = pdf.with_suffix(".png")
  if proc.returncode != 0 or not png.is_file():
    raise SystemExit(f"pdftocairo 失败（{proc.returncode}）：{proc.stderr[-400:]}")

  DEST.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(png, DEST)
  if not keep:
    shutil.rmtree(work_dir, ignore_errors=True)
  return DEST


def main() -> None:
  """命令行入口。"""
  parser = argparse.ArgumentParser(description="生成英文封面 PNG（产物入库）")
  parser.add_argument("--dpi", type=int, default=240, help="光栅化分辨率（默认 240）")
  parser.add_argument("--work", default=str(ROOT / ".tmp" / "cover-en"), help="中间产物目录")
  parser.add_argument("--keep", action="store_true", help="保留中间 tex / pdf / log")
  args = parser.parse_args()

  check_fonts()
  if not BOOK_COVER.is_file():
    raise SystemExit(f"找不到书稿封面：{BOOK_COVER}")
  dest = build(Path(args.work), args.dpi, args.keep)
  print(f"英文封面：{dest.relative_to(ROOT)}（{dest.stat().st_size / 1024:.0f} KB，{args.dpi} dpi）")
  print(f"替换了 {len(REPLACEMENTS)} 处文案；产物要入库，改过封面记得重跑本工具")


if __name__ == "__main__":
  sys.exit(main())
