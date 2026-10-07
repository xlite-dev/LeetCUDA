"""转换管线入口：book 章节 tex → MyST → Sphinx 站点树。

用法（RTD 与本地一致）::

  cd docs/readthedocs && python -m convert

默认路径全部以本文件位置为基准，因此在任何工作目录下调用结果一致。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import booktree, labels, postprocess, preprocess, tikz2svg

#: 站点首页标题与简介。
BOOK_TITLE = "LeetCUDA：CUDA Kernel 优化之路"
BOOK_SUBTITLE = "从编程模型到 CuTe 与 FlashAttention"
BOOK_INTRO = (
  "本站在线呈现《LeetCUDA：CUDA Kernel 优化之路》的全部章节内容，源文件是仓库中的 "
  "LaTeX 书稿（`kernels/interview/book/`）。页面由书稿自动转换生成，正文、公式、"
  "代码清单与插图均取自原书；公式用 MathJax 渲染，代码清单里标注的行号与源文件行号一致。"
)
BOOK_LINKS = (
  "- 书稿与源码：[xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA)\n"
  "- 本地构建 PDF：`cd kernels/interview/book && ./build.sh`"
)

#: 默认 Git 分支（RTD 上用环境变量覆盖）。
_DEFAULT_BRANCH = "main"

_SOURCE_TEMPLATE = "https://github.com/xlite-dev/LeetCUDA/blob/{ref}/kernels/interview/book"


def main(argv: list[str] | None = None) -> int:
  """执行转换管线。

  :param argv: 命令行参数（默认取 ``sys.argv[1:]``）。
  :returns: 进程退出码。
  """
  root = Path(__file__).resolve().parent.parent
  parser = argparse.ArgumentParser(prog="convert", description="LeetCUDA book → Read the Docs 站点")
  parser.add_argument("--book-dir", default=str(root / "../../kernels/interview/book"),
                      help="书根目录（含 book.tex 的目录）")
  parser.add_argument("--out", default=str(root / "build" / "src"), help="站点源树输出目录")
  parser.add_argument("--work", default=str(root / "build" / "tmp"), help="临时目录")
  parser.add_argument("--only", default="", help="只处理指定章节（逗号分隔的 chap_id）")
  parser.add_argument("--max-chapters", type=int, default=0, help="只处理前 N 章（调试用）")
  parser.add_argument("--skip-tikz", action="store_true", help="跳过 TikZ 编译（调试用）")
  parser.add_argument("--force-tikz", action="store_true", help="忽略缓存重编全部 TikZ")
  parser.add_argument("--jobs", type=int, default=tikz2svg.DEFAULT_JOBS,
                      help="TikZ 并行编译进程数（每进程一份 xelatex，受内存限制）")
  parser.add_argument("--pandoc", default="", help="pandoc 可执行文件路径")
  parser.add_argument("--tikz-fonts", choices=list(tikz2svg.FONT_MODES),
                      default=tikz2svg.FONT_PATHS,
                      help="TikZ 图字号处理：paths=字形转路径（默认，任何查看器一致）；"
                           "woff2=内嵌字体子集（更小，但需查看器支持 @font-face）")
  parser.add_argument("--png", action="store_true",
                      help="额外把生成的 SVG 导出成 PNG（便于查看器不支持 SVG 时核对）")
  parser.add_argument("--source-url", default="", help="原书 tex 的 GitHub 链接前缀")
  parser.add_argument("--no-assemble", action="store_true", help="只转换章节，不生成站点装配页")
  args = parser.parse_args(argv)

  book_dir = Path(args.book_dir).resolve()
  out_dir = Path(args.out).resolve()
  work_dir = Path(args.work).resolve()
  if not (book_dir / "book.tex").is_file():
    print(f"错误：{book_dir} 下没有 book.tex", file=sys.stderr)
    return 2

  started = time.monotonic()
  parts = booktree.parse(book_dir / "book.tex")
  registry = labels.build(parts, book_dir)
  chapters = [chapter for part in parts for chapter in part.chapters]
  selected = _select(chapters, args.only, args.max_chapters)
  print(f"章节：{len(chapters)} 篇，本次处理 {len(selected)} 篇；label 注册表 {len(registry.entries)} 条")

  pandoc = preprocess.find_pandoc(args.pandoc or None)
  print(f"pandoc：{pandoc}")
  out_dir.mkdir(parents=True, exist_ok=True)
  work_dir.mkdir(parents=True, exist_ok=True)

  stages: dict[str, float] = {}
  manifests = []
  stage = time.monotonic()
  for chapter in selected:
    manifests.append(preprocess.normalize(chapter, book_dir, work_dir, out_dir, registry))
  stages["归一化"] = round(time.monotonic() - stage, 1)
  print(f"[1/4] 归一化 {len(manifests)} 章：{stages['归一化']}s")

  stage = time.monotonic()
  tikz_report = tikz2svg.Report()
  if args.skip_tikz:
    print("[2/4] 跳过 TikZ 编译（--skip-tikz）")
  else:
    snippets = tikz2svg.collect(manifests, work_dir, out_dir)
    tikz_report = tikz2svg.build(
      snippets, book_dir, work_dir, force=args.force_tikz,
      font_mode=args.tikz_fonts, jobs=args.jobs)
    print(f"[2/4] TikZ：{len(snippets)} 个片段，新编 {len(tikz_report.ok)}，"
          f"缓存 {len(tikz_report.skipped)}，失败 {len(tikz_report.failed)}，"
          f"模式 {tikz_report.mode}，{tikz_report.seconds}s")
  stages["TikZ"] = round(time.monotonic() - stage, 1)

  stage = time.monotonic()
  results = []
  source_url = args.source_url or _SOURCE_TEMPLATE.format(ref=_git_ref())
  md_dir = work_dir / "md"
  for chapter, manifest in zip(selected, manifests):
    raw_md = md_dir / f"{chapter.chap_id}.md"
    warning = preprocess.to_markdown(work_dir / "norm" / f"{chapter.chap_id}.tex", raw_md, pandoc)
    if warning.strip():
      manifest.notes.append(f"pandoc 警告：{warning.strip()[:400]}")
    ctx = postprocess.RenderContext(
      chap=chapter,
      parts=parts,
      manifest=manifest,
      registry=registry,
      work_dir=work_dir,
      tikz_failed=set(tikz_report.failed),
      source_url=source_url,
    )
    result = postprocess.render_chapter(ctx, raw_md.read_text(encoding="utf-8"))
    (out_dir / f"{chapter.chap_id}.md").write_text(result.markdown, encoding="utf-8")
    results.append((chapter, manifest, result))
  stages["渲染"] = round(time.monotonic() - stage, 1)
  print(f"[3/4] 渲染 {len(results)} 章：{stages['渲染']}s")

  stage = time.monotonic()
  converted = {chapter.chap_id for chapter, _, _ in results}
  if not args.no_assemble:
    _assemble_site(parts, out_dir, converted)
  stages["装配"] = round(time.monotonic() - stage, 1)
  print(f"[4/4] 站点装配：{stages['装配']}s")

  if args.png:
    png_dir = root / "build" / "figures-png"
    print(f"PNG 导出：{export_png(out_dir, png_dir)} 张 → {png_dir}")

  dangling, link_total = _check_links(out_dir)
  if dangling:
    print(f"站内链接：{link_total} 条，其中 {len(dangling)} 条锚点缺失")
  report = _write_report(
    root, parts, results, registry, tikz_report, time.monotonic() - started,
    dangling, link_total, stages)
  print(f"报告：{report}")
  return 0


def export_png(src_dir: Path, png_dir: Path, scale: float = 1.6) -> int:
  """把站点里的 SVG 图导出为 PNG（本地核对用）。

  部分查看器（含 VS Code 的 SVG 预览）不支持内嵌 web font，直接看 SVG 会显示成
  重叠乱码；PNG 没有这个问题。只影响本地核对，不参与站点构建。

  :param src_dir: 站点源树（含 ``figures-gen``）。
  :param png_dir: PNG 输出目录。
  :param scale: 相对于 SVG 原始尺寸的放大倍数。
  :returns: 导出数量。
  :raises RuntimeError: 缺少 cairosvg。
  """
  try:
    import cairosvg
  except ImportError as error:
    raise RuntimeError("导出 PNG 需要 cairosvg：pip install cairosvg") from error

  png_dir.mkdir(parents=True, exist_ok=True)
  count = 0
  for svg in sorted((src_dir / "figures-gen").glob("*.svg")):
    text = svg.read_text(encoding="utf-8", errors="replace")
    if "width='0pt'" in text:
      continue
    try:
      cairosvg.svg2png(url=str(svg), write_to=str(png_dir / f"{svg.stem}.png"),
                       scale=scale, background_color="white")
    except Exception:  # cairosvg 对个别格式敏感，跳过即可，不影响站点
      continue
    count += 1
  return count


def _check_links(out_dir: Path) -> tuple[list[tuple[str, str]], int]:
  """检查站内 ``page.html#anchor`` 链接与锚点是否对得上。

  MyST 看不见原始 HTML 写出的锚点，所以这里自己核对一遍：把每个页面里出现的
  站内引用与各页面的 ``<a id>`` 集合求差，悬空引用进报告。

  :param out_dir: 站点源树。
  :returns: ``(悬空引用 [(页面, 引用)], 引用总数)``。
  """
  anchors: dict[str, set[str]] = {}
  pages: set[str] = set()
  for md in sorted(out_dir.glob("*.md")):
    pages.add(md.stem)
    anchors[md.stem] = set(re.findall(r'<a id="([^"]+)"', md.read_text(encoding="utf-8")))

  dangling: list[tuple[str, str]] = []
  total = 0
  for md in sorted(out_dir.glob("*.md")):
    text = md.read_text(encoding="utf-8")
    for page, anchor in re.findall(r"\]\(([A-Za-z0-9_.-]+)\.html#([^)\s]+)", text):
      total += 1
      if page not in pages or anchor not in anchors[page]:
        dangling.append((md.stem, f"{page}.html#{anchor}"))
  return dangling, total


def _git_ref() -> str:
  """取原书 tex 链接用的 Git 引用（RTD 环境变量优先）。

  :returns: 分支名或提交标识。
  """
  import os

  for name in ("READTHEDOCS_GIT_IDENTIFIER", "READTHEDOCS_VERSION_NAME"):
    value = os.environ.get(name, "").strip()
    if value and value != "latest":
      return value
  return _DEFAULT_BRANCH


def _select(chapters: list[booktree.Chapter], only: str, limit: int) -> list[booktree.Chapter]:
  """按参数挑选章节。

  :param chapters: 全部章节。
  :param only: 逗号分隔的 chap_id。
  :param limit: 只取前 N 章。
  :returns: 选中的章节。
  """
  selected = chapters
  if only:
    wanted = {name.strip() for name in only.split(",") if name.strip()}
    selected = [chapter for chapter in selected if chapter.chap_id in wanted]
  if limit > 0:
    selected = selected[:limit]
  return selected


def _assemble_site(parts: list[booktree.Part], out_dir: Path, converted: set[str]) -> None:
  """生成首页与各篇的装配页。

  :param parts: 篇结构。
  :param out_dir: 站点源树。
  :param converted: 已转换的 chap_id 集合。
  """
  part_pages: list[tuple[booktree.Part, str]] = []
  for part in parts:
    available = [chapter for chapter in part.chapters if chapter.chap_id in converted]
    if not available:
      continue
    page = f"part-{part.index}"
    part_pages.append((part, page))
    lines = [f"# {part.title or '目录'}", ""]
    lines.append("```{toctree}")
    lines.append(":maxdepth: 1")
    lines.append("")
    for chapter in available:
      lines.append(chapter.chap_id)
    lines.append("```")
    (out_dir / f"{page}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

  index = [f"# {BOOK_TITLE}", "", f"*{BOOK_SUBTITLE}*", "", BOOK_INTRO, ""]
  cover = _cover_image(out_dir)
  if cover:
    index.extend([f"![{BOOK_TITLE}]({cover})", ""])
  index.extend([BOOK_LINKS, "", "```{toctree}", ":maxdepth: 2", ":caption: 目录", ""])
  for _, page in part_pages:
    index.append(page)
  index.append("```")
  (out_dir / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")


def _cover_image(out_dir: Path) -> str:
  """把封面 PDF 转成 PNG（失败则跳过）。

  :param out_dir: 站点源树。
  :returns: 站点树内的封面相对路径，失败为空串。
  """
  book_cover = Path(__file__).resolve().parent.parent / "../../kernels/interview/book/figures/misc/cover.pdf"
  book_cover = book_cover.resolve()
  if not book_cover.is_file() or shutil.which("pdftocairo") is None:
    return ""
  target = out_dir / "figures" / "misc" / "cover.png"
  if target.is_file():
    return target.relative_to(out_dir).as_posix()
  target.parent.mkdir(parents=True, exist_ok=True)
  proc = subprocess.run(
    ["pdftocairo", "-png", "-r", "120", "-singlefile", str(book_cover), str(target.with_suffix(""))],
    capture_output=True, text=True, check=False)
  if proc.returncode != 0 or not target.is_file():
    return ""
  return target.relative_to(out_dir).as_posix()


def _write_report(
  root: Path,
  parts: list[booktree.Part],
  results: list[tuple[booktree.Chapter, preprocess.Manifest, postprocess.RenderResult]],
  registry: labels.Registry,
  tikz_report: tikz2svg.Report,
  seconds: float,
  dangling: list[tuple[str, str]],
  link_total: int,
  stages: dict[str, float],
) -> Path:
  """写转换报告（markdown + json）。

  :param root: ``docs/readthedocs`` 目录。
  :param parts: 篇结构。
  :param results: 各章结果。
  :param registry: label 注册表。
  :param tikz_report: TikZ 编译报告。
  :param seconds: 总耗时。
  :param dangling: 悬空的站内引用。
  :param link_total: 站内引用总数。
  :param stages: 各阶段耗时。
  :returns: markdown 报告路径。
  """
  report_dir = root / "build"
  report_dir.mkdir(parents=True, exist_ok=True)
  leftover_total: dict[str, int] = {}
  broken: dict[str, list[str]] = {}
  notes: list[str] = []
  code_total = image_total = tikz_total = 0
  for chapter, manifest, result in results:
    code_total += manifest.stats.get("code", 0)
    image_total += manifest.stats.get("image", 0)
    tikz_total += manifest.stats.get("tikz", 0)
    if result.broken_refs:
      broken[chapter.chap_id] = result.broken_refs
    for name, count in result.leftovers.items():
      leftover_total[name] = leftover_total.get(name, 0) + count
    notes.extend(manifest.notes)
    notes.extend(f"{chapter.chap_id}: {note}" for note in result.notes)

  stage_text = "；".join(f"{name} {value:.1f}s" for name, value in stages.items())
  lines = [
    "# LeetCUDA RTD 转换报告",
    "",
    f"- 章节：{len(results)} / {sum(len(part.chapters) for part in parts)}",
    f"- TikZ 片段：{tikz_total}（新编 {len(tikz_report.ok)}，缓存 {len(tikz_report.skipped)}，失败 {len(tikz_report.failed)}）",
    f"- 代码清单：{code_total}；镜像图片：{image_total}",
    f"- label 注册表：{len(registry.entries)} 条（重复 {len(registry.duplicates)}）",
    f"- 总耗时：{seconds:.1f}s（{stage_text}）",
    "",
    "## 未解析引用",
    "",
  ]
  if broken:
    lines.append("| 章 | label |")
    lines.append("| --- | --- |")
    for chap_id, refs in broken.items():
      for ref in refs:
        lines.append(f"| {chap_id} | `{ref}` |")
  else:
    lines.append("（无）")
  lines.extend(["", "## TikZ 失败清单", ""])
  if tikz_report.failed:
    lines.append("| 图片 | 错误 |")
    lines.append("| --- | --- |")
    for image_id, message in tikz_report.failed.items():
      lines.append(f"| `{image_id}` | {message} |")
  else:
    lines.append("（无）")
  lines.extend(["", "## 站内链接自检", "", f"- 站内引用 {link_total} 条，锚点缺失 {len(dangling)} 条", ""])
  if dangling:
    lines.append("| 所在页 | 悬空引用 |")
    lines.append("| --- | --- |")
    for page, ref in dangling[:60]:
      lines.append(f"| {page} | `{ref}` |")
  else:
    lines.append("（无）")
  lines.extend(["", "## 残留 LaTeX 命令（pandoc 未转换）", ""])
  if leftover_total:
    lines.append("| 命令 | 次数 |")
    lines.append("| --- | --- |")
    for name, count in sorted(leftover_total.items(), key=lambda item: item[1], reverse=True)[:40]:
      lines.append(f"| `\\{name}` | {count} |")
  else:
    lines.append("（无）")
  lines.extend(["", "## 逐章明细", "", "| 章 | 标题 | TikZ | 代码 | 图片 | 未解析引用 |", "| --- | --- | --- | --- | --- | --- |"])
  for chapter, manifest, result in results:
    lines.append(
      f"| {chapter.chap_id} | {chapter.title} | {manifest.stats.get('tikz', 0)} | "
      f"{manifest.stats.get('code', 0)} | {manifest.stats.get('image', 0)} | {len(result.broken_refs)} |")
  lines.extend(["", "## 提示", ""])
  lines.extend([f"- {note}" for note in notes[:80]] or ["（无）"])
  report_path = report_dir / "report.md"
  report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  (report_dir / "report.json").write_text(json.dumps({
    "chapters": len(results),
    "tikz_total": tikz_total,
    "tikz_ok": len(tikz_report.ok),
    "tikz_cached": len(tikz_report.skipped),
    "tikz_failed": tikz_report.failed,
    "code_total": code_total,
    "image_total": image_total,
    "labels": len(registry.entries),
    "broken_refs": broken,
    "dangling_anchors": dangling,
    "link_total": link_total,
    "leftovers": leftover_total,
    "seconds": round(seconds, 1),
    "stages": stages,
    "tikz_seconds": tikz_report.seconds,
    "tikz_mode": tikz_report.mode,
    "tikz_font_mode": tikz_report.font_mode,
    "notes": notes,
  }, ensure_ascii=False, indent=2), encoding="utf-8")
  return report_path


if __name__ == "__main__":
  sys.exit(main())
