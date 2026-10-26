"""转换管线入口：book 章节 tex → MyST → Sphinx 站点树。

站点由「主书 + 若干子集合」组成（集合定义见 :mod:`convert.collections`）：主书产物平铺
在站点根，子集合产物落各自的子目录，主书首页的 toctree 末尾追加集合入口。

用法（RTD 与本地一致）::

  cd docs/readthedocs && python -m convert

默认路径全部以本文件位置为基准，因此在任何工作目录下调用结果一致。
"""
from __future__ import annotations

import argparse
import json
import posixpath
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import booktree, figtext, labels, macros, postprocess, preprocess, tikz2svg
from .collections import COLLECTIONS, Collection

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

#: 首页标题右侧的 Trendshift 徽章，片段照抄仓库根 README.md（徽章图由 trendshift.io 提供，
#: 排行榜每天变，只引远程图、不落本地副本）。写成 raw HTML 是为了让它浮动在标题右边：
#: MyST 的图片语法会独占一段，浮动样式只能靠 HTML 容器。CSS 见 `_static/custom.css`。
BOOK_BADGE = (
  '<div class="rtd-trendshift">'
  '<a href="https://trendshift.io/repositories/83964?utm_source=trendshift-badge'
  '&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-83964"'
  ' target="_blank" rel="noopener noreferrer">'
  '<img src="https://trendshift.io/api/badge/trendshift/repositories/83964/daily?language=Cuda"'
  ' alt="xlite-dev/LeetCUDA | Trendshift" width="250" height="53">'
  '</a></div>'
)

#: 默认 Git 分支（RTD 上用环境变量覆盖）。
_DEFAULT_BRANCH = "main"

_SOURCE_TEMPLATE = "https://github.com/xlite-dev/LeetCUDA/blob/{ref}/kernels/interview/book"


@dataclass
class BookSpec:
  """一本要转成站点的书（主书或子集合）。

  :param slug: 站点子目录名，主书为空串。
  :param title: 站点标题（主书用 ``BOOK_TITLE``，集合用集合标题）。
  :param book_dir: 书根目录（根 tex 所在目录）。
  :param book_tex: 根 tex 文件名。
  :param out_dir: 站点源树输出目录。
  :param work_dir: 临时目录。
  :param relative_dir: 相对主书根目录的路径（主书为空串），用于源码溯源链接。
  :param collection: 对应的集合定义（主书为 ``None``）。
  """
  slug: str
  title: str
  book_dir: Path
  book_tex: str
  out_dir: Path
  work_dir: Path
  relative_dir: str = ""
  collection: Collection | None = None

  @property
  def depth(self) -> int:
    """页面在站点树里的目录层数。

    :returns: 主书为 0，``cute-zhihu`` 这类一级集合为 1。
    """
    return 0 if not self.slug else len([part for part in self.slug.split("/") if part])


@dataclass
class BookResult:
  """一本书的转换结果。

  :param spec: 书籍规格。
  :param parts: 篇结构。
  :param registry: label 注册表。
  :param results: 逐章结果。
  :param tikz_report: TikZ 编译报告。
  :param stages: 各阶段耗时。
  :param seconds: 总耗时。
  """
  spec: BookSpec
  parts: list[booktree.Part]
  registry: labels.Registry
  results: list[tuple[booktree.Chapter, preprocess.Manifest, postprocess.RenderResult]]
  tikz_report: tikz2svg.Report
  stages: dict[str, float]
  seconds: float


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
  parser.add_argument("--only", default="", help="只处理主书的指定章节（逗号分隔的 chap_id）")
  parser.add_argument("--max-chapters", type=int, default=0, help="只处理主书前 N 章（调试用）")
  parser.add_argument("--no-collections", action="store_true", help="只转主书，跳过子集合（调试用）")
  parser.add_argument("--collection", default="", help="只转指定集合（逗号分隔的 slug）")
  parser.add_argument("--only-collection", default="",
                      help="只处理集合里的指定章节（逗号分隔的 chap_id）；会作用于本次选中的"
                           "每个集合，多集合时请配合 --collection")
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
  out_root = Path(args.out).resolve()
  work_root = Path(args.work).resolve()
  if not (book_dir / "book.tex").is_file():
    print(f"错误：{book_dir} 下没有 book.tex", file=sys.stderr)
    return 2

  started = time.monotonic()
  pandoc = preprocess.find_pandoc(args.pandoc or None)
  print(f"pandoc：{pandoc}")

  main_spec = BookSpec(slug="", title=BOOK_TITLE, book_dir=book_dir, book_tex="book.tex",
                       out_dir=out_root, work_dir=work_root)
  collection_specs = _collection_specs(args, book_dir, out_root, work_root)
  for spec in (main_spec, *collection_specs):
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    spec.work_dir.mkdir(parents=True, exist_ok=True)

  # 数学宏只从主书抽一次：``macros.extract`` 只认书根的 preamble.tex/book.tex，对子集合
  # 调用会得到空表并覆盖 build/mathjax-macros.json，主书公式静默变形。
  books = [_convert_book(main_spec, root, pandoc, args, only=args.only,
                         macro_target=root / "build" / "mathjax-macros.json")]
  for spec in collection_specs:
    books.append(_convert_book(spec, root, pandoc, args, only=args.only_collection))

  stage = time.monotonic()
  main_book = books[0]
  pages: dict[str, list[str]] = {}
  if not args.no_assemble:
    _assemble_site(main_book.parts, main_book.spec.out_dir, _converted(main_book),
                   [book.spec.slug for book in books[1:]])
    for book in books[1:]:
      if book.spec.collection is None:
        continue
      pages[book.spec.slug] = _assemble_collection(
        book.parts, book.spec.out_dir, _converted(book), book.spec.collection)
  # 装配清单始终重写：``--no-assemble`` 也写空清单，否则 verify 会读到上一次全量构建
  # 留下的清单，把「本次没有装配集合」误判成「集合页应该都在」。
  _write_assembly(root, [] if args.no_assemble else books[1:], pages)
  main_book.stages["装配"] = round(time.monotonic() - stage, 1)
  print(f"[4/4] 站点装配：{main_book.stages['装配']}s")

  if args.png:
    png_dir = root / "build" / "figures-png"
    print(f"PNG 导出：{export_png(out_root, png_dir)} 张 → {png_dir}")

  dangling, link_total = _check_links(out_root)
  if dangling:
    print(f"站内链接：{link_total} 条，其中 {len(dangling)} 条锚点缺失")
  for book in books:
    report = _write_report(
      root, book,
      dangling=dangling if not book.spec.slug else [],
      link_total=link_total if not book.spec.slug else 0)
    print(f"报告：{report}")
  print(f"总耗时：{time.monotonic() - started:.1f}s")
  return 0


def _collection_specs(
  args: argparse.Namespace,
  book_root: Path,
  out_root: Path,
  work_root: Path,
) -> list[BookSpec]:
  """按命令行参数挑出要转的子集合。

  ``--only`` / ``--max-chapters`` 是主书的调试开关，此时整站不完整，一并跳过集合
  （否则主书首页的 toctree 会引用不存在的文档，Sphinx 直接报警告）。

  :param args: 命令行参数。
  :param book_root: 主书根目录。
  :param out_root: 站点源树根目录。
  :param work_root: 临时目录根。
  :returns: 集合的书籍规格（按 ``COLLECTIONS`` 顺序）。
  """
  if args.no_collections or args.only or args.max_chapters > 0:
    return []
  wanted = {item.strip() for item in args.collection.split(",") if item.strip()}
  unknown = wanted - {collection.slug for collection in COLLECTIONS}
  if unknown:
    raise SystemExit(f"错误：未知集合 {sorted(unknown)}，可选 {[c.slug for c in COLLECTIONS]}")
  specs: list[BookSpec] = []
  for collection in COLLECTIONS:
    if wanted and collection.slug not in wanted:
      continue
    root_tex = book_root / collection.book_tex
    specs.append(BookSpec(
      slug=collection.slug,
      title=collection.title,
      book_dir=root_tex.parent,
      book_tex=root_tex.name,
      out_dir=out_root / collection.slug,
      work_dir=work_root / collection.slug,
      relative_dir=root_tex.parent.relative_to(book_root).as_posix(),
      collection=collection,
    ))
  return specs


def _convert_book(
  spec: BookSpec,
  root: Path,
  pandoc: str,
  args: argparse.Namespace,
  only: str = "",
  macro_target: Path | None = None,
) -> BookResult:
  """把一本书转成站点页面。

  :param spec: 书籍规格。
  :param root: ``docs/readthedocs`` 目录。
  :param pandoc: pandoc 路径。
  :param args: 命令行参数。
  :param only: 只处理这些 chap_id（逗号分隔，空串表示全部）。
  :param macro_target: 数学宏表输出路径（只给主书传）。
  :returns: 该书的转换结果。
  """
  started = time.monotonic()
  label = spec.slug or "主书"
  book_tex = spec.book_dir / spec.book_tex
  parts = booktree.parse(book_tex)
  chapters = [chapter for part in parts for chapter in part.chapters]
  if not chapters:
    raise SystemExit(f"错误：{book_tex} 没有解析出任何章节（\\input 只认 chapters/ 与 appendices/）")
  registry = labels.build(parts, spec.book_dir)
  if macro_target is not None:
    count = len(macros.write(spec.book_dir, macro_target))
    print(f"数学宏：{count} 条 → {macro_target.relative_to(root)}")
  selected = _select(chapters, only, args.max_chapters)
  print(f"[{label}] 章节：{len(chapters)} 篇，本次处理 {len(selected)} 篇；"
        f"label 注册表 {len(registry.entries)} 条")

  stages: dict[str, float] = {}
  stage = time.monotonic()
  manifests = [
    preprocess.normalize(chapter, spec.book_dir, spec.work_dir, spec.out_dir, registry)
    for chapter in selected
  ]
  stages["归一化"] = round(time.monotonic() - stage, 1)
  print(f"[{label}] [1/4] 归一化 {len(manifests)} 章：{stages['归一化']}s")

  stage = time.monotonic()
  tikz_report = tikz2svg.Report()
  snippets = tikz2svg.collect(manifests, spec.work_dir, spec.out_dir)
  pruned = tikz2svg.prune_stale(snippets, spec.work_dir, [chapter.chap_id for chapter in selected])
  if pruned:
    print(f"[{label}] [2/4] 清理陈旧片段 {len(pruned)} 个：{', '.join(pruned[:3])}"
          f"{' …' if len(pruned) > 3 else ''}")
  if args.skip_tikz:
    print(f"[{label}] [2/4] 跳过 TikZ 编译（--skip-tikz）")
  elif snippets:
    tikz_report = tikz2svg.build(
      snippets, spec.book_dir, spec.work_dir, force=args.force_tikz,
      font_mode=args.tikz_fonts, jobs=args.jobs)
    print(f"[{label}] [2/4] TikZ：{len(snippets)} 个片段，新编 {len(tikz_report.ok)}，"
          f"缓存 {len(tikz_report.skipped)}，失败 {len(tikz_report.failed)}，"
          f"模式 {tikz_report.mode}，{tikz_report.seconds}s")
    if not spec.slug:
      _build_english_atlas(root, spec, snippets, [chapter.chap_id for chapter in selected], args)
  stages["TikZ"] = round(time.monotonic() - stage, 1)

  stage = time.monotonic()
  results = []
  source_url = args.source_url or _SOURCE_TEMPLATE.format(ref=_git_ref())
  if spec.relative_dir:
    source_url = f"{source_url}/{spec.relative_dir}"
  # raw HTML 表格里的 ``<img>`` 不会被 Sphinx 改写路径，必须按页面层数写相对前缀。
  image_prefix = "../" * spec.depth + "_images/"
  md_dir = spec.work_dir / "md"
  for chapter, manifest in zip(selected, manifests):
    raw_md = md_dir / f"{chapter.chap_id}.md"
    warning = preprocess.to_markdown(spec.work_dir / "norm" / f"{chapter.chap_id}.tex",
                                     raw_md, pandoc)
    if warning.strip():
      manifest.notes.append(f"pandoc 警告：{warning.strip()[:400]}")
    ctx = postprocess.RenderContext(
      chap=chapter,
      parts=parts,
      manifest=manifest,
      registry=registry,
      work_dir=spec.work_dir,
      tikz_failed=set(tikz_report.failed),
      source_url=source_url,
      image_prefix=image_prefix,
    )
    result = postprocess.render_chapter(ctx, raw_md.read_text(encoding="utf-8"))
    (spec.out_dir / f"{chapter.chap_id}.md").write_text(result.markdown, encoding="utf-8")
    results.append((chapter, manifest, result))
  stages["渲染"] = round(time.monotonic() - stage, 1)
  print(f"[{label}] [3/4] 渲染 {len(results)} 章：{stages['渲染']}s")
  return BookResult(spec, parts, registry, results, tikz_report, stages,
                    time.monotonic() - started)


def _build_english_atlas(
  root: Path,
  spec: BookSpec,
  snippets: list,
  chapters: list[str],
  args: argparse.Namespace,
) -> None:
  """编译英文图集（只对主书做：词典按主书的图 id 建）。

  :param root: ``docs/readthedocs`` 目录。
  :param spec: 书籍规格。
  :param snippets: TikZ 片段。
  :param chapters: 本次转换的章标识。
  :param args: 命令行参数。
  """
  dictionary = figtext.load_dictionary(root / "i18n" / "figures-en.json")
  if not dictionary:
    return
  en_work = spec.work_dir / "tikz-en"
  en_tasks, hit, miss = tikz2svg.collect_translated(
    snippets, en_work, root / "_static" / "figures-en", dictionary)
  pruned = tikz2svg.prune_atlas(root / "_static" / "figures-en", en_tasks, chapters)
  if pruned:
    print(f"[2/4] 清理陈旧英文图 {len(pruned)} 张：{', '.join(pruned[:3])}"
          f"{' …' if len(pruned) > 3 else ''}")
  en_report = tikz2svg.build(
    en_tasks, spec.book_dir, en_work, force=args.force_tikz,
    font_mode=args.tikz_fonts, jobs=args.jobs)
  print(f"[2/4] 英文图集：{len(en_tasks)} 张（词典命中 {hit} 处、未命中 {miss} 处），"
        f"新编 {len(en_report.ok)}，缓存 {len(en_report.skipped)}，"
        f"失败 {len(en_report.failed)}，{en_report.seconds}s")


def _converted(book: BookResult) -> set[str]:
  """取出已转换的章 id 集合。

  :param book: 书籍转换结果。
  :returns: chap_id 集合。
  """
  return {chapter.chap_id for chapter, _, _ in book.results}


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


#: 站内链接：Markdown 的 ``[text](page.html#anchor)`` 与装配页里 raw HTML 的
#: ``<a href="../index.html">``（集合首页的「返回主书」）。前者带锚点，后者只查存在。
_MD_LINK_RE = re.compile(r"\]\(([A-Za-z0-9_./-]+)\.html(?:#([^)\s]+))?\)")
_RAW_LINK_RE = re.compile(r'<a\s[^>]*href="([A-Za-z0-9_./-]+\.html)"')


def _check_links(out_dir: Path) -> tuple[list[tuple[str, str]], int]:
  """检查站内链接的页面与锚点。

  MyST 看不见原始 HTML 写出的锚点，所以这里自己核对一遍：把每本书（含子集合子目录）
  每页出现的站内引用与各页面的 ``<a id>`` 集合求差，悬空引用进报告。引用按**页所在
  目录**解析——集合页之间的 ``reed-01-layout.html#…`` 指的是同目录的兄弟页。

  带锚点的引用同时查页与锚点，不带锚点的（装配页里手写的 raw HTML 链接）只查页：
  ``<a href>`` 不会被浏览器主动请求，`visual_check` 的 requestfailed 抓不到。

  :param out_dir: 站点源树。
  :returns: ``(悬空引用 [(页面, 引用)], 引用总数)``。
  """
  pages: set[str] = set()
  anchors: dict[str, set[str]] = {}
  for md in sorted(out_dir.rglob("*.md")):
    docname = md.relative_to(out_dir).with_suffix("").as_posix()
    pages.add(docname)
    anchors[docname] = set(re.findall(r'<a id="([^"]+)"', md.read_text(encoding="utf-8")))

  dangling: list[tuple[str, str]] = []
  total = 0
  for md in sorted(out_dir.rglob("*.md")):
    docname = md.relative_to(out_dir).with_suffix("").as_posix()
    base = posixpath.dirname(docname)
    text = md.read_text(encoding="utf-8")
    refs = [(page, anchor) for page, anchor in _MD_LINK_RE.findall(text)]
    refs.extend((target[:-len(".html")], "") for target in _RAW_LINK_RE.findall(text))
    for page, anchor in refs:
      target = posixpath.normpath(posixpath.join(base, page)) if base else page
      total += 1
      if target not in pages:
        dangling.append((docname, f"{page}.html"))
      elif anchor and anchor not in anchors[target]:
        dangling.append((docname, f"{page}.html#{anchor}"))
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


def _assemble_site(
  parts: list[booktree.Part],
  out_dir: Path,
  converted: set[str],
  collections: list[str] | None = None,
) -> None:
  """生成首页与各篇的装配页。

  :param parts: 篇结构。
  :param out_dir: 站点源树。
  :param converted: 已转换的 chap_id 集合。
  :param collections: 已装配集合的 slug，按顺序追加在篇之后（即侧栏末尾）。
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

  # 徽章写在标题前面：浮动元素只对后出现的行生效，放标题后面会落到副标题那一行。
  index = [BOOK_BADGE, "", f"# {BOOK_TITLE}", "", f"*{BOOK_SUBTITLE}*", "", BOOK_INTRO, ""]
  cover = _cover_image(out_dir)
  if cover:
    index.extend([f"![{BOOK_TITLE}]({cover})", ""])
  index.extend([BOOK_LINKS, "", "```{toctree}", ":maxdepth: 2", ":caption: 目录", ""])
  for _, page in part_pages:
    index.append(page)
  for slug in collections or []:
    index.append(f"{slug}/index")
  index.append("```")
  (out_dir / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")


def _assemble_collection(
  parts: list[booktree.Part],
  out_dir: Path,
  converted: set[str],
  collection: Collection,
) -> list[str]:
  """生成集合首页与各系列的装配页。

  :param parts: 集合的篇结构。
  :param out_dir: 集合的站点源树（``build/src/<slug>``）。
  :param converted: 已转换的 chap_id 集合。
  :param collection: 集合定义。
  :returns: 写出的页面（相对集合目录的 docname，``index`` 在首位）。
  """
  pages: list[str] = []
  for part in parts:
    available = [chapter for chapter in part.chapters if chapter.chap_id in converted]
    if not available:
      continue
    page = f"part-{part.index}"
    lines = [f"# {part.title or '目录'}", "", "```{toctree}", ":maxdepth: 1", ""]
    lines.extend(chapter.chap_id for chapter in available)
    lines.append("```")
    (out_dir / f"{page}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pages.append(page)

  index = [f"# {collection.title}", ""]
  for paragraph in collection.intro:
    index.extend([paragraph, ""])
  # 返回链接用 raw HTML：MyST 会把 ``[…](../index.html)`` 当文档引用去解析，
  # 解析不到就退化成 ``href="#../index.html"`` 的死链。
  index.extend([f'<a href="../index.html">返回《{BOOK_TITLE}》主书</a>', ""])
  index.extend(["```{toctree}", ":maxdepth: 2", ""])
  index.extend(pages)
  index.append("```")
  (out_dir / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
  return ["index", *pages]


def _write_assembly(root: Path, books: list[BookResult], pages: dict[str, list[str]]) -> Path:
  """落一份集合装配清单，供 ``convert.verify`` 核对入口与页面是否齐全。

  :param root: ``docs/readthedocs`` 目录。
  :param books: 集合的转换结果。
  :param pages: 集合 slug → 已写出的页面。
  :returns: 清单路径。
  """
  data = {
    "collections": [
      {
        "slug": book.spec.slug,
        "title": book.spec.title,
        "book_tex": f"{book.spec.relative_dir}/{book.spec.book_tex}".lstrip("/"),
        "pages": pages.get(book.spec.slug, []),
        "chapters": [chapter.chap_id for chapter, _, _ in book.results],
      }
      for book in books
    ]
  }
  target = root / "build" / "collections.json"
  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
  return target


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
  book: BookResult,
  dangling: list[tuple[str, str]] | None = None,
  link_total: int = 0,
) -> Path:
  """写一本书的转换报告（markdown + json，按书分名，互不覆盖）。

  :param root: ``docs/readthedocs`` 目录。
  :param book: 该书的转换结果。
  :param dangling: 悬空的站内引用（全站结果只记在主书报告里）。
  :param link_total: 站内引用总数（同上）。
  :returns: markdown 报告路径。
  """
  dangling = dangling or []
  report_dir = root / "build"
  report_dir.mkdir(parents=True, exist_ok=True)
  stem = "report" if not book.spec.slug else f"report-{book.spec.slug}"
  leftover_total: dict[str, int] = {}
  broken: dict[str, list[str]] = {}
  notes: list[str] = []
  code_total = image_total = tikz_total = 0
  for chapter, manifest, result in book.results:
    code_total += manifest.stats.get("code", 0)
    image_total += manifest.stats.get("image", 0)
    tikz_total += manifest.stats.get("tikz", 0)
    if result.broken_refs:
      broken[chapter.chap_id] = result.broken_refs
    for name, count in result.leftovers.items():
      leftover_total[name] = leftover_total.get(name, 0) + count
    notes.extend(manifest.notes)
    notes.extend(f"{chapter.chap_id}: {note}" for note in result.notes)

  stage_text = "；".join(f"{name} {value:.1f}s" for name, value in book.stages.items())
  total_chapters = sum(len(part.chapters) for part in book.parts)
  lines = [
    f"# {book.spec.title} RTD 转换报告",
    "",
    f"- 章节：{len(book.results)} / {total_chapters}",
    f"- TikZ 片段：{tikz_total}（新编 {len(book.tikz_report.ok)}，"
    f"缓存 {len(book.tikz_report.skipped)}，失败 {len(book.tikz_report.failed)}）",
    f"- 代码清单：{code_total}；镜像图片：{image_total}",
    f"- label 注册表：{len(book.registry.entries)} 条（重复 {len(book.registry.duplicates)}）",
    f"- 总耗时：{book.seconds:.1f}s（{stage_text}）",
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
  if book.tikz_report.failed:
    lines.append("| 图片 | 错误 |")
    lines.append("| --- | --- |")
    for image_id, message in book.tikz_report.failed.items():
      lines.append(f"| `{image_id}` | {message} |")
  else:
    lines.append("（无）")
  lines.extend(["", "## 站内链接自检", ""])
  if book.spec.slug:
    lines.extend(["- 集合页面之间的引用计入主书报告（全站一次扫描）", ""])
  else:
    lines.extend([f"- 站内引用 {link_total} 条，锚点缺失 {len(dangling)} 条", ""])
  if dangling:
    lines.append("| 所在页 | 悬空引用 |")
    lines.append("| --- | --- |")
    for page, ref in dangling[:60]:
      lines.append(f"| {page} | `{ref}` |")
  else:
    lines.append("（无）" if not book.spec.slug else "（见主书报告）")
  lines.extend(["", "## 残留 LaTeX 命令（pandoc 未转换）", ""])
  if leftover_total:
    lines.append("| 命令 | 次数 |")
    lines.append("| --- | --- |")
    for name, count in sorted(leftover_total.items(), key=lambda item: item[1], reverse=True)[:40]:
      lines.append(f"| `\\{name}` | {count} |")
  else:
    lines.append("（无）")
  lines.extend(["", "## 逐章明细", "", "| 章 | 标题 | TikZ | 代码 | 图片 | 未解析引用 |", "| --- | --- | --- | --- | --- | --- |"])
  for chapter, manifest, result in book.results:
    lines.append(
      f"| {chapter.chap_id} | {chapter.title} | {manifest.stats.get('tikz', 0)} | "
      f"{manifest.stats.get('code', 0)} | {manifest.stats.get('image', 0)} | {len(result.broken_refs)} |")
  lines.extend(["", "## 提示", ""])
  lines.extend([f"- {note}" for note in notes[:80]] or ["（无）"])
  report_path = report_dir / f"{stem}.md"
  report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  (report_dir / f"{stem}.json").write_text(json.dumps({
    "book": book.spec.slug or "main",
    "chapters": len(book.results),
    "tikz_total": tikz_total,
    "tikz_ok": len(book.tikz_report.ok),
    "tikz_cached": len(book.tikz_report.skipped),
    "tikz_failed": book.tikz_report.failed,
    "code_total": code_total,
    "image_total": image_total,
    "labels": len(book.registry.entries),
    "broken_refs": broken,
    "dangling_anchors": dangling,
    "link_total": link_total,
    "leftovers": leftover_total,
    "seconds": round(book.seconds, 1),
    "stages": book.stages,
    "tikz_seconds": book.tikz_report.seconds,
    "tikz_mode": book.tikz_report.mode,
    "tikz_font_mode": book.tikz_report.font_mode,
    "notes": notes,
  }, ensure_ascii=False, indent=2), encoding="utf-8")
  return report_path


if __name__ == "__main__":
  sys.exit(main())
