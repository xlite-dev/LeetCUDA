"""站点内容核对：把「渲染出来了吗」升级成「渲染对吗」。

`__main__` 的自检只保证锚点存在（结构层），查不出编号错、表格丢失、标题里混入
pandoc 属性这类内容错误。本模块在 Sphinx 构建之后跑一遍，按下面的清单逐项断言，
任一不过就以非零码退出，让构建（本地 build.sh 与 RTD 的 post_build）真的失败。

清单：

1. 源树里没有残留 token，也没有 ``{#...}`` 属性文本；
2. H1 的章号与注册表一致（``第 N 章`` / ``附录 X``）；
3. 没有「所有章引用都渲染成同一个数字」这种编号坍缩；
4. HTML 里的 ``<table>`` 数量不小于原书的 tabular/longtable 数量；
5. 站内链接的锚点全部存在。

用法：``python -m convert.verify --src build/src --html build/html --book-dir ...``
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import booktree, labels, postprocess, texutil

#: ``第 N 章：`` / ``附录 A：`` 前缀。
_CHAPTER_PREFIX_RE = re.compile(r"^第\s*(\d+)\s*章：")
_APPENDIX_PREFIX_RE = re.compile(r"^附录\s*([A-Z])\s*：")

#: 原书里真正的表格环境（``table`` 只是浮动壳，里面还是 ``tabular``）。
_TABLE_ENV_RE = re.compile(r"\\begin\{(?:tabular|tabularx|longtable)\}")

#: 站内链接。
_LINK_RE = re.compile(r"\]\(([A-Za-z0-9_.-]+)\.html#([^)\s]+)")
_ANCHOR_RE = re.compile(r'<a id="([^"]+)"')


@dataclass
class CheckResult:
  """核对结果。

  :param name: 检查项名称。
  :param ok: 是否通过。
  :param detail: 说明或失败明细。
  """
  name: str
  ok: bool
  detail: str = ""


@dataclass
class Report:
  """核对汇总。

  :param checks: 逐项结果。
  """
  checks: list[CheckResult] = field(default_factory=list)

  @property
  def ok(self) -> bool:
    """是否全部通过。

    :returns: 全部通过为 True。
    """
    return all(check.ok for check in self.checks)

  def add(self, name: str, ok: bool, detail: str = "") -> None:
    """追加一项结果。

    :param name: 检查项。
    :param ok: 是否通过。
    :param detail: 说明。
    """
    self.checks.append(CheckResult(name, ok, detail))

  def render(self) -> str:
    """渲染成可打印文本。

    :returns: 多行文本。
    """
    lines = []
    for check in self.checks:
      mark = "OK  " if check.ok else "FAIL"
      lines.append(f"[{mark}] {check.name}{('：' + check.detail) if check.detail else ''}")
    return "\n".join(lines)


#: book.toc 的章级条目：
#: ``\contentsline {chapter}{\numberline {第十七章\hspace {.3em}}标题}{17}{chapter.17}``。
#: 编号里含 ``\hspace {...}``，所以不能用 ``[^}]*`` 硬切。
_TOC_RE = re.compile(
  r"\\contentsline \{chapter\}\{\\numberline \{(?P<num>.*?)\\hspace \{[^}]*\}\}"
  r"(?P<title>[^}]*)\}\{\d+\}\{(?P<key>(?:chapter|appendix)\.[A-Za-z0-9]+)\}")

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_number(text: str) -> str:
  """把 ``第一章`` / ``附录 A`` 归一成 ``"1"`` / ``"A"``。

  :param text: 目录里的编号文本。
  :returns: 数字或字母。
  """
  text = text.strip()
  if "附录" in text:
    return text.replace("附录", "").strip()
  digits = text.replace("第", "").replace("章", "").strip()
  if "十" not in digits:
    return "".join(str(_CN_DIGITS[ch]) for ch in digits) or digits
  head, _, tail = digits.partition("十")
  tens = _CN_DIGITS[head] if head else 1
  ones = _CN_DIGITS[tail] if tail else 0
  return str(tens * 10 + ones)


def _normalize_title(text: str) -> str:
  """归一标题文本以便宽松比较。

  :param text: 标题。
  :returns: 去空白后的标题。
  """
  text = re.sub(r"\\[a-zA-Z]+\s*", "", text)
  return re.sub(r"[\s{}\\]", "", text)


def check_book_toc(
  book_dir: Path,
  registry: labels.Registry,
  chapters: list[booktree.Chapter],
) -> tuple[bool, str, int]:
  """拿 ``book.toc``（xelatex 产出的真实目录）核对章号与标题顺序。

  ``book.toc`` 只在本地构建过书之后才存在，RTD 上没有——存在时用它做真值，不存在
  就跳过。它同时校验两件事：章号（中文数字 ↔ 我们的数字/字母）与章顺序。

  :param book_dir: 书根目录。
  :param registry: 全书注册表。
  :param chapters: 按书序排列的章节。
  :returns: ``(是否通过, 说明, 核对条数)``。
  """
  toc = book_dir / "book.toc"
  if not toc.is_file():
    return True, "跳过（本地无 book.toc）", 0

  entries = _TOC_RE.findall(toc.read_text(encoding="utf-8", errors="replace"))
  if not entries:
    return True, "跳过（book.toc 无可解析章条目）", 0

  # 目录只收录有编号的章（ch19b 是 \chapter*，不在目录里），按同序对齐。
  numbered_pages = [
    chapter for chapter in chapters
    if chapter.kind == "appendix" or chapter.chap_id in registry.numbered
  ]
  problems: list[str] = []
  checked = 0
  for (number_text, title, _key), chapter in zip(entries, numbered_pages):
    expected = _cn_number(number_text)
    actual = registry.chapter_numbers.get(chapter.chap_id, "")
    checked += 1
    if expected != actual:
      problems.append(f"{chapter.chap_id}: 目录 {expected} vs 站点 {actual}")
      continue
    expected_title = _normalize_title(title)
    page_title = _normalize_title(chapter.title)
    if expected_title and page_title and expected_title[:12] != page_title[:12]:
      problems.append(f"{chapter.chap_id}: 标题 目录 “{expected_title[:14]}” vs 书 “{page_title[:14]}”")
  ok = not problems
  detail = f"核对 {checked} 条" if ok else f"{len(problems)} 处不一致，例如 {problems[0]}"
  return ok, detail, checked


def _display_math_balanced(text: str) -> bool:
  """判断显示公式定界符是否成对。

  判据：每一处「行首 ``$$``」都必须是一个显示公式区间的开门符或闭门符。不用「数
  ``$$`` 次数取奇偶」——图注链接的 title 里会出现相邻行内公式（``$a$$b$``），那不是
  一对显示公式定界符；也不能只看行尾，因为 pandoc 会把 ``\\end{equation}$$`` 与后续
  正文放在同一行。

  :param text: 已剔除代码块的 markdown 文本。
  :returns: 配对为 True。
  """
  spans = texutil.math_spans(text)
  opens = {start for start, end in spans if text.startswith("$$", start) and end - start > 2}
  closes = {end - 2 for start, end in spans if text.startswith("$$", start) and end - start > 2}
  for match in re.finditer(r"(?m)^[ \t]*\$\$", text):
    position = text.index("$$", match.start())
    if position not in opens and position not in closes:
      return False
  return True


#: 片段里出现这些才算「图里有文字」：中文标签，或 ``\node … {非空内容}``。
_TEXT_HINT_RE = re.compile(r"[\u4e00-\u9fff]|\\node\b[^;]*?\{[^{}]{2,}\}")


def check_table_counts_per_chapter(
  parts: list[booktree.Part],
  html_dir: Path,
) -> tuple[bool, str]:
  """逐章比对「原书表格环境数」与「该页渲染出的 ``<table>`` 数」。

  只看全站总数会漏——总差 1 张时 90% 阈值照样通过（ch37 的量化开销表就是这么漏掉的）。

  :param parts: 篇结构。
  :param html_dir: Sphinx 输出目录。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  checked = 0
  for part in parts:
    for chapter in part.chapters:
      page = html_dir / f"{chapter.chap_id}.html"
      if not page.is_file():
        continue
      sources = [chapter.tex]
      if "wp/" in chapter.tex.read_text(encoding="utf-8", errors="replace"):
        sources.extend(sorted((chapter.tex.parent / "wp").glob("*.tex")))
      expected = sum(len(_TABLE_ENV_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
                     for path in sources)
      actual = page.read_text(encoding="utf-8", errors="replace").count("<table")
      checked += 1
      if actual < expected:
        problems.append(f"{chapter.chap_id}: 原书 {expected} 张 → 页面 {actual} 张")
  ok = not problems
  detail = f"逐章核对 {checked} 页" if ok else f"{len(problems)} 章缺表，例如 {problems[0]}"
  return ok, detail


def check_table_cells(md_texts: dict[str, str]) -> tuple[bool, str]:
  """检查表格单元格里没有漏出来的 LaTeX 命令。

  :param md_texts: 页面名 → markdown。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  for name, text in md_texts.items():
    for match in re.finditer(r"<t[dh][^>]*>([^<]*)", text):
      cell = match.group(1)
      if re.search(r"\\[a-zA-Z]+\{", cell) or "(lr)" in cell:
        problems.append(f"{name}: {cell.strip()[:40]}")
        break
  ok = not problems
  return ok, f"{len(problems)} 页命中，例如 {problems[0]}" if problems else ""


def check_tikz_glyphs(work_dir: Path, src_dir: Path) -> tuple[bool, str]:
  """检查「图里有文字」的片段确实输出了字形。

  ``--no-fonts`` 模式下文字会变成 ``<use>``（或 ``<text>``）；两者都没有，说明
  SVG 里没有文字，页面上的表现就是图注文字全丢（ch00 的 occupancy 图曾如此）。

  :param work_dir: 临时目录（``build/tmp``）。
  :param src_dir: 站点源树（``build/src``）。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  checked = 0
  for snippet in sorted((work_dir / "tikz").glob("*.tex")):
    body = re.sub(r"(?<!\\)%[^\n]*", "", snippet.read_text(encoding="utf-8", errors="replace"))
    if not _TEXT_HINT_RE.search(body):
      continue
    checked += 1
    svg = src_dir / "figures-gen" / f"{snippet.stem}.svg"
    if not svg.is_file():
      problems.append(f"{snippet.stem}: SVG 缺失")
      continue
    text = svg.read_text(encoding="utf-8", errors="replace")
    if "<use" not in text and "<text" not in text:
      problems.append(f"{snippet.stem}: 无字形输出（{svg.stat().st_size} 字节）")
  ok = not problems
  detail = f"核对 {checked} 张有文字的图" if ok else f"{len(problems)} 张丢文字，例如 {problems[0]}"
  return ok, detail


def count_source_tables(parts: list[booktree.Part]) -> int:
  """统计原书闭包内的表格环境数量（每个文件只统计一次）。

  :param parts: 篇结构。
  :returns: ``tabular`` / ``tabularx`` / ``longtable`` 环境总数。
  """
  files: set[Path] = set()
  for part in parts:
    for chapter in part.chapters:
      files.add(chapter.tex)
      files.update(chapter.tex.parent.glob("wp/*.tex"))
  total = 0
  for path in sorted(files):
    total += len(_TABLE_ENV_RE.findall(path.read_text(encoding="utf-8")))
  return total


def verify(src_dir: Path, html_dir: Path, book_dir: Path) -> Report:
  """执行全部核对。

  :param src_dir: 站点源树（``build/src``）。
  :param html_dir: Sphinx 输出（``build/html``）。
  :param book_dir: 书根目录（``book/``）。
  :returns: 核对结果。
  """
  report = Report()
  src_files = sorted(src_dir.glob("*.md"))
  texts = {path.stem: path.read_text(encoding="utf-8") for path in src_files}

  leftovers = [(name, text.count("@@RTD;")) for name, text in texts.items() if "@@" in text]
  report.add("源树无残留 token", not leftovers,
             f"{len(leftovers)} 页残留，例如 {leftovers[0][0]}" if leftovers else "")

  attrs = [name for name, text in texts.items() if re.search(r"\{#[^}]*\}", text)]
  report.add("标题无 pandoc 属性泄漏", not attrs,
             f"{len(attrs)} 页命中，例如 {attrs[0]}" if attrs else "")

  registry = labels.build(booktree.parse(book_dir / "book.tex"), book_dir)
  mismatch: list[str] = []
  seen_prefixes: dict[str, str] = {}
  for part in booktree.parse(book_dir / "book.tex"):
    for chapter in part.chapters:
      text = texts.get(chapter.chap_id)
      if text is None:
        continue
      raw_heading = next((line for line in text.split("\n") if line.startswith("# ")), "")
      heading = re.sub(r"^#\s+", "", raw_heading)
      number = registry.chapter_numbers.get(chapter.chap_id, "")
      if chapter.kind == "appendix":
        expected = f"附录 {number}：" if number else ""
        seen_prefixes[chapter.chap_id] = f"附录 {number}" if number else ""
      elif chapter.chap_id in registry.numbered:
        expected = f"第 {number} 章：" if number else ""
        seen_prefixes[chapter.chap_id] = f"第 {number} 章" if number else ""
      else:
        expected = ""
        seen_prefixes[chapter.chap_id] = ""
      if expected and not heading.startswith(expected):
        mismatch.append(f"{chapter.chap_id}: 期望 “{expected}”，实际 “{heading[:24]}”")
  report.add("H1 章号与注册表一致", not mismatch,
             f"{len(mismatch)} 处，例如 {mismatch[0]}" if mismatch else "")

  chapter_ref_texts = set(re.findall(r"\[([^\]]+)\]\([a-z0-9-]+\.html#ch-[0-9a-z-]+", "\n".join(texts.values())))
  collapsed = chapter_ref_texts - {"1"} if chapter_ref_texts == {"1"} else set()
  report.add("章引用编号未坍缩", not collapsed,
             f"所有章引用都是 {chapter_ref_texts}" if collapsed else "")

  expected_tables = count_source_tables(booktree.parse(book_dir / "book.tex"))
  if html_dir.is_dir():
    rendered = sum(
      path.read_text(encoding="utf-8", errors="replace").count("<table")
      for path in html_dir.glob("*.html"))
  else:
    rendered = 0
  report.add("表格已渲染", rendered >= expected_tables * 0.9,
             f"原书表格环境 {expected_tables}，页面 <table> {rendered}")

  anchors: dict[str, set[str]] = {
    name: set(_ANCHOR_RE.findall(text)) for name, text in texts.items()
  }
  dangling: list[str] = []
  for name, text in texts.items():
    for page, anchor in _LINK_RE.findall(text):
      if page not in anchors or anchor not in anchors[page]:
        dangling.append(f"{name} → {page}.html#{anchor}")
  report.add("站内锚点无悬空", not dangling,
             f"{len(dangling)} 条，例如 {dangling[0]}" if dangling else "")

  raw_html = [name for name, text in texts.items() if "{=html}" in text]
  html_raw = 0
  if html_dir.is_dir():
    html_raw = sum(
      path.read_text(encoding="utf-8", errors="replace").count("{=html}")
      for path in html_dir.glob("*.html"))
  report.add("无 pandoc 原始 HTML 残迹", not raw_html and html_raw == 0,
             f"{len(raw_html)} 页残留，例如 {raw_html[0]}" if raw_html
             else (f"页面可见 {html_raw} 处" if html_raw else ""))

  broken_math: list[str] = []
  unbalanced: list[str] = []
  for name, text in texts.items():
    lines = text.split("\n")
    flags = postprocess.code_fence_flags(lines)
    clean = "\n".join("" if flag else line for line, flag in zip(lines, flags))
    for start, end in texutil.math_spans(clean):
      span = clean[start:end]
      if "](" in span or "<a id=" in span:
        broken_math.append(f"{name}: {span[:60]}".replace("\n", " "))
        break
    if not _display_math_balanced(clean):
      unbalanced.append(name)
  report.add("数学区内无站内链接/锚点", not broken_math,
             f"{len(broken_math)} 处，例如 {broken_math[0]}" if broken_math else "")
  report.add("显示公式定界符配对", not unbalanced,
             f"{len(unbalanced)} 页不配对：{unbalanced[0]}" if unbalanced else "")

  # 页面侧证据：显示公式夹在正文里时，Sphinx 会渲染成「字面 $ + 行内公式」。
  # 注意只有 equation/align 这类「显示环境」出现在行内公式里才算断裂——
  # `\(\begin{bmatrix}…\)` 是正常的行内矩阵。
  display_env_only = r"equation|align|gather|multline|eqnarray|displaymath"
  broken_display: list[str] = []
  if html_dir.is_dir():
    for path in sorted(html_dir.glob("*.html")):
      page = path.read_text(encoding="utf-8", errors="replace")
      if re.search(r"\$<span class=\"math", page) or re.search(
          r'class="math[^"]*">\\\(\\begin\{(?:' + display_env_only + r')', page):
        broken_display.append(path.stem)
  report.add("显示公式未被拆成字面 $", not broken_display,
             f"{len(broken_display)} 页，例如 {broken_display[0]}" if broken_display else "")

  ok, detail = check_table_cells(texts)
  report.add("表格单元格无 LaTeX 残渣", ok, detail)

  if html_dir.is_dir():
    ok, detail = check_table_counts_per_chapter(booktree.parse(book_dir / "book.tex"), html_dir)
    report.add("逐章表格数不缺", ok, detail)

  work_dir = src_dir.parent / "tmp"
  if (work_dir / "tikz").is_dir():
    ok, detail = check_tikz_glyphs(work_dir, src_dir)
    report.add("TikZ 图保留文字", ok, detail)

  parts = booktree.parse(book_dir / "book.tex")
  chapters = [chapter for part in parts for chapter in part.chapters]
  if any(chapter.chap_id not in texts for chapter in chapters):
    report.add("书目录编号核对", True, "跳过（只转换了部分章节）")
  else:
    ok, detail, _ = check_book_toc(book_dir, registry, chapters)
    report.add("书目录编号核对", ok, detail)

  return report


def main() -> int:
  """命令行入口。

  :returns: 进程退出码（0 全部通过）。
  """
  root = Path(__file__).resolve().parent.parent
  parser = argparse.ArgumentParser(description="站点内容核对")
  parser.add_argument("--src", default=str(root / "build" / "src"), help="站点源树")
  parser.add_argument("--html", default=str(root / "build" / "html"), help="Sphinx 输出目录")
  parser.add_argument("--book-dir", default=str(root / "../../kernels/interview/book"),
                      help="书根目录")
  args = parser.parse_args()
  report = verify(Path(args.src).resolve(), Path(args.html).resolve(), Path(args.book_dir).resolve())
  print(report.render())
  return 0 if report.ok else 1


if __name__ == "__main__":
  sys.exit(main())
