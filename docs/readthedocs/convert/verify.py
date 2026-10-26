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
import html
import json
import posixpath
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import booktree, figtext, labels, postprocess, preprocess, readme, texutil
from .collections import COLLECTIONS

#: ``第 N 章：`` / ``附录 A：`` 前缀。
_CHAPTER_PREFIX_RE = re.compile(r"^第\s*(\d+)\s*章：")
_APPENDIX_PREFIX_RE = re.compile(r"^附录\s*([A-Z])\s*：")

#: 原书里真正的表格环境（``table`` 只是浮动壳，里面还是 ``tabular``）。
_TABLE_ENV_RE = re.compile(r"\\begin\{(?:tabular|tabularx|longtable)\}")

#: 站内链接（页名可带子目录，如集合页里的 ``cute-zhihu/x.html#…``）。
_LINK_RE = re.compile(r"\]\(([A-Za-z0-9_./-]+)\.html#([^)\s]+)")
_ANCHOR_RE = re.compile(r'<a id="([^"]+)"')

#: 侧栏「下载 PDF」按钮（``_templates/layout.html`` 渲染的静态锚点，按类名认）。
#: 先取整条 ``<a …>`` 开标签再取 ``href``，免得受属性顺序影响。
_PDF_BUTTON_CLASS = "rtd-pdf-download"
_REPO_BUTTON_CLASS = "rtd-repo-link"
_ANCHOR_OPEN_RE = re.compile(r"<a\b[^>]*>", re.IGNORECASE)
_HREF_RE = re.compile(r'\bhref="([^"]*)"', re.IGNORECASE)


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
  book_tex: str = "book.tex",
) -> tuple[bool, str, int]:
  """拿 ``<书>.toc``（xelatex 产出的真实目录）核对章号与标题顺序。

  ``.toc`` 只在本地构建过书之后才存在，RTD 上没有——存在时用它做真值，不存在就跳过。
  它同时校验两件事：章号（中文数字 ↔ 我们的数字/字母）与章顺序。

  :param book_dir: 书根目录。
  :param registry: 全书注册表。
  :param chapters: 按书序排列的章节。
  :param book_tex: 根 tex 文件名（目录文件名跟它同名，如 ``cute-zhihu.toc``）。
  :returns: ``(是否通过, 说明, 核对条数)``。
  """
  toc = book_dir / Path(book_tex).with_suffix(".toc")
  if not toc.is_file():
    return True, f"跳过（本地无 {toc.name}）", 0

  entries = _TOC_RE.findall(toc.read_text(encoding="utf-8", errors="replace"))
  if not entries:
    return True, f"跳过（{toc.name} 无可解析章条目）", 0

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

#: 页面里引用 TikZ 产物的写法：``figures-gen/<image_id>.svg``。
_FIGURE_REF_RE = re.compile(r"figures-gen/([A-Za-z0-9_\-]+)\.svg")


def referenced_figures(src_dir: Path) -> set[str]:
  """页面实际引用的 TikZ 产物名。

  以页面引用为准，而不是以 ``build/tmp/tikz`` 目录为准：缓存里可能留着已删图的旧片段，
  按目录枚举会把它们算进核对（ch20 的 ``-t011`` 曾让「TikZ 图保留文字」误报）。

  :param src_dir: 站点源树（``build/src``）。
  :returns: image_id 集合。
  """
  names: set[str] = set()
  for page in src_dir.rglob("*.md"):
    names |= set(_FIGURE_REF_RE.findall(page.read_text(encoding="utf-8", errors="replace")))
  return names


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

  只核对页面实际引用的图：缓存里可能留着已删图的旧片段（``tikz2svg.prune_stale``
  之外的遗留），那不能说明页面有问题。

  :param work_dir: 临时目录（``build/tmp``）。
  :param src_dir: 站点源树（``build/src``）。
  :returns: ``(是否通过, 说明)``。
  """
  referenced = referenced_figures(src_dir)
  problems: list[str] = []
  checked = 0
  for snippet in sorted((work_dir / "tikz").glob("*.tex")):
    if snippet.stem not in referenced:
      continue
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


#: ``\text{…$…$…}``：数学区里嵌套 ``$``，MyST 会把它切成两个数学节点。
_NESTED_MATH_RE = re.compile(
  r"\\(?:text|textrm|textbf|textit|texttt|textsf|textnormal)\{[^{}]*\$")


def check_nested_math(md_texts: dict[str, str]) -> tuple[bool, str]:
  """检查公式里没有嵌套的 ``$``。

  :param md_texts: 页面名 → markdown。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  for name, text in md_texts.items():
    lines = text.split("\n")
    flags = postprocess.code_fence_flags(lines)
    clean = "\n".join("" if flag else line for line, flag in zip(lines, flags))
    match = _NESTED_MATH_RE.search(clean)
    if match is not None:
      problems.append(f"{name}: {match.group(0)[:40]}")
  ok = not problems
  return ok, f"{len(problems)} 页命中，例如 {problems[0]}" if problems else ""


#: raw HTML 表格块（MyST 原样透传，里面的 Markdown 不会被解析）。
_HTML_TABLE_BLOCK_RE = re.compile(r"<table\b.*?</table>", re.S)


def check_table_inline_markdown(md_texts: dict[str, str]) -> tuple[bool, str]:
  """检查 raw HTML 表格里没有残留 Markdown 行内语法。

  :param md_texts: 页面名 → markdown。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  for name, text in md_texts.items():
    for block in _HTML_TABLE_BLOCK_RE.finditer(text):
      hit = re.search(r"\]\(|\*\*|(?<!\*)\*[^*\n]+\*(?!\*)", block.group(0))
      if hit is not None:
        problems.append(f"{name}: {hit.group(0)[:30]}")
        break
  ok = not problems
  return ok, f"{len(problems)} 页命中，例如 {problems[0]}" if problems else ""


#: 页面里的图片引用：子目录页面写成 ``../_images/x.jpg``，前缀要一起捕获。
_IMAGE_SRC_RE = re.compile(r'src="((?:\.\./)*)((?:figures-gen|_images)/[^"]+)"')


def check_images_present(html_dir: Path) -> tuple[bool, str]:
  """检查页面引用的图片文件确实存在。

  raw HTML 里的 ``<img>`` 不会被 Sphinx 搬运，少了隐藏引用就会 404（表格里的图曾如此）。
  路径按**页面所在目录**解析：集合页在子目录，Sphinx 输出的前缀是 ``../_images/``。

  :param html_dir: Sphinx 输出目录。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  checked = 0
  for path in sorted(html_dir.rglob("*.html")):
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in _IMAGE_SRC_RE.finditer(text):
      checked += 1
      source = match.group(1) + match.group(2)
      if not (path.parent / source).resolve().is_file():
        problems.append(f"{path.relative_to(html_dir).as_posix()}: {source}")
  ok = not problems
  return ok, (f"共 {checked} 张图" if ok
              else f"{len(problems)} 张缺失，例如 {problems[0]}")


def check_pdf_button(html_dir: Path, expected_url: str) -> tuple[bool, str]:
  """检查每页都渲染了「下载 PDF」按钮，且链接与仓库 README 的解析结果一致。

  按钮是模板渲染出的静态锚点：正文页在 ``_templates/breadcrumbs.html`` / ``footer.html``
  里紧挨「下一页」，没有上一页/下一页按钮行的页面（``genindex`` / ``search``）由
  ``_templates/layout.html`` 在侧栏兜底。链接来自 ``README.md`` 的 ``[leetcuda-pdf]``。
  这里逐页比对：按钮是站点上唯一的 PDF 入口，链接写死成旧版本或者整个按钮在构建中消失，
  读者都拿不到书，而页面本身看不出异常。

  :param html_dir: Sphinx 输出目录。
  :param expected_url: ``README.md`` 里 ``[leetcuda-pdf]`` 的地址。
  :returns: ``(是否通过, 说明)``。
  """
  return _check_page_button(html_dir, _PDF_BUTTON_CLASS, expected_url, "按钮")


def _button_hrefs(page: str, css_class: str) -> list[str]:
  """取页面里带指定 class 的锚点 href。

  :param page: 页面 HTML。
  :param css_class: 锚点的 class 标记。
  :returns: 反转义后的 href 列表（缺 href 记空串）。
  """
  found: list[str] = []
  for tag in _ANCHOR_OPEN_RE.findall(page):
    if css_class not in tag:
      continue
    href = _HREF_RE.search(tag)
    # 上游若开了模板 autoescape，``&`` 会写成 ``&amp;``；比对前统一反转义（当前设置下
    # 是无操作——Sphinx 7.4 的模板环境没开 autoescape，但别依赖这点）。
    found.append(html.unescape(href.group(1)) if href else "")
  return found


def _check_page_button(html_dir: Path, css_class: str, expected_url: str,
                       label: str) -> tuple[bool, str]:
  """逐页核对常驻按钮存在且链接一致。

  :param html_dir: Sphinx 输出目录。
  :param css_class: 按钮锚点的 class 标记。
  :param expected_url: 期望的链接。
  :param label: 核对项里对按钮的称呼。
  :returns: ``(是否通过, 说明)``。
  """
  pages = sorted(html_dir.rglob("*.html"))
  missing: list[str] = []
  mismatch: list[str] = []
  for path in pages:
    page = path.read_text(encoding="utf-8", errors="replace")
    name = path.relative_to(html_dir).as_posix()
    found = _button_hrefs(page, css_class)
    if not found:
      missing.append(name)
    elif any(url != expected_url for url in found):
      mismatch.append(f"{name} → {found[0]}")
  problems: list[str] = []
  if not pages:
    problems.append(f"{html_dir} 下没有页面")
  if missing:
    problems.append(f"{len(missing)} 页没有{label}，例如 {missing[0]}")
  if mismatch:
    problems.append(f"{len(mismatch)} 页{label}链接不符（期望 {expected_url}），例如 {mismatch[0]}")
  return not problems, ("；".join(problems) if problems else f"{len(pages)} 页均指向同一链接")


def check_repo_button(html_dir: Path, expected_url: str) -> tuple[bool, str]:
  """检查每页右上角都渲染了「Go to repository」，且链接指向仓库主页。

  按钮写在 ``_templates/breadcrumbs.html`` 的 ``breadcrumbs_aside`` 位（每页都有，
  含 ``genindex`` / ``search``）。链接来自 ``convert.readme.REPO_URL``；主题升级后
  模板被顶掉、或链接写错，页面本身看不出异常，这里逐页兜住。

  :param html_dir: Sphinx 输出目录。
  :param expected_url: 仓库主页地址。
  :returns: ``(是否通过, 说明)``。
  """
  return _check_page_button(html_dir, _REPO_BUTTON_CLASS, expected_url, "仓库按钮")


def check_english_figures(
  snippet_dir: Path,
  en_dir: Path,
  dictionary_path: Path,
  referenced: set[str] | None = None,
) -> tuple[bool, str]:
  """检查英文图集：词典覆盖、成品数量与字形。

  英文图集是构建期按词典重编译的 SVG（``_static/figures-en/``），英文模式下由前端换图。
  这里核对三件事：词典是否覆盖全部单元、**全部命中**的图是否都编出来了、成品里有没有
  字形（空 SVG 在页面上就是一张空白图）。

  :param snippet_dir: TikZ 片段目录（``build/tmp/tikz``）。
  :param en_dir: 英文图集目录。
  :param dictionary_path: 词典路径。
  :param referenced: 页面实际引用的图；给出时只核对它们，忽略缓存里的陈旧片段。
  :returns: ``(是否通过, 说明)``。
  """
  dictionary = figtext.load_dictionary(dictionary_path)
  if not dictionary:
    return True, "无词典，跳过"
  snippets = sorted(snippet_dir.glob("*.tex"))
  if referenced is not None:
    snippets = [path for path in snippets if path.stem in referenced]
  if not snippets:
    return True, "无片段，跳过"
  covered, total, _missing = figtext.coverage(snippets, dictionary)
  problems: list[str] = []
  if covered != total:
    problems.append(f"词典覆盖 {covered}/{total}")
  expected = 0
  for path in snippets:
    body = figtext.strip_comments(path.read_text(encoding="utf-8"))
    _english, hit, miss = figtext.translate_snippet(body, dictionary)
    if hit and not miss:
      expected += 1
  produced = sorted(en_dir.glob("*.svg")) if en_dir.is_dir() else []
  if len(produced) < expected:
    problems.append(f"英文图 {len(produced)}/{expected} 张")
  empty = [path.stem for path in produced if "<use" not in path.read_text(
    encoding="utf-8", errors="replace") and "<text" not in path.read_text(
    encoding="utf-8", errors="replace")]
  if empty:
    problems.append(f"{len(empty)} 张无字形（如 {empty[0]}）")
  ok = not problems
  detail = f"词典 {len(dictionary)} 条，英文图 {len(produced)}/{expected} 张" if ok else "；".join(problems)
  return ok, detail


def check_multirow_headers(
  parts: list[booktree.Part],
  md_texts: dict[str, str],
) -> tuple[bool, str]:
  """逐章核对「多行表头」是否收进了 ``<thead>``。

  pandoc 的 LaTeX reader 只支持单行表头，源文「两行表头 + ``\\midrule``」的表第二行
  会落进表体（ch33/ch35/ch37 的扫描表）。转换器按源文行数把它们收进 ``<thead>``，
  这里按章核对数量，防止回归。

  :param parts: 篇结构。
  :param md_texts: 页面名 → markdown。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  checked = 0
  for part in parts:
    for chapter in part.chapters:
      text = md_texts.get(chapter.chap_id)
      if text is None:
        continue
      sources = [chapter.tex]
      if "wp/" in chapter.tex.read_text(encoding="utf-8", errors="replace"):
        sources.extend(sorted((chapter.tex.parent / "wp").glob("*.tex")))
      expected = 0
      for path in sources:
        body = texutil.strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        for _start, _end, tabular in texutil.find_environments(body, "tabular"):
          if preprocess._table_header_rows(tabular) >= 2:
            expected += 1
      actual = 0
      for block in re.finditer(r"<table\b.*?</table>", text, re.S):
        head = re.search(r"<thead>.*?</thead>", block.group(0), re.S)
        if head and len(re.findall(r"<tr", head.group(0))) >= 2:
          actual += 1
      checked += expected
      if actual < expected:
        problems.append(f"{chapter.chap_id}: 源文 {expected} 张 → 页面 {actual} 张")
  ok = not problems
  detail = f"核对 {checked} 张多行表头表" if ok else f"{len(problems)} 章未收拢，例如 {problems[0]}"
  return ok, detail


def check_emphasis_flanking(md_texts: dict[str, str]) -> tuple[bool, str]:
  """检查正文里没有「配不上对的强调标记」。

  CommonMark 的 flanking 规则在中英混排 + 标点相邻时拒绝配对（``**…。**见``），标记
  会原样漏到页面上。转换器已把这类对改成 HTML 强调，这里核对是否清干净了。

  :param md_texts: 页面名 → markdown。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  total = 0
  for name, text in md_texts.items():
    masked = postprocess._mask_protected(text)
    for pattern, _tag in ((postprocess._STRONG_PAIR_RE, "strong"),
                          (postprocess._EM_PAIR_RE, "em")):
      for match in pattern.finditer(masked):
        content = text[match.start(1):match.end(1)]
        before = text[match.start() - 1] if match.start() > 0 else ""
        after = text[match.end()] if match.end() < len(masked) else ""
        total += 1
        if not postprocess._can_emphasize(before, after, content):
          problems.append(f"{name}: {content[:30]}")
  ok = not problems
  detail = (f"核对 {total} 对强调" if ok
            else f"{len(problems)} 对配不上，例如 {problems[0]}")
  return ok, detail


#: 没被解析掉的 MyST 属性（图片 ``{width=…}``、块级 ``{.class}`` 等）。
_STRAY_ATTR_RE = re.compile(r"\{(?:width|height|align|scale)\s*=|(?<![\w{])\.rtd-")


def check_stray_attributes(html_dir: Path) -> tuple[bool, str]:
  """检查页面上没有漏出来的 MyST 属性字面文本。

  ``![插图](x.svg){width=95%}`` 这种写法里，不带引号的百分比值 MyST 解析不了，
  会原样显示 ``{width=95%}``（全站 25 张图都中招）。这里在产物页面里核对。

  :param html_dir: Sphinx 输出目录。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  for path in sorted(html_dir.rglob("*.html")):
    page = path.read_text(encoding="utf-8", errors="replace")
    match = _STRAY_ATTR_RE.search(page)
    if match is not None:
      problems.append(f"{path.relative_to(html_dir).as_posix()}: {match.group(0)[:30]}")
  ok = not problems
  return ok, f"{len(problems)} 页命中，例如 {problems[0]}" if problems else ""


def check_hard_breaks(texts: dict[str, str]) -> tuple[bool, str]:
  """检查正文里没有漏出来的 LaTeX 硬换行反斜杠。

  原书里的 ``\\\\``（硬换行）会被 pandoc 输出成「行尾反斜杠 + 换行」；合行后那个
  反斜杠会变成可见字面（面试速查的问答曾显示成 ``…cuDNN？\\ A：…``）。转换器已把它
  转成 ``<br>``，这里核对清干净了。代码块、公式与 HTML 表格里的反斜杠是合法的。

  :param texts: 页面名 → markdown。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  total = 0
  for name, text in texts.items():
    lines = text.split("\n")
    flags = postprocess.code_fence_flags(lines)
    masked = "".join(
      "\n" if char == "\n" else " "
      for line, flag in zip(lines, flags) for char in (" " * len(line) if flag else line)
    )
    spans = texutil.math_spans(masked)
    for start, end in spans:
      masked = masked[:start] + " " * (end - start) + masked[end:]
    masked = re.sub(r"<table\b.*?</table>", lambda m: " " * len(m.group(0)), masked, flags=re.S)
    for match in re.finditer(r"\\ ", masked):
      total += 1
      if len(problems) < 3:
        problems.append(f"{name}: …{masked[max(0, match.start() - 30):match.end() + 20]}…")
  ok = not problems
  detail = (f"核对 {total} 处" if ok
            else f"{total} 处字面反斜杠，例如 {problems[0]}")
  return ok, detail


def check_math_foreign_markup(texts: dict[str, str]) -> tuple[bool, str]:
  """检查公式片段里没有被折行转换污染的 HTML 标记。

  转换器把正文硬换行（行尾反斜杠）转成 ``<br>`` 时若漏保护公式，bmatrix/align 的
  ``\\\\`` 会被吃成 ``\\<br>``，MathJax 按 ``\\<`` 未定义报错、整条公式垮掉（线上
  导读章与 ch25 曾大面积中招）。这里在源树核对公式片段里没有 ``<br`` 字面。

  :param texts: 页面名 → markdown。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  spans = 0
  for name, text in texts.items():
    lines = text.split("\n")
    flags = postprocess.code_fence_flags(lines)
    clean = "\n".join("" if flag else line for line, flag in zip(lines, flags))
    for start, end in texutil.math_spans(clean):
      spans += 1
      span = clean[start:end]
      position = span.find("<br")
      if position != -1 and len(problems) < 3:
        excerpt = span[max(0, position - 30):position + 20].replace("\n", " ")
        problems.append(f"{name}: …{excerpt}…")
  ok = not problems
  detail = (f"核对 {spans} 个公式片段" if ok
            else f"{len(problems)}+ 处公式内 <br>，例如 {problems[0]}")
  return ok, detail


def check_math_text_commands(md_texts: dict[str, str]) -> tuple[bool, str]:
  """检查数学区里的 text 命令不含 MathJax 渲染不了的写法。

  公式里的 ``\\text{}``/``\\texttt{}`` 等内容由 MathJax 的 text mode 渲染，它只认
  纯文本与嵌套行内数学：``\\_``、``\\%``、``\\ ``、``\\,``、``\\quad``、嵌套
  ``\\texttt``、``{\\bf …}`` 都会**原样显示**在页面上（不是报错，``mjx-merror``
  抓不到）。转换器（``postprocess._normalize_mathjax_text``）负责把它们搬到数学层；
  这里核对产物是该规范化的**不动点**——书稿更新引入新写法时，这里报出第一处差异。

  :param md_texts: 页面名 → markdown。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  for name, text in md_texts.items():
    fixed = postprocess._normalize_mathjax_text(text)
    if fixed == text:
      continue
    limit = min(len(text), len(fixed))
    index = next((position for position in range(limit) if text[position] != fixed[position]), limit)
    excerpt = text[max(0, index - 40):index + 40].replace("\n", " ")
    problems.append(f"{name}: …{excerpt}…")
    if len(problems) >= 3:
      break
  ok = not problems
  detail = (f"核对 {len(md_texts)} 页的 text 命令" if ok
            else f"{len(problems)}+ 页有未规范化写法，例如 {problems[0]}")
  return ok, detail


#: 伪数学 span：pandoc 把表格单元格里的数学渲染成 ``<span class="math inline">``，
#: 内容已是纯文本或 HTML（``<em>N</em>``、``{0, 2}``），没有 ``$``/``\\(`` 定界符。
_PSEUDO_MATH_RE = re.compile(r'<span class="math inline">(.*?)</span>', re.S)


def check_pseudo_math_spans(md_texts: dict[str, str]) -> tuple[bool, str]:
  """检查没有定界符的 ``math inline`` span 里不含 LaTeX 命令。

  这类 span 没有 ``$...$`` 定界符，MathJax 不处理它们、只当普通文本显示——一旦内容
  里混进 ``\\_``、``\\bf`` 这类 LaTeX 命令，就会原样漏到页面上，而 ``mjx-merror``
  与公式残迹扫描都抓不到（公式区判定里它们不算数学区）。

  :param md_texts: 页面名 → markdown。
  :returns: ``(是否通过, 说明)``。
  """
  problems: list[str] = []
  spans = 0
  for name, text in md_texts.items():
    lines = text.split("\n")
    flags = postprocess.code_fence_flags(lines)
    clean = "\n".join("" if flag else line for line, flag in zip(lines, flags))
    for match in _PSEUDO_MATH_RE.finditer(clean):
      body = match.group(1)
      if "$" in body or "\\(" in body:
        continue  # 自带定界符的是正常公式，MathJax 会渲染
      spans += 1
      if re.search(r"\\[a-zA-Z]+|\\[%&#_]", body) and len(problems) < 3:
        problems.append(f"{name}: …{body[:60]}…")
  ok = not problems
  detail = (f"核对 {spans} 个无定界符 span" if ok
            else f"{len(problems)}+ 处 span 内有 LaTeX 命令，例如 {problems[0]}")
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


def check_translation_dicts(html_dir: Path, root: Path,
                            full_build: bool = True) -> tuple[bool, str]:
  """核对代码块/公式英译词典与页面产物逐字节一致。

  页面（``pre`` / ``.math`` 内）含中文的文本节点必须都能在词典里命中；词典的每个
  key 也应能在页面里找到（部分章节构建时该方向无从判断，跳过）。错配会让英文版
  静默残留中文——典型来源是转换链换了 pandoc 版本，公式 tex 源的形态（如
  ``align`` / ``aligned``）发生漂移，词典 key 与页面 DOM 对不上。

  只统计**主书**（站点根下的页面）：子集合（``<slug>/``）不参与英译词典，纳入统计
  会把它的代码注释与图注全算成未命中，直接判红。

  :param html_dir: Sphinx 输出目录。
  :param root: ``docs/readthedocs`` 根（词典所在）。
  :param full_build: 是否全量构建。
  :returns: (是否通过, 描述)。
  """
  from .cjk_nodes import MathTextCollector, PreTextCollector

  code_entries = json.loads((root / "_static" / "code-en.json").read_text(encoding="utf-8"))
  math_js = (root / "_static" / "math-en.js").read_text(encoding="utf-8")
  math_entries = json.loads(math_js[math_js.index("{"): math_js.rindex(";")])

  def page_nodes(collector_cls) -> set[str]:
    found: set[str] = set()
    # 非递归 glob：集合页在子目录，天然被排除在外。
    for path in sorted(html_dir.glob("*.html")):
      collector = collector_cls()
      collector.feed(path.read_text(encoding="utf-8"))
      found.update(node.strip() for node in collector.nodes if node.strip())
    return found

  details: list[str] = []
  ok_all = True
  for label, collector_cls, entries in (
      ("代码块", PreTextCollector, code_entries),
      ("公式", MathTextCollector, math_entries)):
    nodes = page_nodes(collector_cls)
    miss = nodes - set(entries)
    orphan = (set(entries) - nodes) if full_build else set()
    ok_all = ok_all and not miss and not orphan
    detail = f"词典 {len(entries)} 条，页面中文节点 {len(nodes)}，命中 {len(nodes) - len(miss)}"
    if miss:
      detail += f"，未命中 {len(miss)}（如 {sorted(miss)[0][:40]!r}）"
    if orphan:
      detail += f"，死条目 {len(orphan)}（如 {sorted(orphan)[0][:40]!r}）"
    details.append(f"{label}：{detail}")
  return ok_all, "；".join(details)


@dataclass
class BookContext:
  """一本书在站点树里的位置与真值。

  :param slug: 站点子目录名（主书为空串）。
  :param title: 书名（核对项后缀用）。
  :param book_dir: 书根目录。
  :param book_tex: 根 tex 文件名。
  :param src_dir: 该书的站点源树目录。
  :param html_dir: 该书的 HTML 输出目录。
  :param work_dir: 该书的转换临时目录。
  :param parts: 篇结构。
  :param registry: label 注册表。
  """
  slug: str
  title: str
  book_dir: Path
  book_tex: str
  src_dir: Path
  html_dir: Path
  work_dir: Path
  parts: list[booktree.Part]
  registry: labels.Registry

  def texts(self) -> dict[str, str]:
    """该书各页的 markdown，按 ``chap_id`` 索引。

    :returns: ``chap_id → markdown``。
    """
    return {path.stem: path.read_text(encoding="utf-8")
            for path in sorted(self.src_dir.glob("*.md"))}


def _book_contexts(src_dir: Path, html_dir: Path, book_dir: Path) -> list[BookContext]:
  """按「主书 + 已装配的集合」构造核对上下文。

  集合是否参与由转换器写的装配清单 ``build/collections.json`` 决定：清单里列出才核对，
  ``--no-collections`` / ``--only`` 这类局部构建不会因此变红。

  :param src_dir: 站点源树根。
  :param html_dir: HTML 输出根。
  :param book_dir: 主书根目录。
  :returns: 上下文列表（主书在首位）。
  """
  main_parts = booktree.parse(book_dir / "book.tex")
  contexts = [BookContext(
    slug="", title="主书", book_dir=book_dir, book_tex="book.tex",
    src_dir=src_dir, html_dir=html_dir, work_dir=src_dir.parent / "tmp",
    parts=main_parts, registry=labels.build(main_parts, book_dir))]

  manifest = src_dir.parent / "collections.json"
  if not manifest.is_file():
    return contexts
  entries = json.loads(manifest.read_text(encoding="utf-8")).get("collections", [])
  for entry in entries:
    collection = next((item for item in COLLECTIONS if item.slug == entry.get("slug")), None)
    if collection is None:
      continue
    # 优先用清单里的 book_tex（转换器实际读的那本），注册表只作兜底：两处不一致时
    # 按实际产物核对，别拿注册表去猜。
    root_tex = book_dir / (entry.get("book_tex") or collection.book_tex)
    if not root_tex.is_file():
      continue
    parts = booktree.parse(root_tex)
    contexts.append(BookContext(
      slug=collection.slug, title=collection.title,
      book_dir=root_tex.parent, book_tex=root_tex.name,
      src_dir=src_dir / collection.slug, html_dir=html_dir / collection.slug,
      work_dir=src_dir.parent / "tmp" / collection.slug,
      parts=parts, registry=labels.build(parts, root_tex.parent)))
  return contexts


def _count_rendered_tables(html_dir: Path) -> int:
  """统计一本书页面里的 ``<table>`` 数量（只数本书自己的页面）。

  :param html_dir: 该书的 HTML 目录。
  :returns: ``<table`` 出现次数。
  """
  if not html_dir.is_dir():
    return 0
  return sum(path.read_text(encoding="utf-8", errors="replace").count("<table")
             for path in html_dir.glob("*.html"))


def _nav_hrefs(page: str) -> list[str]:
  """取左侧栏导航里的链接顺序。

  :param page: 页面 HTML。
  :returns: href 列表（无侧栏时为空）。
  """
  start = page.find("wy-menu-vertical")
  if start == -1:
    return []
  end = page.find("</nav>", start)
  return re.findall(r'href="([^"]+)"', page[start:end if end != -1 else len(page)])


def check_collections(manifest: Path, html_dir: Path) -> tuple[bool, str]:
  """核对集合页面齐全、主书侧栏入口排在全书各篇之后。

  只认转换器写的装配清单：清单里没有的集合（``--no-collections``、``--only`` 的局部
  构建）一律跳过，避免调试构建被判定失败；清单里的集合缺页则是真问题。

  :param manifest: 装配清单（``build/collections.json``）。
  :param html_dir: HTML 输出根目录。
  :returns: ``(是否通过, 说明)``。
  """
  if not manifest.is_file():
    return True, "无装配清单，跳过"
  entries = json.loads(manifest.read_text(encoding="utf-8")).get("collections", [])
  if not entries:
    return True, "本次构建未装配集合"

  index_page = html_dir / "index.html"
  hrefs = _nav_hrefs(index_page.read_text(encoding="utf-8", errors="replace")) \
    if index_page.is_file() else []
  part_positions = [index for index, href in enumerate(hrefs)
                    if re.fullmatch(r"part-\d+\.html", href)]

  problems: list[str] = []
  checked = 0
  for entry in entries:
    slug = entry.get("slug", "")
    for page in entry.get("pages", []) + entry.get("chapters", []):
      checked += 1
      if not (html_dir / slug / f"{page}.html").is_file():
        problems.append(f"{slug}/{page}.html 缺失")
    target = f"{slug}/index.html"
    if target not in hrefs:
      problems.append(f"主书侧栏没有 {target} 入口")
    elif part_positions and hrefs.index(target) < max(part_positions):
      problems.append(f"{target} 排在篇页之前（应在附录之后）")
    # 清单只说明"转换器写了哪些页"，不代表首页把它们链出来了：``_assemble_collection``
    # 在某篇一章都没转时会跳过整篇，章节页仍在磁盘上、清单里也在，读者却点不进去。
    index_html = html_dir / slug / "index.html"
    if index_html.is_file():
      body = index_html.read_text(encoding="utf-8", errors="replace")
      linked = set(re.findall(r'href="([a-z0-9-]+)\.html"', body))
      missing = [chapter for chapter in entry.get("chapters", []) if chapter not in linked]
      if missing:
        problems.append(f"{slug} 首页没链到 {len(missing)} 章（如 {missing[0]}）")
  ok = not problems
  detail = (f"{len(entries)} 个集合、{checked} 页齐全" if ok
            else f"{len(problems)} 处问题，例如 {problems[0]}")
  return ok, detail


def _verify_book(report: Report, book: BookContext) -> None:
  """跑一本书专属的核对（章号、表格、目录编号、图内文字）。

  :param report: 核对汇总。
  :param book: 书籍上下文。
  """
  texts = book.texts()
  suffix = f"（{book.title}）" if book.slug else ""
  chapters = [chapter for part in book.parts for chapter in part.chapters]
  full_build = all(chapter.chap_id in texts for chapter in chapters)

  mismatch: list[str] = []
  for part in book.parts:
    for chapter in part.chapters:
      text = texts.get(chapter.chap_id)
      if text is None:
        continue
      raw_heading = next((line for line in text.split("\n") if line.startswith("# ")), "")
      heading = re.sub(r"^#\s+", "", raw_heading)
      number = book.registry.chapter_numbers.get(chapter.chap_id, "")
      if chapter.kind == "appendix":
        expected = f"附录 {number}：" if number else ""
      elif chapter.chap_id in book.registry.numbered:
        expected = f"第 {number} 章：" if number else ""
      else:
        expected = ""
      if expected and not heading.startswith(expected):
        mismatch.append(f"{chapter.chap_id}: 期望 “{expected}”，实际 “{heading[:24]}”")
  report.add(f"H1 章号与注册表一致{suffix}", not mismatch,
             f"{len(mismatch)} 处，例如 {mismatch[0]}" if mismatch else "")

  # 表格相关的三项都以「原书全部表格环境」为真值，局部构建时必须跳过——否则
  # ``--only-collection`` 打样会拿全书表格数去比几页的产物，必然判红。
  expected_tables = count_source_tables(book.parts)
  if not full_build:
    for name in ("表格已渲染", "逐章表格数不缺", "多行表头已收进 thead", "书目录编号核对"):
      report.add(f"{name}{suffix}", True, "跳过（只转换了部分章节）")
  else:
    rendered = _count_rendered_tables(book.html_dir)
    report.add(f"表格已渲染{suffix}", rendered >= expected_tables * 0.9,
               f"原书表格环境 {expected_tables}，页面 <table> {rendered}")

    ok, detail = check_table_counts_per_chapter(book.parts, book.html_dir)
    report.add(f"逐章表格数不缺{suffix}", ok, detail)

    ok, detail = check_multirow_headers(book.parts, texts)
    report.add(f"多行表头已收进 thead{suffix}", ok, detail)

    ok, detail, _ = check_book_toc(book.book_dir, book.registry, chapters, book.book_tex)
    report.add(f"书目录编号核对{suffix}", ok, detail)

  if (book.work_dir / "tikz").is_dir():
    ok, detail = check_tikz_glyphs(book.work_dir, book.src_dir)
    report.add(f"TikZ 图保留文字{suffix}", ok, detail)


def verify(src_dir: Path, html_dir: Path, book_dir: Path) -> Report:
  """执行全部核对。

  通用项（残留 token、公式、表格单元格、图片、按钮）覆盖站点里的每一页，页面键用
  **docname**（``cute-zhihu/reed-01-layout``）而不是文件名——主书与集合都有 ``index``、
  ``part-1``，按 stem 作键会互相覆盖、静默漏页。书籍相关项按书分别核对。

  :param src_dir: 站点源树（``build/src``）。
  :param html_dir: Sphinx 输出（``build/html``）。
  :param book_dir: 书根目录（``book/``）。
  :returns: 核对结果。
  """
  report = Report()
  books = _book_contexts(src_dir, html_dir, book_dir)
  texts = {path.relative_to(src_dir).with_suffix("").as_posix(): path.read_text(encoding="utf-8")
           for path in sorted(src_dir.rglob("*.md"))}

  leftovers = [(name, text.count("@@RTD;")) for name, text in texts.items() if "@@" in text]
  report.add("源树无残留 token", not leftovers,
             f"{len(leftovers)} 页残留，例如 {leftovers[0][0]}" if leftovers else "")

  attrs = [name for name, text in texts.items() if re.search(r"\{#[^}]*\}", text)]
  report.add("标题无 pandoc 属性泄漏", not attrs,
             f"{len(attrs)} 页命中，例如 {attrs[0]}" if attrs else "")

  main_texts = books[0].texts()
  chapter_ref_texts = set(re.findall(r"\[([^\]]+)\]\([a-z0-9-]+\.html#ch-[0-9a-z-]+",
                                     "\n".join(main_texts.values())))
  collapsed = chapter_ref_texts - {"1"} if chapter_ref_texts == {"1"} else set()
  report.add("章引用编号未坍缩", not collapsed,
             f"所有章引用都是 {chapter_ref_texts}" if collapsed else "")

  anchors: dict[str, set[str]] = {
    name: set(_ANCHOR_RE.findall(text)) for name, text in texts.items()
  }
  dangling: list[str] = []
  for name, text in texts.items():
    base = posixpath.dirname(name)
    for page, anchor in _LINK_RE.findall(text):
      target = posixpath.normpath(posixpath.join(base, page)) if base else page
      if target not in anchors or anchor not in anchors[target]:
        dangling.append(f"{name} → {page}.html#{anchor}")
  report.add("站内锚点无悬空", not dangling,
             f"{len(dangling)} 条，例如 {dangling[0]}" if dangling else "")

  raw_html = [name for name, text in texts.items() if "{=html}" in text]
  html_raw = 0
  if html_dir.is_dir():
    html_raw = sum(
      path.read_text(encoding="utf-8", errors="replace").count("{=html}")
      for path in html_dir.rglob("*.html"))
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
    for path in sorted(html_dir.rglob("*.html")):
      page = path.read_text(encoding="utf-8", errors="replace")
      if re.search(r"\$<span class=\"math", page) or re.search(
          r'class="math[^"]*">\\\(\\begin\{(?:' + display_env_only + r')', page):
        broken_display.append(path.relative_to(html_dir).as_posix())
  report.add("显示公式未被拆成字面 $", not broken_display,
             f"{len(broken_display)} 页，例如 {broken_display[0]}" if broken_display else "")

  ok, detail = check_table_cells(texts)
  report.add("表格单元格无 LaTeX 残渣", ok, detail)

  ok, detail = check_nested_math(texts)
  report.add("公式内无嵌套 $", ok, detail)

  ok, detail = check_emphasis_flanking(texts)
  report.add("强调标记成对（flanking）", ok, detail)

  ok, detail = check_table_inline_markdown(texts)
  report.add("HTML 表格内无 Markdown 残留", ok, detail)

  ok, detail = check_hard_breaks(texts)
  report.add("无字面 LaTeX 换行反斜杠", ok, detail)

  ok, detail = check_math_foreign_markup(texts)
  report.add("公式内无 <br> 污染", ok, detail)

  ok, detail = check_math_text_commands(texts)
  report.add("公式 text 命令已规范化", ok, detail)

  ok, detail = check_pseudo_math_spans(texts)
  report.add("伪数学 span 无 LaTeX 残留", ok, detail)

  if html_dir.is_dir():
    ok, detail = check_images_present(html_dir)
    report.add("页面图片均存在", ok, detail)

    ok, detail = check_stray_attributes(html_dir)
    report.add("MyST 属性均已解析", ok, detail)

    # 「下载 PDF」按钮的链接真值取自仓库根 README（解析失败会直接抛错中断核对）。
    ok, detail = check_pdf_button(html_dir, readme.pdf_link())
    report.add("下载 PDF 按钮已渲染", ok, detail)

    ok, detail = check_repo_button(html_dir, readme.REPO_URL)
    report.add("仓库按钮已渲染", ok, detail)

  for book in books:
    _verify_book(report, book)

  main_book = books[0]
  if (main_book.work_dir / "tikz").is_dir():
    # 「TikZ 图保留文字」由 _verify_book 按书跑；英文图集只覆盖主书（词典按主书生成）。
    ok, detail = check_english_figures(
      main_book.work_dir / "tikz", src_dir.parent.parent / "_static" / "figures-en",
      src_dir.parent.parent / "i18n" / "figures-en.json", referenced_figures(src_dir))
    report.add("英文图集完整", ok, detail)

  chapters = [chapter for part in main_book.parts for chapter in part.chapters]
  full_build = all(chapter.chap_id in main_texts for chapter in chapters)
  ok, detail = check_translation_dicts(html_dir, Path(__file__).resolve().parent.parent,
                                       full_build=full_build)
  report.add("英译词典匹配产物", ok, detail)

  ok, detail = check_collections(src_dir.parent / "collections.json", html_dir)
  report.add("集合已装配且入口在附录之后", ok, detail)

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
