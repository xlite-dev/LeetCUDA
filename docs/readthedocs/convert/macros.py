"""从书稿 tex 中提取数学宏，供 MathJax 使用。

书里定义了大量数学宏（``\\def\\v{\\mathbf}``、``\\newcommand{\\abs}[1]{…}``、
``\\def\\Z{\\mathbb{Z}}`` 等，合计近 200 条）。这些宏在纸书上由 LaTeX 展开，而网页
里公式是原样交给 MathJax 的——不告诉 MathJax 这些宏，``\\v{L}`` 会按 MathJax 内置的
重音命令渲染成 ``Ľ``，``\\Z`` 直接报错。

这里把定义抽出来写成 MathJax 的 ``tex.macros`` 配置（转换时生成
``build/mathjax-macros.json``，``conf.py`` 里读它）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .texutil import read_group, strip_comments

#: ``\newcommand{\name}[n]{body}`` / ``\renewcommand*``。
_NEWCOMMAND_RE = re.compile(r"\\(?:re)?newcommand\*?\s*\{\s*\\([a-zA-Z@]+)\s*\}\s*(?:\[(\d+)\])?\s*\{")

#: ``\def\name#1#2{body}``。
_DEF_RE = re.compile(r"\\def\s*\\([a-zA-Z@]+)\s*((?:#\d)*)\s*\{")

#: 这些宏与排版有关，塞给 MathJax 只会添乱。
_SKIP_NAMES = {
  "headrulewidth", "footrulewidth", "baselinestretch", "arraystretch", "parindent",
  "parskip", "tableofcontents", "thechapter", "thesection", "thefigure", "thetable",
  "theequation", "contentsname", "listfigurename", "listtablename", "chaptername",
  "appendixname", "figurename", "tablename", "bibname", "indexname", "abstractname",
  "refname", "headheight", "textwidth", "textheight", "columnsep", "abovecaptionskip",
}

#: 宏体里含这些命令时说明它不是纯数学宏（引用、环境、图片、代码等）。
_SKIP_BODY = ("\\begin{", "\\end{", "\\includegraphics", "\\label", "\\ref", "\\eqref",
              "\\cite", "\\item", "\\lstinline", "\\verb", "\\input", "\\pageref",
              "\\hspace", "\\vspace", "\\caption", "\\footnote", "\\ifcase", "\\pgf")

#: TikZ 图形里的尺寸/颜色参数不是数学宏（如 ``\def\cs{0.42}``、``\def\mfill{tkFillBlue}``），
#: 它们只在离线编译的图里用，喂给 MathJax 反而可能遮蔽数学符号。
_SKIP_VALUE_RE = re.compile(
  r"^(?:[-+]?[\d.]+\s*(?:pt|cm|mm|em|ex)?"
  r"|(?:tk|colorx)[A-Za-z]*(?:![A-Za-z\d\\.{}]*)?"
  r"|\d+)$")


def _clean_body(body: str) -> str:
  """清掉宏体里 MathJax 不认的排版命令。

  :param body: 宏体。
  :returns: 处理后的宏体。
  """
  body = re.sub(r"\\(?:protect|relax|ignorespaces|allowbreak)\b", "", body)
  body = re.sub(r"\\(?:hspace|vspace)\s*\{[^{}]*\}", "", body)
  return re.sub(r"\s+", " ", body).strip()


def extract(book_dir: Path) -> dict[str, object]:
  """扫描书稿，抽出数学宏定义。

  :param book_dir: 书根目录。
  :returns: 可直接放进 MathJax ``tex.macros`` 的字典。
  """
  macros: dict[str, object] = {}
  for path in _tex_files(book_dir):
    text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
    for match in _NEWCOMMAND_RE.finditer(text):
      name = match.group(1)
      nargs = int(match.group(2) or 0)
      try:
        body, _ = read_group(text, match.end() - 1)
      except ValueError:
        continue
      _record(macros, name, body, nargs)
    for match in _DEF_RE.finditer(text):
      name = match.group(1)
      nargs = len(re.findall(r"#\d", match.group(2)))
      try:
        body, _ = read_group(text, match.end() - 1)
      except ValueError:
        continue
      _record(macros, name, body, nargs)
  return macros


def _record(macros: dict[str, object], name: str, body: str, nargs: int) -> None:
  """登记一条宏定义（过滤非数学宏）。

  :param macros: 结果字典。
  :param name: 宏名（不含反斜杠）。
  :param body: 宏体。
  :param nargs: 参数个数。
  """
  if name in _SKIP_NAMES or name in macros:
    return
  if any(token in body for token in _SKIP_BODY):
    return
  cleaned = _clean_body(body)
  if not cleaned or _SKIP_VALUE_RE.match(cleaned):
    return
  macros[name] = cleaned if nargs == 0 else [cleaned, nargs]


def _tex_files(book_dir: Path) -> list[Path]:
  """列出需要扫描的 tex 文件。

  :param book_dir: 书根目录。
  :returns: 文件列表。
  """
  files = [book_dir / "preamble.tex", book_dir / "book.tex"]
  for folder in ("chapters", "chapters/wp", "appendices"):
    directory = book_dir / folder
    if directory.is_dir():
      files.extend(sorted(directory.glob("*.tex")))
  return [path for path in files if path.is_file()]


def write(book_dir: Path, target: Path) -> dict[str, object]:
  """抽取并落盘。

  :param book_dir: 书根目录。
  :param target: 输出 JSON 路径。
  :returns: 抽出的宏。
  """
  macros = extract(book_dir)
  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(macros, ensure_ascii=False, indent=1), encoding="utf-8")
  return macros
