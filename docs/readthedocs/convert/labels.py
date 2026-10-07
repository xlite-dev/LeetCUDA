"""全书 label → 编号/标题注册表，交叉引用还原的依据。

书里 ``\\ref`` 渲染成编号（``\\ref{ch:16}`` → ``16``、``\\eqref{eq:16-brbc}`` →
``(16.3)``、``图~\\ref{fig:10-1}`` → ``图 10.1``），所以注册表按阅读顺序重放
LaTeX 计数器：章 / 节 / 图 / 表 / 公式 / 定理。编号默认是 book 文档类的
``<章>.<序号>``，遇到 ``\\renewcommand{\\theXXX}`` 时按书中定义改写（例如
ch19b 白皮书导读的 ``W.1``）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .booktree import Chapter, Part
from .texutil import flatten_inputs, plain_text, read_group, strip_comments

#: label 前缀 → 引用类型。
PREFIX_KIND = {
  "ch": "chapter",
  "app": "chapter",
  "sec": "section",
  "fig": "figure",
  "tab": "table",
  "eq": "equation",
  "eqn": "equation",
  "thm": "theorem",
  "lem": "theorem",
  "def": "theorem",
}

#: 章节内会重置于零的计数器（book 文档类按章复位）。
_PER_CHAPTER = ("section", "subsection", "subsubsection", "figure", "table", "equation", "theorem", "kaogu")

#: 默认编号模板（ctexbook + 书内定理声明）。
_DEFAULT_FORMATS = {
  "chapter": "{c}",
  "section": "{c}.{s}",
  "subsection": "{c}.{s}.{ss}",
  "subsubsection": "{c}.{s}.{ss}.{sss}",
  "equation": "{c}.{n}",
  "figure": "{c}.{n}",
  "table": "{c}.{n}",
  "theorem": "{c}.{n}",
  "kaogu": "{c}.{n}",
}

#: label 中不属于 token 载荷字符集的字符（`_` 会被 token 与锚点改写，必须统一）。
_LABEL_UNSAFE = re.compile(r"[^A-Za-z0-9\-.:]")


def sanitize(label: str) -> str:
  """把 label 规范成 token 载荷与 HTML 锚点都能用的形式。

  token 字符集不含 ``_``（pandoc 会在 markdown 输出里转义它），锚点 id 同理。
  登记与查询两侧都过这个函数，label 里的 ``_`` 才不会查不到。

  :param label: 原始 label。
  :returns: 规范后的 label。
  """
  return _LABEL_UNSAFE.sub("-", label)

#: 标题命令 → 章节层级（``\wph`` 系列是 ch19b 对白皮书节标题的映射）。
_HEADING_LEVEL = {
  "chapter": "chapter",
  "section": "section",
  "subsection": "subsection",
  "subsubsection": "subsubsection",
  "wph": "section",
  "wpsh": "subsection",
  "wpssh": "subsubsection",
}

#: 宏定义行不参与扫描：``\def\wph#1{\section{#1}}`` 里的 ``\section`` 不是标题。
#: 例外是 ``\renewcommand{\theXXX}``——它正是编号格式定义，必须保留。
_DEFINITION_RE = re.compile(
  r"(?m)^\s*\\(?:def|newcommand|providecommand|newenvironment|lstnewenvironment)\b[^\n]*$"
  r"|^\s*\\renewcommand\b(?!\s*\{\\the)[^\n]*$")

_SCAN_RE = re.compile(r"""
  \\(?P<cmd>appendix|chapter|section|subsection|subsubsection|wpssh|wpsh|wph)\b(?P<star>\*)?
| \\begin\{(?P<env>figure|table|longtable|subfigure|equation|equation\*|align|align\*|theorem|lemma|definition|kaogu)\}
| \\end\{(?P<endenv>figure|table|longtable|subfigure)\}
| \\label\{(?P<label>[^}]*)\}
| \\caption\b
| \\renewcommand\{\\the(?P<fmtname>[a-zA-Z]+)\}
| \\setcounter\{(?P<scname>[a-zA-Z]+)\}\{(?P<scval>\d+)\}
""", re.VERBOSE)


def sanitize(label: str) -> str:
  """把 label 转成 HTML 锚点 id（去掉冒号等非 ``[A-Za-z0-9_.-]`` 字符）。

  :param label: 原 label。
  :returns: 锚点 id。
  """
  return re.sub(r"[^A-Za-z0-9_.-]", "-", label)


@dataclass(frozen=True)
class Entry:
  """一个 label 的解析结果。

  :param label: 原 label。
  :param kind: ``chapter`` / ``section`` / ``figure`` / ``table`` / ``equation`` / ``theorem`` / ``unknown``。
  :param number: 书中渲染出的编号，如 ``16``、``16.3``、``W.1``。
  :param title: 上下文标题（章/节标题或图/表 caption，纯文本）。
  :param page: 所在站点页面（``chap_id``）。
  """
  label: str
  kind: str
  number: str
  title: str
  page: str

  @property
  def anchor(self) -> str:
    """HTML 锚点 id。

    :returns: 锚点 id。
    """
    return sanitize(self.label)


class Registry:
  """全书 label 注册表。"""

  def __init__(self) -> None:
    """初始化空注册表。"""
    self.entries: dict[str, Entry] = {}
    self.duplicates: list[str] = []
    self.chapter_numbers: dict[str, str] = {}
    self.chapter_formats: dict[str, dict[str, str]] = {}
    self.numbered: set[str] = set()

  def add(self, entry: Entry) -> None:
    """登记一个 label（键为规范化后的 label）。

    :param entry: 解析结果。
    """
    key = sanitize(entry.label)
    if key in self.entries:
      self.duplicates.append(entry.label)
    self.entries[key] = entry

  def get(self, label: str) -> Entry | None:
    """查询 label。

    :param label: 原 label（内部先规范化）。
    :returns: 命中项或 None。
    """
    return self.entries.get(sanitize(label))


class _ChapterScan:
  """全书计数器重放。

  一个 scan 实例贯穿整本书：LaTeX 里章号是连续的文档级计数器，只有把 46 篇
  tex 串起来扫，``\\ref{ch:xx}`` 才会得到 16/17/… 这样的真实章号（曾按「每章
  各起一个 scan」实现，结果全书章号恒为 1）。

  :param registry: 目标注册表。
  """

  def __init__(self, registry: Registry) -> None:
    """初始化计数器与编号格式（与 book 文档类默认一致）。

    :param registry: 目标注册表。
    """
    self.registry = registry
    self.counters = dict.fromkeys(_PER_CHAPTER, 0)
    self.counters["chapter"] = 0
    self.formats = dict(_DEFAULT_FORMATS)
    self.appendix = False
    self.chap_id = ""
    self.page = ""
    self.chapter_numbered = False
    self.chapter_title = ""
    self.section_title = ""
    self.caption = ""
    self.env_stack: list[str] = []

  def start_chapter(self, chap_id: str) -> None:
    """切到新的一章。

    章号与页计数器沿用（LaTeX 是连续文档），编号格式回到默认——ch19b 用
    ``\\begingroup`` 包住它对 ``\\thefigure`` 的重定义，出组即失效。

    :param chap_id: 章节标识（同时用作站点页面名）。
    """
    self.chap_id = chap_id
    self.page = chap_id
    self.chapter_numbered = False
    self.chapter_title = ""
    self.section_title = ""
    self.caption = ""
    self.env_stack = []
    self.formats = dict(_DEFAULT_FORMATS)

  def start_appendix(self) -> None:
    """进入附录：后续章号改为字母 A、B、…（等价于书里的 ``\\appendix``）。"""
    self.appendix = True
    self.counters["chapter"] = 0
    self.formats["chapter"] = "{c}"

  @property
  def chapter_str(self) -> str:
    """当前章号字符串（附录为字母）。

    :returns: 章号。
    """
    index = self.counters["chapter"]
    if self.appendix and index > 0:
      return chr(ord("A") + index - 1)
    return str(index)

  def number(self, kind: str) -> str:
    """按类型给出编号文本。

    :param kind: 引用类型。
    :returns: 编号文本。
    """
    template = self.formats.get(kind, "{c}.{n}")
    values = {
      "c": self.chapter_str,
      "s": str(self.counters["section"]),
      "ss": str(self.counters["subsection"]),
      "sss": str(self.counters["subsubsection"]),
      "n": str(self.counters.get(kind, 0)),
    }
    try:
      return template.format(**values)
    except (KeyError, IndexError):
      return values["n"]

  def scan(self, text: str) -> None:
    """重放一章的计数器并登记 label。

    :param text: 该章的展开 tex 文本。
    """
    normalized = _DEFINITION_RE.sub("", strip_comments(text))
    for match in _SCAN_RE.finditer(normalized):
      self._on_match(match, normalized)

  def _on_match(self, match: re.Match[str], text: str) -> None:
    """处理一个扫描命中。

    :param match: 正则命中。
    :param text: 被扫描文本。
    """
    if match.group("cmd"):
      self._on_heading(match, text)
    elif match.group("env"):
      self._on_begin(match.group("env"))
    elif match.group("endenv"):
      self._on_end(match.group("endenv"))
    elif match.group("label") is not None:
      self._on_label(match.group("label"))
    elif match.group("fmtname"):
      self._on_renewcommand(match, text)
    elif match.group("scname"):
      name = match.group("scname")
      if name in self.counters:
        self.counters[name] = int(match.group("scval"))
    elif match.group(0).startswith("\\caption"):
      self._on_caption(match, text)

  def _on_heading(self, match: re.Match[str], text: str) -> None:
    """处理 ``\\chapter`` / ``\\section`` 等标题命令。

    :param match: 正则命中。
    :param text: 被扫描文本。
    """
    command = match.group("cmd")
    if command == "appendix":
      self.appendix = True
      self.counters["chapter"] = 0
      self.formats["chapter"] = "{c}"
      return
    starred = match.group("star") is not None
    cursor = match.end()
    while cursor < len(text) and text[cursor] in " \t\n":
      cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
      return
    raw_title, cursor = read_group(text, cursor)
    title = plain_text(raw_title)
    command = _HEADING_LEVEL.get(command, command)
    if command == "chapter":
      if not starred:
        self.counters["chapter"] += 1
        self.chapter_numbered = True
        for name in _PER_CHAPTER:
          self.counters[name] = 0
        self.formats.update({
          "section": "{c}.{s}", "subsection": "{c}.{s}.{ss}", "subsubsection": "{c}.{s}.{ss}.{sss}",
          "equation": "{c}.{n}", "figure": "{c}.{n}", "table": "{c}.{n}", "theorem": "{c}.{n}",
          "kaogu": "{c}.{n}",
        })
      self.chapter_title = title
      self.section_title = ""
      return
    if starred:
      self.section_title = title
      return
    if command == "section":
      self.counters["section"] += 1
      self.counters["subsection"] = 0
      self.counters["subsubsection"] = 0
    elif command == "subsection":
      self.counters["subsection"] += 1
      self.counters["subsubsection"] = 0
    else:
      self.counters["subsubsection"] += 1
    self.section_title = title

  def _on_begin(self, env: str) -> None:
    """处理环境开始。

    :param env: 环境名。
    """
    if env in ("figure", "table", "longtable", "subfigure"):
      self.env_stack.append(env)
      return
    if env in ("equation", "align"):
      self.counters["equation"] += 1
      return
    if env in ("theorem", "lemma", "definition"):
      self.counters["theorem"] += 1
      return
    if env == "kaogu":
      self.counters["kaogu"] += 1

  def _on_end(self, env: str) -> None:
    """处理环境结束。

    :param env: 环境名。
    """
    while self.env_stack:
      popped = self.env_stack.pop()
      if popped == env:
        return

  def _on_caption(self, match: re.Match[str], text: str) -> None:
    """处理 ``\\caption``，图/表计数器在 caption 处步进（与 LaTeX 一致）。

    :param match: 正则命中。
    :param text: 被扫描文本。
    """
    try:
      raw, _ = read_group(text, match.end())
    except ValueError:
      raw = ""
    self.caption = plain_text(raw)
    top = self.env_stack[-1] if self.env_stack else ""
    if top == "figure":
      self.counters["figure"] += 1
    elif top in ("table", "longtable"):
      self.counters["table"] += 1

  def _on_renewcommand(self, match: re.Match[str], text: str) -> None:
    """处理 ``\renewcommand{\theXXX}{...}``，把编号定义转成模板。

    :param match: 正则命中。
    :param text: 被扫描文本。
    """
    name = match.group("fmtname")
    cursor = match.end()
    while cursor < len(text) and text[cursor] in " \t\n":
      cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
      return
    definition, _ = read_group(text, cursor)
    template = re.sub(
      r"\\arabic\{([a-zA-Z]+)\}",
      lambda m: {"chapter": "{c}", "section": "{s}", "subsection": "{ss}", "subsubsection": "{sss}"}.get(m.group(1), "{n}"),
      definition)
    if name in self.formats:
      self.formats[name] = template

  def _on_label(self, label: str) -> None:
    """登记一个 label。

    :param label: 原 label。
    """
    prefix = label.split(":", 1)[0] if ":" in label else ""
    kind = PREFIX_KIND.get(prefix, "")
    if not kind:
      top = self.env_stack[-1] if self.env_stack else ""
      kind = {"figure": "figure", "table": "table", "longtable": "table", "subfigure": "figure"}.get(top, "unknown")
    if kind == "chapter":
      title = self.chapter_title
    else:
      title = self.caption if kind in ("figure", "table") else self.section_title
    self.registry.add(Entry(
      label=label,
      kind=kind,
      number=self.number(kind),
      title=title,
      page=self.page,
    ))


def build(parts: list[Part], book_dir: Path) -> Registry:
  """按书的顺序扫描全部章节，构建 label 注册表。

  :param parts: ``booktree.parse`` 的结果。
  :param book_dir: 书根目录（``book/``），用于解析 ``\\input``。
  :returns: 全书注册表。
  """
  registry = Registry()
  scan = _ChapterScan(registry)
  for part in parts:
    if part.appendix:
      scan.start_appendix()
    for chapter in part.chapters:
      scan.start_chapter(chapter.chap_id)
      scan.scan(flatten_inputs(chapter.tex, book_dir))
      registry.chapter_numbers[chapter.chap_id] = scan.chapter_str
      registry.chapter_formats[chapter.chap_id] = dict(scan.formats)
      if scan.chapter_numbered:
        registry.numbered.add(chapter.chap_id)
  return registry


def format_number(template: str, chapter: str, index: int) -> str:
  """按编号模板渲染一个对象编号。

  :param template: 形如 ``"{c}.{n}"`` / ``"W.{n}"`` 的模板。
  :param chapter: 章号字符串。
  :param index: 该类型对象在本章内的序号。
  :returns: 编号文本。
  """
  try:
    return template.format(c=chapter, s=index, ss=index, sss=index, n=index)
  except (KeyError, IndexError):
    return str(index)


def chapter_number_prefix(registry: Registry, chap: Chapter) -> str:
  """给出章在书中的显示编号前缀（``第 16 章：`` / ``附录 A：`` / 空）。

  编号直接取注册表里该章自己的计数器值，避免再走一遍「顺序数数」而错位。

  :param registry: 全书注册表。
  :param chap: 目标章。
  :returns: 标题前缀，无编号章返回空串。
  """
  number = registry.chapter_numbers.get(chap.chap_id, "")
  if not number:
    return ""
  if chap.kind == "appendix":
    return f"附录 {number}："
  if chap.chap_id not in registry.numbered:
    return ""
  return f"第 {number} 章："
