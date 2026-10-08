"""tex 归一化：展开 input、抽离 tikz 与代码清单、把 LaTeX 结构改写成 token。

原书 tex 一概不改——本模块只产出 ``<work>/norm/<chap>.tex`` 这一派生副本供
pandoc 读取。抽出的内容分三类落盘：``tikz/`` 待编译的图片段、``code/`` 代码
载荷、``manifest/<chap>.json`` 记录 token 与载荷的映射。

处理顺序（每一步都依赖前一步的产物，顺序不可随意调整）：

1. 展开 ``\\input``（ch19b 白皮书导读把 wp0--wp7 内联进来）；
2. 抽离代码清单（``\\lstinputlisting`` / ``lstlisting`` / ``python`` / ``cpp`` /
   ``verbatim``）——代码里含 ``%``、``$``、``\\``，必须在任何文本改写之前移出；
3. 抽离 ``tikzpicture`` 片段，留在原位的是图片 token；
4. 抽离活图片 ``\\includegraphics``（全书实际只有 wp6 两张）；
5. 拆掉纯排版盒子（``\\resizebox`` / ``minipage`` 等），把宽度意图记到图片 token；
6. figure / table / subfigure / caption / 定理类环境 → token；
7. 数学环境内的 ``\\label`` 提到环境外，避免污染公式；
8. ``\\label`` / ``\\ref`` / ``\\eqref`` → token；
9. 清掉无需出现在网页上的排版命令。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import tokens
from .booktree import Chapter
from .texutil import (
  find_environments,
  flatten_inputs,
  math_spans,
  read_bracket,
  read_group,
  read_group_after_space,
  read_optional,
  split_options,
  strip_comments,
)

#: 代码环境 → 语言（``None`` 表示按选项推断）。
_CODE_ENVS: dict[str, str | None] = {
  "lstlisting": None,
  "python": "python",
  "cpp": "cpp",
  "verbatim": "text",
}

#: 定理类环境 → （网页标签，CSS 类名，是否单独计数）。
ADMON_KINDS: dict[str, tuple[str, str, bool]] = {
  "kaogu": ("实践经验", "kaogu", True),
  "theorem": ("定理", "theorem", True),
  "lemma": ("引理", "theorem", True),
  "definition": ("定义", "theorem", True),
  "remark": ("注记", "remark", False),
  "proof": ("证明", "proof", False),
}

#: 文件扩展名 → Pygments 语言。
_EXT_LANG = {
  ".cu": "cuda", ".cuh": "cuda", ".h": "c", ".c": "c", ".cpp": "cpp", ".cc": "cpp",
  ".py": "python", ".sh": "bash", ".json": "json", ".md": "text", ".txt": "text",
}

#: listings style → 语言（终端/伪代码按纯文本渲染）。
_STYLE_LANG = {"console": "text", "pseudo": "text"}

#: ``language=`` 选项 → Pygments 语言。
_OPT_LANG = {
  "c": "c", "c++": "cpp", "cpp": "cpp", "cuda": "cuda", "python": "python",
  "python3": "python", "bash": "bash", "sh": "bash", "console": "text",
  "text": "text", "json": "json", "markdown": "text",
}

#: 排版噪声命令（无参数），网页上无意义。
_DROP_ZERO_ARG = (
  "protect", "noindent", "centering", "clearpage", "cleardoublepage", "newpage",
  "pagebreak", "linebreak", "nopagebreak", "FloatBarrier", "smallskip", "medskip", "bigskip",
  "par", "leavevmode", "null", "hfill", "vfill", "hfil", "vfil", "phantom", "relax", "tableofcontents",
  "tiny", "scriptsize", "footnotesize", "small", "normalsize", "large", "Large", "LARGE", "huge",
  "Huge", "bfseries", "mdseries", "itshape", "upshape", "slshape", "scshape", "ttfamily", "sffamily",
  "rmfamily", "selectfont", "raggedright", "raggedbottom", "sloppy", "maketitle", "ignorespaces",
)

#: 排版噪声命令（带固定个数的分组参数）。
_DROP_WITH_ARGS = (
  ("needspace", 1), ("vspace", 1), ("hspace", 1), ("markboth", 2), ("markright", 1),
  ("thispagestyle", 1), ("pagestyle", 1), ("setcounter", 2), ("setlength", 2),
  ("addcontentsline", 3), ("fontsize", 2), ("phantomsection", 0), ("renewcommand", 2),
)

#: 环境 → 替换（``None`` 表示整体删掉环境标记）。
_ENV_REWRITE: dict[str, tuple[str, str] | None] = {
  "compactitem": ("itemize", "itemize"),
  "sloppypar": None,
  "adjustbox": None,
  "minipage": None,
  "lstinputlisting": None,
}

_WIDTH_RE = re.compile(r"([0-9.]+)\s*\\(?:textwidth|linewidth|columnwidth)")
_IMG_TOKEN_RE = re.compile(r"@@RTD;IMG;([A-Za-z0-9\-.:]+)((?:;[^@;]*)?)@@")


@dataclass
class CodeBlock:
  """一段代码载荷。

  :param code_id: token 载荷标识。
  :param lang: Pygments 语言名。
  :param first_line: 代码块首行对应的源文件行号，0 表示不显示行号。
  :param source: 展示用来源说明（标题）。
  :param path: 载荷文件，相对 ``work`` 目录。
  """
  code_id: str
  lang: str
  first_line: int
  source: str
  path: str


@dataclass
class ImageRef:
  """一张图片。

  :param image_id: token 载荷标识。
  :param dest: 站点树内的相对路径。
  :param width: 展示宽度（如 ``"22%"``），空串表示用自然尺寸。
  :param origin: 来源说明（tikz 片段或原图路径）。
  """
  image_id: str
  dest: str
  width: str
  origin: str


@dataclass
class Manifest:
  """单章转换清单，供后处理还原 token。

  :param chap_id: 章节标识。
  :param codes: 代码载荷。
  :param images: 图片。
  :param stats: 统计计数（tikz / code / image / 各类环境）。
  :param notes: 需要写进 report 的提示。
  """
  chap_id: str
  codes: dict[str, CodeBlock] = field(default_factory=dict)
  images: dict[str, ImageRef] = field(default_factory=dict)
  stats: dict[str, int] = field(default_factory=dict)
  notes: list[str] = field(default_factory=list)

  def to_json(self) -> str:
    """序列化为 JSON 文本。

    :returns: JSON 字符串。
    """
    return json.dumps({
      "chap_id": self.chap_id,
      "codes": {key: asdict(value) for key, value in self.codes.items()},
      "images": {key: asdict(value) for key, value in self.images.items()},
      "stats": self.stats,
      "notes": self.notes,
    }, ensure_ascii=False, indent=2)


class _Builder:
  """归一化过程中的可变状态。"""

  def __init__(
    self,
    chap: Chapter,
    book_dir: Path,
    work_dir: Path,
    src_dir: Path,
    registry=None,
  ) -> None:
    """初始化。

    :param chap: 目标章。
    :param book_dir: 书根目录（``book/``）。
    :param work_dir: 临时目录（``build/tmp``）。
    :param src_dir: 站点源树（``build/src``）。
    :param registry: 全书 label 注册表，供图内 ``\\ref`` 取编号。
    """
    self.chap = chap
    self.book_dir = book_dir
    self.work_dir = work_dir
    self.src_dir = src_dir
    self.registry = registry
    self.manifest = Manifest(chap_id=chap.chap_id)
    self.repo_root = _find_repo_root(book_dir)
    self._tikz_index = 0
    self._code_index = 0
    self._image_index = 0

  def note(self, message: str) -> None:
    """记录一条需要进 report 的提示。

    :param message: 提示文本。
    """
    self.manifest.notes.append(message)

  def bump(self, key: str, amount: int = 1) -> None:
    """累加统计计数。

    :param key: 计数名。
    :param amount: 增量。
    """
    self.manifest.stats[key] = self.manifest.stats.get(key, 0) + amount

  def new_code(self, lang: str, first_line: int, source: str, text: str) -> str:
    """落盘一段代码载荷并登记。

    :param lang: 语言名。
    :param first_line: 首行行号（0 表示不显示）。
    :param source: 来源说明。
    :param text: 代码正文。
    :returns: code_id。
    """
    self._code_index += 1
    code_id = f"{self.chap.chap_id}-c{self._code_index:03d}"
    rel = f"code/{code_id}.txt"
    target = self.work_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    self.manifest.codes[code_id] = CodeBlock(code_id, lang, first_line, source, rel)
    self.bump("code")
    return code_id

  def new_tikz(self, source: str) -> str:
    """落盘一个 tikz 片段并登记为图片。

    :param source: 完整的 ``tikzpicture`` 环境（含 ``\\begin`` 与选项）。
    :returns: image_id。
    """
    self._tikz_index += 1
    image_id = f"{self.chap.chap_id}-t{self._tikz_index:03d}"
    rel = f"tikz/{image_id}.tex"
    target = self.work_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.strip() + "\n", encoding="utf-8")
    self.manifest.images[image_id] = ImageRef(image_id, f"figures-gen/{image_id}.svg", "", rel)
    self.bump("tikz")
    return image_id

  def new_image(self, source: Path, width: str = "") -> str:
    """把原图镜像进站点树并登记。

    :param source: 原图路径。
    :param width: 宽度意图。
    :returns: image_id。
    """
    self._image_index += 1
    image_id = f"{self.chap.chap_id}-i{self._image_index:03d}"
    try:
      rel_source = source.relative_to(self.book_dir)
    except ValueError:
      rel_source = Path(source.name)
    dest = rel_source
    target = self.src_dir / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    self.manifest.images[image_id] = ImageRef(image_id, dest.as_posix(), width, source.as_posix())
    self.bump("image")
    return image_id


def normalize(
  chap: Chapter,
  book_dir: Path,
  work_dir: Path,
  src_dir: Path,
  registry=None,
) -> Manifest:
  """把一章 tex 归一化成 pandoc 可读的副本。

  :param chap: 目标章。
  :param book_dir: 书根目录。
  :param work_dir: 临时目录。
  :param src_dir: 站点源树。
  :param registry: 全书 label 注册表，供图内 ``\\ref`` 取编号。
  :returns: 该章的转换清单。
  """
  builder = _Builder(chap, book_dir, work_dir, src_dir, registry)
  text = flatten_inputs(chap.tex, book_dir)
  text = _extract_input_listings(text, builder)
  text = _extract_code_envs(text, builder)
  text = _convert_verb(text)
  # 代码载荷已落盘，此处剥掉注释：否则被注释的 ``% \end{tikzpicture}`` 会让环境
  # 配对错位（ch19b 的 wp5 就有这种写法）。注释在网页上没有意义。
  text = strip_comments(text)
  text = _extract_tikz(text, builder)
  text = _extract_images(text, builder)
  text = _unwrap_boxes(text, builder)
  text = _containers(text, builder)
  text = _isolate_math_envs(text)
  text = _isolate_display_dollars(text)
  text = _isolate_display_brackets(text)
  text = _expand_math_columns(text)
  text = _unwrap_shortstack(text)
  text = _lift_math_labels(text, builder)
  text = _tokenize_refs(text, builder)
  text = _normalize_math_primitives(text)
  text = _strip_wrapper_envs(text)
  text = _drop_noise(text)
  text = _tidy(text)

  norm = work_dir / "norm" / f"{chap.chap_id}.tex"
  norm.parent.mkdir(parents=True, exist_ok=True)
  norm.write_text(text, encoding="utf-8")

  manifest_dir = work_dir / "manifest"
  manifest_dir.mkdir(parents=True, exist_ok=True)
  (manifest_dir / f"{chap.chap_id}.json").write_text(builder.manifest.to_json(), encoding="utf-8")
  return builder.manifest


def load_manifest(chap_id: str, work_dir: Path) -> Manifest:
  """读回单章清单。

  :param chap_id: 章节标识。
  :param work_dir: 临时目录。
  :returns: 转换清单。
  """
  raw = json.loads((work_dir / "manifest" / f"{chap_id}.json").read_text(encoding="utf-8"))
  manifest = Manifest(chap_id=raw["chap_id"])
  manifest.codes = {k: CodeBlock(**v) for k, v in raw["codes"].items()}
  manifest.images = {k: ImageRef(**v) for k, v in raw["images"].items()}
  manifest.stats = raw["stats"]
  manifest.notes = raw["notes"]
  return manifest


def _apply_edits(text: str, edits: list[tuple[int, int, str]]) -> str:
  """按位置替换文本（从后往前，避免偏移失效）。

  :param text: 原文本。
  :param edits: ``(start, end, replacement)`` 列表。
  :returns: 替换后的文本。
  """
  for start, end, replacement in sorted(edits, key=lambda item: item[0], reverse=True):
    text = text[:start] + replacement + text[end:]
  return text


def _line_is_commented(text: str, pos: int) -> bool:
  """判断某位置所在行是否为注释行。

  :param text: 文本。
  :param pos: 位置。
  :returns: 是否为注释行。
  """
  start = text.rfind("\n", 0, pos) + 1
  return text[start:pos].lstrip().startswith("%")


def _read_inline_options(body: str) -> tuple[str | None, int]:
  """读取紧跟环境开始的 ``[...]`` 选项（不跨行，避免把正文当选项）。

  :param body: 环境体。
  :returns: ``(选项原文或 None, 内容起始下标)``。
  """
  cursor = 0
  while cursor < len(body) and body[cursor] in " \t":
    cursor += 1
  if cursor < len(body) and body[cursor] == "[":
    try:
      return read_bracket(body, cursor)
    except ValueError:
      return None, 0
  return None, 0


def _find_repo_root(book_dir: Path) -> Path:
  """向上找到含 ``.git`` 的仓库根目录。

  :param book_dir: 书根目录。
  :returns: 仓库根目录（找不到时返回书根目录）。
  """
  for candidate in [book_dir, *book_dir.parents]:
    if (candidate / ".git").exists():
      return candidate
  return book_dir


def _resolve_source(builder: _Builder, raw_path: str) -> Path | None:
  """解析代码清单引用的源码路径（相对书根目录）。

  书中有两处清单指向同级仓库（``../../../../ffpa-attn/...``）。本地开发时该路径
  存在、RTD 上只检出本仓库而不存在，所以这里给出提示，让本地预览与 RTD 构建的
  差异在报告里可见。

  :param builder: 构建状态。
  :param raw_path: 引用原文。
  :returns: 存在的路径或 None。
  """
  candidates = [builder.book_dir / raw_path, builder.chap.tex.parent / raw_path]
  for candidate in candidates:
    if candidate.is_file():
      try:
        candidate.resolve().relative_to(builder.repo_root)
      except ValueError:
        builder.note(
          f"{builder.chap.chap_id}: 清单引用位于本仓库之外（{raw_path}），"
          "RTD 构建时将显示提示块而非代码")
      return candidate
  builder.note(f"{builder.chap.chap_id}: 代码清单引用缺失 {raw_path}")
  return None


def _language_for(options: dict[str, str], path: Path | None) -> tuple[str, bool]:
  """推断代码语言与是否显示行号。

  :param options: listings 选项。
  :param path: 源码路径（内联环境为 None）。
  :returns: ``(语言, 是否显示行号)``。
  """
  style = options.get("style", "")
  if style in _STYLE_LANG:
    return _STYLE_LANG[style], False
  language = options.get("language", "").strip().lower()
  if language in _OPT_LANG:
    lang = _OPT_LANG[language]
    return lang, lang != "text"
  if path is not None:
    return _EXT_LANG.get(path.suffix.lower(), "text"), True
  return "cuda", True


def _parse_ranges(spec: str) -> list[tuple[int, int]]:
  """解析 ``linerange`` 选项。

  :param spec: 形如 ``"{119-139,200-210}"`` 或 ``"1-3"`` 的取值。
  :returns: ``[(起始行, 结束行), ...]``。
  """
  cleaned = spec.strip().strip("{}")
  ranges: list[tuple[int, int]] = []
  for chunk in cleaned.split(","):
    chunk = chunk.strip()
    if not chunk:
      continue
    if "-" in chunk:
      start, _, stop = chunk.partition("-")
      ranges.append((int(start), int(stop)))
    else:
      value = int(chunk)
      ranges.append((value, value))
  return ranges


def _emit_listing(builder: _Builder, path: Path | None, options: dict[str, str], raw_path: str) -> str:
  """按 listings 选项生成代码载荷 token 文本。

  :param builder: 构建状态。
  :param path: 源码路径（None 表示引用缺失）。
  :param options: listings 选项。
  :param raw_path: 引用原文（用于兜底标题）。
  :returns: 替换用的 token 文本。
  """
  if path is None:
    title = options.get("title", "").strip()
    detail = f"：{title}" if title else ""
    return (
      "\n\n```{admonition} 代码清单未内联\n"
      ":class: rtd-admon rtd-source-missing\n\n"
      f"原书此处引用 `{raw_path}`{detail}。该文件不在本仓库内（属外部仓库，"
      "如 ffpa-attn），站点无法取证内联，请到对应仓库查看。\n```\n\n")
  lang, numbering = _language_for(options, path)
  lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
  ranges = _parse_ranges(options.get("linerange", "")) or [(1, len(lines))]
  title = options.get("title", "")
  firstnumber = options.get("firstnumber", "auto").strip()

  blocks: list[str] = []
  for start, stop in ranges:
    snippet = "\n".join(lines[max(start - 1, 0):stop])
    # firstnumber 取值为 auto/空时跟随源码行号；显式数字则用它；last/continue 等
    # listings 合法但本书未用的取值回落到 start，避免整条管线被一个 ValueError 打断。
    if numbering:
      first_line = int(firstnumber) if firstnumber.isdigit() else start
    else:
      first_line = 0
    default_source = f"{path.name} L{start}-{stop}"
    source = default_source if len(ranges) > 1 or not title else title
    code_id = builder.new_code(lang, first_line, source, snippet)
    if title and len(ranges) == 1:
      blocks.append(
        f"{tokens.make(tokens.KIND_CODE_TITLE)} {title} {tokens.make(tokens.KIND_CODE_TITLE_END)}")
    elif len(ranges) > 1 and title:
      blocks.append(
        f"{tokens.make(tokens.KIND_CODE_TITLE)} {title}（{default_source}） {tokens.make(tokens.KIND_CODE_TITLE_END)}")
    else:
      blocks.append(
        f"{tokens.make(tokens.KIND_CODE_TITLE)} {default_source} {tokens.make(tokens.KIND_CODE_TITLE_END)}")
    blocks.append("")
    blocks.append(tokens.make(tokens.KIND_CODE, code_id))
    blocks.append("")
  return "\n".join(blocks)


def _extract_input_listings(text: str, builder: _Builder) -> str:
  """抽取 ``\\lstinputlisting``（命令形式与环境形式）。

  :param text: 章节 tex。
  :param builder: 构建状态。
  :returns: 替换后的文本。
  """
  edits: list[tuple[int, int, str]] = []

  for start, end, body in find_environments(text, "lstinputlisting"):
    if _line_is_commented(text, start):
      continue
    options_text, cursor = _read_inline_options(body)
    if cursor >= len(body):
      builder.note(f"{builder.chap.chap_id}: 无法解析的 lstinputlisting 环境")
      continue
    try:
      raw_path, _ = read_group_after_space(body, cursor)
    except ValueError:
      builder.note(f"{builder.chap.chap_id}: lstinputlisting 环境缺少源文件参数")
      continue
    options = split_options(options_text)
    replacement = _emit_listing(builder, _resolve_source(builder, raw_path), options, raw_path)
    edits.append((start, end, replacement))

  text = _apply_edits(text, edits)

  edits = []
  for match in re.finditer(r"\\lstinputlisting\b", text):
    if _line_is_commented(text, match.start()):
      continue
    options_text, cursor = read_optional(text, match.end())
    try:
      raw_path, end = read_group_after_space(text, cursor)
    except ValueError:
      builder.note(f"{builder.chap.chap_id}: lstinputlisting 缺少源文件参数")
      continue
    options = split_options(options_text or "")
    replacement = _emit_listing(builder, _resolve_source(builder, raw_path), options, raw_path)
    edits.append((match.start(), end, replacement))
  return _apply_edits(text, edits)


def _extract_code_envs(text: str, builder: _Builder) -> str:
  """抽取 ``lstlisting`` / ``python`` / ``cpp`` / ``verbatim`` 环境。

  :param text: 章节 tex。
  :param builder: 构建状态。
  :returns: 替换后的文本。
  """
  edits: list[tuple[int, int, str]] = []
  for env, forced_lang in _CODE_ENVS.items():
    for start, end, body in find_environments(text, env):
      if _line_is_commented(text, start):
        continue
      options_text, cursor = _read_inline_options(body)
      options = split_options(options_text or "")
      payload = body[cursor:].strip("\n")
      if forced_lang is not None:
        lang, numbering = forced_lang, False
      else:
        lang, numbering = _language_for(options, None)
      title = options.get("title", "")
      source = title or f"{env} 代码段"
      first_line = 0
      if numbering and options.get("firstnumber", "").strip().isdigit():
        first_line = int(options["firstnumber"].strip())
      code_id = builder.new_code(lang, first_line, source, payload)
      blocks = []
      if title:
        blocks.append(
          f"{tokens.make(tokens.KIND_CODE_TITLE)} {title} {tokens.make(tokens.KIND_CODE_TITLE_END)}")
        blocks.append("")
      blocks.append(tokens.make(tokens.KIND_CODE, code_id))
      edits.append((start, end, "\n".join(blocks)))
  return _apply_edits(text, edits)


#: ``\verb`` 内容里需要转义的字符（否则 ``%`` 会被当注释、``_``/``$`` 会被当数学）。
_VERB_ESCAPE = {
  "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "%": r"\%",
  "&": r"\&", "#": r"\#", "$": r"\$", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}

#: ``\verb<delim>...<delim>``。
_VERB_RE = re.compile(r"\\verb\*?([^A-Za-z\s])([^\n]*?)\1")


def _convert_verb(text: str) -> str:
  """把 ``\\verb|...|`` 转成 ``\\texttt{...}``。

  ``\\verb`` 的内容必须逐字保留（wp0 里就有 ``\\verb|(i,j) = (k%4,k/4)|``），
  所以先转义成 LaTeX 等价写法，之后的注释剥离与 pandoc 都不会破坏它。

  :param text: 章节 tex。
  :returns: 替换后的文本。
  """
  def replace(match: re.Match[str]) -> str:
    body = match.group(2)
    escaped = "".join(_VERB_ESCAPE.get(char, char) for char in body)
    return f"\\texttt{{{escaped}}}"

  return _VERB_RE.sub(replace, text)


#: 纯排版包装环境：网页上由 CSS 负责，剥掉环境标记即可。
#: 值是该环境 ``\begin{env}`` 之后需要一并丢弃的必选参数个数。
_WRAPPER_ENVS = {
  "center": 0,
  "flushleft": 0,
  "flushright": 0,
  "sloppypar": 0,
  "adjustbox": 1,
  "spacing": 1,
  "minipage": 1,
  "samepage": 0,
}


def _strip_wrapper_envs(text: str) -> str:
  """剥掉纯排版包装环境（``center`` / ``adjustbox`` / ``minipage`` 等）。

  pandoc 会把 ``\\begin{center}`` 变成 fenced div ``::: center``，而 MyST 的
  colon_fence 会把它当指令开头，后面整段内容就被吞成字面文本（ch01 的两张图就
  是这样丢的）。这些环境在网页上没有语义，直接去掉。

  :param text: 章节 tex。
  :returns: 处理后的文本。
  """
  for env, args in _WRAPPER_ENVS.items():
    pattern = re.compile(r"\\begin\{" + env + r"\}(?:\s*\[[^\]]*\])?" + r"(?:\s*\{[^{}]*\})?" * args)
    parts: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
      if _line_is_commented(text, match.start()):
        continue
      parts.append(text[cursor:match.start()])
      cursor = match.end()
    parts.append(text[cursor:])
    text = "".join(parts)
    text = re.sub(r"\\end\{" + env + r"\}", "", text)
  return text


#: 片段内的交叉引用（图里也会写 ``图~\ref{fig:x}``）。
_SNIPPET_REF_RE = re.compile(r"\\(?:eqref|ref)\{([^}]*)\}")


def _extract_tikz(text: str, builder: _Builder) -> str:
  """抽离 ``tikzpicture``，原位留下图片 token。

  :param text: 章节 tex。
  :param builder: 构建状态。
  :returns: 替换后的文本。
  """
  edits: list[tuple[int, int, str]] = []
  for start, end, _body in find_environments(text, "tikzpicture"):
    source = _resolve_snippet_refs(text[start:end], builder.registry)
    image_id = builder.new_tikz(source)
    edits.append((start, end, "\n\n" + tokens.make(tokens.KIND_IMAGE, image_id) + "\n\n"))
  return _apply_edits(text, edits)


def _resolve_snippet_refs(source: str, registry) -> str:
  """把图内 ``\\ref`` 换成实际编号。

  图里也会写 ``图~\\ref{fig:10-1}``；片段是单独编译的，孤立文档里这些 label 不存在，
  LaTeX 只能渲染成 ``??``。这里用全书注册表把编号直接写进片段。

  :param source: tikzpicture 片段源码。
  :param registry: label 注册表（可为 None）。
  :returns: 替换后的片段源码。
  """
  def replace(match: re.Match[str]) -> str:
    entry = registry.get(match.group(1)) if registry is not None else None
    if entry is None or not entry.number:
      return "??"
    if match.group(0).startswith("\\eqref"):
      return f"({entry.number})"
    return entry.number

  return _SNIPPET_REF_RE.sub(replace, source)


def _extract_images(text: str, builder: _Builder) -> str:
  """抽离活 ``\\includegraphics`` 并把原图镜像进站点树。

  :param text: 章节 tex。
  :param builder: 构建状态。
  :returns: 替换后的文本。
  """
  edits: list[tuple[int, int, str]] = []
  for match in re.finditer(r"\\includegraphics\b", text):
    if _line_is_commented(text, match.start()):
      continue
    options_text, cursor = read_optional(text, match.end())
    try:
      raw_path, end = read_group_after_space(text, cursor)
    except ValueError:
      continue
    options = split_options(options_text or "")
    width = _width_percent(options.get("width", ""))
    source = builder.book_dir / raw_path
    if not source.is_file():
      builder.note(f"{builder.chap.chap_id}: 图片缺失 {raw_path}")
      continue
    image_id = builder.new_image(source, width)
    edits.append((match.start(), end, tokens.make(tokens.KIND_IMAGE, image_id, _token_width(width))))
  return _apply_edits(text, edits)


def _width_percent(spec: str) -> str:
  """把 LaTeX 宽度表达式转成百分比。

  :param spec: 如 ``"0.92\\textwidth"`` / ``"\\linewidth"`` / ``"8cm"``。
  :returns: 形如 ``"92%"`` 的字符串，无法判断时为空串。
  """
  if not spec:
    return ""
  match = _WIDTH_RE.search(spec)
  if match is not None:
    return f"{float(match.group(1)) * 100:.0f}%"
  if "\\textwidth" in spec or "\\linewidth" in spec or "\\columnwidth" in spec:
    return "100%"
  return ""


def _token_width(width: str) -> str:
  """把百分比宽度转成 token 载荷（token 字符集不含 ``%``）。

  :param width: 形如 ``"22%"`` 的宽度。
  :returns: 纯数字载荷，非百分比宽度返回空串。
  """
  return width.rstrip("%") if width.endswith("%") else ""


def _set_width(text: str, width: str) -> str:
  """给文本中的图片 token 补上宽度。

  :param text: 片段。
  :param width: 宽度百分比。
  :returns: 替换后的片段。
  """
  if not width:
    return text
  payload = _token_width(width)
  return _IMG_TOKEN_RE.sub(lambda m: tokens.make(tokens.KIND_IMAGE, m.group(1), payload), text)


def _unwrap_boxes(text: str, builder: _Builder) -> str:
  """拆掉 ``\\resizebox`` / ``\\scalebox`` / ``\\adjustbox``，把宽度记到图片 token。

  :param text: 章节 tex。
  :param builder: 构建状态。
  :returns: 替换后的文本。
  """
  for name, take_width in (("resizebox", True), ("scalebox", False), ("adjustbox", True)):
    while True:
      match = re.search(r"\\" + name + r"\b", text)
      if match is None:
        break
      cursor = match.end()
      args: list[str] = []
      width = ""
      try:
        optional, cursor = read_optional(text, cursor)
        if name == "adjustbox" and optional:
          width = _width_percent(optional)
        first, cursor = read_group(text, cursor, "{", "}")
        if take_width and name != "adjustbox":
          width = _width_percent(first)
        if name == "resizebox":
          _, cursor = read_group(text, cursor, "{", "}")
        content, cursor = read_group(text, cursor, "{", "}")
      except ValueError:
        builder.note(f"{builder.chap.chap_id}: {name} 参数解析失败")
        text = text[:match.start()] + text[match.end():]
        continue
      _ = args
      text = text[:match.start()] + _set_width(content, width) + text[cursor:]
  return text


def _containers(text: str, builder: _Builder) -> str:
  """figure / table / subfigure / caption / 定理环境 → token。

  :param text: 章节 tex。
  :param builder: 构建状态。
  :returns: 替换后的文本。
  """
  text = _wrap_environment(text, "figure", tokens.KIND_FIGURE, tokens.KIND_FIGURE_END, builder, "figure")
  text = _wrap_environment(text, "table", tokens.KIND_TABLE, tokens.KIND_TABLE_END, builder, "table")
  # longtable 交给 pandoc 原生转换（它的列声明参数是必须保留的表格结构）
  text = _wrap_subfigures(text, builder)
  text = _wrap_captions(text)
  text = _wrap_admonitions(text, builder)
  text = _rewrite_envs(text)
  return text


def _wrap_environment(
  text: str, env: str, start_kind: str, end_kind: str, builder: _Builder, stat: str) -> str:
  """把环境包成 begin/end token。

  :param text: 章节 tex。
  :param env: 环境名。
  :param start_kind: 起始 token 种类。
  :param end_kind: 结束 token 种类。
  :param builder: 构建状态。
  :param stat: 统计名。
  :returns: 替换后的文本。
  """
  edits: list[tuple[int, int, str]] = []
  for start, end, body in find_environments(text, env):
    if _line_is_commented(text, start):
      continue
    builder.bump(stat)
    cleaned = body
    _position, cursor = read_optional(body, 0)
    if _position is not None:
      cleaned = body[cursor:]
    edits.append((
      start, end,
      "\n\n" + tokens.make(start_kind) + "\n\n" + cleaned + "\n\n" + tokens.make(end_kind) + "\n\n"))
  return _apply_edits(text, edits)


def _wrap_subfigures(text: str, builder: _Builder) -> str:
  """把 ``subfigure`` 包成带宽度载荷的 token。

  :param text: 章节 tex。
  :param builder: 构建状态。
  :returns: 替换后的文本。
  """
  edits: list[tuple[int, int, str]] = []
  for start, end, body in find_environments(text, "subfigure"):
    options_text, cursor = read_optional(body, 0)
    width = ""
    try:
      spec, cursor = read_group(body, cursor)
      width = _width_percent(spec)
    except ValueError:
      builder.note(f"{builder.chap.chap_id}: subfigure 宽度解析失败")
    builder.bump("subfigure")
    _ = options_text
    edits.append((
      start, end,
      "\n\n" + tokens.make(tokens.KIND_SUBFIGURE, _token_width(width)) + "\n\n" + body[cursor:]
      + "\n\n" + tokens.make(tokens.KIND_SUBFIGURE_END) + "\n\n"))
  return _apply_edits(text, edits)


def _wrap_captions(text: str) -> str:
  """把 ``\\caption`` 包成 token。

  :param text: 章节 tex。
  :returns: 替换后的文本。
  """
  edits: list[tuple[int, int, str]] = []
  for match in re.finditer(r"\\caption\b", text):
    if _line_is_commented(text, match.start()):
      continue
    cursor = match.end()
    optional, cursor = read_optional(text, cursor)
    try:
      body, end = read_group_after_space(text, cursor)
    except ValueError:
      continue
    _ = optional
    edits.append((
      match.start(), end,
      f"{tokens.make(tokens.KIND_CAPTION)} {body.strip()} {tokens.make(tokens.KIND_CAPTION_END)}"))
  return _apply_edits(text, edits)


def _wrap_admonitions(text: str, builder: _Builder) -> str:
  """定理类环境 → admonition token。

  :param text: 章节 tex。
  :param builder: 构建状态。
  :returns: 替换后的文本。
  """
  edits: list[tuple[int, int, str]] = []
  for env in ADMON_KINDS:
    for start, end, body in find_environments(text, env):
      if _line_is_commented(text, start):
        continue
      tag, cursor = read_optional(body, 0)
      title = ""
      if cursor < len(body) and body[cursor] == "{":
        try:
          title, cursor = read_group(body, cursor)
        except ValueError:
          title = ""
      note = " ".join(part for part in (f"〔{tag.strip()}〕" if tag else "", title.strip()) if part)
      builder.bump(env)
      edits.append((
        start, end,
        "\n\n" + tokens.make(tokens.KIND_ADMON, env) + (" " + note if note else "")
        + " " + tokens.make(tokens.KIND_ADMON_TITLE_END) + "\n\n"
        + body[cursor:] + "\n\n" + tokens.make(tokens.KIND_ADMON_END) + "\n\n"))
  return _apply_edits(text, edits)


def _rewrite_envs(text: str) -> str:
  """删除/替换纯排版环境标记。

  :param text: 章节 tex。
  :returns: 替换后的文本。
  """
  for env, replacement in _ENV_REWRITE.items():
    if replacement is None:
      text = re.sub(r"\\begin\{" + env + r"\}(\[[^\]]*\])?(\{[^{}]*\})?", "\n\n", text)
      text = text.replace(f"\\end{{{env}}}", "\n\n")
    else:
      text = text.replace(f"\\begin{{{env}}}", f"\\begin{{{replacement[0]}}}")
      text = text.replace(f"\\end{{{env}}}", f"\\end{{{replacement[1]}}}")
  return text


_MATH_LABEL_ENVS = ("equation", "equation*", "align", "align*", "gather", "multline", "eqnarray")


#: 独立成段的数学环境（内含 aligned/cases 等不算独立环境）。
_ISOLATED_MATH_ENVS = ("equation", "align", "gather", "multline", "eqnarray", "alignat", "flalign",
                       "displaymath")


def _isolate_math_envs(text: str) -> str:
  """把数学环境与前后正文分开（独占段落）。

  书里 ``\\begin{equation}`` 常常紧贴正文（``……共 $2K$ 次；所以\\begin{equation}``），
  pandoc 就会把 ``$$…$$`` 就地放在段落中间。MyST 的显示公式必须自成一段，夹在正文里
  会被拆成「字面 ``$`` + 行内公式 + 字面 ``$``」——页面上公式直接垮掉，还多出孤立的
  美元符号。这里只加空行，不改公式内容。

  :param text: 章节 tex。
  :returns: 处理后的文本。
  """
  edits: list[tuple[int, int, str]] = []
  for env in _ISOLATED_MATH_ENVS:
    for name in (env, f"{env}*"):
      for start, end, _body in find_environments(text, name):
        block = text[start:end].strip("\n")
        edits.append((start, end, "\n\n" + block + "\n\n"))
  return _apply_edits(text, edits)


#: 正文里直接写的显示公式定界符（未被 ``$$`` 转义、也不在注释里）。
_DISPLAY_DOLLAR_RE = re.compile(r"(?<!\\)\$\$")


def _isolate_display_dollars(text: str) -> str:
  """把正文里成对的 ``$$…$$`` 独立成段。

  书里除了 ``\\begin{equation}``，也直接写 ``$$…$$``（ch25 的 logical_divide 推导），
  同样是紧贴正文的。与其让 MyST 把它拆成行内公式，不如先加空行。

  :param text: 章节 tex（已剥注释）。
  :returns: 处理后的文本。
  """
  markers = [match.start() for match in _DISPLAY_DOLLAR_RE.finditer(text)]
  if len(markers) < 2:
    return text
  edits: list[tuple[int, int, str]] = []
  for index in range(0, len(markers) - 1, 2):
    start, end = markers[index], markers[index + 1] + 2
    edits.append((start, end, "\n\n" + text[start:end].strip("\n") + "\n\n"))
  return _apply_edits(text, edits)


#: 正文里以 ``\[ … \]`` 书写的显示公式。
#: 必须排除 ``\\[2pt]``（换行+间距命令）里的那个 ``\[``，否则会把整段括进去。
_DISPLAY_BRACKET_RE = re.compile(r"(?<!\\)\\\[(.*?)(?<!\\)\\\]", re.S)


def _isolate_display_brackets(text: str) -> str:
  """把 ``\\[…\\]`` 显示公式独立成段。

  与 ``$$…$$`` 同理：紧贴正文时 pandoc 会把它输出成段中的 ``$$…$$``，MyST 会拆成
  「字面 ``$`` + 行内公式 + 字面 ``$``」（ch25 的 logical_divide 推导就是这样垮的）。

  :param text: 章节 tex（已剥注释）。
  :returns: 处理后的文本。
  """
  edits: list[tuple[int, int, str]] = []
  for match in _DISPLAY_BRACKET_RE.finditer(text):
    edits.append((match.start(), match.end(), "\n\n" + match.group(0).strip("\n") + "\n\n"))
  return _apply_edits(text, edits)


#: array 宏包的「整列数学模式」声明，例如 ``>{$}c<{$}``、``>{\small}l``。
_COLUMN_DECORATION_RE = re.compile(r">\{(?P<pre>[^{}]*)\}|<\{(?P<post>[^{}]*)\}")


def _split_top_level(text: str, separator: str) -> list[str]:
  """按顶层分隔符切分（忽略花括号与嵌套环境内部的分隔符）。

  表格行里的 ``&`` / ``\\\\`` 可能属于嵌套的 ``bmatrix`` 等环境
  （``\\begin{bmatrix}1 & 8\\end{bmatrix}``），不能当单元格/行分隔符。

  :param text: 表格体。
  :param separator: ``"&"`` 或 ``"\\\\"``。
  :returns: 切分结果。
  """
  parts: list[str] = []
  current: list[str] = []
  depth = 0
  envs: list[str] = []
  index = 0
  length = len(text)
  while index < length:
    if text.startswith("\\begin{", index):
      end = text.find("}", index)
      envs.append(text[index:end + 1])
      current.append(text[index:end + 1])
      index = end + 1
      continue
    if text.startswith("\\end{", index):
      end = text.find("}", index)
      if envs:
        envs.pop()
      current.append(text[index:end + 1])
      index = end + 1
      continue
    if text.startswith(separator, index) and depth == 0 and not envs:
      parts.append("".join(current))
      current = []
      index += len(separator)
      continue
    char = text[index]
    if char == "\\" and not text.startswith(separator, index):
      current.append(text[index:index + 2])
      index += 2
      continue
    if char == "{":
      depth += 1
    elif char == "}":
      depth -= 1
    current.append(char)
    index += 1
  parts.append("".join(current))
  return parts


def _expand_math_columns(text: str) -> str:
  """把 ``>{$}c<{$}`` 展开成普通列声明 + 逐格 ``$…$``。

  ``>{$}c<{$}`` 的含义是「该列每个单元格都在数学模式里」。pandoc 解析不了这种声明：
  与 ``\\shortstack`` 同时出现时整张表会被丢掉（wp2 的线性形式表就这样消失了），
  即使侥幸出表，``\\v{L}`` 也会被当成重音命令渲染成 ``Ľ``。展开后语义等价、
  pandoc 能正常出表。

  :param text: 章节 tex。
  :returns: 处理后的文本。
  """
  edits: list[tuple[int, int, str]] = []
  for env in ("tabular", "tabularx", "longtable"):
    for start, end, _body in find_environments(text, env):
      block = text[start:end]
      expanded = _expand_one_table(block, env)
      if expanded != block:
        edits.append((start, end, expanded))
  return _apply_edits(text, edits)


def _expand_one_table(block: str, env: str) -> str:
  """展开单张表里的数学模式列声明。

  :param block: ``\\begin{env}...\\end{env}`` 原文。
  :param env: 环境名。
  :returns: 展开后的表（无数学模式列时原样返回）。
  """
  head = f"\\begin{{{env}}}"
  cursor = len(head)
  while cursor < len(block) and block[cursor] == "[":
    close = block.find("]", cursor)
    if close == -1:
      return block
    cursor = close + 1
  if cursor >= len(block) or block[cursor] != "{":
    return block
  try:
    spec, after_spec = read_group(block, cursor)
  except ValueError:
    return block

  columns: list[tuple[str, bool]] = []
  math_columns: list[int] = []
  index = 0
  pending_math = False
  while index < len(spec):
    match = _COLUMN_DECORATION_RE.match(spec, index)
    if match is not None:
      if match.group("pre") is not None and "$" in match.group("pre"):
        pending_math = True
      index = match.end()
      continue
    char = spec[index]
    if char in "lcrXpmb":
      columns.append((char, pending_math))
      if pending_math:
        math_columns.append(len(columns) - 1)
      pending_math = False
    elif char == "|":
      pending_math = False
    index += 1
  if not math_columns:
    return block

  new_spec = "".join(("|" if char == "|" else char) for char in spec if char in "|lcrXpmb")
  new_spec = _rebuild_spec(spec, columns)
  body = block[after_spec:block.rfind(f"\\end{{{env}}}")]
  rows = _split_top_level(body, "\\\\")
  rebuilt_rows = []
  for row in rows:
    cells = _split_top_level(row, "&")
    for position in math_columns:
      if position >= len(cells):
        continue
      cell = cells[position]
      if "\\multicolumn" in cell or "$" in cell:
        continue
      stripped = cell.strip()
      if not stripped or stripped.startswith("\\hline") or stripped.startswith("\\midrule"):
        continue
      leading = cell[:len(cell) - len(cell.lstrip())]
      trailing = cell[len(cell.rstrip()):]
      cells[position] = f"{leading}${stripped}${trailing}"
    rebuilt_rows.append("&".join(cells))
  return head + new_spec + "\\\\".join(rebuilt_rows) + f"\\end{{{env}}}"


def _rebuild_spec(original: str, columns: list[tuple[str, bool]]) -> str:
  """按原顺序重建列声明（去掉 ``>{...}`` / ``<{...}`` 装饰，保留竖线位置）。

  :param original: 原列声明。
  :param columns: ``(列字母, 是否数学模式)`` 列表。
  :returns: 新列声明。
  """
  letters = [letter for letter, _math in columns]
  result: list[str] = []
  used = 0
  index = 0
  while index < len(original):
    match = _COLUMN_DECORATION_RE.match(original, index)
    if match is not None:
      index = match.end()
      continue
    char = original[index]
    if char in "|@!":
      result.append(char)
    elif char in "lcrXpmb":
      if used < len(letters):
        result.append(letters[used])
        used += 1
    index += 1
  while used < len(letters):
    result.append(letters[used])
    used += 1
  return "{" + "".join(result) + "}"


#: 格内换行命令（``\shortstack{A\\ B}`` 表示单元格里分两行）。
_SHORTSTACK_RE = re.compile(r"\\shortstack\b")


def _unwrap_shortstack(text: str) -> str:
  """把 ``\\shortstack{…\\\\…}`` 压成单行文本。

  它是「单元格内分两行」的写法，全书 77 处全在表格里。pandoc 会把内容里的 ``\\\\``
  当成表格换行，把一行拆成两行（wp2 的线性形式表就多出两行空壳）。压成一行后既不出
  错，信息也不丢。

  :param text: 章节 tex。
  :returns: 处理后的文本。
  """
  out: list[str] = []
  index = 0
  while True:
    match = _SHORTSTACK_RE.search(text, index)
    if match is None:
      out.append(text[index:])
      break
    out.append(text[index:match.start()])
    cursor = match.end()
    while cursor < len(text) and text[cursor] in " \t\n":
      cursor += 1
    if cursor < len(text) and text[cursor] == "[":
      close = text.find("]", cursor)
      if close == -1:
        out.append(match.group(0))
        index = match.end()
        continue
      cursor = close + 1
      while cursor < len(text) and text[cursor] in " \t\n":
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
      out.append(match.group(0))
      index = match.end()
      continue
    try:
      inner, end = read_group(text, cursor)
    except ValueError:
      out.append(match.group(0))
      index = match.end()
      continue
    out.append(" ".join(part.strip() for part in inner.split("\\\\") if part.strip()))
    index = end
  return "".join(out)


def _lift_math_labels(text: str, builder: _Builder) -> str:
  """把数学环境内的 ``\\label`` 提到环境外（否则会污染公式）。

  :param text: 章节 tex。
  :param builder: 构建状态。
  :returns: 替换后的文本。
  """
  for env in _MATH_LABEL_ENVS:
    while True:
      target = None
      for start, end, body in find_environments(text, env):
        if "\\label{" in body:
          target = (start, end, body)
          break
      if target is None:
        break
      start, end, body = target
      labels = re.findall(r"\\label\{([^}]*)\}", body)
      cleaned = re.sub(r"\\label\{[^}]*\}", "", body)
      prefix = "".join(tokens.make(tokens.KIND_LABEL, label) + "\n\n" for label in labels)
      builder.bump("math_label", len(labels))
      rebuilt = f"\\begin{{{env}}}{cleaned}\\end{{{env}}}"
      text = text[:start] + "\n\n" + prefix + rebuilt + "\n\n" + text[end:]
  return text


def _plain_number(builder: _Builder, label: str, parenthesized: bool) -> str:
  """把引用直接写成编号文本（数学区内用）。

  :param builder: 构建状态。
  :param label: 引用 label。
  :param parenthesized: 是否加圆括号（``\\eqref`` 语义）。
  :returns: 编号文本，解析不到时返回 ``??``（与 LaTeX 的表现一致）。
  """
  entry = builder.registry.get(label) if builder.registry is not None else None
  if entry is None or not entry.number:
    builder.note(f"{builder.chap.chap_id}: 数学区内的引用解析不到（{label}）")
    return "??"
  return f"({entry.number})" if parenthesized else entry.number


#: ``\label`` / ``\ref`` / ``\eqref`` / ``\pageref``。
_REF_CMD_RE = re.compile(r"\\(label|eqref|pageref|ref)\{([^}]*)\}")


def _tokenize_refs(text: str, builder: _Builder) -> str:
  """``\\label`` / ``\\ref`` / ``\\eqref`` → token。

  **数学区内的引用不走 token**：例如 ch34 的
  ``\\xrightarrow[\\text{第\\ref{ch:35}章…}]{…}``，站内链接塞进公式会让 MathJax
  直接解析失败（页面上留下半截公式加一串链接文本）。这类位置直接写编号。

  :param text: 章节 tex。
  :param builder: 构建状态。
  :returns: 替换后的文本。
  """
  spans = math_spans(text)

  def inside_math(pos: int) -> bool:
    return any(start <= pos < end for start, end in spans)

  chunks: list[str] = []
  cursor = 0
  for match in _REF_CMD_RE.finditer(text):
    command, label = match.group(1), match.group(2)
    chunks.append(text[cursor:match.start()])
    if command == "label":
      chunks.append(" " + tokens.make(tokens.KIND_LABEL, label) + " ")
    elif inside_math(match.start()):
      chunks.append(_plain_number(builder, label, command == "eqref"))
      builder.bump("math_ref")
    elif command == "eqref":
      chunks.append(" " + tokens.make(tokens.KIND_EQREF, label) + " ")
    else:
      chunks.append(" " + tokens.make(tokens.KIND_REF, label) + " ")
    cursor = match.end()
  chunks.append(text[cursor:])
  result = "".join(chunks)
  builder.bump("ref", len(re.findall(r"@@RTD;(?:REF|EREF);", result)))
  return result


#: plain TeX 的上下标原语（pandoc 与 MathJax 都不认，需换成 LaTeX 写法）。
_PRIMITIVE_SUB_RE = re.compile(r"\\sb(?![a-zA-Z])")
_PRIMITIVE_SUP_RE = re.compile(r"\\sp(?![a-zA-Z])")


def _normalize_math_primitives(text: str) -> str:
  """把 ``\\sb`` / ``\\sp`` 换成 ``_`` / ``^``。

  书中大量使用 ``$c\\sb{ij}$`` 这类 plain TeX 写法；pandoc 原样透传，而 MathJax
  不认识这两个原语，公式会渲染失败。文本模式下这两条命令本就是 LaTeX 错误，所以
  书中出现处必然在数学区，可以放心替换。

  :param text: 章节 tex。
  :returns: 替换后的文本。
  """
  text = _PRIMITIVE_SUB_RE.sub("_", text)
  return _PRIMITIVE_SUP_RE.sub("^", text)


#: booktabs 的局部横线与间距命令：pandoc 不认识它们的参数，会把
#: ``\cmidrule(lr){2-3}`` 当成普通文本塞进表格单元格（ch37 的量化开销表就多出一行
#: ``2-3(lr)4-5``），必须提前删掉。``\toprule/\midrule/\bottomrule`` 保留，
#: pandoc 认识它们。
_PARTIAL_RULES = (
  r"\\cmidrule\s*(?:\([^)]*\))?\s*\{[^{}]*\}",
  r"\\cline\s*\{[^{}]*\}",
  r"\\specialrule\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\{[^{}]*\}",
  r"\\addlinespace\s*(?:\[[^\]]*\])?",
  r"\\arrayrulecolor\s*\{[^{}]*\}",
  r"\\morecmidrules",
)


def _drop_noise(text: str) -> str:
  """清掉网页上无意义的排版命令。

  :param text: 章节 tex。
  :returns: 清理后的文本。
  """
  text = _drop_providecommands(text)
  for pattern in _PARTIAL_RULES:
    text = re.sub(pattern, "", text)
  text = re.sub(r"\\allowbreak\s*\{\}", "", text)
  text = re.sub(r"\\allowbreak\b", "", text)
  for name in _DROP_ZERO_ARG:
    text = re.sub(r"\\" + name + r"\b", " ", text)
  for name, count in _DROP_WITH_ARGS:
    text = _drop_command(text, name, count)
  text = text.replace("\\textbackslash", "\\")
  return text


def _drop_providecommands(text: str) -> str:
  """删除 ``\\providecommand`` 兼容宏定义（如单章编译用的 ``\\url`` 降级）。

  只处理 ``\\providecommand``：``\\newcommand`` / ``\\def`` 里的数学宏（
  ch19b 的 ``\\abs`` / ``\\norm`` 等）必须留给 pandoc 展开。

  :param text: 章节 tex。
  :returns: 清理后的文本。
  """
  pattern = re.compile(r"\\providecommand\s*\*?\s*\{")
  while True:
    match = pattern.search(text)
    if match is None:
      return text
    cursor = match.end() - 1
    try:
      _, cursor = read_group(text, cursor)
      _, cursor = read_optional(text, cursor)
      if cursor < len(text) and text[cursor] == "{":
        _, cursor = read_group(text, cursor)
    except ValueError:
      text = text[:match.start()] + text[match.end():]
      continue
    text = text[:match.start()] + text[cursor:]


def _drop_command(text: str, name: str, groups: int) -> str:
  """删除带固定个数分组参数的命令。

  :param text: 章节 tex。
  :param name: 命令名（不含反斜杠）。
  :param groups: 分组参数个数。
  :returns: 替换后的文本。
  """
  pattern = re.compile(r"\\" + name + r"\b")
  while True:
    match = pattern.search(text)
    if match is None:
      return text
    cursor = match.end()
    end = cursor
    ok = True
    for _ in range(groups):
      optional, cursor = read_optional(text, cursor)
      _ = optional
      try:
        _, cursor = read_group(text, cursor)
      except ValueError:
        ok = False
        break
      end = cursor
    if not ok:
      text = text[:match.start()] + text[match.end():]
      continue
    text = text[:match.start()] + text[end:]


def _tidy(text: str) -> str:
  """收敛空行与行尾空白。

  :param text: 章节 tex。
  :returns: 整理后的文本。
  """
  text = re.sub(r"[ \t]+\n", "\n", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  return text.strip() + "\n"


def find_pandoc(explicit: str | None = None) -> str:
  """定位 pandoc 可执行文件。

  :param explicit: 显式指定的路径。
  :returns: pandoc 路径。
  :raises RuntimeError: 找不到 pandoc。
  """
  if explicit:
    return explicit
  found = shutil.which("pandoc")
  if found:
    return found
  try:
    import pypandoc
  except ImportError as exc:
    raise RuntimeError("未找到 pandoc：请安装 pandoc 或 pip install pypandoc-binary") from exc
  return pypandoc.get_pandoc_path()


def to_markdown(norm_tex: Path, out_md: Path, pandoc: str, top_level: str = "chapter") -> str:
  """调用 pandoc 把归一化 tex 转成 markdown。

  :param norm_tex: 归一化 tex。
  :param out_md: 输出 markdown。
  :param pandoc: pandoc 路径。
  :param top_level: pandoc 的 ``--top-level-division`` 取值。
  :returns: pandoc 的 stderr（可能为空）。
  :raises RuntimeError: pandoc 退出码非 0。
  """
  out_md.parent.mkdir(parents=True, exist_ok=True)
  # 只保留 pipe 表：MyST 不认识 pandoc 的 grid/simple/multiline 表，留着会在页面上
  # 变成一堆 `+---+` 文本。关掉这三种后 pandoc 会把表格降级为 pipe 表（单元格内的
  # 换行折成空格）。
  command = [
    pandoc,
    "-f", "latex",
    "-t", "markdown-grid_tables-multiline_tables-simple_tables",
    "--wrap=none",
    "--markdown-headings=atx",
    f"--top-level-division={top_level}",
    str(norm_tex),
    "-o", str(out_md),
  ]
  proc = subprocess.run(command, capture_output=True, text=True, check=False)
  if proc.returncode != 0:
    raise RuntimeError(f"pandoc 失败（{norm_tex.name}）：{proc.stderr.strip()[:800]}")
  return proc.stderr
