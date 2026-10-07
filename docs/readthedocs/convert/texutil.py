"""LaTeX 扫描工具：括号配对读取、注释剥离、环境定位、输入展平。

这些函数只做文本层面的扫描，不理解 LaTeX 语义——转换管线需要它们把原书 tex
切分成可独立处理的片段（tikz 图、代码清单、环境块），原文件一概不改。
"""
from __future__ import annotations

import re
from pathlib import Path

#: 未转义的 ``%`` 到行尾即注释。
_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")

#: ``\input`` / ``\include`` 引用。
_INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]*)\}")

#: 无参数的命令名（用于纯文本降级）。
_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?")

#: 常见包装命令：只保留其内容。
_WRAPPER_RE = re.compile(
  r"\\(?:text(?:bf|it|rm|sf|tt|sc|sl|up|md|normal)|emph|mbox|hbox|math(?:rm|bf|it|cal|sf|tt)|"
  r"operatorname|ensuremath|displaystyle|left|right|bigl|bigr|Bigl|Bigr|sb|sp|quad|qquad)\b")


def strip_comments(text: str) -> str:
  """删除 LaTeX 注释（未转义的 ``%`` 至行尾）。

  :param text: tex 文本。
  :returns: 去注释后的文本。
  """
  return _COMMENT_RE.sub("", text)


def read_group(text: str, pos: int, opener: str = "{", closer: str = "}") -> tuple[str, int]:
  """读取从 ``pos`` 起的配对分组内容。

  :param text: tex 文本。
  :param pos: ``opener`` 所在下标。
  :param opener: 起始符。
  :param closer: 结束符。
  :returns: ``(组内文本, 结束符之后的下标)``。
  :raises ValueError: 起始符缺失或分组不闭合。
  """
  if pos >= len(text) or text[pos] != opener:
    raise ValueError(f"位置 {pos} 处不是 {opener!r}")
  depth = 0
  i = pos
  while i < len(text):
    char = text[i]
    if char == "\\":
      i += 2
      continue
    if char == opener:
      depth += 1
    elif char == closer:
      depth -= 1
      if depth == 0:
        return text[pos + 1:i], i + 1
    i += 1
  raise ValueError(f"从位置 {pos} 起的分组不闭合")


def read_bracket(text: str, pos: int) -> tuple[str, int]:
  """读取配对的中括号参数，忽略数学区内的括号。

  listings 的 ``title={... $[M,M{+}B)$ ...}`` 会在方括号里再出现方括号，而
  ``$[M,M{+}B)$`` 这类数学里的中括号并不配对——所以扫描时要感知 ``$`` 数学模式
  与花括号嵌套，否则会把选项读到文件末尾。

  :param text: tex 文本。
  :param pos: ``[`` 所在下标。
  :returns: ``(参数内容, 结束符之后的下标)``。
  :raises ValueError: 中括号不闭合。
  """
  if pos >= len(text) or text[pos] != "[":
    raise ValueError(f"位置 {pos} 处不是 [")
  depth = 0
  in_math = False
  i = pos
  while i < len(text):
    char = text[i]
    if char == "\\":
      i += 2
      continue
    if char == "$":
      in_math = not in_math
      i += 1
      continue
    if not in_math:
      if char in "[{":
        depth += 1
      elif char in "]}":
        depth -= 1
        if depth == 0:
          return text[pos + 1:i], i + 1
    i += 1
  raise ValueError(f"从位置 {pos} 起的中括号不闭合")


def read_optional(text: str, pos: int) -> tuple[str | None, int]:
  """读取可选的 ``[...]`` 参数。

  括号不闭合时（正文里恰好出现一个 ``[``）视为「没有可选参数」，返回原位置，
  避免把普通文本误判成参数。

  :param text: tex 文本。
  :param pos: 起始下标（允许前置空白）。
  :returns: ``(参数内容或 None, 新下标)``。
  """
  i = pos
  while i < len(text) and text[i] in " \t\n":
    i += 1
  if i < len(text) and text[i] == "[":
    try:
      body, end = read_bracket(text, i)
    except ValueError:
      return None, pos
    return body, end
  return None, pos


def is_commented_at(text: str, pos: int) -> bool:
  """判断某个位置是否落在 LaTeX 注释里。

  :param text: tex 文本。
  :param pos: 待判断的位置。
  :returns: 该位置所在行的行首到 ``pos`` 之间有未转义的 ``%`` 时为 True。
  """
  line_start = text.rfind("\n", 0, pos) + 1
  line = text[line_start:pos]
  index = 0
  while index < len(line):
    if line[index] == "\\":
      index += 2
      continue
    if line[index] == "%":
      return True
    index += 1
  return False


def read_group_after_space(text: str, pos: int) -> tuple[str, int]:
  """跳过空白后读取分组参数。

  LaTeX 允许在命令与参数之间换行，listings 的写法就有「选项结束后换行再接源文件
  路径」的形式，直接 ``read_group`` 会失败。

  :param text: tex 文本。
  :param pos: 起始下标。
  :returns: ``(组内文本, 结束符之后的下标)``。
  :raises ValueError: 跳过空白后不是 ``{`` 或分组不闭合。
  """
  i = pos
  while i < len(text) and text[i] in " \t\n":
    i += 1
  return read_group(text, i)


def find_environments(text: str, name: str) -> list[tuple[int, int, str]]:
  """定位所有同名环境的跨度（支持嵌套）。

  :param text: tex 文本。
  :param name: 环境名，如 ``"tikzpicture"``。
  :returns: ``[(起始下标, 结束下标, 环境体), ...]``，按下标升序。
  """
  begin_re = re.compile(r"\\begin\{" + re.escape(name) + r"\}")
  end_re = re.compile(r"\\end\{" + re.escape(name) + r"\}")
  spans: list[tuple[int, int, str]] = []
  pos = 0
  while True:
    start = begin_re.search(text, pos)
    if start is None:
      return spans
    depth = 1
    cursor = start.end()
    while depth > 0:
      nxt_begin = begin_re.search(text, cursor)
      nxt_end = end_re.search(text, cursor)
      if nxt_end is None:
        raise ValueError(f"环境 {name} 缺少 \\end")
      if nxt_begin is not None and nxt_begin.start() < nxt_end.start():
        depth += 1
        cursor = nxt_begin.end()
      else:
        depth -= 1
        cursor = nxt_end.end()
        if depth == 0:
          spans.append((start.start(), nxt_end.end(), text[start.end():nxt_end.start()]))
          pos = nxt_end.end()


def flatten_inputs(path: Path, root: Path, depth: int = 0) -> str:
  """递归展开 ``\\input`` / ``\\include``，等价于 LaTeX 的包含语义。

  相对路径按书的主文件所在目录（``root``）解析，与 xelatex 在 ``book/`` 下
  编译时的行为一致。

  :param path: 待展开的 tex 文件。
  :param root: 书根目录（``book/``）。
  :param depth: 当前递归深度，用于兜底防环。
  :returns: 展开后的 tex 文本。
  :raises FileNotFoundError: 引用的文件不存在。
  """
  text = path.read_text(encoding="utf-8")
  if depth > 8:
    return text

  def replace(match: re.Match[str]) -> str:
    if is_commented_at(text, match.start()):
      return match.group(0)
    target = match.group(1).strip()
    if not target:
      return ""
    if not target.endswith(".tex"):
      target += ".tex"
    included = root / target
    if not included.is_file():
      raise FileNotFoundError(f"{path.name} 引用了不存在的文件：{target}")
    return flatten_inputs(included, root, depth + 1)

  return _INPUT_RE.sub(replace, text)


def plain_text(tex: str) -> str:
  """把 tex 片段降级为纯文本（用于图片 alt、链接 title、文件名）。

  :param tex: tex 片段。
  :returns: 单行纯文本。
  """
  text = strip_comments(tex)
  text = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", text)
  text = _WRAPPER_RE.sub(" ", text)
  text = _CMD_RE.sub("", text)
  text = text.replace("{", "").replace("}", "")
  text = text.replace("~", " ").replace("\\", "")
  text = re.sub(r"\s+", " ", text)
  return text.strip()


def split_options(options: str) -> dict[str, str]:
  """解析 ``key=value,key={...}`` 形式的可选参数。

  :param options: 可选参数原文（不含方括号）。
  :returns: 键值映射，值为空串表示无值开关。
  """
  result: dict[str, str] = {}
  pos = 0
  while pos < len(options):
    match = re.compile(r"\s*([A-Za-z]+)\s*=").match(options, pos)
    if match is None:
      match = re.compile(r"\s*([A-Za-z]+)\s*(?=,)").match(options, pos)
      if match is None:
        break
      result[match.group(1)] = ""
      pos = match.end()
      if pos < len(options) and options[pos] == ",":
        pos += 1
      continue
    key = match.group(1)
    cursor = match.end()
    while cursor < len(options) and options[cursor] in " \t\n":
      cursor += 1
    if cursor >= len(options):
      result[key] = ""
      break
    if options[cursor] == "{":
      value, cursor = read_group(options, cursor)
    elif options[cursor] == "[":
      value, cursor = read_group(options, cursor, "[", "]")
    else:
      stop = options.find(",", cursor)
      if stop == -1:
        stop = len(options)
      value = options[cursor:stop].strip()
      cursor = stop
    result[key] = value.strip()
    pos = cursor
    while pos < len(options) and options[pos] in " ,":
      pos += 1
  return result
