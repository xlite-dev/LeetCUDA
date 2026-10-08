"""插图文字的中英对照处理（英文图集用）。

TikZ 图是构建时现编译的，所以图内文字也能一起做成英文：把 ``\\node{…}`` 里的文字
按词典替换后重编译一套 SVG（``_static/figures-en/``），英文模式下由前端换图。

替换要**保住 TeX 结构**：``\\\\``（换行）、``$…$``（数学）、``\\texttt{…}`` 这类命令
在送翻译前先替换成占位符 ``⟦n⟧``，翻完再还原——否则机器翻译会拆散命令、把数学改掉
（实测 ``\\texttt{-{}-set full}\\\\名字前缀`` → ``\\texttt{-{}-set full}\\\\name prefix``
只有先掩码才稳）。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .texutil import read_group

#: 带参数、且内容可能是正文的文本命令。
_TEXT_CMD_RE = re.compile(
  r"\\(?:texttt|textbf|textit|textrm|mathrm|text|kern|shortstack|hspace|vspace|raisebox)"
  r"\s*\{"
)

#: TeX 转义字符。
_ESCAPE_RE = re.compile(r"\\[%&_#$~{}]")

#: 无参数的字号/字形命令。
_STYLE_CMD_RE = re.compile(
  r"\\(?:tiny|scriptsize|footnotesize|small|normalsize|large|Large|itshape|bfseries|"
  r"ttfamily|sffamily|rmfamily|centering|raggedright)\b"
)

#: 占位符（机器翻译会原样保留这对括号，实测可靠）。
_PLACEHOLDER_RE = re.compile(r"⟦(\d+)⟧")

#: ``\node`` / ``node``（TikZ 的 ``-- node {…}`` 写法没有反斜杠）。
_NODE_RE = re.compile(r"(?<!\\)\\?node\b")

#: 含中文。
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def mask(text: str) -> tuple[str, list[str]]:
  """把 TeX 结构替换成占位符，只把正文留给机器翻译。

  处理规则：

  - ``\\\\``（换行）、``$…$``（数学）、``\\%`` 这类转义、字号命令：整段掩掉；
  - ``\\textbf{…}`` 这类**内容可能是正文**的命令：掩掉命令名与花括号、内容继续
    递归处理——书里有 400 多字中文写在 ``\\textbf{}``/``\\texttt{}`` 里，整段掩掉
    就翻不到了。

  :param text: 节点文字。
  :returns: ``(掩码后的文本, 被替换的片段)``。
  """
  store: list[str] = []
  return _mask_into(text, store), store


def _mask_into(text: str, store: list[str]) -> str:
  """单遍扫描的掩码实现（递归处理命令内容，占位符编号共用一套）。

  :param text: 文本。
  :param store: 片段表。
  :returns: 掩码后的文本。
  """
  out: list[str] = []
  index = 0
  length = len(text)
  while index < length:
    if text.startswith("\\\\", index):
      out.append(_keep("\\\\", store))
      index += 2
      continue
    if text[index] == "$":
      stop = text.find("$", index + 1)
      if stop != -1:
        out.append(_keep(text[index:stop + 1], store))
        index = stop + 1
        continue
    match = _TEXT_CMD_RE.match(text, index)
    if match is not None:
      try:
        content, end = read_group(text, match.end() - 1)
      except ValueError:
        out.append(text[index])
        index += 1
        continue
      if _CJK_RE.search(content):
        out.append(_keep(match.group(0), store))
        out.append(_mask_into(content, store))
        out.append(_keep("}", store))
      else:
        out.append(_keep(match.group(0) + content + "}", store))
      index = end
      continue
    match = _ESCAPE_RE.match(text, index) or _STYLE_CMD_RE.match(text, index)
    if match is not None:
      out.append(_keep(match.group(0), store))
      index = match.end()
      continue
    out.append(text[index])
    index += 1
  return "".join(out)


def _keep(fragment: str, store: list[str]) -> str:
  """登记片段并返回其占位符。

  :param fragment: 原样保留的片段。
  :param store: 片段表。
  :returns: 占位符文本。
  """
  store.append(fragment)
  return f"⟦{len(store) - 1}⟧"


def unmask(text: str, store: list[str]) -> str:
  """还原占位符。

  :param text: 掩码后的文本。
  :param store: :func:`mask` 返回的片段表。
  :returns: 还原后的文本。
  """
  return _PLACEHOLDER_RE.sub(lambda match: store[int(match.group(1))], text)


def node_spans(body: str) -> list[tuple[int, int, str]]:
  """给出片段里全部 ``node`` 文本的 ``(start, end, text)``。

  ``\\node[opts] (name) at (x,y) {text}`` 的形态不固定（可选参数、名字、位置子句都
  可能出现，顺序也不定），所以逐字符跳到第一个 ``{``。

  :param body: TikZ 片段源码。
  :returns: 按出现顺序排列的文本区间。
  """
  spans: list[tuple[int, int, str]] = []
  for match in _NODE_RE.finditer(body):
    index, length = match.end(), len(body)
    while index < length:
      char = body[index]
      if char in " \t\n":
        index += 1
        continue
      if char in "[(":
        opener = char
        closer = "]" if char == "[" else ")"
        depth = 0
        while index < length:
          if body[index] == opener:
            depth += 1
          elif body[index] == closer:
            depth -= 1
            if depth == 0:
              index += 1
              break
          index += 1
        continue
      if char == "{":
        try:
          inner, end = read_group(body, index)
        except ValueError:
          break
        spans.append((index + 1, end - 1, inner))
        break
      if char == ";":
        break
      index += 1
  return spans


def collect_units(snippet_paths: list[Path]) -> Counter[str]:
  """统计需要翻译的单元（去重计数）。

  单元是**掩码后的节点文字**（保留整句上下文，翻译质量比逐词好），外加节点之外
  残留的中文片段——书里有 53 个字写在 ``\\foreach`` 的列表里（ch23 的 swizzle 图）。

  :param snippet_paths: 片段文件列表。
  :returns: 单元 → 出现次数。
  """
  units: Counter[str] = Counter()
  for path in snippet_paths:
    body = strip_comments(path.read_text(encoding="utf-8"))
    covered = node_spans(body)
    for start, end, text in covered:
      if _CJK_RE.search(text):
        units[mask(text)[0]] += 1
    for match in _CJK_RE.finditer(body):
      if any(start <= match.start() < end for start, end, _text in covered):
        continue
      units[mask(body[match.start():match.start() + 60])[0]] += 1
  return units


def strip_comments(text: str) -> str:
  """去掉 TeX 注释（``%`` 到行尾，``\\%`` 不算）。

  :param text: 片段源码。
  :returns: 去注释后的文本。
  """
  return re.sub(r"(?<!\\)%[^\n]*", "", text)


def load_dictionary(path: Path) -> dict[str, str]:
  """读取词典（读入时做一次 TeX 特殊字符规范化）。

  :param path: JSON 词典路径。
  :returns: 掩码后的中文单元 → 掩码后的英文单元。
  """
  if not path.is_file():
    return {}
  return {key: sanitize(value) for key, value in json.loads(
    path.read_text(encoding="utf-8")).items()}


#: 机器译文里需要转义的 TeX 特殊字符（``^`` 除外——它在译文里总是 ``\^{}`` 形式）。
_SANITIZE_ESCAPES = {
  "%": "\\%", "&": "\\&", "#": "\\#", "_": "\\_", "$": "\\$",
  "~": r"\textasciitilde{}",
}


def sanitize(text: str) -> str:
  """转义机器译文里未转义的 TeX 特殊字符。

  译文直接放进 TikZ 节点里编译，所以裸的 ``%``（注释掉整行）、``&``、``#``、``_``、
  ``$``、``~`` 会直接让 xelatex 报错或静默吃掉内容（ch10/ch19 的英文图就是这样编不
  出来的）。占位符 ``⟦n⟧`` 不在转义范围内；已经转义过的（``\\%``、``\\^{}``）保持原样。

  花括号只在**嵌套不合法**时才转义：译文里的 ``\\^{}``、``\\S{}``、``\\textbf{…}`` 是
  配对的，一律转义会把它们弄坏。

  :param text: 译文。
  :returns: 转义后的文本。
  """
  out: list[str] = []
  for index, char in enumerate(text):
    if char in _SANITIZE_ESCAPES and not _escaped(text, index):
      out.append(_SANITIZE_ESCAPES[char])
      continue
    out.append(char)
  result = "".join(out)
  if nesting_sound(result):
    return result
  # 译文本身把括号搬错了位置：整体转义成字面括号，至少保证能编译。
  escaped: list[str] = []
  for index, char in enumerate(result):
    if char in "{}" and not _escaped(result, index):
      escaped.append("\\" + char)
      continue
    escaped.append(char)
  return "".join(escaped)


def _escaped(text: str, index: int) -> bool:
  """判断某位置的字符是否被反斜杠转义。

  要看**前面连续反斜杠的个数**：``\\{``（换行后紧跟真括号）里那个括号没被转义，
  而 ``\\{`` 之外的 ``\{`` 才是字面括号。只看紧邻一个字符会判错，把 ``\\{…}`` 这种
  合法分组当成括号不平衡。

  :param text: 文本。
  :param index: 字符下标。
  :returns: 被转义为 True。
  """
  count = 0
  position = index - 1
  while position >= 0 and text[position] == "\\":
    count += 1
    position -= 1
  return count % 2 == 1


def save_dictionary(path: Path, data: dict[str, str]) -> None:
  """写入词典（按键排序，便于 diff 与人工校对）。

  :param path: JSON 词典路径。
  :param data: 词典内容。
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  ordered = {key: data[key] for key in sorted(data)}
  path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
                  encoding="utf-8")


def nesting_sound(text: str) -> bool:
  """检查花括号嵌套是否合法（深度不出现负值、结尾归零）。

  译文里的括号数量可能是平衡的、但**位置**被搬错了——机器翻译会把 ``\\textbf{`` 与
  它的 ``}`` 挪到句子不同位置（``chunk } … \\textbf{: …``），数量检查看不出来，编译
  时才会以 TikZ 的 "Giving up on this path" 或 ``\\pgfutil@next`` 报错收场。

  :param text: 还原占位符之后的译文。
  :returns: 合法为 True。
  """
  depth = 0
  for index, char in enumerate(text):
    if char not in "{}" or _escaped(text, index):
      continue
    if char == "{":
      depth += 1
    else:
      depth -= 1
      if depth < 0:
        return False
  return depth == 0


def translate_snippet(body: str, dictionary: dict[str, str]) -> tuple[str, int, int]:
  """把片段里的节点文字替换成英文。

  只做**整段精确匹配**：不认识的单元原样保留（宁可留中文，也不要半中半英地把
  句子拆坏）。

  :param body: TikZ 片段源码。
  :param dictionary: 词典。
  :returns: ``(英文片段, 命中数, 未命中数)``。
  """
  spans = node_spans(body)
  edits: list[tuple[int, int, str]] = []
  hit = miss = 0
  for start, end, text in spans:
    if not _CJK_RE.search(text):
      continue
    masked, store = mask(text)
    english = dictionary.get(masked)
    if not english:
      miss += 1
      continue
    restored = unmask(english, store)
    if not nesting_sound(restored):
      # 括号被翻译挪错位置：宁可保留中文，也不发出编不出来的英文图。
      miss += 1
      continue
    hit += 1
    edits.append((start, end, restored))
  out = body
  for start, end, replacement in sorted(edits, reverse=True):
    out = out[:start] + replacement + out[end:]
  return out, hit, miss


def coverage(
  snippet_paths: list[Path],
  dictionary: dict[str, str],
) -> tuple[int, int, list[str]]:
  """统计词典覆盖情况。

  :param snippet_paths: 片段文件列表。
  :param dictionary: 词典。
  :returns: ``(已覆盖单元数, 总单元数, 未覆盖单元样例)``。
  """
  units = collect_units(snippet_paths)
  missing = [unit for unit in units if unit not in dictionary]
  return len(units) - len(missing), len(units), missing
