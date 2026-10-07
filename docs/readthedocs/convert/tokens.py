"""预处理与后处理之间交换的占位 token。

pandoc 既不理解这些 token，也不会改写它们，前提是 token 的载荷字符集被限制在
``A-Za-z0-9-.:`` 且字段用 ``;`` 分隔——pandoc 会改写的字符一律不用：``%`` 是
LaTeX 注释起始符、``_``/``*``/``|`` 在 markdown 输出里会被转义、``~`` 会被
转成不换行空格、``\\``/``{}`` 会被当作命令或分组。

token 形如 ``@@RTD;<KIND>[;<field>...]@@``。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: 匹配任意 token；组 1 是 KIND，组 2 是 ``;field`` 尾部。
TOKEN_RE = re.compile(r"@@RTD;([A-Z_]+)((?:;[^@;]*)*)@@")

#: token 载荷中不允许出现的字符（替换为 ``-``）。
UNSAFE_RE = re.compile(r"[^A-Za-z0-9\-.:]")

KIND_LABEL = "LBL"
KIND_REF = "REF"
KIND_EQREF = "EREF"
KIND_IMAGE = "IMG"
KIND_TIKZ = "TIKZ"
KIND_CODE = "CODE"
KIND_CODE_TITLE = "CODETITLE"
KIND_CODE_TITLE_END = "CODETITLEEND"
KIND_ADMON = "ADMON"
KIND_ADMON_TITLE_END = "ADMONTITLEEND"
KIND_ADMON_END = "ADMONEND"
KIND_FIGURE = "FIG"
KIND_FIGURE_END = "FIGEND"
KIND_TABLE = "TAB"
KIND_TABLE_END = "TABEND"
KIND_CAPTION = "CAP"
KIND_CAPTION_END = "CAPEND"
KIND_SUBFIGURE = "SUBFIG"
KIND_SUBFIGURE_END = "SUBFEND"

#: 需要整块收集（begin/end 配对）的 token 种类。
BLOCK_KINDS = {
  KIND_ADMON: KIND_ADMON_END,
  KIND_FIGURE: KIND_FIGURE_END,
  KIND_TABLE: KIND_TABLE_END,
  KIND_SUBFIGURE: KIND_SUBFIGURE_END,
  KIND_CAPTION: KIND_CAPTION_END,
  KIND_CODE_TITLE: KIND_CODE_TITLE_END,
}


@dataclass(frozen=True)
class Token:
  """一个已解析的 token。

  :param kind: token 种类，见本模块的 ``KIND_*`` 常量。
  :param fields: 载荷字段。
  :param raw: token 原文。
  """
  kind: str
  fields: tuple[str, ...]
  raw: str


def make(kind: str, *fields: str) -> str:
  """构造一个 token。

  :param kind: token 种类。
  :param fields: 载荷字段；非法字符被替换为 ``-``。
  :returns: token 文本。
  """
  body = "".join(";" + UNSAFE_RE.sub("-", field) for field in fields)
  return f"@@RTD;{kind}{body}@@"


def parse(raw: str) -> Token:
  """解析单个 token 文本。

  :param raw: 形如 ``@@RTD;KIND;a;b@@`` 的文本。
  :returns: 解析结果。
  """
  match = TOKEN_RE.fullmatch(raw)
  if match is None:
    raise ValueError(f"不是合法 token: {raw!r}")
  fields = tuple(f for f in match.group(2).split(";") if f != "")
  return Token(match.group(1), fields, raw)


def find(text: str) -> list[Token]:
  """按出现顺序取出文本中的全部 token。

  :param text: 任意文本。
  :returns: token 列表。
  """
  return [Token(m.group(1), tuple(f for f in m.group(2).split(";") if f != ""), m.group(0))
          for m in TOKEN_RE.finditer(text)]


def strip(text: str) -> str:
  """删除文本中的全部 token。

  :param text: 任意文本。
  :returns: 删除 token 后的文本。
  """
  return TOKEN_RE.sub("", text)
