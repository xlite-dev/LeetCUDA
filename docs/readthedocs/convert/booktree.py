"""从 ``book.tex`` 解析站点导航骨架。

章节清单由 ``book.tex`` 的 ``\\part`` / ``\\input`` 顺序生成，不硬编码：书里新增
章节后重跑转换即可自动入站。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .texutil import plain_text, read_group, strip_comments


@dataclass
class Chapter:
  """一个站点页面（章或附录）。

  :param chap_id: 页面标识，取 tex 文件名（不含扩展名）。
  :param tex: 原书 tex 路径。
  :param kind: ``"chapter"`` 或 ``"appendix"``。
  :param title: 章标题（纯文本）。
  :param label: ``\\chapter`` 后的 ``\\label``，无则空串。
  :param numbered: 是否为编号章（``\\chapter*`` 为 False）。
  """
  chap_id: str
  tex: Path
  kind: str
  title: str
  label: str
  numbered: bool


@dataclass
class Part:
  """一个 ``\\part``。

  :param index: 从 1 起的序号。
  :param title: 篇标题（纯文本）。
  :param chapters: 篇内页面。
  :param appendix: 是否为附录篇（章号用字母）。
  """
  index: int
  title: str
  chapters: list[Chapter] = field(default_factory=list)
  appendix: bool = False


#: ``\part{...}`` 与 ``\input{...}``。
_EVENT_RE = re.compile(r"\\(part|input)\s*\*?\s*\{")


def parse(book_tex: Path) -> list[Part]:
  """解析 ``book.tex`` 得到 ``[Part, ...]``。

  :param book_tex: ``book.tex`` 路径。
  :returns: 按阅读顺序排列的篇与页。
  """
  root = book_tex.parent
  text = strip_comments(book_tex.read_text(encoding="utf-8"))
  parts: list[Part] = []
  current: Part | None = None
  for match in _EVENT_RE.finditer(text):
    if match.group(1) == "part":
      title, _ = read_group(text, match.end() - 1)
      current = Part(len(parts) + 1, plain_text(title))
      parts.append(current)
      continue
    target, _ = read_group(text, match.end() - 1)
    if not target.startswith(("chapters/", "appendices/")):
      continue
    if current is None:
      current = Part(len(parts) + 1, "")
      parts.append(current)
    tex = root / (target if target.endswith(".tex") else target + ".tex")
    info = _chapter_info(tex, target)
    if info.kind == "appendix":
      current.appendix = True
    current.chapters.append(info)
  return parts


def _chapter_info(tex: Path, target: str) -> Chapter:
  """读取章标题、label 与编号属性。

  :param tex: 章节 tex 路径。
  :param target: ``book.tex`` 里的 ``\\input`` 目标（用于判定章/附录）。
  :returns: 章节元信息。
  """
  text = strip_comments(tex.read_text(encoding="utf-8"))
  match = re.search(r"\\chapter(\*)?\s*\{", text)
  if match is None:
    title, label = tex.stem, ""
  else:
    raw_title, cursor = read_group(text, match.end() - 1)
    label_match = re.search(r"\\label\{([^}]*)\}", text[cursor:cursor + 300])
    title = plain_text(raw_title)
    label = label_match.group(1) if label_match is not None else ""
  return Chapter(
    chap_id=tex.stem,
    tex=tex,
    kind="appendix" if target.startswith("appendices/") else "chapter",
    title=title,
    label=label,
    numbered=match is not None and match.group(1) is None,
  )
