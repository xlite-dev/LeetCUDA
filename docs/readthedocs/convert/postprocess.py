"""markdown 后处理：token → MyST。

输入是 pandoc 产出的 markdown（内嵌 token），输出是 Sphinx 可直接构建的 MyST
文档：HTML 锚点、站内交叉引用链接、admonition、图片、带源文件行号的代码块、图注
与表注。

锚点一律用原始 HTML（``<a id="..."></a>``）而不是 MyST 目标语法：原始 HTML 会被
Sphinx 原样写进 HTML 输出，链接目标确定，也不与文档标题的解析规则纠缠。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from . import tokens
from .booktree import Chapter, Part
from .texutil import math_spans
from .labels import Registry, chapter_number_prefix, format_number, sanitize
from .preprocess import ADMON_KINDS, Manifest

#: 图片 alt 文本（正文图注由 caption 段落承担）。
_DEFAULT_ALT = "插图"

#: 页脚样式类名。
_FOOTER_CLASS = "rtd-footer"


@dataclass
class RenderContext:
  """渲染一章所需的外部信息。

  :param chap: 目标章。
  :param parts: 全书篇结构（用于章号前缀）。
  :param manifest: 该章转换清单。
  :param registry: 全书 label 注册表。
  :param work_dir: 临时目录（代码载荷所在处）。
  :param tikz_failed: 编译失败的 image_id。
  :param source_url: 原书 tex 的 GitHub 链接前缀。
  """
  chap: Chapter
  parts: list[Part]
  manifest: Manifest
  registry: Registry
  work_dir: Path
  tikz_failed: set[str] = field(default_factory=set)
  source_url: str = ""


@dataclass
class RenderResult:
  """渲染结果。

  :param markdown: 最终 MyST 文本。
  :param broken_refs: 未解析的引用 label。
  :param leftovers: 未转换的 LaTeX 命令计数。
  :param notes: 渲染期发现的问题（未闭合块等）。
  :param stats: 其他统计。
  """
  markdown: str
  broken_refs: list[str] = field(default_factory=list)
  leftovers: dict[str, int] = field(default_factory=dict)
  notes: list[str] = field(default_factory=list)
  stats: dict[str, int] = field(default_factory=dict)


class _Renderer:
  """把一章的 token 化 markdown 渲染成 MyST。"""

  def __init__(self, ctx: RenderContext) -> None:
    """初始化。

    :param ctx: 渲染上下文。
    """
    self.ctx = ctx
    self.pending: list[str] = []
    self.broken: list[str] = []
    self.notes: list[str] = []
    self.counter = {"kaogu": 0, "theorem": 0}
    self.subfigure_letter = 0

  def render(self, md_text: str) -> RenderResult:
    """渲染整章。

    :param md_text: pandoc 产出的 markdown。
    :returns: 渲染结果。
    """
    leftovers = scan_leftovers(md_text)
    blocks = self._render_items(self._split(md_text))
    body = "\n\n".join(block for block in blocks if block.strip())
    if self.pending:
      body = "\n\n".join(self.pending) + "\n\n" + body
      self.pending = []
    body = _collapse_math_blank_lines(body)
    body = _normalize_display_math(body)
    body = _strip_raw_inline_html(body)
    body, table_images = _htmlify_table_inlines(body)
    hidden = _hidden_image_refs(table_images, body)
    if hidden:
      body = f"{body}\n\n{hidden}"
    body = _clean_heading_attributes(body)
    body = _escape_colon_fences(body)
    body = _degrade_leftovers(body)
    body = _fix_emphasis_flanking(body)
    body = self._prefix_title(body)
    body = self._append_footer(body)
    return RenderResult(
      markdown=body.strip() + "\n",
      broken_refs=sorted(set(self.broken)),
      leftovers=leftovers,
      notes=list(self.notes),
      stats={"broken_refs": len(set(self.broken))},
    )

  def _split(self, md: str) -> list[tuple[str, object]]:
    """把 markdown 切成文本片段与 token 交替的序列。

    :param md: markdown 文本。
    :returns: ``[("text", str) | ("tok", Token), ...]``。
    """
    items: list[tuple[str, object]] = []
    pos = 0
    for match in tokens.TOKEN_RE.finditer(md):
      if match.start() > pos:
        items.append(("text", md[pos:match.start()]))
      items.append(("tok", tokens.parse(match.group(0))))
      pos = match.end()
    if pos < len(md):
      items.append(("text", md[pos:]))
    return items

  def _render_items(self, items: list[tuple[str, object]]) -> list[str]:
    """渲染一段（可含块级 token）内容。

    :param items: 切分后的序列。
    :returns: 块列表。
    """
    blocks: list[str] = []
    buffer: list[tuple[str, object]] = []

    def flush() -> None:
      if not buffer:
        return
      pending = buffer[:]
      buffer.clear()
      blocks.extend(self._render_paragraphs(pending))

    index = 0
    while index < len(items):
      kind, payload = items[index]
      if kind != "tok":
        buffer.append(items[index])
        index += 1
        continue
      tok = payload
      if tok.kind == tokens.KIND_LABEL and tok.fields:
        self.pending.append(f'<a id="{self._anchor(tok.fields[0])}"></a>')
        index += 1
        continue
      if tok.kind in tokens.BLOCK_KINDS and tok.kind != tokens.KIND_CAPTION:
        flush()
        match = self._match(items, index, tokens.BLOCK_KINDS[tok.kind])
        if match >= len(items):
          self.notes.append(f"{tok.kind} 块未闭合，已按到段末处理")
        inner = items[index + 1:match]
        block = self._render_block(tok, inner)
        if block.strip():
          blocks.append(self._take_pending(block))
        index = match + 1
        continue
      if tok.kind == tokens.KIND_CODE:
        flush()
        blocks.append(self._take_pending(self._render_code(tok)))
        index += 1
        continue
      buffer.append(items[index])
      index += 1
    flush()
    return blocks

  def _match(self, items: list[tuple[str, object]], start: int, end_kind: str) -> int:
    """找到配对的结束 token 下标。

    :param items: 序列。
    :param start: 起始下标（起始 token 所在处）。
    :param end_kind: 结束 token 种类。
    :returns: 结束 token 下标；找不到时返回序列长度。
    """
    depth = 0
    for index in range(start + 1, len(items)):
      kind, payload = items[index]
      if kind != "tok":
        continue
      if payload.kind == end_kind and depth == 0:
        return index
      if payload.kind in tokens.BLOCK_KINDS:
        depth += 1
      elif payload.kind in tokens.BLOCK_KINDS.values():
        depth -= 1
    return len(items)

  def _render_block(self, tok: tokens.Token, inner: list[tuple[str, object]]) -> str:
    """渲染一个块级 token。

    :param tok: 起始 token。
    :param inner: 块内内容。
    :returns: MyST 文本。
    """
    if tok.kind == tokens.KIND_FIGURE:
      return self._render_figure(inner)
    if tok.kind == tokens.KIND_TABLE:
      return self._render_table(inner, tok.fields[0] if tok.fields else "")
    if tok.kind == tokens.KIND_SUBFIGURE:
      return self._render_subfigure(inner, tok.fields[0] if tok.fields else "")
    if tok.kind == tokens.KIND_ADMON:
      return self._render_admonition(tok.fields[0] if tok.fields else "kaogu", inner)
    if tok.kind == tokens.KIND_CODE_TITLE:
      title = self._render_inline(inner)
      return f"**{title}**" if title.strip() else ""
    if tok.kind == tokens.KIND_CAPTION:
      caption = self._render_inline(inner)
      return f"{{.rtd-caption}}\n{caption}" if caption.strip() else ""
    return ""

  def _render_figure(self, items: list[tuple[str, object]]) -> str:
    """渲染 figure 环境（含 subfigure 分支）。

    :param items: figure 内容。
    :returns: MyST 文本。
    """
    self.subfigure_letter = 0
    sub_blocks: list[str] = []
    rest: list[tuple[str, object]] = []
    index = 0
    while index < len(items):
      kind, payload = items[index]
      if kind == "tok" and payload.kind == tokens.KIND_SUBFIGURE:
        match = self._match(items, index, tokens.KIND_SUBFIGURE_END)
        width = f"{payload.fields[0]}%" if payload.fields and payload.fields[0] else ""
        sub_blocks.append(self._render_subfigure(items[index + 1:match], width))
        index = match + 1
        continue
      rest.append(items[index])
      index += 1

    caption, labels, content = self._split_caption(rest)
    number = self._object_number(labels)
    blocks = [block for block in sub_blocks if block.strip()]
    blocks.extend(self._render_items(content))
    if caption.strip():
      prefix = f"图 {number}：" if number else "图："
      blocks.append(f"{{.rtd-caption}}\n{prefix}{caption.strip()}")
    anchors = self._anchors(labels)
    if anchors:
      blocks.insert(0, anchors)
    return "\n\n".join(blocks)

  def _render_table(self, items: list[tuple[str, object]], header_rows: str = "") -> str:
    """渲染 table / longtable 环境（书里表注在表上方）。

    :param items: table 内容。
    :param header_rows: 源文里的表头行数（≥2 时把这几行收进 ``<thead>``）。
    :returns: MyST 文本。
    """
    caption, labels, content = self._split_caption(items)
    number = self._object_number(labels)
    blocks: list[str] = []
    anchors = self._anchors(labels)
    if anchors:
      blocks.append(anchors)
    if caption.strip():
      prefix = f"表 {number}：" if number else "表："
      blocks.append(f"{{.rtd-caption}}\n{prefix}{caption.strip()}")
    blocks.extend(self._render_items(content))
    text = "\n\n".join(blocks)
    if header_rows.isdigit() and int(header_rows) >= 2:
      text = _wrap_table_header(text, int(header_rows))
    return text

  def _render_subfigure(self, items: list[tuple[str, object]], width: str) -> str:
    """渲染一个 subfigure。

    :param items: subfigure 内容。
    :param width: 声明的宽度百分比。
    :returns: MyST 文本。
    """
    caption, labels, content = self._split_caption(items)
    self.subfigure_letter += 1
    letter = chr(ord("a") + (self.subfigure_letter - 1) % 26)
    blocks = self._render_items_with_width(content, width)
    if caption.strip():
      blocks.append(f"{{.rtd-subcaption}}\n（{letter}）{caption.strip()}")
    anchors = self._anchors(labels)
    if anchors:
      blocks.insert(0, anchors)
    return "\n\n".join(blocks)

  def _render_admonition(self, kind: str, items: list[tuple[str, object]]) -> str:
    """渲染定理类环境为 admonition。

    :param kind: 环境名。
    :param items: 环境内容（首元素为标题区，其后为正文）。
    :returns: MyST 文本。
    """
    end_index = len(items)
    for index, (item_kind, payload) in enumerate(items):
      if item_kind == "tok" and payload.kind == tokens.KIND_ADMON_TITLE_END:
        end_index = index
        break
    title = self._render_inline(items[:end_index])
    body_items = items[end_index + 1:]
    body_blocks = self._render_items(body_items)
    label, css, numbered = ADMON_KINDS.get(kind, ("提示", "remark", False))
    heading = f"{label} {self._admon_number(kind, numbered)}".strip()
    lines = [f"```{{admonition}} {heading}", f":class: rtd-admon rtd-{css}"]
    if title.strip():
      lines.extend(["", f"**{title.strip()}**"])
    content = "\n\n".join(block for block in body_blocks if block.strip())
    if content:
      lines.extend(["", content])
    lines.append("```")
    return "\n".join(lines)

  def _render_paragraphs(self, items: list[tuple[str, object]], width: str = "") -> list[str]:
    """按空行切段渲染流式内容。

    :param items: 文本与内联 token 交替的序列。
    :param width: 图片宽度覆盖值。
    :returns: 块列表。
    """
    blocks: list[str] = []
    for paragraph in self._split_paragraphs(items):
      blocks.extend(self._render_paragraph(paragraph, width))
    return blocks

  def _split_paragraphs(self, items: list[tuple[str, object]]) -> list[list[tuple[str, object]]]:
    """按空行把序列切成段落。

    :param items: 序列。
    :returns: 段落列表（每段是 ``(kind, payload)`` 列表）。
    """
    paragraphs: list[list[tuple[str, object]]] = []
    current: list[tuple[str, object]] = []
    for kind, payload in items:
      if kind != "text":
        current.append((kind, payload))
        continue
      for part in re.split(r"(\n\s*\n)", payload):
        if part == "":
          continue
        if re.fullmatch(r"\n\s*\n", part):
          paragraphs.append(current)
          current = []
        else:
          current.append(("text", part))
    paragraphs.append(current)
    return [paragraph for paragraph in paragraphs if paragraph]

  def _render_paragraph(self, items: list[tuple[str, object]], width: str = "") -> list[str]:
    """渲染单个段落，并把它自带的锚点放在段落前。

    :param items: 段落内容。
    :param width: 图片宽度覆盖值。
    :returns: 块列表（段落只有 label 时返回空列表，锚点留给下一个块）。
    """
    anchors = [
      f'<a id="{self._anchor(payload.fields[0])}"></a>'
      for kind, payload in items
      if kind == "tok" and payload.kind == tokens.KIND_LABEL and payload.fields
    ]
    rendered = _join_cjk_lines(self._render_inline_items(items, width)).strip()
    if not rendered:
      self.pending.extend(anchors)
      return []
    prefix = "\n\n".join(self.pending + anchors)
    self.pending = []
    return [f"{prefix}\n\n{rendered}" if prefix else rendered]

  def _render_items_with_width(self, items: list[tuple[str, object]], width: str) -> list[str]:
    """渲染内容并给其中的图片指定宽度。

    :param items: 内容。
    :param width: 宽度百分比。
    :returns: 块列表。
    """
    return self._render_paragraphs(items, width)

  def _render_inline(self, items: list[tuple[str, object]]) -> str:
    """渲染为单行内联 markdown（caption / 标题用）。

    :param items: 内容。
    :returns: 单行 markdown。
    """
    return re.sub(r"\s+", " ", _join_cjk_lines(self._render_inline_items(items))).strip()

  def _render_inline_items(self, items: list[tuple[str, object]], width: str = "") -> str:
    """替换内联 token（label 由段落层收集，这里跳过）。

    :param items: 内容。
    :param width: 图片宽度覆盖值。
    :returns: markdown 文本。
    """
    chunks: list[str] = []
    index = 0
    while index < len(items):
      kind, payload = items[index]
      if kind == "text":
        chunks.append(payload)
        index += 1
        continue
      tok = payload
      if tok.kind == tokens.KIND_REF and tok.fields:
        chunks.append(self._reference(tok.fields[0], parenthesized=False))
      elif tok.kind == tokens.KIND_EQREF and tok.fields:
        chunks.append(self._reference(tok.fields[0], parenthesized=True))
      elif tok.kind == tokens.KIND_IMAGE:
        chunks.append(self._image(tok, width))
      elif tok.kind == tokens.KIND_CAPTION:
        inner_end = self._match(items, index, tokens.KIND_CAPTION_END)
        caption = self._render_inline(items[index + 1:inner_end])
        if caption.strip():
          chunks.append(f"*{caption.strip()}*")
        index = inner_end
      index += 1
    return "".join(chunks)

  def _split_caption(
    self, items: list[tuple[str, object]]) -> tuple[str, list[str], list[tuple[str, object]]]:
    """从内容中摘出 caption 与 label。

    :param items: 内容。
    :returns: ``(caption, labels, 其余内容)``。
    """
    captions: list[str] = []
    labels: list[str] = []
    rest: list[tuple[str, object]] = []
    index = 0
    while index < len(items):
      kind, payload = items[index]
      if kind == "tok" and payload.kind == tokens.KIND_CAPTION:
        match = self._match(items, index, tokens.KIND_CAPTION_END)
        inner = items[index + 1:match]
        for inner_kind, inner_payload in inner:
          if (inner_kind == "tok" and inner_payload.kind == tokens.KIND_LABEL
              and inner_payload.fields):
            labels.append(inner_payload.fields[0])
        captions.append(self._render_inline(inner))
        index = match + 1
        continue
      if kind == "tok" and payload.kind == tokens.KIND_LABEL:
        labels.append(payload.fields[0])
        index += 1
        continue
      rest.append(items[index])
      index += 1
    return " ".join(part for part in captions if part.strip()), labels, rest

  def _object_number(self, labels: list[str]) -> str:
    """由 label 查图/表编号。

    :param labels: 内容中的 label。
    :returns: 编号文本（查不到为空串）。
    """
    for label in labels:
      entry = self.ctx.registry.get(label)
      if entry is not None and entry.number:
        return entry.number
    return ""

  def _admon_number(self, kind: str, numbered: bool) -> str:
    """定理类环境的编号。

    :param kind: 环境名。
    :param numbered: 是否编号。
    :returns: 编号文本（不编号为空串）。
    """
    if not numbered:
      return ""
    chap_id = self.ctx.chap.chap_id
    chapter = self.ctx.registry.chapter_numbers.get(chap_id, "")
    formats = self.ctx.registry.chapter_formats.get(chap_id, {})
    if kind == "kaogu":
      self.counter["kaogu"] += 1
      template = formats.get("kaogu", "{c}.{n}")
      return format_number(template, chapter, self.counter["kaogu"])
    self.counter["theorem"] += 1
    template = formats.get("theorem", "{c}.{n}")
    return format_number(template, chapter, self.counter["theorem"])

  def _reference(self, label: str, parenthesized: bool) -> str:
    """把一个 label 引用渲染成站内链接。

    :param label: 目标 label。
    :param parenthesized: 是否带括号（``\\eqref``）。
    :returns: markdown 链接。
    """
    entry = self.ctx.registry.get(label)
    if entry is None or not entry.number:
      self.broken.append(label)
      return f"`{label}`"
    text = f"({entry.number})" if parenthesized else entry.number
    title = entry.title.replace('"', "'").replace("\n", " ").strip()
    url = f"{entry.page}.html#{entry.anchor}"
    if title:
      return f'[{text}]({url} "{title}")'
    return f"[{text}]({url})"

  def _image(self, tok: tokens.Token, width: str) -> str:
    """渲染图片（含 TikZ 编译失败的兜底）。

    :param tok: 图片 token。
    :param width: 宽度覆盖值。
    :returns: markdown 或兜底块。
    """
    image_id = tok.fields[0]
    image = self.ctx.manifest.images.get(image_id)
    if image is None:
      self.broken.append(image_id)
      return f"`{image_id}`"
    if image_id in self.ctx.tikz_failed:
      return self._figure_fallback(image_id, image.origin)
    token_width = tok.fields[1] if len(tok.fields) > 1 else ""
    if not width and token_width:
      width = f"{token_width}%"
    effective = width or image.width
    if effective:
      # 带宽度的图走 ``{image}`` 指令：Markdown 的 ``![…](…){width=95%}`` 里那个不带
      # 引号的百分比值 MyST 解析不了，会原样漏成字面 ``{width=95%}``（全站 25 张图），
      # 加了引号又会给图片套一层自链接。指令形式干净，且仍是 docutils 图片节点，
      # Sphinx 照常把文件搬进 ``_images``。
      return (f"```{{image}} {image.dest}\n"
              f":alt: {_DEFAULT_ALT}\n"
              f":width: {effective}\n"
              "```")
    return f"![{_DEFAULT_ALT}]({image.dest})"

  def _figure_fallback(self, image_id: str, origin: str) -> str:
    """TikZ 编译失败时的兜底展示。

    :param image_id: 图片标识。
    :param origin: 片段文件（相对 ``work``）。
    :returns: admonition 块。
    """
    source = self.ctx.work_dir / origin
    code = source.read_text(encoding="utf-8") if source.is_file() else ""
    fence = "````" if "```" in code else "```"
    return (
      "```{admonition} 插图未能渲染\n"
      ":class: rtd-admon rtd-figure-fallback\n\n"
      f"原图 `{image_id}` 的 TikZ 代码编译失败，以下为原始代码：\n\n"
      f"{fence}tex\n{code.rstrip()}\n{fence}\n"
      "```"
    )

  def _render_code(self, tok: tokens.Token) -> str:
    """渲染代码块。

    :param tok: 代码 token。
    :returns: MyST 代码块。
    """
    code_id = tok.fields[0]
    block = self.ctx.manifest.codes.get(code_id)
    if block is None:
      self.broken.append(code_id)
      return f"`{code_id}`"
    payload = (self.ctx.work_dir / block.path).read_text(encoding="utf-8")
    fence = "````" if "```" in payload else "```"
    header = f"{fence}{{code-block}} {block.lang}"
    if block.first_line:
      header += f"\n:lineno-start: {block.first_line}"
    return f"{header}\n\n{payload.rstrip()}\n{fence}"

  def _anchor(self, label: str) -> str:
    """计算锚点 id。

    :param label: 目标 label。
    :returns: 锚点 id。
    """
    return sanitize(label)

  def _anchors(self, labels: list[str]) -> str:
    """把一组 label 渲染成锚点块。

    :param labels: label 列表。
    :returns: 锚点块（无 label 时为空串）。
    """
    lines = [f'<a id="{self._anchor(label)}"></a>' for label in labels if label]
    return "\n".join(lines)

  def _take_pending(self, block: str) -> str:
    """把待输出锚点挂到下一个块之前。

    :param block: 块文本。
    :returns: 拼好的块。
    """
    if not self.pending:
      return block
    anchors = "\n".join(self.pending)
    self.pending = []
    return f"{anchors}\n\n{block}"

  def _prefix_title(self, body: str) -> str:
    """给章节标题补上书里的编号前缀。

    :param body: 正文。
    :returns: 处理后的正文。
    """
    prefix = chapter_number_prefix(self.ctx.registry, self.ctx.chap)
    if not prefix:
      return body
    return re.sub(r"(?m)^# ", f"# {prefix}", body, count=1)

  def _append_footer(self, body: str) -> str:
    """追加原书 tex 溯源页脚。

    :param body: 正文。
    :returns: 处理后的正文。
    """
    relative = self.ctx.chap.tex.name
    parent = self.ctx.chap.tex.parent.name
    source = f"{parent}/{relative}"
    if self.ctx.source_url:
      link = f"[`{source}`]({self.ctx.source_url}/{source})"
    else:
      link = f"`{source}`"
    return f"{body}\n\n---\n\n{{.{_FOOTER_CLASS}}}\n*源文件：{link}（内容以原书 tex 为准）*"


def render_chapter(ctx: RenderContext, md_text: str) -> RenderResult:
  """渲染一章。

  :param ctx: 渲染上下文。
  :param md_text: pandoc 产出的 markdown。
  :returns: 渲染结果。
  """
  return _Renderer(ctx).render(md_text)


#: CJK 与全角标点的字符类（用于断行/空格合并）。
_CJK_CLASS = r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
_CJK_RE = re.compile(_CJK_CLASS)

#: 对换行敏感的整行：管道表、grid 表、pandoc 输出的 HTML 表。这些行不能参与
#: CJK 折行合并，否则整张表会被压成一行（曾导致全站 0 个表格）。
_BLOCK_LINE_RE = re.compile(
  r"(?m)^[ \t]*(?:"
  r"\|.*"
  r"|\+[-=+ ]+\+"
  r"|</?[a-zA-Z][^>\n]*>.*"
  r")[ \t]*\n")

#: 标题行尾的 pandoc 块属性：``### 延伸阅读 {#延伸阅读 .unnumbered}``。
_HEADING_ATTR_RE = re.compile(r"(?m)^([ \t]*#{1,6}[ \t]+.*?)[ \t]*\{[^{}]*\}[ \t]*$")

#: 独占一行的块属性（pandoc 会给无标题的 raw block 写这种行）。
_ONLY_ATTR_RE = re.compile(r"^[ \t]*\{[^{}]*\}[ \t]*$")


def _clean_heading_attributes(md_text: str) -> str:
  """剥掉 pandoc 写在标题行尾的块属性。

  pandoc 给 ``\\chapter*``/``\\section*`` 输出 ``{#id .unnumbered}``；含 CJK 的 id
  MyST 解析不了，会原样显示在 H1/H3、页面 ``<title>`` 和全站侧栏目录里。锚点由本模块
  自己写的 ``<a id=...>`` 提供，这些属性没有用处。

  :param md_text: markdown 文本。
  :returns: 清理后的文本。
  """
  lines: list[str] = []
  for line in md_text.split("\n"):
    if line.lstrip().startswith("#"):
      line = _HEADING_ATTR_RE.sub(r"\1", line).rstrip()
    elif _ONLY_ATTR_RE.match(line):
      continue
    lines.append(line)
  return "\n".join(lines)


def _join_cjk_lines(text: str) -> str:
  """合并段落内的软换行：中文之间的换行直接相连，其余按空格连接。

  原书 tex 里中文段落是硬换行的，pandoc 会把换行保留成 markdown 的软换行，
  渲染出来两个汉字之间就会多一个空格，与排版结果不符。

  :param text: 段落文本。
  :returns: 合并后的段落文本。
  """
  protected: list[str] = []

  def protect(match: re.Match[str]) -> str:
    protected.append(match.group(0))
    return f"\x00{len(protected) - 1}\x00"

  # 表格与 HTML 行对换行敏感，先整行保护，不参与后面的折行合并。
  guarded = _BLOCK_LINE_RE.sub(protect, text)
  guarded = re.sub(r"`[^`]*`|\$\$.*?\$\$|\$[^$\n]*\$", protect, guarded, flags=re.S)
  # 原书里的 ``\\`` 是硬换行，pandoc 输出成「行尾反斜杠 + 换行」。直接合行的话那个
  # 反斜杠会变成可见字面（面试速查的问答会显示成 ``…cuDNN？\ A：…``），所以先转
  # ``<br>`` 再合行。
  guarded = re.sub(r"(?m)\\(?=\n)", "<br>", guarded)

  def replace(match: re.Match[str]) -> str:
    left, right = match.group(1), match.group(2)
    if left == ">" and right == "<":
      return f"{left}{right}"  # ``<br>`` 与紧随的自动链接/标签之间不留空格
    if _CJK_RE.match(left) and _CJK_RE.match(right):
      return f"{left}{right}"
    return f"{left} {right}"

  guarded = re.sub(r"([^\n])\n[ \t]*([^\n])", replace, guarded)
  # ``<br>`` 后面不留空白（合行时可能补了空格，或被后续自动链接的占位符挡住）。
  guarded = re.sub(r"<br>[ \t]+", "<br>", guarded)
  # 合行后剩下的换行都是段落边界：段尾的 ``<br>`` 没有意义，去掉。
  guarded = re.sub(r"<br>[ \t]*(?=\n|$)", "", guarded)
  # 汉字之间不需要空格：pandoc 已把 tex 的断行折成空格，这里按中文排版规则去掉。
  guarded = re.sub(rf"({_CJK_CLASS})[ \t]+(?={_CJK_CLASS})", r"\1", guarded)
  return re.sub(r"\x00(\d+)\x00", lambda match: protected[int(match.group(1))], guarded)


#: 围栏行（````` ```{code-block} cuda ````` / 裸 ````` ``` `````）。
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})\s*(\S.*)?$")

#: info string 以这些开头的围栏才算代码块；其余（``{admonition}``/``{toctree}``/``{figure}``）
#: 是 MyST 指令，里面是正常 markdown，后处理必须照常处理。
_CODE_FENCE_PREFIXES = ("{code-block", "{code", "{.code")


def code_fence_flags(lines: list[str]) -> list[bool]:
  """逐行标注哪些行属于代码块内容。

  只有代码块的内容要整体跳过（里面是逐字代码）；指令围栏（``{admonition}`` 等）
  的内容是正常 markdown——含公式、链接、图片、pandoc 原始 HTML 残迹——必须参与
  后处理。早期把两者一起跳过，导致指令内的 ``<!-- -->{=html}`` 与公式原样漏到页面上。

  :param lines: 按行拆分的 markdown。
  :returns: 与输入等长的布尔列表，True 表示该行位于代码块内。
  """
  flags: list[bool] = []
  stack: list[bool] = []
  for line in lines:
    match = _FENCE_RE.match(line.lstrip())
    if match:
      info = (match.group(2) or "").strip()
      if stack and not info:
        stack.pop()
        flags.append(False)
        continue
      stack.append(info.startswith(_CODE_FENCE_PREFIXES))
      flags.append(False)
      continue
    flags.append(bool(stack) and stack[-1])
  return flags


#: pandoc 用 `` `X`{=html} `` 表示原始 HTML 行内（``X`` 为 ``<!-- -->`` 时只是分隔符）。
_RAW_INLINE_HTML_RE = re.compile(r"`([^`]*)`\{=html\}")

#: 残留下来的原始格式属性。
_RAW_ATTR_RE = re.compile(r"\{=(?:html|latex|tex)\}")


def _strip_raw_inline_html(md_text: str) -> str:
  """还原 pandoc 的原始 HTML 行内标记。

  pandoc 会在 ``$\\ge$`` 与后面的数字之间插一段 `` `<!-- -->`{=html} `` 作分隔符；
  MyST 把整串当行内代码原样显示，页面上就出现 ``<!-- -->{=html}`` 这种字面文本
  （appB 的表格、ch09 的勘误框、ch26b/ch36 的性能数字里都有）。这里只脱掉
  ``{=html}`` 包装：注释仍是注释（浏览器不显示），其它内容按原始 HTML 透传。

  :param md_text: markdown 文本。
  :returns: 处理后的文本。
  """
  lines = md_text.split("\n")
  flags = code_fence_flags(lines)
  out: list[str] = []
  for line, is_code in zip(lines, flags):
    if is_code:
      out.append(line)
      continue
    out.append(_RAW_ATTR_RE.sub("", _RAW_INLINE_HTML_RE.sub(r"\1", line)))
  return "\n".join(out)


def _collapse_math_blank_lines(md_text: str) -> str:
  """合并显示公式内部的空行。

  pandoc 会把书里 ``align`` 环境内部的空行原样保留（``$$\\begin{align}`` 与
  ``\\end{align}$$`` 之间有空行）。MyST 的 ``$$...$$`` 必须落在同一段内，空行会把
  公式截断——页面上就会漏出 ``$$\\begin{align}`` 这类字面文本，后面的公式也一起垮掉。

  :param md_text: markdown 文本。
  :returns: 处理后的文本。
  """
  lines = md_text.split("\n")
  flags = code_fence_flags(lines)
  out: list[str] = []
  inside = False
  for line, is_code in zip(lines, flags):
    if is_code:
      out.append(line)
      continue
    stripped = line.strip()
    if not inside:
      if stripped.startswith("$$") and not (stripped.endswith("$$") and len(stripped) > 4):
        inside = True
      out.append(line)
      continue
    if not stripped:
      continue
    out.append(line)
    if "$$" in stripped:
      inside = False
  return "\n".join(out)


def _normalize_display_math(md_text: str) -> str:
  """把显示公式的 ``$$`` 定界符放到独占行。

  MyST 的显示公式只认「``$$`` 独占一行（或整段就是 ``$$…$$``）」这一种写法：
  pandoc 常输出「开门符后面紧跟内容、闭门符在最后一行行尾」的形式
  （``$$(4,\\ \\mathrm{MMA\\_M})\\n …\\bigr),$$``），MyST 会把它拆成
  「字面 ``$`` + 行内公式 + 字面 ``$``」，页面上公式散架还多出美元符号。
  实测（myst-parser 4.0）四种写法只有独占行的那种正确。

  :param md_text: markdown 文本。
  :returns: 处理后的文本。
  """
  lines = md_text.split("\n")
  flags = code_fence_flags(lines)
  out: list[str] = []
  inside = False
  for line, is_code in zip(lines, flags):
    stripped = line.strip()
    if is_code:
      out.append(line)
      continue
    if not inside:
      if stripped.startswith("$$"):
        body = stripped[2:]
        if body.endswith("$$") and len(body) > 2:
          out.extend(["$$", body[:-2].strip(), "$$"])
          continue
        if body.strip():
          indent = line[:len(line) - len(line.lstrip())]
          out.append("$$")
          out.append(indent + body)
          inside = True
          continue
        if not body.strip():
          inside = True
          out.append("$$")
          continue
      out.append(line)
      continue
    if "$$" in stripped:
      position = stripped.index("$$")
      before, after = stripped[:position], stripped[position + 2:]
      if before.strip():
        out.append(before)
      out.append("$$")
      inside = False
      if after.strip():
        out.append(after)
      continue
    out.append(line)
  return "\n".join(out)


#: Markdown 行内链接（含可选 title）。
_INLINE_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)(?:\s+[\"“]([^\"”]*)[\"”])?\)")

#: Markdown 行内图片（须先于链接处理，否则会被当成链接）。
_INLINE_IMAGE_RE = re.compile(r"!\[([^\]\n]*)\]\(([^)\s]+)(?:\s+[\"“]([^\"”]*)[\"”])?\)")

#: raw HTML 表格块。
_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.S)


def _htmlify_table_inlines(md_text: str) -> tuple[str, list[str]]:
  """把 raw HTML 表格里的 Markdown 行内语法转成 HTML。

  pandoc 直出的 ``<table>`` 由 MyST 原样透传，单元格里的 Markdown 不会被解析：
  跨章引用（``[31](ch35-…#ch-35 "标题")``）、图片（``![插图](figures-gen/x.svg)``）、
  强调都会漏成字面文本（ch37 的 FP8/FP4 对照表、ch19b 的视图对照表都中招）。

  :param md_text: markdown 文本。
  :returns: ``(处理后的文本, 表格里用到的图片源路径)``。
  """
  used_images: list[str] = []

  def fix_table(match: re.Match[str]) -> str:
    block = match.group(0)

    def to_image(image: re.Match[str]) -> str:
      alt, source = image.group(1), image.group(2)
      used_images.append(source)
      # Sphinx 只搬运 docutils 图片节点引用的文件，所以指向 _images 之余，
      # 还要在页尾补一个隐藏的图片节点让 Sphinx 真去拷（见 _hidden_image_refs）。
      name = source.rsplit("/", 1)[-1]
      return (f'<img src="_images/{_escape_attr(name)}"'
              f' alt="{_escape_attr(alt or _DEFAULT_ALT)}" style="max-width: 100%;">')

    def to_anchor(link: re.Match[str]) -> str:
      label, url, title = link.group(1), link.group(2), link.group(3) or ""
      title_attr = f' title="{_escape_attr(title)}"' if title else ""
      return f'<a href="{_escape_attr(url)}"{title_attr}>{label}</a>'

    block = _INLINE_IMAGE_RE.sub(to_image, block)
    block = _INLINE_LINK_RE.sub(to_anchor, block)
    block = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", block)
    block = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", block)
    return re.sub(r"`([^`\n]+)`", r"<code>\1</code>", block)

  return _HTML_TABLE_RE.sub(fix_table, md_text), used_images


def _hidden_image_refs(sources: list[str], already: str) -> str:
  """为表格里的图片补隐藏的 docutils 图片节点。

  raw HTML 里的 ``<img>`` 不会被 Sphinx 处理，它也就不会把图片拷进 ``_images``；
  页面上就会出现 404。这里补一个隐藏的 ``{image}`` 指令，既是拷贝的来源，也不占版面。

  :param sources: 表格里用到的图片源路径。
  :param already: 已经渲染好的正文（用于跳过已引用过的图片）。
  :returns: 追加的 markdown（无需要时为空串）。
  """
  pending = [source for source in dict.fromkeys(sources)
             if f"]({source}" not in already and f"{{image}} {source}" not in already]
  if not pending:
    return ""
  blocks = ["```{image} " + source + "\n:class: rtd-hidden-ref\n```" for source in pending]
  return "\n\n".join(blocks)


def _escape_attr(text: str) -> str:
  """转义 HTML 属性值。

  :param text: 原文。
  :returns: 转义后的文本。
  """
  return (text.replace("&", "&amp;").replace('"', "&quot;")
          .replace("<", "&lt;").replace(">", "&gt;"))


#: HTML 表格块（pandoc 直出，用于把多行表头收进 thead）。
_HTML_TABLE_WRAP_RE = re.compile(r"<table\b.*?</table>", re.S)

#: 表格行。
_TABLE_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.S)


def _wrap_table_header(md_text: str, header_rows: int) -> str:
  """把表格的前 ``header_rows`` 行收进 ``<thead>``，并把其中的单元格改成 ``<th>``。

  pandoc 的 LaTeX reader 只支持单行表头：源文是「两行表头 + ``\\midrule``」时（ch33 的
  扫描表、ch35 的 tile×流水表、ch37 的量化开销表），第二行表头会落进表体——页面上就
  少了表头底色与加粗，读者容易把它当数据行。

  :param md_text: 渲染后的 markdown。
  :param header_rows: 源文里的表头行数。
  :returns: 处理后的 markdown。
  """

  def fix(match: re.Match[str]) -> str:
    block = match.group(0)
    head_match = re.search(r"<thead>.*?</thead>", block, re.S)
    body_match = re.search(r"<tbody>.*?</tbody>", block, re.S)
    if body_match is None:
      return block
    head_rows = _TABLE_ROW_RE.findall(head_match.group(0)) if head_match else []
    body_rows = _TABLE_ROW_RE.findall(body_match.group(0))
    missing = header_rows - len(head_rows)
    if missing <= 0 or not body_rows:
      return block
    moved = body_rows[:missing]
    rest = body_rows[missing:]
    if len(rest) == 0:
      return block
    head_text = "\n".join(_cell_to_header(row) for row in head_rows + moved)
    body_text = "\n".join(rest)
    new_head = f"<thead>\n{head_text}\n</thead>\n<tbody>\n{body_text}\n</tbody>"
    if head_match:
      start, end = head_match.start(), body_match.end()
      return block[:start] + new_head + block[end:]
    return block[:body_match.start()] + new_head + block[body_match.end():]

  return _HTML_TABLE_WRAP_RE.sub(fix, md_text)


def _cell_to_header(row: str) -> str:
  """把一行数据单元格改成表头单元格。

  :param row: ``<tr>...</tr>``。
  :returns: 替换后的行。
  """
  return row.replace("<td", "<th").replace("</td>", "</th>")


def _protected_mask(md_text: str) -> bytearray:
  """标出「受保护」的字符位置：代码围栏、行内代码、公式。

  这些位置里的 ``*`` 是字面字符（C 指针、通配符、数学乘法），强调相关的改写不能碰。
  返回等长位掩码而不是区间：正文很长（ch19b 单章 100 多处公式），逐字符线性找区间
  会让渲染慢好几倍。

  公式判定复用 :func:`texutil.math_spans`——「什么算公式」必须与其它模块一致：
  自己数 ``$$`` 会把相邻行内公式（``$a$$b$``）当成显示公式定界符，配对一错位就会
  撑出几千字的保护区间，把后面的正文整段盖住。

  :param md_text: markdown 文本。
  :returns: 与文本等长的掩码（1 = 受保护）。
  """
  length = len(md_text)
  mask = bytearray(length)
  lines = md_text.split("\n")
  flags = code_fence_flags(lines)
  offset = 0
  fence_ranges: list[tuple[int, int]] = []
  for line, is_code in zip(lines, flags):
    if is_code:
      mask[offset:offset + len(line)] = b"\x01" * len(line)
      fence_ranges.append((offset, offset + len(line)))
    offset += len(line) + 1

  def overlaps_fence(start: int, end: int) -> bool:
    return any(not (end <= fence_start or start >= fence_end)
               for fence_start, fence_end in fence_ranges)

  for start, end in math_spans(md_text):
    if end > length or overlaps_fence(start, end):
      continue
    for position in range(start, end):
      mask[position] = 1

  index = 0
  while index < length:
    if mask[index] or md_text[index] != "`":
      index += 1
      continue
    stop = md_text.find("`", index + 1)
    if stop == -1 or "\n" in md_text[index:stop]:
      index += 1
      continue
    for position in range(index, stop + 1):
      mask[position] = 1
    index = stop + 1
  return mask


#: 掩码字符：等长替换代码/公式后用来扫描强调，不在正文里出现。
_MASK_CHAR = "\ue000"

#: 成对的强调标记（内容不含 ``*``，避免误碰嵌套写法）。
_STRONG_PAIR_RE = re.compile(r"\*\*(?=\S)([^*\n]+?)\*\*")
_EM_PAIR_RE = re.compile(r"(?<!\*)\*(?=\S)([^*\n]+?)\*(?!\*)")


def _mask_protected(md_text: str) -> str:
  """把代码与公式等长替换成掩码字符。

  等长是关键：偏移不变，强调对可以照常跨过行内公式（``**坑一：$N$ 传错**``），
  而掩码里没有 ``*``，所以代码/公式里的 ``*`` 不会被误配对。

  :param md_text: markdown 文本。
  :returns: 掩码后的文本（与原文等长）。
  """
  mask = _protected_mask(md_text)
  if not any(mask):
    return md_text
  chars = list(md_text)
  for index, flag in enumerate(mask):
    if flag and chars[index] != "\n":
      chars[index] = _MASK_CHAR
  return "".join(chars)


def _is_space(char: str) -> bool:
  """是否为空白（或不存在）。

  :param char: 单个字符（可为空串）。
  :returns: 空白或空串为 True。
  """
  return char == "" or char.isspace()


def _is_punct(char: str) -> bool:
  """是否为 Unicode 标点或符号（CommonMark 的 punctuation 口径）。

  :param char: 单个字符（可为空串）。
  :returns: 标点/符号为 True。
  """
  return char != "" and unicodedata.category(char)[0] in ("P", "S")


def _fix_emphasis_flanking(md_text: str) -> str:
  """把「按 CommonMark 规则配不上对」的强调改成 HTML。

  中英混排 + 标点相邻时，``**`` 的 flanking 判定会拒绝配对，标记就原样漏到页面上：

  - ``**坑一：…输出。**见 4.6 节…``：闭合 ``**`` 前面是 ``。``（标点）、后面是 ``见``
    （非空白非标点），不算 right-flanking，配不上 → 页面上显示字面 ``**``；
  - ``中**（强调）**``：开启 ``**`` 前面是汉字、后面是 ``（``，不算 left-flanking。

  实测全站强强调 2854 处里有 900 处属于这两类。这里只改这些配不上的对（改成
  ``<strong>`` / ``<em>``，与字符集无关），配得上的保持 Markdown 原样——页脚那种
  「``*…链接…*``」的写法因此不受影响（HTML 里再嵌 Markdown 链接不会被解析）。

  :param md_text: markdown 文本。
  :returns: 处理后的文本。
  """

  masked = _mask_protected(md_text)
  edits: list[tuple[int, int, str]] = []
  for pattern, tag in ((_STRONG_PAIR_RE, "strong"), (_EM_PAIR_RE, "em")):
    for match in pattern.finditer(masked):
      if any(start <= match.start() and match.end() <= end for start, end, _text in edits):
        continue  # 已被外层强调改写（如 **a *b* c**）
      # 判定用原文（掩码只负责找出候选对）：CommonMark 看到的是真实字符，
      # 公式的 ``$`` 属于标点，会影响 flanking 结论。
      content = md_text[match.start(1):match.end(1)]
      before = md_text[match.start() - 1] if match.start() > 0 else ""
      after = md_text[match.end()] if match.end() < len(masked) else ""
      if _can_emphasize(before, after, content):
        continue
      edits.append((match.start(), match.end(), f"<{tag}>{content}</{tag}>"))
  for start, end, replacement in sorted(edits, reverse=True):
    md_text = md_text[:start] + replacement + md_text[end:]
  return md_text


def _can_emphasize(before: str, after: str, content: str) -> bool:
  """判断这对强调标记按 CommonMark 规则能否配对。

  :param before: 开启标记前的字符。
  :param after: 闭合标记后的字符。
  :param content: 强调内容。
  :returns: 能配对为 True。
  """
  if not content or _is_space(content[0]) or _is_space(content[-1]):
    return False
  left_flanking = not (  # 开启标记：不能「后面是标点、前面既非空白也非标点」
    _is_punct(content[0]) and not (_is_space(before) or _is_punct(before)))
  right_flanking = not (  # 闭合标记：不能「前面是标点、后面既非空白也非标点」
    _is_punct(content[-1]) and not (_is_space(after) or _is_punct(after)))
  return left_flanking and right_flanking


def _escape_colon_fences(md_text: str) -> str:
  """转义行首的 ``:::``，避免被 MyST 当成 colon fence。

  本站自己一律用反引号围栏，所以正文里出现的 ``:::`` 都是 pandoc 的 fenced div
  残留（例如 ``\\begin{center}``）。不转义的话，它会把后面整段内容吞进指令里。

  :param md_text: markdown 文本。
  :returns: 转义后的文本。
  """
  return re.sub(r"(?m)^([ \t]*)(:{3,})", r"\1\\\2", md_text)


def _masked_for_scan(md_text: str) -> str:
  """屏蔽代码围栏、行内代码与数学区，只留下正文。

  残留统计必须区分「本该转成 HTML 的 LaTeX」和「公式里的 LaTeX 命令」——后者是
  正常的。pandoc 用 ``--wrap=none`` 输出，公式里的换行会原样保留，所以
  ``$...$`` 也可能跨行，逐行正则不够用，这里做一次字符级扫描。

  :param md_text: pandoc 产出的 markdown。
  :returns: 屏蔽后的文本。
  """
  lines = md_text.splitlines()
  flags = code_fence_flags(lines)
  fenced = ["" if is_code else line for line, is_code in zip(lines, flags)]
  text = tokens.TOKEN_RE.sub(" ", "\n".join(fenced))

  out: list[str] = []
  index = 0
  length = len(text)
  while index < length:
    char = text[index]
    if char == "\\":
      if text.startswith("\\$", index):
        out.append("\\$")
        index += 2
        continue
      if text.startswith("\\[", index) or text.startswith("\\(", index):
        closer = "\\]" if text[index + 1] == "[" else "\\)"
        end = text.find(closer, index + 2)
        if end != -1:
          out.append(" ")
          index = end + 2
          continue
      out.append(char)
      index += 1
      continue
    if char == "$":
      if text.startswith("$$", index):
        end = text.find("$$", index + 2)
        if end != -1:
          out.append(" ")
          index = end + 2
          continue
      end = _closing_dollar(text, index + 1)
      if end != -1:
        out.append(" ")
        index = end + 1
        continue
    if char == "`":
      end = text.find("`", index + 1)
      if end != -1 and "\n" not in text[index:end]:
        out.append(" ")
        index = end + 1
        continue
    out.append(char)
    index += 1
  return "".join(out)


#: 漏网的数学命令降级表（只作用于非数学区，避免污染公式）。
_LEFTOVER_MAP = (
  (re.compile(r"\\mathbf\{([^{}]*)\}"), r"**\1**"),
  (re.compile(r"\\mathit\{([^{}]*)\}"), r"*\1*"),
  (re.compile(r"\\mathrm\{([^{}]*)\}"), r"\1"),
  (re.compile(r"\\text(?:rm|bf|it|tt|sf)?\{([^{}]*)\}"), r"\1"),
  (re.compile(r"\\texttt\{([^{}]*)\}"), r"`\1`"),
  (re.compile(r"\\times\b"), "×"),
  (re.compile(r"\\cdot\b"), "·"),
  (re.compile(r"\\to\b"), "→"),
  (re.compile(r"\\approx\b"), "≈"),
  (re.compile(r"\\ldots\b"), "…"),
  (re.compile(r"\\qquad\b|\\quad\b"), " "),
)


def _degrade_leftovers(md_text: str) -> str:
  """把数学区之外漏网的 LaTeX 命令降级成等价纯文本。

  pandoc 偶有解析失败（含不平衡 ``$`` 的表格单元），残留的 ``\\mathbf{...}`` 之类
  会原样显示在页面上。这里只处理非数学区，公式里的命令一律不碰。

  :param md_text: 渲染后的 markdown。
  :returns: 降级后的文本。
  """
  segments: list[str] = []
  plain_start = 0
  index = 0
  length = len(md_text)
  while index < length:
    opener = ""
    closer = ""
    if md_text.startswith("$$", index):
      opener, closer = "$$", "$$"
    elif md_text.startswith("\\[", index):
      opener, closer = "\\[", "\\]"
    elif md_text.startswith("\\(", index):
      opener, closer = "\\(", "\\)"
    elif md_text[index] == "$" and (index == 0 or md_text[index - 1] != "\\"):
      opener, closer = "$", "$"
    if opener:
      end = md_text.find(closer, index + len(opener))
      if end != -1:
        segments.append(_degrade_plain(md_text[plain_start:index]))
        segments.append(md_text[index:end + len(closer)])
        index = end + len(closer)
        plain_start = index
        continue
    index += 1
  segments.append(_degrade_plain(md_text[plain_start:]))
  return "".join(segments)


def _degrade_plain(text: str) -> str:
  """对普通文本段应用降级替换（跳过代码围栏与行内代码）。

  :param text: 普通文本段。
  :returns: 替换后的文本。
  """
  lines = text.split("\n")
  flags = code_fence_flags(lines)
  result: list[str] = []
  for line, is_code in zip(lines, flags):
    result.append(line if is_code else _degrade_line(line))
  return "\n".join(result)


def _degrade_line(line: str) -> str:
  """对一行普通文本应用降级替换（行内代码保持原样）。

  :param line: 一行 markdown。
  :returns: 替换后的行。
  """
  parts = re.split(r"(`[^`]*`)", line)
  for position in range(0, len(parts), 2):
    for pattern, replacement in _LEFTOVER_MAP:
      parts[position] = pattern.sub(replacement, parts[position])
  return "".join(parts)


def _closing_dollar(text: str, start: int) -> int:
  """找下一个未转义的 ``$``，遇到空行即放弃（避免一个杂散 ``$`` 吃掉整段正文）。

  :param text: 文本。
  :param start: 起始下标。
  :returns: 下标，找不到返回 -1。
  """
  index = start
  while index < len(text):
    char = text[index]
    if char == "\\":
      index += 2
      continue
    if char == "$":
      return index
    if text.startswith("\n\n", index):
      return -1
    index += 1
  return -1


def scan_leftovers(md_text: str) -> dict[str, int]:
  """统计 pandoc 未能转换、残留的 LaTeX 命令。

  :param md_text: pandoc 产出的 markdown。
  :returns: 命令名 → 出现次数（按次数降序）。
  """
  masked = _masked_for_scan(md_text)
  counts: dict[str, int] = {}
  for match in re.finditer(r"\\([a-zA-Z]+)", masked):
    counts[match.group(1)] = counts.get(match.group(1), 0) + 1
  return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
