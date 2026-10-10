"""从 Sphinx 产出的静态 HTML 里抽取「含中文的代码块 / 公式文本节点」。

两处消费：``convert.verify`` 用它核对英译词典与页面产物逐字节一致；
``tools/code_dict`` 人工产线用它从 ``build/html`` 抽取节点供翻译。输入都是
静态 HTML——此时 MathJax 还没渲染，``.math`` 里的文本节点就是 tex 源，
与 ``translate.js`` 在英文模式下替换的时机（DOM 解析完、MathJax 渲染前）一致。
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

#: 汉字（用于判定「含中文的节点」）。
CJK = re.compile(r"[\u4e00-\u9fff]")

#: 汉字 + 中文标点 + 全角形式（译文里禁止残留）。
CJK_BAD = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")

#: 连续的中文段（公式骨架校验时替换成占位符）。
CJK_RUN = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]+")


class PreTextCollector(HTMLParser):
  """收集 ``<pre>`` 内的文本节点（html.parser 默认解码实体，即 textContent）。"""

  def __init__(self) -> None:
    super().__init__(convert_charrefs=True)
    self.depth = 0
    self.nodes: list[str] = []

  def handle_starttag(self, tag: str, attrs) -> None:
    if tag == "pre":
      self.depth += 1

  def handle_endtag(self, tag: str) -> None:
    if tag == "pre" and self.depth:
      self.depth -= 1

  def handle_data(self, data: str) -> None:
    if self.depth and CJK.search(data):
      self.nodes.append(data)


class MathTextCollector(HTMLParser):
  """收集 class 含 ``math`` 的元素（Sphinx 公式容器）内的文本节点。

  Sphinx mathjax3 输出 ``<div/span class="math notranslate nohighlight">tex 源</...>``，
  MathJax 渲染后才换成 ``mjx-container``——静态 html 里文本节点就是 tex 源。
  """

  def __init__(self) -> None:
    super().__init__(convert_charrefs=True)
    self.depth = 0
    self.nodes: list[str] = []

  def handle_starttag(self, tag: str, attrs) -> None:
    if self.depth:
      self.depth += 1
      return
    for name, value in attrs:
      if name == "class" and value and "math" in value.split():
        self.depth = 1
        break

  def handle_endtag(self, tag: str) -> None:
    if self.depth:
      self.depth -= 1

  def handle_data(self, data: str) -> None:
    if self.depth and CJK.search(data):
      self.nodes.append(data)
