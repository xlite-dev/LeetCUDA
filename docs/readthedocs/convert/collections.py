"""并入站点的子集合：主书之外、整本转成站点的独立书。

``COLLECTIONS`` 是唯一真值源：``convert.__main__`` 据此转换与装配（主书目录末尾追加
一个入口），``convert.verify`` 据此逐书核对。新增一本集合只改这里一处。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Collection:
  """一本并入站点的子集合。

  :param slug: 站点子目录名（兼 toctree 的文档前缀），如 ``cute-zhihu``。
  :param title: 集合首页 H1，也是主书侧栏里的入口文字。
  :param book_tex: 相对主书根目录的集合根 tex，如 ``cute-zhihu/cute-zhihu.tex``。
  :param intro: 首页说明段落（markdown 原文）。
  """
  slug: str
  title: str
  book_tex: str
  intro: tuple[str, ...] = ()


COLLECTIONS: tuple[Collection, ...] = (
  Collection(
    slug="cute-zhihu",
    title="CuTe文章集合",
    book_tex="cute-zhihu/cute-zhihu.tex",
    intro=(
      "本集合忠实整理知乎上多位作者的 CUDA/CuTe 系列文章，共 32 篇，按系列与主题分四部分编排："
      "reed 的 CuTe 系列（13 篇）、竹熙佳处的 CuTe 教程系列（7 篇）、reed 的"
      "「NVidia GPU 指令集架构」系列（8 篇），以及 GPU 架构·内存·Attention 精度主题的 4 篇。"
      "同一部分内按系列依赖与发表顺序编排，每章章首标注作者与原文链接。",
      "整理过程中忠实保留原文内容与图片，除修正明显的公式、图号与拼写笔误、补全缺失的图注外"
      "不对原文作改动。全部文章著作权归原作者所有，本集合仅用于开源学习，不得用于任何商业用途。",
    ),
  ),
)
