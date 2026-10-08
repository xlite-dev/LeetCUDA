"""LeetCUDA book → Read the Docs 站点转换管线。

把 ``kernels/interview/book/`` 下的章节 tex 逐章转换为 MyST markdown，并生成
Sphinx 站点树。原书 tex 与源码保持零改动：转换只在 ``<rtd>/build/`` 下的派生
副本上进行。

管线：``booktree``（导航骨架）→ ``labels``（label/编号注册表）→
``preprocess``（归一化 tex + 抽离 tikz/代码）→ ``tikz2svg``（TikZ 图编译 SVG）→
``postprocess``（markdown 还原为 MyST）→ ``__main__``（编排与站点装配）。
"""
from __future__ import annotations

__version__ = "0.1.0"
