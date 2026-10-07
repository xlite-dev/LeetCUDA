"""LeetCUDA book Read the Docs 站点配置。

源树（``build/src``）由 ``python -m convert`` 从书稿 tex 生成，因此这里的
``exclude_patterns`` 只需挡住构建产物；站点标题、主题、MathJax 版本在此固定。
"""
from __future__ import annotations

project = "LeetCUDA：CUDA Kernel 优化之路"
author = "LeetCUDA Project"
copyright = "LeetCUDA Project"
language = "zh_CN"

extensions = [
  "myst_parser",
  "sphinx.ext.mathjax",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "build", "README.md"]

myst_enable_extensions = [
  "amsmath",
  "attrs_block",
  "attrs_inline",
  "colon_fence",
  "deflist",
  "dollarmath",
  "html_image",
  "substitution",
  "tasklist",
]
myst_heading_anchors = 3
myst_dmath_allow_labels = True

numfig = False
html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_last_updated_fmt = ""
html_title = project
html_theme_options = {
  "navigation_depth": 4,
  "collapse_navigation": False,
  "sticky_navigation": True,
  "titles_only": False,
  "prev_next_buttons_location": "both",
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
  "tex": {
    "tags": "ams",
    "inlineMath": [["\\(", "\\)"], ["$", "$"]],
    "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    "processEscapes": True,
  },
  "options": {"enableMenu": False},
}

# 两类已知误报：
# 1. myst.xref_missing——站内链接是 ``page.html#anchor`` 形式，锚点由原始 HTML 写出，
#    MyST 看不见，会把它们当未解析的交叉引用；断链由转换报告与 verify 的锚点对账把关。
# 2. myst.header——原书有 40 处标题跳级（``\subsubsection`` 直接跟在 ``\section``
#    后面等），MyST 会报 non-consecutive header level，对渲染无影响。
suppress_warnings = ["myst.xref_missing", "myst.header", "misc.highlighting_failure"]
