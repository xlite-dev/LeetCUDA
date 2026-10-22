"""LeetCUDA book Read the Docs 站点配置。

源树（``build/src``）由 ``python -m convert`` 从书稿 tex 生成，因此这里的
``exclude_patterns`` 只需挡住构建产物；站点标题、主题、MathJax 版本在此固定。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 侧栏「下载 PDF」按钮的链接：唯一事实源是仓库根 README.md 的 `[leetcuda-pdf]` 链接定义
# （链接随版本发布变化）。构建期现读现用，解析失败即构建失败，不静默降级成「没有按钮」。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert.readme import REPO_URL, pdf_link  # noqa: E402

leetcuda_pdf_url = pdf_link()

# 右上角「Go to repository」按钮的仓库地址（构建核对复用同一个常量）。
leetcuda_repo_url = REPO_URL

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
  # 三层：主书是「篇 → 章」两层（章内小节放在右栏，见 _templates/page.html 的 `body` 块），
  # 子集合多一层「集合首页 → 系列 → 文章」，要三层才能像主书那样把章节名列在系列下面。
  # ``titles_only`` 必须为 True：否则 depth=3 会把主书每一章的 ``\section`` 也列进左栏
  # （实测条目从 58 条涨到 542 条）。小节目录本来就在右栏，左栏只要标题。
  "navigation_depth": 3,
  "collapse_navigation": False,
  "sticky_navigation": True,
  "titles_only": True,
  "prev_next_buttons_location": "both",
}

templates_path = ["_templates"]

# 中英切换按钮（Google 网站翻译，按需加载；只在公网可访问时可用）。
# math-en.js 必须排在 translate.js 之前：它同步定义公式英译词典
# （window.LEETCUDA_MATH_EN），translate.js 在 DOM 解析完时同步替换公式 tex 源，
# 要赶在 defer 的 MathJax 渲染之前（见 _static/translate.js 的 swapMathText）。
html_js_files = ["math-en.js", "translate.js"]

# 侧栏「下载 PDF」按钮的地址，模板（_templates/layout.html）据此渲染静态链接。
html_context = {
  "leetcuda_pdf_url": leetcuda_pdf_url,
  "leetcuda_repo_url": leetcuda_repo_url,
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# 书里定义了近 90 个数学宏（`\v` 是向量、`\Z` 是整数集、`\abs`/`\norm` 带参数…）。
# 不给 MathJax 这些定义，`\v{L}` 会按内置重音命令渲染成 `Ľ`、`\Z` 直接报错。
# 宏表由转换器从书稿 tex 里抽出（convert/macros.py），构建时生成该 JSON。
_macro_file = Path(__file__).resolve().parent / "build" / "mathjax-macros.json"
_math_macros = (
  json.loads(_macro_file.read_text(encoding="utf-8")) if _macro_file.is_file() else {}
)

mathjax3_config = {
  "tex": {
    "tags": "ams",
    "inlineMath": [["\\(", "\\)"], ["$", "$"]],
    "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    "processEscapes": True,
    "macros": _math_macros,
  },
  "options": {"enableMenu": False},
}

def _ensure_mathjax_on_all_pages(app, pagename, templatename, context, doctree):
  """给「自身没有数学」的页面也补上 MathJax。

  Sphinx 只给文档自身含数学的页面插 MathJax script，而侧栏目录里的章节标题可能含
  公式（第 37 章的 ``$D$``）——这些标题出现在**每一页**的导航里，没有 script 的
  页面（``appD-source-index``、``genindex``）上它们会字面显示。config 与 script 与
  Sphinx 自己注入的形式一致（``mathjax3_config`` + ``mathjax_path``），由
  ``_templates/layout.html`` 的 ``scripts`` 块追加在 ``math-en.js`` / ``translate.js``
  之后，顺序仍满足「词典与 tex 源替换先于 MathJax 渲染」。
  """
  scripts = context.get("script_files") or []
  for item in scripts:
    filename = getattr(item, "filename", None)
    if filename is None and isinstance(item, dict):
      filename = item.get("filename", "")
    if "mathjax" in str(filename or item).lower():
      return
  context["leetcuda_mathjax_fallback"] = {
    "config": json.dumps(mathjax3_config, ensure_ascii=False),
    "path": mathjax_path,
  }


def setup(app):
  app.connect("html-page-context", _ensure_mathjax_on_all_pages)


# 两类已知误报：
# 1. myst.xref_missing——站内链接是 ``page.html#anchor`` 形式，锚点由原始 HTML 写出，
#    MyST 看不见，会把它们当未解析的交叉引用；断链由转换报告与 verify 的锚点对账把关。
# 2. myst.header——原书有 40 处标题跳级（``\subsubsection`` 直接跟在 ``\section``
#    后面等），MyST 会报 non-consecutive header level，对渲染无影响。
suppress_warnings = ["myst.xref_missing", "myst.header", "misc.highlighting_failure"]
