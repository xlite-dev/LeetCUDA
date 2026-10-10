# 版式与前端

改这里的任何东西，跑构建前先 `touch build/src/index.md`（见 `pipeline.md` 的「增量构建不重拷 `_static/`」），改完用 `tools/visual_check` 的版式几何核对确认没有回归。

## 三栏版式与居中

- **三栏：左「篇 → 章」+ 正文 + 右「本页目录」**：左栏只到章节级（`conf.py` 的 `navigation_depth = 2`），小节级目录放右栏，两处不重复。
- **整体居中**：rtd 主题把 `.wy-nav-content` 的 `margin` 归零，三栏整体贴左（2560px 屏右侧空 1460px）。`custom.css` 把「侧栏 + 正文」当作一组：侧栏 `left` 与正文容器 `margin-left` 用同一个 `calc()` 偏移，两者紧贴、空白均分到两侧。三栏需要 300 + 800 + 28 + 220 = 1348px，窄于此宽度隐藏右栏；≥1600px 正文放宽到 960px。
- **图片与表格居中**：书里插图多包在 `\begin{center}` 里（那层排版环境在转换时剥掉了），表格是 pandoc 直出的 HTML，两者默认都靠左。CSS 里给 `p > img:only-child` 与表格设了居中——只处理独立成段和表格单元格里的图，正文中夹在句子里的图不动；窄表按内容宽度居中，宽表被 `max-width: 100%` 截到栏宽后在表内横向滚动。
- **过宽公式**：`mjx-container[display="true"]` 用 `min-width: 0 !important` 覆盖 MathJax 的内联 `min-width`，让它在容器内滚动，而不是撑出整页横向滚动条。

## 模板覆盖规则（踩过，别再踩）

覆盖哪个块就写哪个模板文件，写错位置会被顶掉且不报错：

| 目标 | 必须写在 | 原因 |
| --- | --- | --- |
| `body` 块（正文区，插右栏 `{{ toc }}`） | `_templates/page.html` | Sphinx 基础主题的 `page.html` 也定义了这个块，写在 `layout.html` 里会被它顶掉 |
| `sidebartitle` 块（侧栏兜底按钮行） | `_templates/layout.html` | `genindex.html` / `search.html` 直接 `extends layout.html` 而不经 `page.html`，写 `page.html` 里这两个页面会漏 |
| 上一页 / 下一页按钮行（插「下载 PDF」） | `_templates/breadcrumbs.html`、`_templates/footer.html`（**整文件复制**） | 那一行不在任何 `{% block %}` 里——`breadcrumbs.html` 的块只有 `breadcrumbs` / `breadcrumbs_aside`，都在上面的面包屑 `<ul>` 内。**升级 sphinx-rtd-theme 时必须回来对齐这两个副本**（版本在 `requirements.txt` 里钉死） |

## 「下载 PDF」按钮（挨着「下一页」）

按钮在**上一页 / 下一页那一行的「下一页」右边**，`prev_next_buttons_location = both` 所以顶部与页脚两处都有；**每页都有**：

| 页面 | 按钮落位 |
| --- | --- |
| 正文页（有下一页） | `.rst-breadcrumbs-buttons` 与 `.rst-footer-buttons` 里，紧跟「下一页」右侧（同为 `float-right`，DOM 里排在「下一页」之后，浮动因此把它挤到「下一页」与容器右缘之间） |
| 末页（只有上一页） | 同一行的最右（没有「下一页」可依附，不影响可读性） |
| `genindex` / `search`（没有上一页/下一页那一行） | 回退到 `_templates/layout.html` 的 `sidebartitle` 块：侧栏搜索框下方 `.rtd-sidebar-actions`，此时「English」也落在这一行（左下载、右切换） |

- **静态锚点，不依赖 JS**：正文页那两处写在 `_templates/breadcrumbs.html` / `_templates/footer.html`（整文件复制主题模板，见上面的覆盖规则表）；兜底那处写在 `layout.html`，用 `{% if not (prev or next) %}` 判条件。
- **链接的唯一事实源是仓库根 `README.md` 的 `[leetcuda-pdf]` 链接定义**（不是书稿 tex）。构建期解析（`convert/readme.py`，`conf.py` 与 `convert/verify.py` 共用同一 helper），发新版只改 README 那一行，站点跟着变；解析不到、或地址不是 http(s)，`conf.py` 直接抛错让构建红灯，不静默少一个按钮。
- **按钮之间一律不留缝**（用户 2026-10-10 两次要求，先首页三个、后「上一页 + English」）：`custom.css` 里这一行的按钮**都不给外边距**——`.rtd-pdf-download` 无 `margin-right`，`.rtd-translate-toggle` 无 `margin-left`（历史上有过 8px，已删）。挨在一起靠各自的 1px 边框分界；间距只由侧栏那一行的 flex `gap` 提供。行高沿用同一条 24px 规则（`.rst-breadcrumbs-buttons > .btn` / `.rst-footer-buttons > .btn`）。
- **想换到「下一页」左边**：把 `breadcrumbs.html` / `footer.html` 里的那段锚点挪到 `{%- if next %}` **之前**即可（同为 `float-right` 时 DOM 在前的先占右缘）。
- 构建后由 `convert/verify.py` 的「下载 PDF 按钮已渲染」逐页核对：按钮在，且链接与 README 解析值一致（`html.unescape` 后比对）——三处落位都算数。

## 中英切换按钮

按钮是 `translate.js` 造的 `btn btn-neutral` 图标按钮，三种落位（`buildButton()`）：

| 页面 | 落位 |
| --- | --- |
| 有「上一页」的页面（章节页） | 插在**每一处**「上一页」后面——主题把按钮行渲染在顶部与页脚两处（`prev_next_buttons_location = both`），所以它也是两处 |
| 首页（没有「上一页」，但有「下一页」行） | 插进那两处按钮行的右侧按钮组、落在「下载 PDF」**右边**；左侧导航栏里不再有 |
| `genindex` / `search`（连按钮行都没有） | 退回侧栏：模板给的 `.rtd-sidebar-actions`（「下载 PDF」也在那一行），没有该容器（旧产物）才自建 `.rtd-translate` 包装 |

⚠️ 按钮行是浮动布局：`float: right` 的元素**先出现的贴右缘**，后出现的挤到它左边。所以首页那处「显示在下载按钮右边」在 DOM 上得插在下载锚点**前面**，并带 `rtd-translate-toggle-right`（`custom.css` 里给它 `float: right`）。这一行的按钮一律紧贴、不留缝（见上面「下载 PDF」一节的间距说明）。

翻译机制、等待判据与状态提示见 `english.md`。
