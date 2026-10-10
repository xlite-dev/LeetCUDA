# 版式与前端

改这里的任何东西，跑构建前先 `touch build/src/index.md`（见 `pipeline.md` 的「增量构建不重拷 `_static/`」），改完用 `tools/visual_check` 的版式几何核对确认没有回归。

## 三栏版式与居中

- **三栏：左「篇 → 章」+ 正文 + 右「本页目录」**：左栏只到章节级（`conf.py` 的 `navigation_depth = 2`），小节级目录放右栏，两处不重复。
- **整体居中**：rtd 主题把 `.wy-nav-content` 的 `margin` 归零，三栏整体贴左（2560px 屏右侧空 1460px）。`custom.css` 把「侧栏 + 正文」当作一组：侧栏 `left` 与正文容器 `margin-left` 用同一个 `calc()` 偏移，两者紧贴、空白均分到两侧。三栏需要 300 + 800 + 28 + 220 = 1348px，窄于此宽度隐藏右栏；≥1600px 正文放宽到 960px。
- **图片与表格居中**：书里插图多包在 `\begin{center}` 里（那层排版环境在转换时剥掉了），表格是 pandoc 直出的 HTML，两者默认都靠左。CSS 里给 `p > img:only-child` 与表格设了居中——只处理独立成段和表格单元格里的图，正文中夹在句子里的图不动；窄表按内容宽度居中，宽表被 `max-width: 100%` 截到栏宽后在表内横向滚动。
- **过宽公式**：`mjx-container[display="true"]` 用 `min-width: 0 !important` 覆盖 MathJax 的内联 `min-width`，让它在容器内滚动，而不是撑出整页横向滚动条。

## 模板覆盖规则（踩过，别再踩）

覆盖哪个块就写哪个模板文件，写错位置会被顶掉且不报错：

| 目标块 | 必须写在 | 原因 |
| --- | --- | --- |
| `body`（正文区，插右栏 `{{ toc }}`） | `_templates/page.html` | Sphinx 基础主题的 `page.html` 也定义了这个块，写在 `layout.html` 里会被它顶掉 |
| `sidebartitle`（搜索框下方，插「下载 PDF」） | `_templates/layout.html` | `genindex.html` / `search.html` 直接 `extends layout.html` 而不经 `page.html`，写 `page.html` 里这两个页面的侧栏会漏掉 |

## 侧栏「下载 PDF」按钮

左栏搜索框下方是「下载 PDF」按钮，**每页都有**。页面没有「上一页」按钮时（首页就是），中英切换按钮「English」会和它并排在同一行（左下载、右切换）；有「上一页」的章节页里切换按钮在正文顶部，那一行就只剩下载按钮。

- **链接的唯一事实源是仓库根 `README.md` 的 `[leetcuda-pdf]` 链接定义**（不是书稿 tex）。构建期解析（`convert/readme.py`，`conf.py` 与 `convert/verify.py` 共用同一 helper），发新版只改 README 那一行，站点跟着变；解析不到、或地址不是 http(s)，`conf.py` 直接抛错让构建红灯，不静默少一个按钮。
- **渲染在 `_templates/layout.html` 的 `sidebartitle` 块**（静态链接，不依赖 JS），容器是 `.rtd-sidebar-actions`。
- **与「English」同行**：`translate.js` 的侧栏回退分支把切换按钮追加进 `.rtd-sidebar-actions`，`custom.css` 用 flex 排成「左下载、右切换」。章节页的切换按钮在正文顶部（跟在「上一页」后面），那一行就只剩下载按钮。
- 构建后由 `convert/verify.py` 的「下载 PDF 按钮已渲染」逐页核对：按钮在，且链接与 README 解析值一致（`html.unescape` 后比对）。

## 中英切换按钮

按钮是 `translate.js` 造的 `btn btn-neutral` 图标按钮，插入规则：**有「上一页」按钮就插在它后面（章节页在正文顶部，`document.querySelectorAll` 会命中顶部与页脚两处），页面没有「上一页」时才退回侧栏**——侧栏优先用模板给的 `.rtd-sidebar-actions`，没有该容器（旧产物）才自建 `.rtd-translate` 包装。

翻译机制、等待判据与状态提示见 `english.md`。
