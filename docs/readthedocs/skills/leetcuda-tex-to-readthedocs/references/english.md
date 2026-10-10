# 英文版：正文机器翻译 + 图内文字英文图集 + 封面

## 正文：Google 网站翻译

侧栏搜索框下方有一个 `English / 中文` 按钮，点击后整站切到机器翻译的英文版。

- **实现**：`_static/translate.js` 写 `googtrans=/zh-CN/en` cookie 并 reload；官方组件（`translate.google.com/translate_a/element.js`）**只在检测到该 cookie 时才加载**——没切过英文的读者不会引入第三方脚本。
- **保护代码与公式**：切换前给 `pre` / `code` / `.highlight` / 公式容器加 `notranslate`，避免机器翻译改坏它们（Sphinx 已给数学区加了这个类）。
- **加载时机**：英文模式下 `translate.js`（本来就在 `<head>` 里）在解析阶段就发起 `element.js` 请求，不等 `DOMContentLoaded`——`<head>` 里的 MathJax 是 `defer` 的，`DOMContentLoaded` 要等它下载并执行完，线上实测把这一步拖到 3.3–12.8 s；提前注入后请求落在 95–136 ms（此时 `body` 还没建好，容器与脚本挂到 `document.head`）。按钮与状态提示同理，放在 `readystatechange` 到 `interactive` 时建，不必等 MathJax。
- **等待与兜底**：正文替换中位 14.6 s、范围 9.0–22.0 s，另有 45 s 与 90 s 都没翻完的样本（`element.js` 卡住、`el_main` 请求没发出）。Google 插的横幅 iframe 比正文替换早十几秒，**不能当「翻完」判据**；判据是正文自己变了：首个标题已翻成英文，**且**跨页取样（等距取 30 个标题/段落）的汉字数掉到起始值的 70% 以下——只看标题不够，正文是分批落地的，标题可能先到。自发起请求起 30 s 还没等到，按钮旁给出「Google Translate is slow, read in Chinese」，之后转入 2 s 一次的慢速轮询，翻完即收掉。
- **Google 只翻一部分正文**：ch01 上 417 个汉字（约占正文一成）在线上与本地**数字完全一致**地永久留中文，滚动不会补上。所以判据不能写成「一个汉字都不剩」（代码块本来就留着中文），也不能写成「取样节点干净占比 ≥ 0.7」（实测收敛值只有 0.48）。
- **横幅要藏两层**：顶部 Google 翻译横幅（「已翻译为以下语言 / 显示原文 / 选项」）被 `_static/custom.css` 隐藏。**Google 换过实现**：老组件是 `iframe.goog-te-banner-frame`，现在（2026-10-09 线上实测）是 `div.skiptranslate` 包装里的 `iframe.skiptranslate`（其余类名是混淆的 `VIpgJd-…`，不能认）——只写老类名藏不掉，两条规则都留着。另外 Google 会给 `body` 写内联 40px 的 `top` 让位，必须一并清掉（横幅是 `position: fixed` + 超大 z-index）：只藏横幅会在顶上留一条空白，只清偏移不藏横幅则直接盖在正文上。
- **前提是站点能被公网访问**：Google 翻译是让 Google 的服务器去抓页面，`localhost` 与内网地址它抓不到；本地预览时按钮给出提示（本地想看英文可用浏览器自带的翻译），部署到 RTD 后正常可用。
- **不想要这个按钮**：删掉 `conf.py` 里的 `html_js_files` 一行即可，其余不受影响。局限：Google 的网站翻译组件多年未更新，对技术书只能算「读得懂」；按钮是附加功能，翻译坏了不影响中文站。

## 英文图集（图内文字）

图内文字**无法被机器翻译**：TikZ 图编译成 SVG 时用了 `--no-fonts`（字形转矢量轮廓），图里根本没有文本节点；即便有，Google 的网站翻译也只改写 HTML 文本节点，不碰 SVG。所以英文版只能靠**构建期按词典重编译一套 SVG**。

- **词典**：`i18n/figures-en.json`（掩码后的中文单元 → 掩码后的英文单元，1793 条）。用 `python -m tools.translate_figures --proxy http://localhost:7890` 生成：把节点文字掩码后分批送翻译接口（与站点英文模式同一引擎，风格一致），逐条校验占位符完整、可还原，按批保存、可中断续跑。这一步需要联网，是**一次性人工工具**，不参与构建。
- **掩码**：`convert/figtext.py` 在送翻译前把 `\\`（换行）、`$…$`（数学）、`\texttt{…}` 的**命令名与花括号**替换成占位符 `⟦n⟧`，内容留给翻译——否则机器翻译会拆散命令、改坏数学。译文里的 `%`/`&`/`_`/`~` 等特殊字符会被转义。
- **只收下能编译的图**：翻译可能把 `\textbf{` 与它的 `}` 搬到句子不同位置（括号数量仍平衡、位置错了），所以替换时校验**括号嵌套合法性**，编译时校验「PDF 有文字 ⇒ SVG 必须有字形」。任何一处不过，该图就不进英文图集——读者看到的是中文原图，而不是半中半英的图。
- **产物与换图**：`python -m convert` 会额外编译 196 张英文 SVG 到 `_static/figures-en/`（Sphinx 整目录复制，浏览器直接可达，不入库）；英文模式下 `_static/translate.js` 把 `_images/<名字>.svg` 换成 `_static/figures-en/<名字>.svg`，加载失败则回退中文原图。
- **已知边界**：图里写在数学内部的少量中文（`$\text{低秩 GEMM}$` 这类，全书 6 个字）仍是中文——它在数学区间里，翻译会破坏公式。

## 英文封面（离线生成，产物入库）

封面不在上面这套流程里——它是书稿的独立文档 `figures/misc/cover-tikz.tex`，编译成 PDF 后由 `convert._cover_image()` 光栅化成 PNG，字同样是矢量轮廓；而且封面用的 Humor Sans / Comic Neue 只装在作者本机、RTD 镜像里没有（中文用的 LXGW WenKai 25 MB 也不适合入库）。

所以英文封面是**离线生成一次、产物入库**：`python -m tools.build_cover_en`（默认 240 dpi）写到 `_static/figures-en/cover.png`，`.gitignore` 只放行这一个文件。换图用的是同一套逻辑（`_images/cover.png` → `_static/figures-en/cover.png`，加载失败回退中文封面），不用改 JS。

**封面文案改了要重跑该工具**：替换按原文片段定位，命中数不为 1、或产物里还残留汉字就直接报错停下，不会出一张半中半英的封面。
