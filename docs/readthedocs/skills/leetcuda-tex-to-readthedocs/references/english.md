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

## 代码块注释英译（`_static/code-en.json` + `translate.js`）

代码块在 `notranslate` 保护下机器翻译碰不到，与图内文字一样只能走构建期词典：**正文机器翻译 + 代码块按词典替换**是并行的两条线。

- **词典**：`_static/code-en.json`（入库，Sphinx 整目录拷贝）。key = 代码块文本节点 **trim 后的原文**（含代码与中文，如 `nsys stats -r cuda_gpu_kern_sum report.nsys-rep   # per-kernel 时间占比（最常用）`），value = 英文整段——代码/命令/标识符/数字逐字保留，只翻中文，中文标点转英文。全站 771 条唯一节点（40 页）。
- **产线**：`python -m tools.code_dict extract --groups 8 --out <dir>` 从 `build/html` 抽出全部含中文的 `pre` 文本节点（唯一化、按文件装箱均衡分组）→ 各组逐条人工/模型翻译（保持注释前缀与缩进）→ `python -m tools.code_dict merge --parts <dir>/parts` 合并并校验（key 全覆盖、value 无汉字/中文标点、多行节点中纯代码行逐字保留、含中文的 key 不得原样照抄）。**任何一项不过 merge 直接报错退出**，不写半吊子词典。
- **替换**：`translate.js` 的 `swapCodeComments()`——英文模式下 fetch 词典，`TreeWalker` 遍历 `pre` 文本节点，trim 后命中才替换（`nodeValue.replace(trimmed, hit)` 保留前后空白，缩进不动）；未命中的注释保持中文（读者看到完整中文而不是半中半英）；词典拉取失败静默降级。**不依赖 Google 可达，本地预览同样生效**，所以浏览器验收可以直接在 localhost 做（英文模式下 `pre` 内汉字数应为 0）。
- **书稿/源码更新后**：代码块内容变了，重跑一遍产线更新词典——extract 会列出全部当前节点，merge 会对不出词条的节点报 missing。
- **范围**：只处理块级 `pre`；行内 `` `code` `` 的中文不在其中（量小，且属于正文翻译范畴）。

## 公式内文字英译（`_static/math-en.js` + `translate.js`）

公式被 `notranslate` 保护、机器翻译碰不到，公式里的中文（几乎都在 `\text{}` 内）同样走词典：**产线与代码块共用 `tools/code_dict.py`，用 `--what math` 切换**。

- **词典**：`_static/math-en.js`，格式 `window.LEETCUDA_MATH_EN = {...}`（JS 赋值而非 JSON）。key = 公式容器（Sphinx 的 `.math`）内 **tex 源原文**（trim 后），value = 英文版 tex。全站 123 条唯一节点（33 页）。
- **时序是关键**：MathJax 是 defer 的，`interactive` 之后就可能开始渲染、把 tex 源文本节点换成 `mjx-container`——fetch 异步词典可能输给它。所以 `conf.py` 把 `math-en.js` 排在 `translate.js` **之前**同步加载，`swapMathText()` 在 DOM 一解析完就**同步**替换 `.math` 里的 tex 源，抢在渲染之前；MathJax 渲染出来的自然就是英文。代码块不受 MathJax 影响，继续走 fetch（`code-en.json`）。
- **校验比代码块强**：骨架校验——取原文的非中文片段序列，要求在译文中按原顺序逐字出现（`re.fullmatch` 拼接 `.*?`），中文位置允许任意英文，外加花括号计数；LaTeX 命令/花括号/空白/换行逐字不动，只翻中文。（注：早期实现「两侧中文段替换成占位符再比对」是错的——译文里是英文，永不可能相等，已修复。）
- **验收**：无论 MathJax 是否渲染成功（本地 CDN 不通时 tex 源原样显示），英文模式下 `.math`/`mjx-container` 内文本节点汉字数应为 0——headless 全站扫即可，不依赖渲染。
- **词典与产物的一致性（两段构建链必须一致）**：词典 key 绑定页面 tex 源的**形态**，所以本地与 RTD 必须用**同一个 pandoc 版本**——`convert.find_pandoc` 优先用 `pypandoc_binary` 自带的 pandoc（requirements 钉死），PATH 只兜底：RTD 构建镜像自带系统 pandoc，若按 PATH 优先，旧版会把顶层 `\begin{align}` 输出成 `\begin{aligned}`，与本地（3.9，保留 `align`）漂移，**公式词典全部错配、英文站公式静默残留中文**（2026-10 线上实际踩过，代码块不受影响）。`convert.verify` 的「英译词典匹配产物」项守着这条：页面中文节点必须命中词典、词典 key 必须存在于页面（双向核对，含部分章节构建时的降级），错配直接让构建红灯。

## 英文模式下的 UI 细节（Google 翻译的副作用）

Google 网站翻译会重写页面里的文本节点：把文本包进 `<font style="vertical-align: inherit">`，并**吞掉首尾空白**。凡是「靠空格文本节点撑出来」的间距，英文模式下都会塌掉：

- **导航按钮的图标与文字间距**：模板（`_templates/breadcrumbs.html` / `footer.html` / `layout.html`）与 `translate.js` 的按钮里，图标前后**不留空格文本节点**，间距由 `custom.css` 的 margin 给（4px ≈ 原空格宽度）。「下一页」的图标在文字后面，用 `fa-tail` 标记类区分方向——不能靠 `:last-child` 判断（文本节点不是元素子节点，图标在文字前后都是「唯一元素子」，`:last-child` 会把所有图标一并命中，首版踩过）。
- **选择器不能依赖直接子关系**：Google 会把整段 inline 内容（**图标 span 也在内**）包进 `<font style="vertical-align: inherit">`——`.btn > .fa` 这类直接子选择器在英文模式下会整体失配、margin 全塌（模拟包裹后 8/8 按钮 margin 归零），一律用后代选择器（`.btn .fa`）。
- **垂直对齐的真正来源是 `vertical-align` 继承**：`<a class="btn">` 是主题默认的 `middle`，图标 span 是 `baseline`；英文模式下 Google 给文字包的 `<font style="vertical-align: inherit">` 让文字继承到 `middle`，于是**文字比图标低约 1px**——用户报的「图标与文字没有对齐（不居中）」的根因（2026-10 线上实测 dy≈1px）。修法：`a` 设 `vertical-align: baseline`（模板/`translate.js` 内联 + `custom.css` 兜底），图标取 `inherit`，两者永远取自同一父级；实测包裹/未包裹三情形 dy 全为 0。
- **间距最终内联在图标上**（兜底）：模板与 `translate.js` 同时给图标写 `style="margin-right:4px"`（`fa-tail` 用 `margin-left:4px`）。内联不依赖选择器命中、不受 CSS 缓存版本影响——线上两轮"CSS 方案被打穿"后的最终手段；`custom.css` 那段规则保留为统一机制。验收含最坏情形：**移除 custom.css 后再整段 `<font>` 包裹，margin 仍 4px**。
- **验收方式**：headless 里 `translate.google.com` 不可达，Google 翻译不会激活；用模拟改写复现与验收，覆盖两种形态——「文本被 `<font>` 包裹 + 吞首尾空格」（首版复现了 gap 4→0）与「`<a>` 全部子节点整体被 `<font>` 包裹」（暴露直接子选择器失配）。改后两种形态 gap 恒 4px，中文模式按钮宽度与改动前一致。

## 英文封面（离线生成，产物入库）

封面不在上面这套流程里——它是书稿的独立文档 `figures/misc/cover-tikz.tex`，编译成 PDF 后由 `convert._cover_image()` 光栅化成 PNG，字同样是矢量轮廓；而且封面用的 Humor Sans / Comic Neue 只装在作者本机、RTD 镜像里没有（中文用的 LXGW WenKai 25 MB 也不适合入库）。

所以英文封面是**离线生成一次、产物入库**：`python -m tools.build_cover_en`（默认 240 dpi）写到 `_static/figures-en/cover.png`，`.gitignore` 只放行这一个文件。换图用的是同一套逻辑（`_images/cover.png` → `_static/figures-en/cover.png`，加载失败回退中文封面），不用改 JS。

**封面文案改了要重跑该工具**：替换按原文片段定位，命中数不为 1、或产物里还残留汉字就直接报错停下，不会出一张半中半英的封面。
