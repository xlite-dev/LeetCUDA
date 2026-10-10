# 验收与预览

## 三层验收，缺一层就会漏

| 层 | 手段 | 能发现 |
| --- | --- | --- |
| ① 文件级 | `python -m convert.verify --src build/src --html build/html` | 章号、表格数、字面残迹、悬空锚点、未解析属性（23 项） |
| ② 浏览器级 | `python -m tools.visual_check --html build/html --out build/shots/chapters` | MathJax 报错、页面上可见的字面残迹、控制台报错、404 资源 |
| ③ 版式几何 | 已并进 `visual_check` | 图片/表格是否居中、有没有元素（过宽公式、宽表）撑出正文栏 |

ch33/ch35/ch37 的表格、ch01 的丢图、ch00 的图内文字丢失都只有 ②③ 层能发现；只看 `sphinx-build` 的告警数会漏掉大量内容错误（**0 警告 ≠ 页面正确**）。

**总数阈值会掩盖单点丢失**：表格数 126 → 125 时「≥90%」照样通过，ch37 的量化开销表就是这么漏掉的；改成逐章比对后立刻暴露。

## 核对清单（23 项，任一不过即以非零码退出）

`build.sh` 在 Sphinx 之后跑一次，RTD 的 `post_build` 同样会跑：

| 检查项 | 说明 |
| --- | --- |
| 源树无残留 token | 转换 token 全部还原 |
| 标题无 pandoc 属性泄漏 | 标题里不残留 `{#id .unnumbered}` |
| H1 章号与注册表一致 | `第 N 章：` / `附录 X：` 与计数器一致 |
| 章引用编号未坍缩 | 防「所有章引用都渲染成同一个数字」 |
| 表格已渲染 | 页面 `<table>` 数 ≥ 原书表格环境的 90% |
| 站内锚点无悬空 | 全部 `page.html#anchor` 目标存在 |
| 无 pandoc 原始 HTML 残迹 | 页面上不出现 `{=html}` / `<!-- -->` 字面文本 |
| 数学区内无站内链接/锚点 | 公式里塞 Markdown 链接会让 MathJax 崩 |
| 显示公式定界符配对 | 每个行首 `$$` 都是公式区间的端点 |
| 显示公式未被拆成字面 `$` | 页面里不出现「孤立 `$` + 行内公式」 |
| 表格单元格无 LaTeX 残渣 | 格子里不漏 `\cmidrule(lr){2-3}` 这类命令 |
| 公式内无嵌套 `$` | `\text{…$…$…}` 会让 MyST 把公式切成两段 |
| 强调标记成对（flanking） | 中英混排下 `**…。**见` 配不上对，字面标记会漏到页面上 |
| HTML 表格内无 Markdown 残留 | raw HTML 表格里不残留 `[](…)` / `**…**` / `![…](…)` |
| 逐章表格数不缺 | 按章比对原书表格环境数与页面 `<table>` 数（总数阈值会漏单张表） |
| 多行表头已收进 thead | 源文两行表头的表，页面 `<thead>` 也要有两行 |
| 页面图片均存在 | 页面引用的 `_images/…` 文件都在（raw HTML 图片需补隐藏引用） |
| MyST 属性均已解析 | 页面上不出现字面 `{width=…}` / `{.class}` 这类未解析属性 |
| 无字面 LaTeX 换行反斜杠 | 正文里不出现漏出来的 `\ `（`\\` 硬换行没转成 `<br>`） |
| 下载 PDF 按钮已渲染 | 每页侧栏都有该按钮，且链接与根 README 的 `[leetcuda-pdf]` 一致 |
| TikZ 图保留文字 | 有文字的片段必须真的输出字形（防陈旧/残缺 SVG 留在页面上） |
| 英文图集完整 | 词典覆盖、英文 SVG 数量与字形（`_static/figures-en/`） |
| 书目录编号核对 | 本地有 `book.toc` 时，逐章比对书目录真值（章号 + 标题） |

## 浏览器验收（visual_check）

```bash
pip install playwright && python -m playwright install chromium   # 一次性
python -m tools.visual_check --html build/html --out build/shots/chapters \
    --proxy http://localhost:7890          # 无代理时省略 --proxy
python -m tools.visual_check --only index,ch00-profiling --out build/shots   # 只验几页
    --sheet build/shots/sheet.png          # 把截图拼成接触表，便于肉眼过一遍排版
```

流程：逐页打开、等 MathJax 渲染完，然后统计公式报错（`mjx-merror` 的 TeX 报错原文）、可见的字面残迹（排除代码块与公式容器）、控制台报错、404 资源，并量一遍版式几何，最后整页截图留档。公式与字体走 CDN，验收机需要能出网（否则用 `--proxy`）。

## 本地预览

```bash
conda activate cdit && cd /workspace/dev/vipshop/LeetCUDA/docs/readthedocs
./build.sh
python -m http.server -d build/html 8000   # 或直接双击 build/html/index.html
```

## 远程 SSH 开发的预览（VS Code 内置浏览器）

站点产物是静态 HTML，远程开发时本机浏览器打不开服务器上的 `build/html`。

```bash
python -m http.server -d build/html 8000    # 在远端跑着；改完重跑 build.sh 刷新页面即可
```

`Ctrl/Cmd+Shift+P` → **`Simple Browser: Show`**（中文「简单浏览器：显示」）→ URL 填 `http://localhost:8000`。

**必须先转发端口**：内置浏览器和普通浏览器一样在**你本机**渲染，`localhost` 指的是你的电脑、不是远程服务器；没转发就是 `ERR_CONNECTION_REFUSED`。两种转发方式：

- **端口面板**：`Ctrl/Cmd+Shift+P` → `Ports: Focus on Ports View`（或底部面板「端口 / PORTS」标签页）→「转发端口 / Forward a Port」→ 填 `8000`，然后按面板里 **本地地址 / Local Address** 那一列打开——本机 8000 被占用时 VS Code 会改用 8001 之类，别照抄 8000。
- **在集成终端里起服务**：把 `http.server` 放到 VS Code 集成终端跑，端口检测会自动把它加入端口面板并转发。**由脚本在后台起的服务检测不到，必须手动转发**（这正是打开 `localhost:8000` 报 `ERR_CONNECTION_REFUSED` 的原因）。

直接访问服务器网卡地址（如 `http://10.99.177.11:8000`）通常走不通（办公网、安全组拦截）：现象是页面先空白、随后 `ERR_CONNECTION_TIMED_OUT`。**判断依据在服务端**——`http.server` 的访问日志里压根没有这次请求，说明流量没到站点、不是页面问题；这种情况只能走端口转发。

另一条零网络兜底（连不到 8000 时）：仓库里装了 `Preview`（`searking.preview-vscode`，支持 HTML），在编辑器里打开 `build/html/index.html` 后 `Ctrl/Cmd+Shift+P` → **`Open Preview`**（中文「打开预览」），由本机 VS Code 渲染，不经网络。注意它加载 `_static/` 相对资源的行为视实现而定，样式可能不全——要完整样式还是走端口转发。

两点限制：内置浏览器只认 http(s)，`file://` 打不开（`translate.js` 也会按 `file:` 判定「站点不可被 Google 抓取」而禁用中英切换）；公式与字体走 CDN（`cdn.jsdelivr.net`），内网机器上页面只显示 TeX 源码、图注字体退化，这是环境限制、不是转换出错。

## 改核对项

新增检查写进 `convert/verify.py`：实现 `check_xxx(...) -> tuple[bool, str]`，在 `verify()` 里 `report.add("中文检查项名", ok, detail)`，必要时在 `main()` 里接参数。`build.sh` 与 RTD 的 `post_build` 都直接调 `python -m convert.verify`，不需要改别的地方。
