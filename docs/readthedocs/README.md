# LeetCUDA 书的 Read the Docs 站点

把 `kernels/interview/book/` 下的书稿 tex 逐章转换成网页：公式走 MathJax 实时渲染，
TikZ 图编译成 SVG，代码清单按 `linerange` 从冻结源码抽出来贴进页面。

站点：<https://leetcuda.readthedocs.io>

原书 tex、源码、图片一律不改：转换读原文件，产物只写进 `build/`（已 gitignore）。

## 本地预览

```bash
conda activate cdit
cd docs/readthedocs
pip install -r requirements.txt        # sphinx / myst-parser / sphinx-rtd-theme
conda install -c conda-forge pandoc    # 需要真正的 pandoc 可执行文件
./build.sh                             # 转换 + 构建，打印入口路径
python -m http.server -d build/html 8000   # 或直接双击 build/html/index.html
```

外部依赖：`pandoc`、`xelatex`（TeX Live，含 ctex/中文）、`dvisvgm`。可选
`pip install cairosvg`，仅用于 `--png` 导出。

常用参数：

```bash
python -m convert --only ch00-profiling,ch16-fa2-splitq-mma   # 只转几章
python -m convert --jobs 8                                    # TikZ 并行度
python -m convert --force-tikz                                # 忽略缓存重编全部 TikZ
python -m convert --tikz-fonts woff2                          # 图内嵌字体（更小，见下）
python -m convert --png                                       # 额外导出 PNG 便于查看
python -m convert --skip-tikz --no-assemble                   # 只跑 tex→md，调试用
```

报告落在 `build/report.md`（同时有 `report.json`）：TikZ 失败清单、未解析引用、
悬空锚点、残留 LaTeX、各阶段耗时、pandoc 警告。

`build.sh` 在 Sphinx 之后还会跑一次内容核对，也可单独跑：

```bash
python -m convert.verify --src build/src --html build/html
```

核对清单（任一项不过就以非零码退出，RTD 的 `post_build` 同样会跑）：

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
| HTML 表格内无 Markdown 残留 | raw HTML 表格里不残留 `[](…)` / `**…**` / `![…](…)` |
| 逐章表格数不缺 | 按章比对原书表格环境数与页面 `<table>` 数（总数阈值会漏单张表） |
| 页面图片均存在 | 页面引用的 `_images/…` 文件都在（raw HTML 图片需补隐藏引用） |
| TikZ 图保留文字 | 有文字的片段必须真的输出字形（防陈旧/残缺 SVG 留在页面上） |
| 书目录编号核对 | 本地有 `book.toc` 时，逐章比对书目录真值（章号 + 标题） |

以上是文件层面的检查，不依赖浏览器。要确认真实渲染效果（MathJax 是否报错、页面上
有没有漏出的字面文本、图片有没有 404），再跑一次浏览器验收：

```bash
pip install playwright && python -m playwright install chromium   # 一次性
python -m tools.visual_check --html build/html --out build/shots/chapters \
    --proxy http://127.0.0.1:7890          # 无代理时省略 --proxy
```

它逐页打开、等 MathJax 渲染完，然后统计：公式报错（`mjx-merror` 的 TeX 报错原文）、
可见的字面残迹（排除代码块与公式容器）、控制台报错、404 资源，并整页截图留档。
`--sheet build/shots/sheet.png` 可把截图拼成接触表，便于肉眼过一遍排版。

## 目录

```
docs/readthedocs/
├── conf.py              Sphinx 配置（MyST + MathJax v3 + sphinx-rtd-theme）
├── requirements.txt     站点构建依赖（版本钉死）
├── build.sh             本地一键：转换 → sphinx-build
├── _static/custom.css   提示块与图注样式
└── convert/             转换包
    ├── booktree.py      解析 book.tex → 篇/章顺序（导航骨架）
    ├── texutil.py       LaTeX 扫描助手（分组、注释、环境配对）
    ├── tokens.py        占位 token 协议（预处理 ↔ 后处理）
    ├── labels.py        全书 label → 类型/编号/标题（重放 LaTeX 计数器）
    ├── preprocess.py    tex 归一化 + 抽离 tikz/代码 + 镜像图片
    ├── tikz2svg.py      TikZ → SVG（并行、增量缓存）
    ├── postprocess.py   markdown → MyST（锚点、链接、提示块、代码块）
    ├── verify.py        内容核对（章号、表格、属性泄漏、锚点）
    └── __main__.py      编排 + 站点装配 + 报告
```

## 管线

`book.tex` → 逐章归一化（展平 `\input`、剥离注释、抽离 TikZ 与代码载荷、把
`\label`/`\ref`/定理环境替换成占位 token）→ pandoc 转 MyST markdown → 还原 token
（锚点、站内链接、admonition、代码块、图注）→ Sphinx 构建 → 内容核对。

TikZ 独立成链：每个片段套 standalone 文档编译成 PDF，再让 dvisvgm 读 PDF 出 SVG，
按内容哈希增量缓存，多进程并行（227 张约 45 秒 / 8 进程）。

### 编号与表格

- **章号是文档级的**：`labels.py` 用一个扫描器把 46 篇 tex 按书序串起来重放
  LaTeX 计数器，`\ref{ch:16}` 才会得到 17 而不是恒为 1；每章开头重置编号格式
  （对应书里 ch19b 的 `\begingroup` 作用域）。
- **表格交给 pandoc 出 HTML**：关掉 `grid_tables/multiline_tables/simple_tables`，
  pandoc 会为复杂表（`p{}` 列、`\multicolumn`）直接写 `<table>`，MyST 原样透传，
  比 pipe 表更能保住多行单元格；单元格里的 `$...$` 由 MathJax 在浏览器端渲染。
- **多行表头**：pandoc 的 LaTeX reader 只支持单行表头，源文「两行表头 + `\midrule`」
  时第二行会落进表体（还曾把 `\cmidrule(lr){2-3}` 的参数渲染成一行 `2-3(lr)4-5`）。
  现在只删 `\cmidrule` 的参数、保留命令本身（pandoc 靠它识别表头），并在预处理时把
  表头行数写进 TABLE token，后处理据此把这几行收进 `<thead>` 并把单元格改成 `<th>`。

### 绕开的三个坑（都实测过，别再走一遍）

1. **不能让 dvisvgm 直接读 xelatex 的 XDV**：CJK 字形推进量解析不可靠，图中文字
   会互相重叠。同一份内容先转 PDF 再读就完全正确，所以现在一律 `dvisvgm --pdf`。
2. **不能用多页文档批量编译再按页导出**：PDF 输入的 `--bbox` 选项无效，每张 SVG
   都会是整页大小，图形缩在左上角。standalone 文档天然紧致裁剪。
3. **`--tikz-fonts` 的两种模式**：

   | 模式 | 227 图合计 | 浏览器 | VS Code 预览 | 图中文字可选中 |
   | --- | --- | --- | --- | --- |
   | `paths`（默认） | 21 MB | 正常 | 正常 | 否 |
   | `woff2` | 约 5 MB | 正常 | 乱码 | 是 |

   两者都不丢视觉信息，`paths` 只是把字形转成矢量轮廓。缓存键含模式，切换会自动重编。

### gitignore 陷阱

仓库根与 `docs/` 两级 `.gitignore` 都有 `*.txt`、`*.tex`、`build*` 规则，会静默吞掉
本目录的 `requirements.txt` 与 `build.sh`。`docs/readthedocs/.gitignore` 里用
`!requirements.txt`、`!build.sh` 逐条解禁，改动后建议 `git check-ignore -v` 复核。

## 书内数学宏

书稿里定义了近 90 个数学宏（`\v` 是向量、`\Z` 是整数集、`\abs`/`\norm` 带参数…）。
不给定义的话，MathJax 会把 `\v{L}` 按内置重音命令渲染成 `Ľ`、`\Z` 直接报错。

转换时从书稿 tex 抽出宏表（`convert/macros.py`）写到 `build/mathjax-macros.json`，
`conf.py` 再把它塞进 `mathjax3_config["tex"]["macros"]`。图里用的 TikZ 尺寸/颜色参数
会被过滤掉，免得遮蔽数学符号。

## 已知边界

- **仓库外的代码清单**：`ch27` 有两处 `\lstinputlisting` 指向同级仓库 `ffpa-attn/`，
  本地能读到、RTD 上只检出本仓库。RTD 构建时这两处显示成提示块并给出原路径。
- **书稿自身的悬空引用**：`ch19b`（CuTe 白皮书导读）有 12 条 `\ref` 指向未收录的
  公式/章节/表格 label，书稿本身也解析不到，页面降级成 code 形式并记入报告。
- **图形内部文字**：`paths` 模式下是矢量轮廓，不能选中；正文文字不受影响。
- **表格样式**：`<table>` 是 pandoc 直出的 HTML，样式较素（无斑马纹/边框定制），
  单元格内的 Markdown 语法不生效（HTML 原文透传）。
- 转换产物不入库，RTD 每次构建现转（apt 装 texlive + 227 张 TikZ 编译 ≈ 3–5 分钟）。

## 在 Read the Docs 上新建项目

1. 把本仓库推送到 GitHub（含根目录 `.readthedocs.yaml`）。
2. 登录 <https://readthedocs.org/>，**Add project → Import a project**，选
   `xlite-dev/LeetCUDA`；仓库列表里没有就用 **Import manually** 填仓库 URL。
3. 项目名与 slug 都填 `leetcuda`（slug 全局唯一，被占用时只能换名，站点地址随之为
   `https://<slug>.readthedocs.io`）。
4. **Admin → Advanced settings → Default branch** 选实际分支（如 `main` 或 `dev`）；
   RTD 导入时会自动在 GitHub 上装 webhook，此后 push 即触发构建。
5. 触发首次构建（Build version）。构建日志里能看到转换报告的输出；失败先看
   `pre_build` 阶段——多半是 apt 包或 pandoc 版本问题。
6. 可选：**Admin → Advanced settings → Language** 选 `Chinese (Simplified)`；
   开启 PR 预览构建便于审阅改稿效果。

> 不开 RTD 的 PDF 格式：那条路走 Sphinx LaTeX + latexmk，本书的 CJK 与自定义宏撑不过去，
> 整本 PDF 仍由 `book/build.sh` 产出。
