# 转换管线

## 目录结构

```
docs/readthedocs/
├── conf.py                 Sphinx 配置（MyST + MathJax v3 + sphinx-rtd-theme）
├── requirements.txt        站点构建依赖（版本钉死）
├── build.sh                本地一键：转换 → sphinx-build → 内容核对
├── .readthedocs.yaml       RTD 构建：pre_build / build / post_build
├── _static/custom.css      版式（三栏居中）、表格、提示块、图注样式
├── _static/translate.js    中英切换 + 英文换图 + 侧栏按钮归位
├── _templates/page.html         版式：正文右侧插入本页目录（{{ toc }}）
├── _templates/breadcrumbs.html  整文件复制主题模板 + 顶部按钮行的「下载 PDF」（挨着「下一页」）
├── _templates/footer.html       同上，页脚按钮行（两处都要跟主题版本对齐）
├── _templates/layout.html       侧栏兜底行：「下载 PDF」+「English」（仅无按钮行的页面）
├── i18n/figures-en.json    英文图集词典（1793 条，人工工具生成）
├── tools/                  build_cover_en.py / translate_figures.py / visual_check.py
└── convert/                转换包
    ├── __main__.py         编排 + 站点装配 + 报告
    ├── booktree.py         解析 book.tex → 篇/章顺序（导航骨架）
    ├── texutil.py          LaTeX 扫描助手（分组、注释、环境配对、math_spans）
    ├── tokens.py           占位 token 协议（预处理 ↔ 后处理）
    ├── labels.py           全书 label → 类型/编号/标题（重放 LaTeX 计数器）
    ├── macros.py           抽书稿数学宏 → build/mathjax-macros.json
    ├── readme.py           根 README 的 [leetcuda-pdf] 链接（下载按钮唯一事实源）
    ├── figtext.py          图内文字掩码 / 还原 / 词典匹配
    ├── preprocess.py       tex 归一化 + 抽离 tikz/代码 + 镜像图片
    ├── tikz2svg.py         TikZ → SVG（并行、增量缓存）
    ├── postprocess.py      markdown → MyST（锚点、链接、提示块、代码块）
    └── verify.py           内容核对（23 项）
```

## 数据流

主链：`book.tex` → 逐章归一化（展平 `\input`、剥离注释、抽离 TikZ 与代码载荷、把 `\label`/`\ref`/定理环境替换成占位 token）→ pandoc 转 MyST markdown → 还原 token（锚点、站内链接、admonition、代码块、图注）→ Sphinx 构建 → 内容核对。

TikZ 独立成链：每个片段套 standalone 文档编译成 PDF，再让 dvisvgm 读 PDF 出 SVG，按内容哈希增量缓存，多进程并行（227 张约 45 秒 / 8 进程）。

报告落 `build/report.md`（同时有 `report.json`）：TikZ 失败清单、未解析引用、悬空锚点、残留 LaTeX、各阶段耗时、pandoc 警告。

## 命令速查

```bash
conda activate cdit && cd /workspace/dev/vipshop/LeetCUDA/docs/readthedocs

./build.sh                                     # 转换 + 构建 + 核对，打印预览路径
./build.sh --skip-tikz                         # 复用已有 SVG
python -m convert --only ch00-profiling,ch16-fa2-splitq-mma   # 只转几章
python -m convert --jobs 8                     # TikZ 并行度
python -m convert --force-tikz                 # 忽略缓存重编全部 TikZ
python -m convert --tikz-fonts woff2           # 图内嵌字体（更小，见 pitfalls）
python -m convert --png                        # 额外导出 PNG 便于查看（需 cairosvg）
python -m convert --skip-tikz --no-assemble    # 只跑 tex→md，调试用
python -m convert.verify --src build/src --html build/html    # 只跑内容核对
```

外部依赖：`pandoc`、`xelatex`（TeX Live，含 ctex/中文）、`dvisvgm`；可选 `cairosvg`（仅 `--png`）。

## 编号与表格

- **章号是文档级的**：`labels.py` 用一个扫描器把 46 篇 tex 按书序串起来重放 LaTeX 计数器，`\ref{ch:16}` 才会得到 17 而不是恒为 1；每章开头重置编号格式（对应书里 ch19b 的 `\begingroup` 作用域）。
- **表格交给 pandoc 出 HTML**：关掉 `grid_tables/multiline_tables/simple_tables`，pandoc 会为复杂表（`p{}` 列、`\multicolumn`）直接写 `<table>`，MyST 原样透传，比 pipe 表更能保住多行单元格；单元格里的 `$...$` 由 MathJax 在浏览器端渲染。
- **多行表头**：pandoc 的 LaTeX reader 只支持单行表头，源文「两行表头 + `\midrule`」时第二行会落进表体（还曾把 `\cmidrule(lr){2-3}` 的参数渲染成一行 `2-3(lr)4-5`）。现在只删 `\cmidrule` 的参数、保留命令本身（pandoc 靠它识别表头），并在预处理时把表头行数写进 TABLE token，后处理据此把这几行收进 `<thead>` 并把单元格改成 `<th>`。

## 书内数学宏

书稿里定义了近 90 个数学宏（`\v` 是向量、`\Z` 是整数集、`\abs`/`\norm` 带参数…）。不给定义的话，MathJax 会把 `\v{L}` 按内置重音命令渲染成 `Ľ`、`\Z` 直接报错。

转换时从书稿 tex 抽出宏表（`convert/macros.py`）写到 `build/mathjax-macros.json`，`conf.py` 再把它塞进 `mathjax3_config["tex"]["macros"]`。图里用的 TikZ 尺寸/颜色参数会被过滤掉，免得遮蔽数学符号。

## 两个环境陷阱

- **增量构建不重拷 `_static/`**：只改 `_static/` 里的文件（`custom.css` / `translate.js`）时，没有页面过期的话增量构建报 `no targets are out of date`，静态文件**不重新复制**，预览看到的还是旧样式。先 `touch build/src/index.md`（或删 `build/html/_static`）再跑一次即可。RTD 每次全新构建，不受影响。
- **gitignore 会静默吞文件**：仓库根与 `docs/` 两级 `.gitignore` 都有 `*.txt`、`*.tex`、`build*` 规则，会吞掉本目录的 `requirements.txt` 与 `build.sh`；`docs/readthedocs/.gitignore` 里用 `!requirements.txt`、`!build.sh` 逐条解禁。改动这些文件后建议 `git check-ignore -v` 复核。
