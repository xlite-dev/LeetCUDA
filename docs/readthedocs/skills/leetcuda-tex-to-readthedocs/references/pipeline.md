# 转换管线

## 目录结构

```
docs/readthedocs/
├── conf.py                 Sphinx 配置（MyST + MathJax v3 + sphinx-rtd-theme）
├── requirements.txt        站点构建依赖（版本钉死）
├── build.sh                本地一键：转换 → sphinx-build → 内容核对
├── .readthedocs.yaml       RTD 构建：pre_build / build / post_build
├── _static/custom.css      版式（三栏居中）、表格、提示块、图注样式
├── _static/translate.js    中英切换 + 英文换图 + 代码块注释词典替换 + 侧栏按钮归位
├── _static/code-en.json    代码块注释英译词典（人工产线生成，见 references/english.md）
├── _static/math-en.js      公式内文字英译词典（--what math 产线生成，conf.py 同步加载）
├── _templates/page.html         版式：正文右侧插入本页目录（{{ toc }}）
├── _templates/breadcrumbs.html  整文件复制主题模板 + 顶部按钮行的「下载 PDF」（挨着「下一页」）
├── _templates/footer.html       同上，页脚按钮行（两处都要跟主题版本对齐）
├── _templates/layout.html       侧栏兜底行：「下载 PDF」+「English」（仅无按钮行的页面）
├── i18n/figures-en.json    英文图集词典（1793 条，人工工具生成）
├── tools/                  build_cover_en.py / translate_figures.py / visual_check.py / code_dict.py
└── convert/                转换包
    ├── __main__.py         编排 + 「主书 + 子集合」多书转换 + 站点装配 + 分书报告
    ├── collections.py      子集合注册表（slug / title / 根 tex / 首页说明），__main__ 与 verify 共用
    ├── booktree.py         解析书根 tex → 篇/章顺序（导航骨架）
    ├── texutil.py          LaTeX 扫描助手（分组、注释、环境配对、math_spans）
    ├── tokens.py           占位 token 协议（预处理 ↔ 后处理）
    ├── labels.py           全书 label → 类型/编号/标题（重放 LaTeX 计数器）
    ├── macros.py           抽书稿数学宏 → build/mathjax-macros.json（只从主书抽，勿对集合调用）
    ├── readme.py           根 README 的 [leetcuda-pdf] 链接（下载按钮唯一事实源）
    ├── figtext.py          图内文字掩码 / 还原 / 词典匹配
    ├── preprocess.py       tex 归一化 + 抽离 tikz/代码 + 镜像图片
    ├── tikz2svg.py         TikZ → SVG（并行、增量缓存）
    ├── postprocess.py      markdown → MyST（锚点、链接、提示块、代码块）
    └── verify.py           内容核对（通用项递归全站 + 每本书专属项 + 集合装配）
```

## 数据流

主链：书根 tex → 逐章归一化（展平 `\input`、展开 `\urldef`、剥离注释、抽离 TikZ 与代码载荷、把 `\label`/`\ref`/定理环境替换成占位 token）→ pandoc 转 MyST markdown → 还原 token（锚点、站内链接、admonition、代码块、图注）→ Sphinx 构建 → 内容核对。

## 主书 + 子集合（多书装配）

站点 = **一本主书 + 若干子集合**，一条 `python -m convert` 全转（RTD 的 `pre_build` 不变，不分叉）：

| 角色 | 注册处 | 站点目录 | 页面 | 报告 |
| --- | --- | --- | --- | --- |
| 主书 | `book.tex`（`--book-dir`） | `build/src/`（平铺） | `index.md` + `part-N.md` + 每章 | `build/report.md` / `.json` |
| 子集合 | `convert/collections.py` 的 `COLLECTIONS` | `build/src/<slug>/` | 同上（`index.md` 是集合首页） | `build/report-<slug>.md` / `.json` |

装配规则：

- **入口位置**：主书 `index.md` 的 toctree 在全部 `part-N` 之后追加 `<slug>/index`，即侧栏「附录」之后的最后一项。集合是三层结构（集合首页 → 4 个 `part` 页 → 32 章），所以 `conf.py` 用 `navigation_depth: 3` + `titles_only: True`（否则主书每章的 `\section` 也会进左栏）。展开/收起用主题自带机制：系列前有「+」，点开才列出章节，当前系列随页面自动展开，与主书「篇 → 章」完全一致——**不要再加 CSS 强制展开**。细节见 `layout-frontend.md`。
- **集合首页**：H1 = `Collection.title`，正文 = `Collection.intro` 段落 + raw HTML `<a href="../index.html">返回…主书</a>`（用 Markdown 链接会被 MyST 当文档引用解析，退化成 `href="#…"` 死链）+ `:maxdepth: 2` 的 toctree（4 个系列 + 32 篇标题直接列出）。
- **章号**：集合用自己那份 `labels` 注册表重放计数器，输出与 PDF 一致的「第 N 章：」前缀（cute-zhihu 是 1..32 连续编号，跨 `\part` 不重置）。
- **宏表只写一次**：`macros.extract` 只认书根的 `preamble.tex` / `book.tex`，对子集合调用会得到空表并覆盖 `build/mathjax-macros.json`，主书公式静默变形。`_convert_book` 只在 `macro_target` 非空（主书）时写。
- **路径**：图片按「相对书根」镜像进 `<slug>/figures/…`，页面内的 `figures/…` 相对链接天然成立；raw HTML 表格里的 `<img>` 不走 Sphinx 改写，`RenderContext.image_prefix` 按页面层数补 `../_images/`。
- **源码溯源**：集合页脚链接是 `…/kernels/interview/book/<slug>/chapters/x.tex`（`BookSpec.relative_dir` 拼在 `source_url` 后）。
- **装配清单**：转换期写 `build/collections.json`（slug / 页面 / 章节），`verify` 据此核对页面齐全与入口位置；`--no-collections`、`--only`、`--max-chapters` 的局部构建写空清单，核对自动跳过。
- **英文边界**：子集合不参与 `code-en.json` / `math-en.js` 词典（`check_translation_dicts` 只取站点根页面）；`translate.js` 的换图只对 `_images/*.svg`（TikZ 产物）生效，集合的 jpg 不受影响、也不会打出 404。

TikZ 独立成链：每个片段套 standalone 文档编译成 PDF，再让 dvisvgm 读 PDF 出 SVG，按内容哈希增量缓存，多进程并行（227 张约 45 秒 / 8 进程）。

报告按书分文件：主书 `build/report.md` + `report.json`，集合 `build/report-<slug>.md` + `.json`（TikZ 失败清单、未解析引用、悬空锚点、残留 LaTeX、各阶段耗时、pandoc 警告）。

## 命令速查

```bash
conda activate cdit && cd /workspace/dev/vipshop/LeetCUDA/docs/readthedocs

./build.sh                                     # 转换（主书 + 集合）+ 构建 + 核对，打印预览路径
./build.sh --skip-tikz                         # 复用已有 SVG
python -m convert --only ch00-profiling,ch16-fa2-splitq-mma   # 只转主书几章（此时跳过集合）
python -m convert --no-collections             # 只转主书，跳过子集合（调试用）
python -m convert --collection cute-zhihu      # 只转指定集合（逗号分隔可多选）
python -m convert --collection cute-zhihu --only-collection reed-08-hopper-mbarrier   # 集合单章打样
python -m convert --jobs 8                     # TikZ 并行度
python -m convert --force-tikz                 # 忽略缓存重编全部 TikZ
python -m convert --tikz-fonts woff2           # 图内嵌字体（更小，见 pitfalls）
python -m convert --png                        # 额外导出 PNG 便于查看（需 cairosvg）
python -m convert --skip-tikz --no-assemble    # 只跑 tex→md，调试用
python -m convert.verify --src build/src --html build/html    # 只跑内容核对
```

外部依赖：`pandoc`、`xelatex`（TeX Live，含 ctex/中文）、`dvisvgm`；可选 `cairosvg`（仅 `--png`）。

## 编号与表格

- **章号是文档级的**：`labels.py` 用一个扫描器把一本书的全部 tex 按书序串起来重放 LaTeX 计数器，`\ref{ch:16}` 才会得到 17 而不是恒为 1；每章开头重置编号格式（对应书里 ch19b 的 `\begingroup` 作用域）。**每本书各自一份注册表**：主书 46 篇、cute-zhihu 32 篇互不影响，集合的「第 N 章」与它自己的 `.toc` 对齐。
- **表格交给 pandoc 出 HTML**：关掉 `grid_tables/multiline_tables/simple_tables`，pandoc 会为复杂表（`p{}` 列、`\multicolumn`）直接写 `<table>`，MyST 原样透传，比 pipe 表更能保住多行单元格；单元格里的 `$...$` 由 MathJax 在浏览器端渲染。
- **多行表头**：pandoc 的 LaTeX reader 只支持单行表头，源文「两行表头 + `\midrule`」时第二行会落进表体（还曾把 `\cmidrule(lr){2-3}` 的参数渲染成一行 `2-3(lr)4-5`）。现在只删 `\cmidrule` 的参数、保留命令本身（pandoc 靠它识别表头），并在预处理时把表头行数写进 TABLE token，后处理据此把这几行收进 `<thead>` 并把单元格改成 `<th>`。

## 书内数学宏

书稿里定义了近 90 个数学宏（`\v` 是向量、`\Z` 是整数集、`\abs`/`\norm` 带参数…）。不给定义的话，MathJax 会把 `\v{L}` 按内置重音命令渲染成 `Ľ`、`\Z` 直接报错。

转换时从书稿 tex 抽出宏表（`convert/macros.py`）写到 `build/mathjax-macros.json`，`conf.py` 再把它塞进 `mathjax3_config["tex"]["macros"]`。图里用的 TikZ 尺寸/颜色参数会被过滤掉，免得遮蔽数学符号。

## 两个环境陷阱

- **增量构建不重拷 `_static/`**：只改 `_static/` 里的文件（`custom.css` / `translate.js`）时，没有页面过期的话增量构建报 `no targets are out of date`，静态文件**不重新复制**，预览看到的还是旧样式。先 `touch build/src/index.md`（或删 `build/html/_static`）再跑一次即可。RTD 每次全新构建，不受影响。
- **gitignore 会静默吞文件**：仓库根与 `docs/` 两级 `.gitignore` 都有 `*.txt`、`*.tex`、`build*` 规则，会吞掉本目录的 `requirements.txt` 与 `build.sh`；`docs/readthedocs/.gitignore` 里用 `!requirements.txt`、`!build.sh` 逐条解禁。改动这些文件后建议 `git check-ignore -v` 复核。
