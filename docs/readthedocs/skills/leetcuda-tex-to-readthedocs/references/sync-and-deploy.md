# 内容同步、部署与已知边界

## 内容更新与自动同步（改书稿前必读）

书稿 tex 是唯一事实源，站点产物不入库、每次构建现转，但**不是所有内容都随 tex 自动更新**——英文图集与封面是例外：

| 内容 | 机制 | tex 修改后 |
| --- | --- | --- |
| 中文正文 / 公式 / 表格 / 数学宏 | 构建期从 tex 现转 | 自动更新 |
| 中文 TikZ 图 | 构建期现编译，缓存键 = 片段内容哈希 | 自动更新（改图即重编） |
| 英文正文 | 运行时 Google 翻译抓最新页面 | 自动更新（机器翻译质量） |
| 「下载 PDF」按钮 | 构建期读仓库根 `README.md` 的 `[leetcuda-pdf]` | 自动更新（改的是 README，不是 tex） |
| 英文图集（图内文字） | 构建期按静态词典 `i18n/figures-en.json` 重编译 | **不自动**：改图内中文文字、新增含中文的图 → 词典未覆盖 → 构建红灯 |
| 封面（中/英） | 读入库的 `cover.pdf` / 入库的 `cover.png` | **不自动**：须重编译/重跑工具并 commit |

### 改书稿后的检查清单

1. 只改正文 / 公式 / 表格：push 即可，中英文自动同步。
2. 改 TikZ 图内中文文字、新增含中文的图：先跑 `python -m tools.translate_figures --proxy …` 补词条并 commit `i18n/figures-en.json`，再 push。跳过这步，`post_build` 的「英文图集完整」核对（`convert/verify.py` 的 `check_english_figures`，查词典覆盖率）不过，RTD 构建必红。只重排、节点中文文字未动的图不受影响——词典匹配的是掩码后的节点文字单元，不是整段源码。
3. 改封面文案：站点不重编封面 tex，只把入库的 `figures/misc/cover.pdf` 光栅化（`convert._cover_image()`），须重新编译 cover.pdf 并 commit；英文封面另跑 `python -m tools.build_cover_en`，commit `_static/figures-en/cover.png`。
4. push 后看 RTD 构建结果：内容核对任一项不过构建即失败，这是质量护栏，**修 tex，不要绕过核对**。

## 在 Read the Docs 上新建项目

1. 把本仓库推送到 GitHub（含根目录 `.readthedocs.yaml`）。
2. 登录 <https://readthedocs.org/>，**Add project → Import a project**，选 `xlite-dev/LeetCUDA`；仓库列表里没有就用 **Import manually** 填仓库 URL。
3. 项目名与 slug 都填 `leetcuda`（slug 全局唯一，被占用时只能换名，站点地址随之为 `https://<slug>.readthedocs.io`）。
4. **Admin → Advanced settings → Default branch** 选实际分支（如 `main` 或 `dev`）；RTD 导入时会自动在 GitHub 上装 webhook，此后 push 即触发构建。
5. 触发首次构建（Build version）。构建日志里能看到转换报告的输出；失败先看 `pre_build` 阶段——多半是 apt 包或 pandoc 版本问题。
6. 可选：**Admin → Advanced settings → Language** 选 `Chinese (Simplified)`；开启 PR 预览构建便于审阅改稿效果。

**不要开 RTD 的 PDF 格式**：那条路走 Sphinx LaTeX + latexmk，本书的 CJK 与自定义宏撑不过去；整本 PDF 仍由 `book/build.sh` 产出。

### RTD 构建的三个阶段（`.readthedocs.yaml`）

| 阶段 | 做什么 | 失败时先怀疑 |
| --- | --- | --- |
| `pre_build` | `cd docs/readthedocs && python -m convert --jobs 4` | apt 包（texlive/dvisvgm/mupdf-tools/poppler-utils）、pandoc 版本 |
| `build` | `sphinx-build -T -b html -d …/build/doctrees -c docs/readthedocs …/build/src $READTHEDOCS_OUTPUT/html` | 模板、`conf.py`（含下载按钮链接解析失败） |
| `post_build` | `cd docs/readthedocs && python -m convert.verify --src build/src --html $READTHEDOCS_OUTPUT/html` | 逐项打印的核对项名称 |

两个别碰的地方：**不要给 RTD 装 apt 的 pandoc**（Ubuntu 24.04 仓库是 3.1.3，会覆盖 `find_pandoc` 命中的 pip `pypandoc_binary` 自带 pandoc 3.9，输出漂移 → `post_build` 核对失败）；`requirements.txt` 的版本是钉死的。

## 已知边界

- **仓库外的代码清单**：`ch27` 有两处 `\lstinputlisting` 指向同级仓库 `ffpa-attn/`，本地能读到、RTD 上只检出本仓库。RTD 构建时这两处显示成提示块并给出原路径。
- **书稿自身的悬空引用**：`ch19b`（CuTe 白皮书导读）有 12 条 `\ref` 指向未收录的公式/章节/表格 label，书稿本身也解析不到，页面降级成 code 形式并记入报告。
- **图形内部文字**：`paths` 模式下是矢量轮廓，不能选中；正文文字不受影响。
- **表格样式**：`<table>` 是 pandoc 直出的 HTML，样式由 `_static/custom.css` 补齐（边框、表头底色、斑马纹、窄表居中、宽表在表内横向滚动）；单元格内的 Markdown 语法不生效（HTML 原文透传），转换器会把表格里的链接/图片/强调转成 HTML。
- **转换产物不入库**，RTD 每次构建现转（apt 装 texlive + 227 张 TikZ 编译 ≈ 3–5 分钟）。
