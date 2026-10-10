---
name: leetcuda-tex-to-readthedocs
description: "LeetCUDA 书稿 LaTeX 到 Read the Docs 站点的转换管线维护 skill（站点目录 LeetCUDA/docs/readthedocs/）。当任务涉及：改完书稿后让站点同步、改转换器 convert/、本地构建与预览 build.sh、内容核对 convert.verify、浏览器验收 tools.visual_check、版式与模板（_templates、_static/custom.css）、中英切换与英文图集（translate.js、i18n/figures-en.json、build_cover_en）、「下载 PDF」按钮、RTD 构建失败排查、.readthedocs.yaml 或 requirements.txt 时，使用本 skill。给出改动落点、命令、三层验收与已证伪路线。"
user-invocable: true
argument-hint: '说明要改什么（书稿内容 / 转换器 / 版式 / 中英切换 / 封面 / 核对项），或直接给出报错的核对项名称与日志。'
---

# LeetCUDA 书稿 → Read the Docs 站点

把 `kernels/interview/book/` 下的书稿 tex 逐章转换成网页：公式走 MathJax 实时渲染，TikZ 图编译成 SVG，代码清单按 `linerange` 从冻结源码抽出来贴进页面。站点 <https://leetcuda.readthedocs.io>。

**原书 tex、源码、图片一律不改**：转换只读原文件，产物只写进 `docs/readthedocs/build/`（已 gitignore）。

## 开始前

1. `conda activate cdit` —— 站点构建统一用这个环境（Sphinx 7.4.7 + myst-parser 4.0.0 + sphinx-rtd-theme 3.0.2，版本见 `requirements.txt`）。
2. 站点根 = `/workspace/dev/vipshop/LeetCUDA/docs/readthedocs/`，下文命令都在该目录下执行。
3. 先按「改动落点表」定位落点，再按需查 `references/`；不要凭直觉改转换器。

## 改动落点表

| 改了什么 | 要做的事 | 细节 |
| --- | --- | --- |
| 中文正文 / 公式 / 表格 / 数学宏 | push 即可：构建期从 tex 现转，中英文自动同步 | — |
| TikZ 图内中文文字、新增含中文的图 | 先补词典再 push，否则 RTD 的 `post_build`「英文图集完整」必红 | `references/english.md` |
| 封面文案 | 重编译入库的 `figures/misc/cover.pdf` 并 commit；英文封面另跑 `tools.build_cover_en` 并 commit 产物 | `references/english.md` |
| 转换器 `convert/*.py` | 改完必须跑全量 `./build.sh`（含 23 项核对）+ 浏览器验收 | `references/pipeline.md`、`references/acceptance.md` |
| 版式 / 模板 / CSS / JS | 先 `touch build/src/index.md` 再 `./build.sh`，否则静态文件不重拷、预览还是旧样式 | `references/layout-frontend.md` |
| 「下载 PDF」按钮（落位/样式）或其链接 | 按钮默认在「下一页」右边（顶部 + 页脚两行）；链接只改仓库根 `README.md` 的 `[leetcuda-pdf]` 一行，站点与核对都跟着变 | `references/layout-frontend.md` |
| RTD 构建失败 | 先看 `pre_build`（apt 包 / pandoc 版本），再看 `post_build` 打印的核对项 | `references/sync-and-deploy.md` |
| 核对项本身 | 新增/调整 `convert/verify.py` 的 `check_*`，接入 `verify()` 与 `main()` | `references/acceptance.md` |

## 本地构建与预览

```bash
conda activate cdit
cd /workspace/dev/vipshop/LeetCUDA/docs/readthedocs
pip install -r requirements.txt          # 首次或依赖变更后
./build.sh                               # 转换 → sphinx-build → 内容核对
python -m http.server -d build/html 8000 # 本机浏览器看 build/html/index.html
```

TikZ 全量编译约 3–5 分钟，调试时务必缩短：

```bash
./build.sh --skip-tikz                          # 复用已有 SVG，只重跑文本转换 + 构建 + 核对
python -m convert --only ch00-profiling,ch16-fa2-splitq-mma   # 只转几章
python -m convert --jobs 8                      # TikZ 并行度
python -m convert --force-tikz                  # 忽略缓存重编全部 TikZ
python -m convert --skip-tikz --no-assemble     # 只跑 tex→md
```

远程 SSH 开发要在 VS Code 里看效果，**必须做端口转发**（内置浏览器在你本机渲染），见 `references/acceptance.md`。

## 验收：三层，缺一层就会漏

1. **文件级**：`python -m convert.verify --src build/src --html build/html`（23 项，`build.sh` 已含）。
2. **浏览器级**：`python -m tools.visual_check --html build/html --out build/shots/chapters` —— MathJax 报错、可见字面残迹、控制台报错、404 资源、整页截图。
3. **版式几何**（已并进 visual_check）：图片/表格是否居中、有没有元素（过宽公式、宽表）撑出正文栏。

两条铁律：**任一项不过就修 tex/转换器，不许绕过核对**；**`0 警告 ≠ 页面正确`**，总数阈值会掩盖单点丢失（表格 126→125 时「≥90%」照样通过，改成逐章比对立刻暴露）。

## 参考文件

- `references/pipeline.md` — 转换管线全貌（编排、章号/表格、TikZ→SVG、数学宏）、命令速查、目录结构、增量构建与 gitignore 陷阱。
- `references/layout-frontend.md` — 三栏版式与居中、模板覆盖规则（page.html / layout.html / 两个整文件副本各管什么）、图片/表格/过宽公式、「下载 PDF」按钮的落位与兜底、中英切换按钮的落位。
- `references/english.md` — Google 翻译切换机制与判定阈值、英文图集（词典/掩码/校验/换图）、英文封面离线生成。
- `references/acceptance.md` — 三层验收细节、23 项核对清单、visual_check 参数、本地与远程（VS Code 端口转发）预览。
- `references/pitfalls.md` — 已证伪路线与踩坑（TikZ、公式、正文结构），附页面上可见的症状。
- `references/sync-and-deploy.md` — 内容更新与自动同步对照表、改稿检查清单、RTD 建站步骤、已知边界。
