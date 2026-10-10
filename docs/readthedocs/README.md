# LeetCUDA 书的 Read the Docs 站点

把 `kernels/interview/book/` 下的书稿 tex 逐章转换成网页：公式走 MathJax 实时渲染，TikZ 图编译成 SVG，代码清单按 `linerange` 从冻结源码抽出来贴进页面。

站点：<https://leetcuda.readthedocs.io>

**原书 tex、源码、图片一律不改**：转换读原文件，产物只写进 `build/`（已 gitignore）。

## 技术细节都在 skill 里，本文件不再维护

转换管线、命令参数、版式与模板规则、中英切换、英文图集与封面、23 项内容核对、浏览器验收、踩坑与已证伪路线、RTD 部署与改稿检查清单——全部在 skill 里，本文件只留入口。

- **skill 名**：`leetcuda-tex-to-readthedocs`（动手前先加载它）
- **skill 位置**：`skills/leetcuda-tex-to-readthedocs/`（`SKILL.md` + `references/`），随本仓库一起走
- **IDE 里的入口**：软链在 `/workspace/dev/vipshop/.github/skills/leetcuda-tex-to-readthedocs`

**维护约定**：改本目录下任何东西（`convert/`、`_templates/`、`_static/`、`conf.py`、`tools/`、核对项）之前先加载 skill；新得到的技术结论、踩坑、实测数字写进 skill 的对应 reference，**不回写本文件**——本文件保持为纯入口。

## 最短上手

```bash
conda activate cdit
cd docs/readthedocs
pip install -r requirements.txt          # sphinx / myst-parser / sphinx-rtd-theme
./build.sh                               # 转换 → sphinx-build → 内容核对
python -m http.server -d build/html 8000 # 预览
```

TikZ 全量编译约 3–5 分钟；调试时用 `./build.sh --skip-tikz` 复用已有 SVG。其余参数与注意事项见 skill 的 `references/pipeline.md`。
