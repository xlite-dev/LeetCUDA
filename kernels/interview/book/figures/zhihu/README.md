# `figures/zhihu/` — 知乎原图归档规范（RFC-B.4）

> 作用：书稿引用知乎文章中的图（或作为 drawio 重建的原始素材）时，**必须先归档原图**再进入重建/引用流程，保证可追溯、可回退、版权来源清晰。
> 关联：BOOK_PLAN §7.2（单图 DoD 六条）、RFC-B.4a（drawio 重建管线试点）；总索引见 `book/references/zhihu-inventory.md`（附录 E 的「图片数」列由此目录回填）。

## 1. 目录规范

```
figures/zhihu/
  <author>-<slug>/            # 一篇文章一个目录；author 为知乎作者名，slug 为文章短名（kebab-case）
    meta.md                   # 元数据 sidecar（模板见 §3），文件名固定
    original/                 # 原始图（从知乎文章直接保存，不改名不改内容）
      fig01.png
      fig02.png
    rebuilt/                  # drawio 重建产物（.drawio + 导出的 .png/.svg），仅重建图放这里
      fig01.drawio
      fig01.png
```

命名约定：

- `<author>`：知乎作者名，保留中文（如 `reed`、`竹熙佳处`、`frankshi`）；同名冲突时加序号后缀。
- `<slug>`：文章短名，kebab-case，英文优先（如 `cute-swizzle`、`tiled-copy`、`how-to-optimize-gemm`）。
- 图文件：`figNN.<ext>`，`NN` 为文章内出现顺序（01 起），`ext` 与原图格式一致（`png`/`jpg`/`webp`）。
- 只归档**书稿实际引用或拟引用**的图；未引用图不入库，避免目录膨胀。

## 2. 流程（单图）

1. **取图**：从文章页保存原图（登录态下右键保存或浏览器快照提取），放入 `original/`，文件名按 §1 命名。
2. **登记**：在 `meta.md` 追加一行（或新建 sidecar 条目），填写来源 URL、作者、文章标题、引用章节、归档日期；`替换状态` 初始为 `否`。
3. **重建（可选，B.4a）**：需要重绘的图（如声明式示意图），在 `rebuilt/` 出 `.drawio` + 导出图，并把 `meta.md` 的 `替换状态` 改为 `是`，同时登记 rebuilt 文件名与重建日期。
4. **引用**：书稿只引用 `rebuilt/`（重建后）或 `original/`（未重建）中的图；`\includegraphics` 路径写相对 `book/` 的路径。
5. **回填**：文章「图片数」= 该目录下被引用的图数，回填到 `references/zhihu-inventory.md` 附录 E 对应行。

## 3. `meta.md` 模板

```markdown
# <文章标题>

- 来源 URL: https://zhuanlan.zhihu.com/p/<id>
- 作者: @<author>
- 文章标题: <原文标题>
- 归档日期: YYYY-MM-DD
- 引用章节: ch<NN>（<用途，如「smem swizzle 前后排布图」>）

## 图片清单

| 图文件 | 文章内位置 | 引用章节 | 替换状态 | 重建文件 | 备注 |
|---|---|---|---|---|---|
| original/fig01.png | 图 7 | ch12 | 否 | — | 原图带知乎水印 |
| original/fig02.png | 图 9 | ch23 | 是 | rebuilt/fig02.drawio / fig02.png | 2026-09-11 重建，见 RFC-B.4a |
```

字段说明：

| 字段 | 取值 | 说明 |
|---|---|---|
| 来源 URL | 原文链接（不带 `utm_*`） | 必须是知乎原文，不接受转载镜像 |
| 作者 / 文章标题 | 与 `zhihu-inventory.md` 一致 | 便于交叉检索 |
| 引用章节 | `chNN` 列表 | 一张图可被多章引用 |
| 替换状态 | `是` / `否` | `是`=已有 drawio 重建产物并替代原图引用 |
| 重建文件 | 相对本目录路径 | 仅 `替换状态=是` 时填写 |

## 4. 约束

- **原图只读**：`original/` 内的文件不得覆盖、压缩或重命名；需要处理请输出到 `rebuilt/`。
- **水印可见**：知乎原图水印**保留**，不得裁剪消除；重建图在 `meta.md` 备注原图水印情况（B.4a 验收关注点）。
- **路径稳定**：书稿一旦引用某图，路径不再变更；确需替换走「新增文件 + 更新 meta.md + 全局替换引用」三步。
- **目录不自动发现**：不为未引用的图建目录；不建 `zhihu-bak/` 之类备份目录。
