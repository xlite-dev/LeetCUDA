# 跨仓库源码冻结快照

书稿有若干 `\lstinputlisting` 用 `../../../../<repo>/...` 引用与 LeetCUDA 同级的
checkout（见 `kernels/interview/book/appendices/appC-build-guide.tex`）。xelatex 编译
PDF 时读者必须本地检出这些仓库；RTD 只检出 LeetCUDA，所以站点用本目录的冻结副本
内联这些清单，否则页面上只会剩提示块。

## 键规则

`sources/<repo>/<path>` 对应同级 checkout 的 `<repo>/<path>`：`book_dir / raw_path`
相对 LeetCUDA 仓库根越出的那一级目录是 `<repo>`，其余部分原样保留。

`convert/preprocess.py::snapshot_key` 对越出仓库根的引用**只**读快照，不回退本地
checkout——本地 checkout 会随上游演进漂移，快照才是本地预览与线上共同的事实源。

## 锚定 commit

| 仓库 | commit | 依据 |
| --- | --- | --- |
| cache-dit | `db125215` | ch39/ch40 头注释、appC |
| ffpa-attn | `861d75e` | ch27 头注释、appC |

## 内容

| 文件 | 引用章 | sha256 |
| --- | --- | --- |
| `cache-dit/csrc/kernels/svdq/cute/fp4/quantize/quantize_impl.cuh` | ch39 | `d7e703efdf49116ac9f16cc1e0b02e102c7b01a564a167990b60443340a96ad4` |
| `cache-dit/csrc/kernels/svdq/cute/fp4/quantize/quantize_uni.cu` | ch39 | `fee86eaf82d80c5bc69b7ba8822964e99101d1670458013f8d43b8c5b6be073e` |
| `cache-dit/csrc/kernels/svdq/cute/fp4/gemm/gemm_w4a4.cuh` | ch40 | `e6a9074b45ed75b6a5e741e54aeb816ae6693b6b79c62dbe9f985f248647e962` |
| `cache-dit/csrc/kernels/svdq/cute/fp4/gemm/gemm_w4a4.cu` | ch40 | `f2c81066d4a6b7d99ce2d3973762ad5f004f531529786b46e1f8cc09a098eef7` |
| `ffpa-attn/csrc/cuffpa/cute/fp8/fp8_pscale.cuh` | ch27 | `3cb88c6b044b8c81bfde1a625495be7ae9394c8b4971c6604ca459f47918fe56` |
| `ffpa-attn/csrc/cuffpa/cute/softmax.cuh` | ch27 | `72c339e36dcbfc218964fed0e653d74da111951ac0ce9670eb05bae1901d784c` |

## 导出与更新

```bash
cd docs/readthedocs
git -C ../../../cache-dit show db125215:<path> > sources/cache-dit/<path>
git -C ../../../ffpa-attn show 861d75e:<path> > sources/ffpa-attn/<path>
sha256sum sources/<repo>/<path>   # 与上表对齐
```

需要重导的时机：书稿新增或改动跨仓库清单引用、上游锚定 commit 变更。缺快照的
跨仓库引用会让 `convert.verify` 的「跨仓库清单已内联」判红，不会静默放过。

## 提交注意

本目录是上游源码的逐字节副本，**任何格式化钩子都不能碰**：仓库根的 `.pre-commit-config.yaml`
里 `clang-format`、`trailing-whitespace`、`end-of-file-fixer` 默认会改写 `.cu`/`.cuh`，
实测一次提交就把 4 个快照的 sha256 改掉（与锚定 commit 不再一致）。提交时要么用
`git commit --no-verify`（本仓库 2 空格缩进与 black 的既有约定同此），要么把本目录加进
钩子顶层 `exclude: ^docs/readthedocs/sources/`。改完用 `sha256sum` 复核上表。
