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
| cache-dit（FP8，ch41） | `453c1b05` | ch41 头注释、appC（PR #1371） |
| cache-dit（FP4/SVDQ） | `eced0158` | ch39/ch40/ch41 头注释、appC（该 PR 未改动） |
| ffpa-attn | `861d75e` | ch27 头注释、appC |

## 内容

| 文件 | 引用章 | sha256 |
| --- | --- | --- |
| `cache-dit/csrc/kernels/svdq/cute/fp4/quantize/quantize_impl.cuh` | ch39 | `a768a022b3c59607654b7225f2d0898f45d043f926c78b5b5afbc84fe815dc11` |
| `cache-dit/csrc/kernels/svdq/cute/fp4/quantize/quantize_uni.cu` | ch39 | `fee86eaf82d80c5bc69b7ba8822964e99101d1670458013f8d43b8c5b6be073e` |
| `cache-dit/csrc/kernels/svdq/cute/fp4/gemm/gemm_w4a4.cuh` | ch40 | `e6a9074b45ed75b6a5e741e54aeb816ae6693b6b79c62dbe9f985f248647e962` |
| `cache-dit/csrc/kernels/svdq/cute/fp4/gemm/gemm_w4a4.cu` | ch40 | `f2c81066d4a6b7d99ce2d3973762ad5f004f531529786b46e1f8cc09a098eef7` |
| `cache-dit/csrc/kernels/fp8/gemm/gemm.cuh` | ch41 | `c1e6d5224cf92e7e078895c2e11666b79e9bddb17e2f50d5d1b2ed7645d593f0` |
| `cache-dit/csrc/kernels/fp8/quantize.cu` | ch41 | `a549d2ab06a11c2d36a1c710067d252705a4d3f483bfa55fa8f7f5e3caa5e52a` |
| `cache-dit/csrc/kernels/fp8/gemm/per_row.cu` | ch41 | `639340ad52ae420f42b564766ccdcd46f0f5d8c450b48e67c4acd8a645c29602` |
| `cache-dit/src/cache_dit/quantization/fp8/fused.py` | ch41 | `710e2a333c72941c33fd701d307bfb6ccdb330ee04351d062fa0d67190ec512e` |
| `cache-dit/csrc/kernels/fp4/gemm.cuh` | ch41 | `a787ee0d5b4ede351543e1d910e853f340e099fa7b627497d2f7fe5a0ca37979` |
| `cache-dit/csrc/kernels/fp4/quantize.cuh` | ch41 | `2bdd6c71e47c6b70b92cae9e6edd77fda5cc3767443a73601adf076643db38eb` |
| `ffpa-attn/csrc/cuffpa/cute/fp8/fp8_pscale.cuh` | ch27 | `3cb88c6b044b8c81bfde1a625495be7ae9394c8b4971c6604ca459f47918fe56` |
| `ffpa-attn/csrc/cuffpa/cute/softmax.cuh` | ch27 | `72c339e36dcbfc218964fed0e653d74da111951ac0ce9670eb05bae1901d784c` |

## 导出与更新

```bash
cd docs/readthedocs
# FP8（ch41，含融合 MLP 一节）
git -C ../../../cache-dit show 453c1b05:<path> > sources/cache-dit/<path>
# FP4 与 SVDQuant
git -C ../../../cache-dit show eced0158:<path> > sources/cache-dit/<path>
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
