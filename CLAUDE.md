# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库定位

LeetCUDA 是一个 CUDA 学习笔记仓库：`kernels/` 下每个 kernel 一个目录，包含 CUDA 实现、PyTorch 绑定和 benchmark 脚本。**没有顶层构建系统**，每个 kernel 目录都通过 `torch.utils.cpp_extension.load` 自行 JIT 编译。完整 kernel 列表和参考性能见上游 README。

## 三套文件约定（`$kernel` / `my_$kernel` / `practice_$kernel`）

本 fork 在多个 kernel 目录下加了一套个人学习用的工作流。同一个 kernel 目录里（如 `kernels/sigmoid/`、`kernels/relu/`、`kernels/reduce/`）会有三个变体——动手改之前先看清自己在哪一层：

| 层级 | 文件 | 用途 |
| --- | --- | --- |
| **参考** | `$kernel.cu` / `$kernel.py` | 上游原版。性能基线，渐进式实现的参考。在还没接入这套工作流的目录（如 `elementwise/` 和 `kernels/*` 下大多数目录）可能只有这一套。除非同步上游，否则别动。 |
| **学习 (my)** | `my_$kernel.cu` / `my_$kernel.py` / `my_$kernel.sh` | 完整保留所有渐进版本，但把所有 `__global__` 函数体掏空（只剩签名 + `// TODO`）。Python driver 增加 `check_correctness`、`--benchmark`、`--profiling`（NVTX 包裹）三类入口。 |
| **练手 (practice)** | `practice_$kernel.cu` / `practice_$kernel.py` | 每种 dtype 只保留**最佳** kernel，去掉名字里的优化标签（如 `sigmoid_f32` 而不是 `sigmoid_f32x4`、`relu_f16` 而不是 `relu_f16x8_pack`）。用于反复练手。 |

注：`kernels/reduce/` 是个特例——既有 `block_all_reduce.{cu,py}`（上游完整版），又有 `all_reduce.{cu,py}`（精简后的 f16 渐进参考）；该目录下的 `my_` / `practice_` 文件对应的是 f16 reduce 这条链。

把这套工作流推广到新 kernel 目录时，约定如下：
- `my_$kernel.cu` 与 `$kernel.cu` 完全一致，仅把每个 `__global__` 的函数体替换成 `// TODO`（PyTorch 绑定、helper 宏、`__device__` 辅助函数都保留）。
- `practice_$kernel.cu` 只保留每种 dtype 的最佳 kernel 签名（一种 dtype 一个），名字去掉优化后缀，但仍然走 `TORCH_BINDING_*` 宏接进 PyTorch。
- `my_$kernel.py` 参照 `kernels/sigmoid/my_sigmoid.py` / `kernels/relu/my_relu.py`：用 `argparse` 暴露 `--benchmark`、`--no-check`、`--profiling <kernel_name>`、`--dtype`、`--S`、`--K`；build 目录放在 `build/<lib_name>/`。
- `my_$kernel.sh` 是 `ncu` 的薄封装：`ncu --nvtx --nvtx-include "profiling/" --set full --import-source yes -k regex:"<name>_kernel" -o "$name" -- python3 my_$kernel.py --profiling "$name"`。

## 常用命令

所有命令都在某个具体的 kernel 目录里执行。

```bash
# 限定单一架构 JIT，否则 PyTorch 会编译 Volta..Hopper 全部架构（很慢）。
export TORCH_CUDA_ARCH_LIST=Ada   # 也可以是 Ampere、Hopper 等

# 参考实现：完整的 Ss × Ks 形状扫描 benchmark
python3 sigmoid.py

# 学习实现：benchmark + 正确性校验
python3 my_sigmoid.py --benchmark
python3 my_sigmoid.py --benchmark --no-check     # 跳过正确性校验
python3 my_sigmoid.py --profiling sigmoid_f16x8_pack --dtype float16
python3 my_sigmoid.py --profiling sigmoid_f32x4     --dtype float32 --S 4096 --K 4096

# 一键 ncu profiling，结果落到当前目录的 <name>.ncu-rep
./my_sigmoid.sh                                  # 默认 kernel + dtype
./my_sigmoid.sh sigmoid_f16x8_pack float16

# 练手实现：只有每种 dtype 的最佳 kernel，名字也是简洁版
python3 practice_sigmoid.py
python3 practice_sigmoid.py --no-check
```

`ncu` 报告用 Nsight Compute UI 打开。`my_$kernel.py` 的 build 产物放在 `build/<lib_name>/`，避免和参考 `$kernel.py` 的编译缓存冲突。

## 构建相关

- 每个 `*.py` driver 都通过 `torch.utils.cpp_extension.load` 编译同目录的 `*.cu`。**仓库根目录没有 `setup.py` / CMake**；`kernels/hgemm/` 和 `ffpa-attn/` 这类子项目自带 `setup.py` / `makefile`，是独立的。
- 全仓库通用的 nvcc flags：`-O3`、`--use_fast_math`、四个 `-U__CUDA_NO_HALF*__` 取消宏（让 half/bf16 的 operator overload 能用）、`--expt-relaxed-constexpr`、`--expt-extended-lambda`。要做 profiling 时再加 `-lineinfo`（`my_*.py` 都已经加了）。
- FP8 / WGMMA / cute 系列 kernel 需要较新的工具链（CUDA 12.x、SM89/SM90）。除参考实现外的工作大多以 Ada/Ampere 为目标——动手前先看 kernel 自己的 README。

## Pre-commit（`.pre-commit-config.yaml`）

`clang-format`（仓库根的 style 文件）、`isort`、`black --line-length 80`，加上标准的 pre-commit-hooks，每次 commit 都会跑。还启用了 `no-commit-to-branch` 钩子——一律在 feature 分支上工作，不要直接提到 `main`。

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

## 新增 / 修改 kernel 时的约定

- `__global__` kernel 命名编码了变体：`<op>_<elem-dtype>[x<vec-width>][_pack]_kernel`（如 `sigmoid_f16x8_pack_kernel`、`block_all_reduce_sum_f16x8_pack_f32_kernel`）。Torch 绑定名去掉 `_kernel`，并通过 `.cu` 文件底部的 `TORCH_BINDING_*` 宏接出去——保持这个模式。
- 向量化 load/store 全仓共用同一组 cast 宏：`FLOAT4`、`HALF2`、`LDST128BITS`（用 `float4*` cast 来做 128-bit 搬运）。不要另起炉灶。
- `my_$kernel.py` 的 benchmark 默认跑 `Ss × Ks ∈ {1024, 2048, 4096}²` 的形状网格。新 kernel 接入时也按这个网格走，性能数据才能横向比较。
- 数值上敏感的 kernel（sigmoid、gelu、softmax）记得 clamp 到对应 dtype 的 `MIN_EXP_*` / `MAX_EXP_*` 常量——参见 `kernels/sigmoid/my_sigmoid.cu`。
