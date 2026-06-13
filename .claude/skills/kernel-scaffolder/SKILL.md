---
name: kernel-scaffolder
description: 在 LeetCUDA 仓库的某个 kernel 目录里，把上游参考实现 ($kernel.cu / $kernel.py) 转成两套学习用文件——my_$kernel.{cu,py,sh}（保留所有变体，掏空 __global__ 函数体作为 TODO）和 practice_$kernel.{cu,py}（每种 dtype 只留最佳 kernel）。在用户说"为 X 创建 my/practice 文件"、"把这套学习工作流推广到 X kernel"、"为 X 生成 my_X 和 practice_X"、"按 CLAUDE.md 的三层约定接入新 kernel"、或者类似的"按已有约定 scaffold 一个新 kernel 目录"诉求时使用。也覆盖只想生成其中一套（只 my 或只 practice）的场景。
---

# kernel-scaffolder

把 LeetCUDA 仓库里某个 kernel 目录下的上游参考实现 `$kernel.cu` / `$kernel.py`，按仓库 CLAUDE.md 描述的约定生成两套学习用文件：

- **`my_$kernel.{cu,py,sh}`** — 保留所有渐进版本（原 cu 里的所有 `__global__` kernel 全部留下），但把每个 kernel 函数体掏空换成 `// TODO: implement <这个变体在做什么>`。helper macros / `__device__` 辅助 / `TORCH_BINDING_*` 宏 / PyBind 注册全部原样保留，让 build 链能直接跑。Python driver 加上 `argparse` 暴露 `--benchmark` / `--no-check` / `--profiling <name>` / `--dtype` / `--S` / `--K`，build 目录搬到 `build/<lib_name>/`，nvcc 加 `-lineinfo`。
- **`practice_$kernel.{cu,py}`** — 每种 dtype 只留**最佳** kernel，函数名去掉优化后缀（如 `sigmoid_f32x4_kernel` → `sigmoid_f32_kernel`），函数体同样掏空标 TODO。Python driver 极简，只有 correctness + benchmark。
- **`my_$kernel.sh`** — 一行 ncu 包装，方便 profiling。

为什么要分两层：`my_*` 是渐进式学习用的——所有变体都在，对照原版一个一个填；`practice_*` 是反复练手用的——只有"最优"那一种，名字也去掉提示，逼自己回忆出最佳实现长什么样。

## 何时使用这个 skill

适用：用户在 LeetCUDA（或其 fork）里某个 `kernels/<op>/` 目录下，已经有上游 `$kernel.cu` + `$kernel.py`，希望按仓库 CLAUDE.md 的三层约定（reference / my / practice）生成 my 和/或 practice 文件。

不适用：
- 当前 kernel 目录里压根没有 `$kernel.cu` 或 `$kernel.py`（不是从空白起手 scaffold，而是从已有参考实现派生）；
- 用户想要生成的是参考实现本身（写一个新 op 的原版 cu/py，那是另一类工作）；
- 目录结构和 CLAUDE.md 描述不一致（比如 `kernels/hgemm/` 这种带自己的 `setup.py` 的子项目）。

如果是其中一种情况，先和用户确认意图再继续。

## 总体工作流

1. **定位输入**。从用户的话里抽取目标 kernel 目录。两种合法表达都接受：
   - **目录路径**：`kernels/gelu/` 或 `/code/LeetCUDA/kernels/gelu/`，从该目录下找 `*.cu` / `*.py` 推断 kernel 名。
   - **kernel 名**：`gelu`，默认对应 `kernels/gelu/`。
   只要其中一种确定，就够用。如果两者都没给清楚，问用户一句确认。

2. **找上游参考实现**。在目标目录里找 `<kernel>.cu` 和 `<kernel>.py`。如果只找到一个，问用户怎么处理；如果都找不到，停下来问用户参考实现在哪。注意 `kernels/reduce/` 是个特例——上游有 `block_all_reduce.{cu,py}`（完整版）和 `all_reduce.{cu,py}`（精简的 f16 链），用户说"reduce" 默认指的是 `all_reduce`。

3. **解析参考 cu**。提取以下信息（用 `Read` 看完整文件，不要 `head/grep` 局部看）：
   - 每个 `__global__ void <name>_kernel(...)` 的签名和它在做什么（函数体）。从函数体里推断"这个变体在做什么"——比如标量 / float4 vec / half2 / 4×half2 / 128-bit pack 等——后面要写进 TODO 注释。
   - 文件顶部的 `#define`（`FLOAT4`、`HALF2`、`LDST128BITS`、`MAX_EXP_*` 等）和任何 `__device__` helper。
   - `TORCH_BINDING_*` 宏的定义、参数（`packed_type, th_type, element_type, n_elements`）和实际调用。
   - `PYBIND11_MODULE` 里 expose 的所有名字。
   - 输入/输出张量布局（一进一出 vs 一进一出标量 vs 多进多出，是否要 atomicAdd 之类）。

4. **生成 `my_$kernel.cu`**。按 §my_kernel.cu 规则生成。

5. **生成 `practice_$kernel.cu`**。按 §practice_kernel.cu 规则生成；最佳 kernel 用 §选最佳 kernel 的启发式规则挑。

6. **生成 `my_$kernel.py`**。从参考 `$kernel.py` 起手，加 argparse 入口、check_correctness、run_profiling，build dir 改到 `build/<lib_name>/`，nvcc flags 加 `-lineinfo`。详见 §my_kernel.py。

7. **生成 `practice_$kernel.py`**。极简版，只跑最佳 kernel 的 correctness + benchmark。详见 §practice_kernel.py。

8. **生成 `my_$kernel.sh`**。一行 ncu 包装，详见 §my_kernel.sh。

9. **检查产物**。每写完一组，做一遍机械检查：
   - `my_*.cu` 里所有原 kernel 都还在吗？所有 `TORCH_BINDING_*(...)` 调用、`PYBIND11_MODULE` 里的 `m.def`、helper macro 都在吗？
   - `practice_*.cu` 里 dtype × 1 个 kernel，名字是不是都去掉了优化后缀？
   - Python driver 的 `lib.<func>` 引用都和 cu 里 expose 的名字对得上吗？
   - 把扩展加载需要的 include / 宏都保留了吗？

10. **告诉用户接下来怎么用**。简短总结：生成了哪几个文件、推荐先跑什么命令验证（`python3 my_$kernel.py --benchmark --no-check` 或 `--no-check` 跳过校验，因为 `__global__` 还是空的，跑完 build 但 correctness 会 FAIL，是预期）。

整个过程不要直接编译——目标是产出**结构正确的脚手架**，让用户接下来填 kernel 函数体。如果用户问"能不能也帮我把函数体填上"，那是另一个任务，可以另起一轮做。

## 选最佳 kernel 的启发式规则

CLAUDE.md 描述的命名约定是 `<op>_<elem-dtype>[x<vec-width>][_pack]_kernel`。在仓库已有的 sigmoid / relu 中，"最佳"的实际选择对应：

| dtype | 最佳变体 | 选择理由 |
| --- | --- | --- |
| f32 | `<op>_f32x4_kernel` | float4 一次搬 128 bit，element-wise op 的最优形态 |
| f16 | `<op>_f16x8_pack_kernel` | 128-bit pack（`LDST128BITS`）+ 8 元素 SIMD，比 unpack 的 f16x8 寄存器压力更低 |
| bf16 | `<op>_bf16x8_pack_kernel` | 同 f16，如果有的话 |
| f8 / 其他低精度 | 走 pack 版 | 同样优先 128-bit pack |

挑选算法：

1. **先按"算法语义"分组**。如果 kernel 名里编码了不同的算法变体（如 `softmax` / `safe_softmax` / `online_safe_softmax`，三种都是 softmax 但数值稳定性 / pass 数不同），把每个算法当独立的 op 处理 — practice 版**每种算法**都要保留各自的最佳 dtype 实现。理由：这些是不同的练习对象，naive softmax 练基本 reduce、safe 练 max-subtract、online 练 single-pass。把它们合并成一个就丢了练习意义。
2. 在每个算法分组里，把同一个 dtype 的所有 kernel 列出来。
3. 在该 dtype 的候选里，按 **`_pack` > `x<最大向量宽度>` > 标量** 的顺序挑出排名最高的一个。
4. 如果某个 dtype 没有 `_pack` 也没有 vec 版（只有标量，比如某些原型）——就保留那个标量版。
5. 如果出现拿不准的情况（比如 `_pack` 和某个 `x<更大宽度>` 同时存在），把候选给用户看一下，问一句"f16 我选 f16x8_pack 行吗？"

reduce 这条链的"最佳"在仓库里直接是 `all_reduce_sum_f16x8_pack_kernel`（只有这一个），照搬即可。

## §my_kernel.cu

完整保留参考实现的所有结构，只把每个 `__global__` 的函数体替换成 TODO。规则：

- include / namespace / 顶部 `#define` 全部保留。但把"会引诱用户直接抄答案"的 `FLOAT4` / `HALF2` / `BFLOAT2` / `LDST128BITS` / `INT4` 之类的 cast 宏注释掉（参考已有 `my_sigmoid.cu` / `my_relu.cu`），数值常量（`MAX_EXP_*` 之类）保留。
- 参考实现里如果有 `__device__` helper（如 `warp_reduce_sum_f32`），保留。可以再补一两个明显有用的 `__device__` helper（如 `__device__ __forceinline__ <op>(...)`）作为提示——但只在不会让答案过于显然的前提下做。
- 对每个 `__global__ void <name>_kernel(...)`：保留签名，函数体替换成
  ```cpp
  __global__ void sigmoid_f32x4_kernel(float *x, float *y, int N) {
    // TODO: implement fp32 vec4 sigmoid (FLOAT4 load/store)
  }
  ```
  TODO 注释要简短描述这个变体的特征（向量宽度 / 是否 pack / 用什么宏 / 关键操作），让人填的时候有方向但不至于直接给出答案。
- 所有 `#define STRINGFY` / `TORCH_BINDING_COMMON_EXTENSION` / `CHECK_TORCH_TENSOR_DTYPE` / `TORCH_BINDING_<OP>` 宏定义、所有 `TORCH_BINDING_<OP>(...)` 实例化、`PYBIND11_MODULE` 里的所有 `TORCH_BINDING_COMMON_EXTENSION(...)` 全保留，原封不动。

如果需要，可以参考 `references/templates.md` 里的"my_kernel.cu 骨架"部分对照修订。

## §practice_kernel.cu

按 §选最佳 kernel 的启发式 挑出每个 dtype 的最佳 kernel，然后写一个精简版：

- include 只保留必需的（`<cuda_fp16.h>` / `<cuda_runtime.h>` / `<torch/extension.h>` / `<torch/types.h>` 等）。
- 顶部加一段 banner 注释，说明这是 practice 版、每种 dtype 一个 kernel、名字是简洁版（去掉优化后缀），并简要列出每个 dtype 的最佳变体是什么以及为什么。
- 每个最佳 kernel 改名：去掉 `x<width>` 和 `_pack` 后缀。
  - `sigmoid_f32x4_kernel` → `sigmoid_f32_kernel`
  - `sigmoid_f16x8_pack_kernel` → `sigmoid_f16_kernel`
  - `all_reduce_sum_f16x8_pack_kernel` → `all_reduce_sum_f16_kernel`
- **当心 acc-type 后缀 ≠ 优化标签**。有些 kernel 名字编码了 accumulator dtype（典型例子：`dot_prod_f16x8_pack_f32_kernel`，末尾的 `_f32` 是"FP32 accumulator"，不是优化变体）。约定：识别"优化标签"只看 `x<num>` / `_pack` 这两类后缀；中间或末尾的 `_f32` / `_f16` 段如果**不是**这两类，就当成参与签名语义的一部分保留下来。比如 `dot_prod_f16x8_pack_f32` 简化成 `dot_prod_f16_f32`（保留 acc 后缀），不是 `dot_prod_f16`。如果不确定看一眼参考 cu 里的 kernel 是不是用相同 elem-dtype 但不同 acc-dtype 出现了多次——出现过就肯定是 acc 后缀。
- 函数体替换成 `// TODO(practice): best <DTYPE> <op> — <最优形态简述>`。
- TORCH_BINDING 宏可以**精简**（不需要 `if (ndim != 2)` 全分支，但要保留 `n_elements` 参数和 dtype 检查）——参照已有的 `practice_sigmoid.cu` / `practice_relu.cu`。把 `n_elements` 设成最佳 kernel 的向量宽度（标量=1，f32x4=4，f16x8_pack=8）。
- 只 bind 简洁名（`sigmoid_f32` / `sigmoid_f16`），dtype × 1 个。
- 如果参考实现是 reduce 风格（kernel 是 templated `<int NUM_THREADS>`，TORCH_BINDING 用 dispatch macro），保留这个结构，参照 `kernels/reduce/practice_all_reduce.cu`。

## §my_kernel.py

参考 `kernels/sigmoid/my_sigmoid.py` 和 `kernels/relu/my_relu.py`。要点：

- 顶部加 `_HERE` / `_BUILD_DIR`，`_BUILD_DIR = build/<kernel>_lib/`，os.makedirs 创建。
- `torch.utils.cpp_extension.load` 的 `name=` 用 `<kernel>_lib`，`sources=[my_<kernel>.cu]` 走绝对路径，`build_directory=_BUILD_DIR`，`extra_cuda_cflags` 在原版基础上加 `-lineinfo`。
- 三个核心函数：
  - `run_benchmark(perf_func, x, tag, out=None, warmup=10, iters=1000, show_all=False)`
  - `run_profiling(perf_func, x, tag, out=None, warmup=10)` —— 包 `nvtx.range_push("profiling")` / `range_pop()`
  - `check_correctness(perf_func, x, tag, out=None, atol=1e-5, rtol=1e-5)` —— `ref = torch.<op>(x)`（element-wise op）或 `ref = torch.sum(x, dtype=torch.float32)`（reduce）等，根据 op 决定。
- `run_benchmark_for_all_test(check=True)`：`Ss = Ks = [1024, 2048, 4096]`，对每个 (S, K) 跑：
  - FP32 路径：参考 cu 里所有 f32 系 kernel + torch baseline，先 check_correctness（如果 check=True），再 run_benchmark。
  - FP16 路径：同理，`atol/rtol=1e-3`。
  - 如果有 bf16，加 bf16 路径。
  - 如果是 reduce 风格（输出标量，没有 out 参数），照搬 `kernels/reduce/my_all_reduce.py` 的写法。
- `run_profiling_for_test(kernel_name, dtype, S=4096, K=4096)`：分支到对应 kernel 调 `run_profiling`。
- `if __name__ == "__main__":` 里 argparse 加 `--benchmark` / `--profiling` / `--dtype` / `--S` / `--K` / `--no-check`。

## §practice_kernel.py

参考 `kernels/sigmoid/practice_sigmoid.py` / `kernels/relu/practice_relu.py` / `kernels/reduce/practice_all_reduce.py`。要点：

- 顶部 docstring 说明这是 practice 版（一句话）。
- `_BUILD_DIR = build/practice_<kernel>_lib/`。
- `lib = load(name="practice_<kernel>_lib", sources=[practice_<kernel>.cu], ...)`。
- 只有两个函数：`check_correctness` 和 `run_benchmark`，签名和 my_*.py 里一致。
- `run(check=True)`：对每个 (S, K) 跑每个 dtype 的最佳 kernel + torch baseline，累加 `all_ok`。
- 末尾打印 ALL PASS / SOME FAIL 总结。
- argparse 只有 `--no-check`，`exit(0 if ok else 1)`。

## §my_kernel.sh

```bash
#!/usr/bin/env bash
set -e

name=${1:-<kernel>_<默认 dtype 后缀，比如 f16>}
dtype=${2:-<默认 dtype，比如 float16>}

ncu --nvtx \
  --nvtx-include "profiling/" \
  --set full \
  --import-source yes \
  -o "$name" \
  -- python3 my_<kernel>.py --profiling "$name" --dtype "$dtype"
```

如果是 reduce 风格（没有 `--dtype` 参数），用：

```bash
#!/usr/bin/env bash
set -e

name=${1:-<best_kernel_name>}

ncu --nvtx \
  --nvtx-include "profiling/" \
  -k regex:"$name"_kernel \
  --set full \
  --import-source yes \
  -f \
  -o "$name" \
  -- python3 my_<kernel>.py --profiling "$name"
```

记得 `chmod +x`（或者 Write 完之后让用户自己 chmod，提一下）。

## 对照模板

`references/templates.md` 里有完整的几个模板片段（element-wise op 风格、reduce 风格），生成时不确定的话先看那里再下笔。

## 边界情况

- **kernel 里有 templated `__global__`**（如 `template <const int NUM_THREADS> __global__ void ...`）：保留模板参数和默认值，函数体掏空。
- **多输入 / 多输出**：维持原签名，TODO 注释里说清楚每个参数代表什么。
- **kernel 函数体里有内嵌的 `#define`**（如 sigmoid 参考里把 `FLOAT4` 内嵌在 kernel 里）：在 my 版里把它放到 kernel 函数体内、`// TODO` 上面，作为提示。
- **CLAUDE.md 没在仓库里**：照样按这个 skill 描述的约定走，但提醒用户检查仓库根的 `CLAUDE.md` 看是否有更新的约定。
- **目录里已经有 `my_*` 或 `practice_*` 文件**：先 `Read` 一下，如果内容像是用户写过的填充版（不只是脚手架），停下来问用户要不要覆盖；如果只是空脚手架可以直接覆盖，但还是先告诉用户一声。
