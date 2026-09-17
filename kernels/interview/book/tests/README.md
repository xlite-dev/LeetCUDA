# book/tests — 每章最小测试

规范源：BOOK_PLAN §6。目标：**取代巨型 notes-v2.cu harness 的正确性验证角色**（notes-v2.cu 保留为集成 bench harness，见附录 C 角色声明）。

## 约定

- 每章一个 `chNN_<slug>.cu`，单文件独立编译：`init → kernel → CPU fp64 参考 → PASS/FAIL`。
- **无 cuBLAS/cuDNN 依赖**（正确性对照一律 CPU fp64）。
- 容差三档（`common_test.h`）：`TOL_F32ACC=1e-3` / `TOL_F16ACC=5e-2` / `TOL_TF32=1e-2`。
- 测试规模 ≤512（`BOOK_TEST_MAX_N`）。
- arch-gated kernel 用 `book_require_sm()` 保护，目标 arch 不在位输出 `SKIP(chNN): requires sm_XX` 并返回 0。
- 行数上限：基础章（ch1-8）≤300 行；GEMM/FA/CuTe 章（ch9-26）≤500 行；共享逻辑进 `common_test.h`。
- 源码头文件以 `#include "../../base.cuh"` 形式引用（相对本目录）。

## 用法

```bash
cd kernels/interview/book/tests
./build_tests.sh --arch sm_120a --all   # 构建运行全部已存在测试
./build_tests.sh --ch ch04              # 单章
./build_tests.sh --ch ch13 --no-run     # 只编译（如远端无 H800 时的 ch13）
```

## 进度

随 RFC-C/D/E/F 各章任务逐个落地；执行卡片中的「测试 ← notes-v2.cu 抽取源」行号见 `book/skills/write-leetcuda-book/RFC.md`。
