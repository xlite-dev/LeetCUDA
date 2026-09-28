// =============================================================================
// notes-v2.cu — CUDA Kernel 面试背题笔记（多 TU 布局：本文件仅 main + 模块原型）
// =============================================================================
//
// 整理自 LeetCUDA 项目（https://github.com/xlite-dev/LeetCUDA），涵盖：
//   - 面试高频 CUDA kernel 的完整实现（~30 个 kernel）
//   - 每类 kernel 附带详细的面试要点注释（WHY + HOW）
//   - 优化技术的递进式讲解（naive → tiling → vectorize → tensor core → ws）
//   - BLAS 语义：N=col-major(Normal), T=row-major(Transposed)
//
// 10 个 Phase 覆盖：
//   Phase 0 — 面试框架速查（GPU 架构 / Memory Hierarchy / Roofline / 优化清单）
//   Phase 1 — 基础原语：Warp Reduce / Block Reduce / Dot Product（含 broadcast 增强版）
//   Phase 2 — Elementwise：ReLU / Elementwise Add / Histogram（基础 + float4 向量化 + atomic）
//   Phase 3 — Softmax：naive → safe → online + RMS/Layer Norm
//   Phase 4 — RoPE：旋转位置编码（Llama 风格 theta=10000）
//   Phase 5 — Mat Transpose：基础版 + BCF merge_write 最佳版（Bank Conflict专题）
//   Phase 6 — GEMV：SGEMV K32/K128/K16（warp-per-row）
//   Phase 7 — GEMM ★：SGEMM → HGEMM → MMA m16n8k16(TN布局) → WGMMA m64n128k16
//   Phase 8 — FlashAttention-2split_q（FA-2, 含 online softmax + P@V 寄存器复用）
//
// 多 TU 布局（并行编译，构建命令见文件尾）：
//   本文件            — main() 与全部模块入口原型（声明+调用）
//   *.cuh             — kernel 与 host 封装（不变，书籍源码冻结对象）
//   {base,sgemv,sgemm,hgemm,fp8_gemm,fp4_gemm,flash_attn,ffpa_attn}.cu
//                     — 各模块 host 侧 test/bench（模板实例化所在 TU）
//   utils.cu          — 跨模块共享符号（bench 全局配置 + 公共辅助函数）
// =============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

// FALayout：与 flash_attn.cu 中的定义保持一致（枚举值顺序不可变）
enum class FALayout {
  All, Pad, SwizzleQ, SwizzleK, SwizzleV, SwizzleQK, SwizzleQV, SwizzleKV, Swizzle
};

// main 专用的模式开关与规模参数（仅 main 读写，不跨 TU）
static bool g_bench_hgemm = false;
static bool g_bench_hgemm_all = false;
static bool g_bench_fa = false;
static bool g_bench_all = false;
static bool g_swizzle_eq_check = false;
static int g_bench_M = 8192, g_bench_N = 8192, g_bench_K = 8192;
static int g_bench_B = 1, g_bench_H = 32, g_bench_Nfa = 8192, g_bench_D = 128;

// 共享全局（定义在 utils.cu；g_fa_layout 定义在 flash_attn.cu）
extern bool g_debug;
extern bool g_verbose;
extern int g_warmup;
extern int g_repeat;
extern bool g_bench_fa3_cute_only;
extern bool g_fa_skip_check;
extern FALayout g_fa_layout;

// ---- base.cu ----
void test_block_reduce(int N);
void test_dot(int N);
void test_relu(int N);
void test_elementwise(int N);
void test_histogram(int N);
void test_merge_attn_states(int num_tokens, int num_heads, int head_size);
void test_softmax(int N);
void test_rms_norm(int N, int K);
void test_layer_norm(int N, int K);
void test_rope(int seq_len, int N);
void test_mat_transpose(int row, int col);
void test_mat_transpose_padded(int row, int col);
void test_swizzle_equiv();

// ---- sgemv.cu / sgemm.cu ----
void test_sgemv(int M, int K);
void test_sgemm(int M, int N, int K);

// ---- hgemm.cu ----
void test_hgemm_mma(int M, int N, int K);
void test_hgemm_swizzle(int M, int N, int K);
#if defined(NOTES_V2_ENABLE_CUTE)
void test_hgemm_cute(int M, int N, int K);
#endif
#if defined(NOTES_V2_ENABLE_WGMMA)
void test_hgemm_wgmma(int M, int N, int K);
#endif
#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
void test_hgemm_tma_mma_ws(int M, int N, int K);
#endif
void bench_hgemm_mma(int M, int N, int K);
void bench_hgemm_swizzle(int M, int N, int K);
#if defined(NOTES_V2_ENABLE_CUTE)
void bench_hgemm_cute(int M, int N, int K);
#endif
#if defined(NOTES_V2_ENABLE_WGMMA)
void bench_hgemm_wgmma(int M, int N, int K);
#endif
#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
void bench_hgemm_tma_mma_ws(int M, int N, int K);
#endif

// ---- fp8_gemm.cu ----
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
void test_fp8_gemm(int M, int N, int K);
void bench_fp8_gemm(int M, int N, int K);
void bench_fp8_gemm_tile_sweep(int M, int N, int K);
#endif

// ---- fp4_gemm.cu ----
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS) && \
    defined(NOTES_V2_ENABLE_SM120_FP4)
void test_fp4_gemm(int M, int N, int K);
void bench_fp4_gemm(int M, int N, int K);
void bench_fp4_gemm_tile_sweep(int M, int N, int K);
#endif

// ---- flash_attn.cu ----
void test_flash_attn(int seqlen, int head_dim);
#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
void test_flash_attn_tma_mma_ws(int seqlen, int head_dim);
void test_flash_attn_3_tma_ws(int seqlen, int head_dim);
#endif
void bench_flash_attn(int B, int H, int N, int D);
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
void run_fa3_cute_tests();
void run_fa3_cute_tma_smoke_tests();
void run_fa2_cute_tests();
void run_fa2_cute_cpasync_tests();
void run_pd_cute_tests();
#endif

// ---- ffpa_attn.cu ----
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
void run_sdnw_cute_tests();
#endif

// ================================================================
// 以下是测试代码，验证 Phase 1 - Phase 8 的kernel的正确性，不评估性能。
// ================================================================

int main(int argc, char *argv[]) {
#if defined(NOTES_V2_ENABLE_WGMMA) || defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  cuInit(0);
#endif

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--bench-hgemm") == 0) {
      g_bench_hgemm = true;
    } else if (strcmp(argv[i], "--bench") == 0) {
      g_bench_hgemm = true;
      g_bench_fa = true;
    } else if (strcmp(argv[i], "--bench-fa") == 0) {
      g_bench_fa = true;
    } else if (strcmp(argv[i], "--bench-fa3-cute") == 0) {
      g_bench_fa = true;
      g_bench_fa3_cute_only = true;
    } else if (strcmp(argv[i], "--bench-hgemm-all") == 0) {
      g_bench_hgemm_all = true;
    } else if (strcmp(argv[i], "--bench-fa-all") == 0) {
      g_bench_fa = true;
      g_fa_layout = FALayout::All;
    } else if (strcmp(argv[i], "--bench-all") == 0) {
      g_bench_all = true;
      g_fa_layout = FALayout::All;
    } else if (strcmp(argv[i], "--mnk") == 0 && i + 1 < argc) {
      sscanf(argv[++i], "%d,%d,%d", &g_bench_M, &g_bench_N, &g_bench_K);
    } else if (strcmp(argv[i], "--bhnd") == 0 && i + 1 < argc) {
      sscanf(argv[++i], "%d,%d,%d,%d", &g_bench_B, &g_bench_H, &g_bench_Nfa, &g_bench_D);
    } else if (strcmp(argv[i], "--fa-layout") == 0 && i + 1 < argc) {
      const char *layout = argv[++i];
      if (strcmp(layout, "all") == 0)
        g_fa_layout = FALayout::All;
      else if (strcmp(layout, "pad") == 0)
        g_fa_layout = FALayout::Pad;
      else if (strcmp(layout, "swizzle-q") == 0)
        g_fa_layout = FALayout::SwizzleQ;
      else if (strcmp(layout, "swizzle-k") == 0)
        g_fa_layout = FALayout::SwizzleK;
      else if (strcmp(layout, "swizzle-v") == 0)
        g_fa_layout = FALayout::SwizzleV;
      else if (strcmp(layout, "swizzle-qk") == 0)
        g_fa_layout = FALayout::SwizzleQK;
      else if (strcmp(layout, "swizzle-qv") == 0)
        g_fa_layout = FALayout::SwizzleQV;
      else if (strcmp(layout, "swizzle-kv") == 0)
        g_fa_layout = FALayout::SwizzleKV;
      else if (strcmp(layout, "swizzle") == 0)
        g_fa_layout = FALayout::Swizzle;
      else {
        fprintf(stderr, "Unsupported FA layout: %s\n", layout);
        return EXIT_FAILURE;
      }
    } else if (strcmp(argv[i], "--fa-skip-check") == 0) {
      g_fa_skip_check = true;
    } else if (strcmp(argv[i], "--swizzle-eq-check") == 0) {
      g_swizzle_eq_check = true;
    } else if (strcmp(argv[i], "--debug") == 0) {
      g_debug = true;
    } else if (strcmp(argv[i], "--verbose") == 0) {
      g_verbose = true;
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      g_warmup = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
      g_repeat = atoi(argv[++i]);
    }
  }

  if (g_swizzle_eq_check) {
    printf("=== notes-v2.cu swizzle v1/v2 equivalence check ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    test_swizzle_equiv();
    printf("=== Done ===\n");
    return 0;
  }

#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  if (argc >= 2 && strcmp(argv[1], "--fa3-cute") == 0) {
    printf("=== CuTe FA3 TMA MMA WS correctness ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    run_fa3_cute_tests();
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "--fa3-cute-tma-smoke") == 0) {
    printf("=== CuTe TMA copy smoke ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    run_fa3_cute_tma_smoke_tests();
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "--fa2-cute-cpasync") == 0) {
    printf("=== CuTe FA2 MMA Stages (cp.async) correctness ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    run_fa2_cute_cpasync_tests();
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "--fa2-cute") == 0) {
    printf("=== CuTe FA2 TMA MMA WS correctness ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    run_fa2_cute_tests();
    return 0;
  }
#endif

  if (g_bench_hgemm || g_bench_fa || g_bench_all) {
    printf("=== notes-v2.cu bench mode ===\n");
    printf("HGEMM: M=%d N=%d K=%d   FA: B=%d H=%d N=%d D=%d\n",
           g_bench_M, g_bench_N, g_bench_K,
           g_bench_B, g_bench_H, g_bench_Nfa, g_bench_D);
    printf("| %-56s | %-9s | %-19s |\n", "Kernel", "Max Err", "TFLOPS/cu{BLAS,DNN}");
    printf(
      "|----------------------------------------------------------|-----------|---------------------|\n"
    );

    if (g_bench_hgemm || g_bench_hgemm_all || g_bench_all) {
      if (g_bench_hgemm_all || g_bench_all) {
        bench_hgemm_mma(g_bench_M, g_bench_N, g_bench_K);
        bench_hgemm_swizzle(g_bench_M, g_bench_N, g_bench_K);
#if defined(NOTES_V2_ENABLE_WGMMA)
        bench_hgemm_wgmma(g_bench_M, g_bench_N, g_bench_K);
#endif
#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
        bench_hgemm_tma_mma_ws(g_bench_M, g_bench_N, g_bench_K);
#endif
      }
#if defined(NOTES_V2_ENABLE_CUTE)
      bench_hgemm_cute(g_bench_M, g_bench_N, g_bench_K);
#endif
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
      bench_fp8_gemm(g_bench_M, g_bench_N, g_bench_K);
#endif
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS) && \
    defined(NOTES_V2_ENABLE_SM120_FP4)
      bench_fp4_gemm(g_bench_M, g_bench_N, g_bench_K);
#endif
    }
    if (g_bench_fa || g_bench_all)
      bench_flash_attn(g_bench_B, g_bench_H, g_bench_Nfa, g_bench_D);

    printf("=== Bench done ===\n");
    return 0;
  }

#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  if (argc >= 2 && strcmp(argv[1], "--pd-cute") == 0) {
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
    // Phase 8: persist-D WS + persistent CTA + scale fused 快速入口
    run_pd_cute_tests();
#endif
    printf("=== persist-D cute tests done ===\n");
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "--sdnw-cute") == 0) {
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
    // ch26c: non-WS TMA Split-D 快速入口 (CPU fp64 ref)
    run_sdnw_cute_tests();
#endif
    printf("=== split-D non-WS cute tests done ===\n");
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "--fp8-gemm") == 0) {
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
    // Phase 9: FP8 GEMM CuTe 快速入口 (4 种 scale 组合 + WS + 尾部 shape)
    printf("=== FP8 GEMM CuTe correctness (Phase 9) ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    test_fp8_gemm(512, 512, 512);
#endif
    printf("=== FP8 GEMM tests done ===\n");
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "--fp8-gemm-sweep") == 0) {
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
    // ch35: tile 几何(BM x BN) x 流水深度扫描，shape 由 --mnk 控制
    bench_fp8_gemm_tile_sweep(g_bench_M, g_bench_N, g_bench_K);
#else
    printf("FP8 GEMM sweep requires NOTES_V2_ENABLE_CUTE + TMA_MMA_WS\n");
#endif
    printf("=== FP8 GEMM sweep done ===\n");
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "--fp4-gemm") == 0) {
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS) && \
    defined(NOTES_V2_ENABLE_SM120_FP4)
    // Phase 10: NVFP4 GEMM CuTe 快速入口（单级/两级 x A 行/B 列 + WS + 尾部 shape）
    printf("=== FP4 GEMM CuTe correctness (Phase 10) ===\n");
    printf("误差列 = relFro 相对 Frobenius 误差（vs CPU fp64）\n");
    printf("| %-56s | %-9s |\n", "Kernel", "relFro");
    printf("|----------------------------------------------------------|----------|\n");
    test_fp4_gemm(512, 512, 512);
#else
    printf("FP4 GEMM requires NOTES_V2_ENABLE_CUTE + TMA_MMA_WS + "
           "NOTES_V2_ENABLE_SM120_FP4\n");
#endif
    printf("=== FP4 GEMM tests done ===\n");
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "--fp4-gemm-sweep") == 0) {
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS) && \
    defined(NOTES_V2_ENABLE_SM120_FP4)
    // ch37: BN x 流水深度扫描，BM 固定 128（SF 行块约束），shape 由 --mnk 控制
    bench_fp4_gemm_tile_sweep(g_bench_M, g_bench_N, g_bench_K);
#else
    printf("FP4 GEMM sweep requires NOTES_V2_ENABLE_CUTE + TMA_MMA_WS + "
           "NOTES_V2_ENABLE_SM120_FP4\n");
#endif
    printf("=== FP4 GEMM sweep done ===\n");
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "--tma-mma-ws") == 0) {
    int M = 128, N = 128, K = 64;
    if (argc > 4) {
      M = atoi(argv[2]);
      N = atoi(argv[3]);
      K = atoi(argv[4]);
    }
    printf("=== SM120 TMA MMA WS validation ===\n");
    printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
    printf("|----------------------------------------------------------|----------|\n");
    test_hgemm_tma_mma_ws(M, N, K);
    return 0;
  }
#endif
  int M = 1024, N = 1024, K = 1024;
  if (argc > 3) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

  printf("=== notes-v2.cu verification harness ===\n");
  printf("| %-56s | %-9s |\n", "Kernel", "Max Err");
  printf("|----------------------------------------------------------|-----------|\n");

  test_block_reduce(N);
  test_dot(N);
  test_relu(1024);
  test_elementwise(1024);
  test_histogram(1024);
  test_merge_attn_states(512, 16, 128);
  test_softmax(256);
  test_rms_norm(8, 128);
  test_layer_norm(8, 128);
  test_rope(8, 128);
  test_mat_transpose(256, 256);
  test_mat_transpose_padded(256, 256);
  test_sgemv(256, 128);
  test_sgemm(M, N, K);
  test_hgemm_mma(M, N, K);
  test_hgemm_swizzle(M, N, K);
#if (defined(NOTES_V2_ENABLE_CUTE))
  test_hgemm_cute(M, N, K);
#endif
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  test_fp8_gemm(M, N, K);
#endif
#if (defined(NOTES_V2_ENABLE_WGMMA))
  test_hgemm_wgmma(M, N, K);
#endif
#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  test_hgemm_tma_mma_ws(M, N, K);
#endif
  test_flash_attn(1024, 64);
#if defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  test_flash_attn_tma_mma_ws(1024, 64);
  test_flash_attn_tma_mma_ws(1024, 128);
  test_flash_attn_3_tma_ws(1024, 64);
  test_flash_attn_3_tma_ws(1024, 128);
#endif
#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)
  // Phase 8: persist-D WS + persistent CTA + scale fused (dense 多 q-tile /
  // causal / GQA / 尾部 tile; Nq=2048 H=8 -> 128 tiles > 96 SM 触发多 iter)
  run_pd_cute_tests();
#endif

  printf("=== All tests done ===\n");
  return 0;
}

// =============================================================================
// Quick build & run reference
// =============================================================================
// 多 TU 并行编译（./build.sh 逐 TU 调 nvcc，ccache 可用则自动启用；
// base.cuh 的非模板 kernel 会进入多个 TU，build.sh 用 objcopy weaken 去重，
// 因此不要再直接用单条 nvcc 编全部——统一走 build.sh）：
//
//   ./build.sh --arch sm_120a --jobs 8   # Blackwell (RTX 5090 / PRO 5000/6000)
//   ./build.sh --arch sm_90a --jobs 8    # Hopper (H100/H200, WGMMA)
//   ./build.sh --arch sm_89 --jobs 8     # Ada (RTX 40 系列)
//   ./build.sh --arch sm_86 --jobs 8     # Ampere (RTX 30 系列)
//   ./build.sh --arch all --jobs 8       # 全部预定义 arch
//   ./build.sh --arch sm_XX --jobs 8     # 通用 arch（无 NOTES_V2_XXX 宏）
//
// 常用运行入口：
//   ./bin/notes_v2_sm120a.bin                          # 全量 verification
//   ./bin/notes_v2_sm120a.bin --bench --mnk 8192       # HGEMM/FP8/FP4/FA bench
//   ./bin/notes_v2_sm120a.bin --bench-hgemm-all --mnk 4096,4096,4096
//   ./bin/notes_v2_sm120a.bin --bench-fa-all --bhnd 1,48,4096,64
//   ./bin/notes_v2_sm120a.bin --pd-cute                # persist-D 正确性
//   ./bin/notes_v2_sm120a.bin --sdnw-cute              # split-D non-WS 正确性
//   ./bin/notes_v2_sm120a.bin --fp8-gemm-sweep --mnk 4096,4096,4096
//   ./bin/notes_v2_sm120a.bin --fp4-gemm-sweep --mnk 4096,4096,4096
//   ./bin/notes_v2_sm120a.bin --swizzle-eq-check       # swizzle v1/v2 等价性
