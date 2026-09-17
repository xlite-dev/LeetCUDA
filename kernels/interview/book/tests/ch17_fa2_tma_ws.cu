// book/tests/ch17_fa2_tma_ws.cu — ch17 最小测试：FA2 TMA + Warp Specialization
// 测试逻辑抽取自 notes-v2.cu test_flash_attn_tma_mma_ws_impl(L2041)，参考实现由
// cuDNN SDPA 换成 CPU fp64（softmax(QK^T/sqrt(d))V，double 累加）。
// 覆盖：
//   正确性: D=64/128 × seqlen=128/512 × F16Acc/F32Acc（kStagesK=2, kStagesV=1）
//   bench  : B=1,H=32,N=4096/16384,D=64/128（Sk=1/2/3, Sv=1），TFLOPS（bench 子命令）
// 约束：seqlen % Br(128) == 0 且 seqlen % Bc(64) == 0（v1 不支持尾 tile）；
//       正确性规模 <= 512（BOOK_TEST_MAX_N，CPU fp64 三重循环时限）。
// 链接说明：cuTensorMapEncodeTiled 是 driver API，build_tests.sh 不带 -lcuda，
//   故经 dlopen("libcuda.so.1") + dlsym 运行时解析；descriptor 字段填法与
//   common.cuh create_tensor_map(L733) 逐字段一致（SWIZZLE_128B、box=(64, major)）。
// include 顺序：flash_attn.cuh 依赖 hgemm.cuh L390 的 swizzle()（源码既有依赖，
//   notes-v2.cu L23-27 同序），故先 hgemm.cuh 再 flash_attn.cuh。
#define NOTES_V2_ENABLE_TMA_MMA_WS 1
#include "../../hgemm.cuh"
#include "../../flash_attn.cuh"
#include "common_test.h"
#include <dlfcn.h>
#include <string>
#include <vector>

// ---- TMA descriptor via dlopen（避免链接期 -lcuda）----
static void *g_libcuda = nullptr;

static CUresult encode_tiled_dl(CUtensorMap *tma_map, half *gmem_ptr,
                                int blocks_height, int blocks_width,
                                int major, int minor) {
  using EncodeFn = decltype(&cuTensorMapEncodeTiled);
  static EncodeFn enc = nullptr;
  if (!enc) {
    if (!g_libcuda) {
      g_libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
      if (!g_libcuda) g_libcuda = dlopen("libcuda.so", RTLD_LAZY);
    }
    enc = reinterpret_cast<EncodeFn>(dlsym(g_libcuda, "cuTensorMapEncodeTiled"));
  }
  uint64_t shape[5] = {(uint64_t)minor * blocks_width,
                       (uint64_t)major * blocks_height, 1, 1, 1};
  uint64_t stride[5] = {sizeof(half), sizeof(half) * minor * blocks_width, 0, 0, 0};
  uint32_t box[5] = {uint32_t(minor), uint32_t(major), 1, 1, 1};
  uint32_t box_stride[5] = {1, 1, 1, 1, 1};
  return enc(tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (void *)gmem_ptr,
             shape, stride + 1, box, box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

static CUtensorMap *create_tma_dl(half *src, int blocks_height, int blocks_width,
                                  int major, int minor) {
  CUtensorMap *d_map;
  BOOK_CUDA_CHECK(cudaMalloc(&d_map, sizeof(CUtensorMap)));
  CUtensorMap h_map;
  CUresult r = encode_tiled_dl(&h_map, src, blocks_height, blocks_width, major, minor);
  if (r != CUDA_SUCCESS) {
    printf("cuTensorMapEncodeTiled failed: %d\n", (int)r);
    exit(1);
  }
  BOOK_CUDA_CHECK(cudaMemcpy(d_map, &h_map, sizeof(CUtensorMap),
                             cudaMemcpyHostToDevice));
  return d_map;
}

// ---- tile 配置（与 notes-v2 test L2046-L2054 同参）----
static constexpr int kMmaTileQ = 8, kMmaTileK = 1;
static constexpr int kValTileQ = 1, kValTileK = 8, kValTileP = 1;
static constexpr int kBr = 16 * kMmaTileQ * kValTileQ;  // 128
static constexpr int kBc = 8 * kMmaTileK * kValTileK;   // 64
static constexpr int kTmaBoxMinor = 64;
static constexpr int kNumThreads = 384;  // 128 producer + 256 consumer

// 单次正确性运行：launch + D2H + fp64 对照。返回 false 表示 SMEM SKIP。
template <int kHeadDim, int kMmaAccF32, int Sk, int Sv>
static bool run_fa(half *d_q, half *d_k, half *d_v, half *d_o,
                   const std::vector<double> &ref, int seqlen, int H,
                   CUtensorMap *tq, CUtensorMap *tk, CUtensorMap *tv) {
  auto fk = flash_attn_tma_mma_ws_stages_split_q<
      kHeadDim, 16, 8, 16, kMmaAccF32, kMmaTileQ, kMmaTileK, kMmaTileQ, 1,
      kValTileQ, kValTileK, kValTileP, kHeadDim / 8, Sk, Sv, kNumThreads>;
  const size_t count = size_t(ref.size());
  const size_t smem_bytes =
      (size_t(kBr) * kHeadDim + size_t(Sk) * kBc * kHeadDim +
       size_t(Sv) * kBc * kHeadDim) * sizeof(half);
  int dev = 0, max_smem = 0;
  cudaFuncAttributes attr{};
  BOOK_CUDA_CHECK(cudaGetDevice(&dev));
  BOOK_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  BOOK_CUDA_CHECK(cudaFuncGetAttributes(&attr, fk));
  if (smem_bytes + attr.sharedSizeBytes > size_t(max_smem)) {
    printf("SKIP(ch17): smem %zu B > optin %d B\n", smem_bytes, max_smem);
    return false;
  }
  BOOK_CUDA_CHECK(cudaFuncSetAttribute(
      fk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  dim3 grid(seqlen / kBr, H), block(kNumThreads);
  fk<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, seqlen, H, tq, tk, tv);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<half> h_o(count);
  BOOK_CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, count * sizeof(half),
                             cudaMemcpyDeviceToHost));
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) out[i] = __half2float(h_o[i]);
  char name[96];
  snprintf(name, sizeof(name), "fa2_tma_ws D=%d N=%d Sk=%d Sv=%d %s", kHeadDim,
           seqlen, Sk, Sv, kMmaAccF32 ? "F32Acc" : "F16Acc");
  book_check(out.data(), ref.data(), (int)count,
             kMmaAccF32 ? TOL_F32ACC : TOL_F16ACC, name);
  return true;
}

// 正确性 case：CPU fp64 参考一次，F16Acc/F32Acc 两档实例化各跑一次
template <int kHeadDim>
static void run_case(int seqlen) {
  const int B = 1, H = 8;
  const size_t count = size_t(B) * H * seqlen * kHeadDim;
  if (seqlen % kBr != 0 || seqlen % kBc != 0) {
    printf("SKIP(ch17): seqlen %d not aligned to Br=%d/Bc=%d\n", seqlen, kBr,
           kBc);
    return;
  }
  std::vector<half> hq(count), hk(count), hv(count);
  srand(42);
  for (size_t i = 0; i < count; ++i) {
    hq[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hk[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hv[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }
  // CPU fp64 参考：softmax(QK^T * scale) V
  std::vector<double> q(count), k(count), v(count), ref(count);
  for (size_t i = 0; i < count; ++i) {
    q[i] = __half2float(hq[i]);
    k[i] = __half2float(hk[i]);
    v[i] = __half2float(hv[i]);
  }
  const double scale = 1.0 / sqrt((double)kHeadDim);
  std::vector<double> s(seqlen), p(seqlen);
  for (int bi = 0; bi < B * H; ++bi)
    for (int qi = 0; qi < seqlen; ++qi) {
      double smax = -INFINITY;
      for (int kj = 0; kj < seqlen; ++kj) {
        double acc = 0.0;
        for (int d = 0; d < kHeadDim; ++d)
          acc += q[bi * seqlen * kHeadDim + qi * kHeadDim + d] *
                 k[bi * seqlen * kHeadDim + kj * kHeadDim + d];
        s[kj] = acc * scale;
        if (s[kj] > smax) smax = s[kj];
      }
      double sum = 0.0;
      for (int kj = 0; kj < seqlen; ++kj) {
        p[kj] = exp(s[kj] - smax);
        sum += p[kj];
      }
      for (int d = 0; d < kHeadDim; ++d) {
        double o = 0.0;
        for (int kj = 0; kj < seqlen; ++kj)
          o += p[kj] * v[bi * seqlen * kHeadDim + kj * kHeadDim + d];
        ref[bi * seqlen * kHeadDim + qi * kHeadDim + d] = o / sum;
      }
    }

  half *d_q, *d_k, *d_v, *d_o;
  BOOK_CUDA_CHECK(cudaMalloc(&d_q, count * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_k, count * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_v, count * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_o, count * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_q, hq.data(), count * sizeof(half),
                             cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_k, hk.data(), count * sizeof(half),
                             cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_v, hv.data(), count * sizeof(half),
                             cudaMemcpyHostToDevice));
  // box innermost 固定 64 half(128B)；D=128 时沿 head_dim 发 2 次 TMA
  CUtensorMap *tq = create_tma_dl(d_q, B * H * seqlen / kBr,
                                  kHeadDim / kTmaBoxMinor, kBr, kTmaBoxMinor);
  CUtensorMap *tk = create_tma_dl(d_k, B * H * seqlen / kBc,
                                  kHeadDim / kTmaBoxMinor, kBc, kTmaBoxMinor);
  CUtensorMap *tv = create_tma_dl(d_v, B * H * seqlen / kBc,
                                  kHeadDim / kTmaBoxMinor, kBc, kTmaBoxMinor);

  run_fa<kHeadDim, 0, 2, 1>(d_q, d_k, d_v, d_o, ref, seqlen, H, tq, tk, tv);
  run_fa<kHeadDim, 1, 2, 1>(d_q, d_k, d_v, d_o, ref, seqlen, H, tq, tk, tv);

  BOOK_CUDA_CHECK(cudaFree(d_q));
  BOOK_CUDA_CHECK(cudaFree(d_k));
  BOOK_CUDA_CHECK(cudaFree(d_v));
  BOOK_CUDA_CHECK(cudaFree(d_o));
  BOOK_CUDA_CHECK(cudaFree(tq));
  BOOK_CUDA_CHECK(cudaFree(tk));
  BOOK_CUDA_CHECK(cudaFree(tv));
}

// bench：单配置 warmup+iters 平均延迟，返回 TFLOPS（FLOPS = 4*B*H*N^2*D）
template <int kHeadDim, int kMmaAccF32, int Sk, int Sv = 1>
static double bench_one(int H, int seqlen, int warmup, int iters) {
  const int B = 1;
  const size_t count = size_t(B) * H * seqlen * kHeadDim;
  std::vector<half> hq(count), hk(count), hv(count);
  srand(42);
  for (size_t i = 0; i < count; ++i) {
    hq[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hk[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hv[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }
  half *d_q, *d_k, *d_v, *d_o;
  BOOK_CUDA_CHECK(cudaMalloc(&d_q, count * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_k, count * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_v, count * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_o, count * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_q, hq.data(), count * sizeof(half),
                             cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_k, hk.data(), count * sizeof(half),
                             cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_v, hv.data(), count * sizeof(half),
                             cudaMemcpyHostToDevice));
  CUtensorMap *tq = create_tma_dl(d_q, B * H * seqlen / kBr,
                                  kHeadDim / kTmaBoxMinor, kBr, kTmaBoxMinor);
  CUtensorMap *tk = create_tma_dl(d_k, B * H * seqlen / kBc,
                                  kHeadDim / kTmaBoxMinor, kBc, kTmaBoxMinor);
  CUtensorMap *tv = create_tma_dl(d_v, B * H * seqlen / kBc,
                                  kHeadDim / kTmaBoxMinor, kBc, kTmaBoxMinor);

  auto fk = flash_attn_tma_mma_ws_stages_split_q<
      kHeadDim, 16, 8, 16, kMmaAccF32, kMmaTileQ, kMmaTileK, kMmaTileQ, 1,
      kValTileQ, kValTileK, kValTileP, kHeadDim / 8, Sk, Sv, kNumThreads>;
  const size_t smem_bytes =
      (size_t(kBr) * kHeadDim + size_t(Sk) * kBc * kHeadDim +
       size_t(Sv) * kBc * kHeadDim) * sizeof(half);
  BOOK_CUDA_CHECK(cudaFuncSetAttribute(
      fk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  dim3 grid(seqlen / kBr, B * H), block(kNumThreads);
  cudaEvent_t beg, end;
  BOOK_CUDA_CHECK(cudaEventCreate(&beg));
  BOOK_CUDA_CHECK(cudaEventCreate(&end));
  for (int w = 0; w < warmup; ++w)
    fk<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, seqlen, H, tq, tk, tv);
  BOOK_CUDA_CHECK(cudaEventRecord(beg));
  for (int i = 0; i < iters; ++i)
    fk<<<grid, block, smem_bytes>>>(d_q, d_k, d_v, d_o, seqlen, H, tq, tk, tv);
  BOOK_CUDA_CHECK(cudaEventRecord(end));
  BOOK_CUDA_CHECK(cudaEventSynchronize(end));
  float ms = 0.0f;
  BOOK_CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
  ms /= iters;
  double tflops =
      4.0 * B * H * (double)seqlen * seqlen * kHeadDim / (double(ms) * 1e-3) / 1e12;
  BOOK_CUDA_CHECK(cudaEventDestroy(beg));
  BOOK_CUDA_CHECK(cudaEventDestroy(end));
  BOOK_CUDA_CHECK(cudaFree(d_q));
  BOOK_CUDA_CHECK(cudaFree(d_k));
  BOOK_CUDA_CHECK(cudaFree(d_v));
  BOOK_CUDA_CHECK(cudaFree(d_o));
  BOOK_CUDA_CHECK(cudaFree(tq));
  BOOK_CUDA_CHECK(cudaFree(tk));
  BOOK_CUDA_CHECK(cudaFree(tv));
  return tflops;
}

static int bench() {
  const int H = 32, warmup = 5, iters = 20;
  cudaDeviceProp prop;
  BOOK_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d), B=1 H=%d, warmup=%d iters=%d (avg), FLOPS=4*B*H*N^2*D\n",
         prop.name, prop.major, prop.minor, H, warmup, iters);
  struct Cfg { const char *tag; int d; int n; double t; };
  Cfg cfgs[] = {
      {"Sk=1 Sv=1 F16Acc", 128, 4096, 0}, {"Sk=2 Sv=1 F16Acc", 128, 4096, 0},
      {"Sk=3 Sv=1 F16Acc", 128, 4096, 0}, {"Sk=2 Sv=1 F32Acc", 128, 4096, 0},
      {"Sk=3 Sv=1 F32Acc", 128, 4096, 0}, {"Sk=2 Sv=1 F16Acc", 128, 16384, 0},
      {"Sk=3 Sv=1 F16Acc", 128, 16384, 0}, {"Sk=2 Sv=1 F32Acc", 128, 16384, 0},
      {"Sk=2 Sv=1 F16Acc", 64, 4096, 0},  {"Sk=3 Sv=1 F16Acc", 64, 4096, 0},
      {"Sk=2 Sv=1 F16Acc", 64, 16384, 0},
  };
  for (auto &c : cfgs) {
    if (c.d == 128 && strstr(c.tag, "F16Acc")) {
      if (!strcmp(c.tag, "Sk=1 Sv=1 F16Acc")) c.t = bench_one<128, 0, 1>(H, c.n, warmup, iters);
      else if (!strcmp(c.tag, "Sk=2 Sv=1 F16Acc")) c.t = bench_one<128, 0, 2>(H, c.n, warmup, iters);
      else c.t = bench_one<128, 0, 3>(H, c.n, warmup, iters);
    } else if (c.d == 128) {
      c.t = !strcmp(c.tag, "Sk=2 Sv=1 F32Acc") ? bench_one<128, 1, 2>(H, c.n, warmup, iters)
                                               : bench_one<128, 1, 3>(H, c.n, warmup, iters);
    } else {
      c.t = !strcmp(c.tag, "Sk=2 Sv=1 F16Acc") ? bench_one<64, 0, 2>(H, c.n, warmup, iters)
                                               : bench_one<64, 0, 3>(H, c.n, warmup, iters);
    }
    printf("| %-34s | D=%-3d N=%-5d | %7.1f TFLOPS |\n", c.tag, c.d, c.n, c.t);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "bench") return bench();
  if (!book_require_sm(90, "ch17")) return 0;  // TMA/mbarrier needs sm_90+
  int dev = 0, optin = 0;
  BOOK_CUDA_CHECK(cudaGetDevice(&dev));
  BOOK_CUDA_CHECK(cudaDeviceGetAttribute(
      &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  printf("max shared memory per block (optin): %d B (%.1f KB)\n", optin,
         optin / 1024.0);
  run_case<64>(128);   // Tc=2:  stage 回绕 0->1
  run_case<64>(512);   // Tc=8:  3 轮完整回绕 + epilogue
  run_case<128>(128);  // kTmaChunks=2 路径（chunk-major swizzle）
  run_case<128>(512);
  return g_failures == 0 ? 0 : 1;
}
