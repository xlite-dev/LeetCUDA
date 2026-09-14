// book/tests/ch19_ffpa_split_d.cu — ch19 最小测试：FFPA Split-D（大 head_dim 分块注意力）
// 测试逻辑抽取自 kernels/interview/bench_ffpa.cu（TMA+WS 版 L205 起、cp.async 版 L329 起），
// 参考实现由 cuDNN SDPA 换成 CPU fp64：softmax(QK^T / sqrt(D)) V，double 全程累加。
// 覆盖（两个 kernel 共用同一份参考，各自对照一次）：
//   正确性: D=64（单 chunk 退化）/ 128（2 chunk）/ 192（3 chunk，chunk 数与 Sk=2 互质 →
//           奇偶相位交错）/ 512（8 chunk，O 累加器 256 reg/thread 的寄存器极值）
//   规模  : H=2，N=256（D=64/128）、192（D=192）、128（D=512），均满足 N % 64 == 0
//   bench  : B=1 H=32，(D,N) ∈ {(128,4096), (512,4096), (512,8192)}，cp.async 与 TMA+WS 对照
// 约束：seqlen % 64 == 0 且 head_dim % 64 == 0（未对齐打印 SKIP）；正确性规模 <= 512
//   （BOOK_TEST_MAX_N，CPU fp64 三重循环时限）。
// 链接说明：TMA 版的 descriptor 由 CuTe make_tma_copy 构造，其 cuTensorMapEncodeTiled 调用
//   走 CUTLASS_CUDA_DRIVER_WRAPPER_CALL（运行时解析），build_tests.sh 不链接 -lcuda 亦可。
// include 顺序：flash_attn.cuh 依赖 hgemm.cuh L388 的 swizzle 派发器（notes-v2.cu L23-27 同序）。
#define NOTES_V2_ENABLE_CUTE 1
#define NOTES_V2_ENABLE_TMA_MMA_WS 1
#include "../../hgemm.cuh"
#include "../../flash_attn.cuh"
#include "../../ffpa_attn.cuh"
#include "common_test.h"
#include <string>
#include <vector>

using namespace cute;

// CPU fp64 参考：O = softmax(Q K^T / sqrt(D)) V，逐 (b,h,q) 行直接算（packed BHND 布局）
static void ffpa_ref_fp64(const std::vector<half> &hq, const std::vector<half> &hk,
                          const std::vector<half> &hv, std::vector<double> &ref,
                          int B, int H, int N, int D) {
  const double scale = 1.0 / sqrt((double)D);
  std::vector<double> s(N), p(N);
  for (int bi = 0; bi < B * H; ++bi) {
    const size_t base = size_t(bi) * N * D;
    for (int qi = 0; qi < N; ++qi) {
      double smax = -INFINITY;
      for (int kj = 0; kj < N; ++kj) {
        double acc = 0.0;
        for (int d = 0; d < D; ++d)
          acc += __half2float(hq[base + size_t(qi) * D + d]) *
                 __half2float(hk[base + size_t(kj) * D + d]);
        s[kj] = acc * scale;
        if (s[kj] > smax) smax = s[kj];
      }
      double sum = 0.0;
      for (int kj = 0; kj < N; ++kj) {
        p[kj] = exp(s[kj] - smax);
        sum += p[kj];
      }
      for (int d = 0; d < D; ++d) {
        double acc = 0.0;
        for (int kj = 0; kj < N; ++kj)
          acc += p[kj] * __half2float(hv[base + size_t(kj) * D + d]);
        ref[base + size_t(qi) * D + d] = acc / sum;
      }
    }
  }
}

// 一次 launch + D2H + 对照；返回 false 表示 SKIP（arch / smem 不可行）
template <int kHeadDim, int Sk, int Sv>
static bool run_async(half *d_q, half *d_k, half *d_v, half *d_o,
                      const std::vector<double> &ref, int rows, int seqlen, int H) {
  using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  constexpr int kBr = 64, kBc = 64;
  const size_t smem_bytes =
      (size_t(Sk) * (cosize(SmemLayoutQ{}) + cosize(SmemLayoutKV{})) +
       size_t(Sv) * cosize(SmemLayoutKV{})) * sizeof(cutlass::half_t);
  auto fk = ffpa_split_d_cute<kHeadDim, Sk, Sv>;
  int dev = 0, max_smem = 0;
  cudaFuncAttributes attr{};
  BOOK_CUDA_CHECK(cudaGetDevice(&dev));
  BOOK_CUDA_CHECK(
      cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  BOOK_CUDA_CHECK(cudaFuncGetAttributes(&attr, fk));
  if (smem_bytes + attr.sharedSizeBytes > size_t(max_smem)) {
    printf("SKIP(ch19): cp.async D=%d smem %zu B > optin %d B\n", kHeadDim,
           smem_bytes, max_smem);
    return false;
  }
  BOOK_CUDA_CHECK(cudaFuncSetAttribute(
      fk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  dim3 grid(seqlen / kBr, H), block(128);  // 128 = 4 warp，与 Layout<4,1,1> 对齐
  fk<<<grid, block, smem_bytes>>>(reinterpret_cast<cutlass::half_t *>(d_q),
                                  reinterpret_cast<cutlass::half_t *>(d_k),
                                  reinterpret_cast<cutlass::half_t *>(d_v),
                                  reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  const size_t count = size_t(rows) * kHeadDim;
  std::vector<half> h_o(count);
  BOOK_CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, count * sizeof(half), cudaMemcpyDeviceToHost));
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) out[i] = __half2float(h_o[i]);
  char name[96];
  snprintf(name, sizeof(name), "ffpa split-d async D=%d N=%d Sk=%d Sv=%d", kHeadDim,
           seqlen, Sk, Sv);
  book_check(out.data(), ref.data(), (int)count, TOL_F16ACC, name);
  return true;
}

// TMA+WS 版：descriptor 由 make_tma_copy 构造（box = 64x64，与 SmemLayout 的 SW128 tile 一致）
template <int kHeadDim, int Sk, int Sv>
static bool run_tma(half *d_q, half *d_k, half *d_v, half *d_o,
                    const std::vector<double> &ref, int rows, int seqlen, int H) {
  using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  constexpr int kBr = 64, kBc = 64;
  auto make_tma_q = [=]() {
    auto tensor = make_tensor(make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(d_q)),
                              make_shape(rows, Int<kHeadDim>{}),
                              make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, SmemLayoutQ{}, Shape<_64, _64>{}, _1{});
  };
  auto make_tma_kv = [=](half *pointer) {
    auto tensor = make_tensor(make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(pointer)),
                              make_shape(rows, Int<kHeadDim>{}),
                              make_stride(Int<kHeadDim>{}, _1{}));
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, SmemLayoutKV{}, Shape<_64, _64>{}, _1{});
  };
  auto tma_q = make_tma_q();
  auto tma_k = make_tma_kv(d_k);
  auto tma_v = make_tma_kv(d_v);
  auto fk = ffpa_attn_tma_mma_ws_split_d_cute<kHeadDim, decltype(tma_q), decltype(tma_k),
                                              decltype(tma_v), Sk, Sv>;
  const size_t smem_bytes =
      (size_t(Sk) * (cosize(SmemLayoutQ{}) + cosize(SmemLayoutKV{})) +
       size_t(Sv) * cosize(SmemLayoutKV{})) * sizeof(cutlass::half_t);
  int dev = 0, max_smem = 0;
  cudaFuncAttributes attr{};
  BOOK_CUDA_CHECK(cudaGetDevice(&dev));
  BOOK_CUDA_CHECK(
      cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  BOOK_CUDA_CHECK(cudaFuncGetAttributes(&attr, fk));
  if (smem_bytes + attr.sharedSizeBytes > size_t(max_smem)) {
    printf("SKIP(ch19): tma ws D=%d smem %zu B > optin %d B\n", kHeadDim, smem_bytes,
           max_smem);
    return false;
  }
  BOOK_CUDA_CHECK(cudaFuncSetAttribute(
      fk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  dim3 grid(seqlen / kBr, H), block(256);  // 128 producer + 128 consumer
  fk<<<grid, block, smem_bytes>>>(tma_q, tma_k, tma_v,
                                  reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
  BOOK_CUDA_CHECK(cudaGetLastError());
  BOOK_CUDA_CHECK(cudaDeviceSynchronize());
  const size_t count = size_t(rows) * kHeadDim;
  std::vector<half> h_o(count);
  BOOK_CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, count * sizeof(half), cudaMemcpyDeviceToHost));
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) out[i] = __half2float(h_o[i]);
  char name[96];
  snprintf(name, sizeof(name), "ffpa split-d tma ws D=%d N=%d Sk=%d Sv=%d", kHeadDim,
           seqlen, Sk, Sv);
  book_check(out.data(), ref.data(), (int)count, TOL_F16ACC, name);
  return true;
}

// 正确性 case：同一份 CPU 参考跑两个 kernel（各 Sk=2/Sv=2），并打印切分常量
template <int kHeadDim>
static void run_case(int H, int seqlen) {
  using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<kHeadDim>;
  using OFrag =
      decltype(partition_fragment_C(typename Traits::TiledMmaPV{}, Shape<_64, _64>{}));
  constexpr int kOElemsPerFrag = decltype(size(OFrag{}))::value;
  constexpr int kDChunks = kHeadDim / 64;
  const size_t count = size_t(H) * seqlen * kHeadDim;
  printf("D=%d: d_chunk=64 x %d chunks, O_acc=%d f32/thread/chunk (total %d), "
         "rows=%zu, N=%d\n",
         kHeadDim, kDChunks, kOElemsPerFrag, kDChunks * kOElemsPerFrag,
         size_t(H) * seqlen, seqlen);

  std::vector<half> hq(count), hk(count), hv(count);
  srand(42);
  for (size_t i = 0; i < count; ++i) {
    hq[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hk[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    hv[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
  }
  std::vector<double> ref(count);
  ffpa_ref_fp64(hq, hk, hv, ref, 1, H, seqlen, kHeadDim);

  const int rows = H * seqlen;
  half *d_q, *d_k, *d_v, *d_o;
  BOOK_CUDA_CHECK(cudaMalloc(&d_q, count * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_k, count * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_v, count * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMalloc(&d_o, count * sizeof(half)));
  BOOK_CUDA_CHECK(cudaMemcpy(d_q, hq.data(), count * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_k, hk.data(), count * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_v, hv.data(), count * sizeof(half), cudaMemcpyHostToDevice));

  if (book_require_sm(80, "ch19"))
    run_async<kHeadDim, 2, 2>(d_q, d_k, d_v, d_o, ref, rows, seqlen, H);
  if (book_require_sm(90, "ch19-tma"))
    run_tma<kHeadDim, 2, 2>(d_q, d_k, d_v, d_o, ref, rows, seqlen, H);

  BOOK_CUDA_CHECK(cudaFree(d_q));
  BOOK_CUDA_CHECK(cudaFree(d_k));
  BOOK_CUDA_CHECK(cudaFree(d_v));
  BOOK_CUDA_CHECK(cudaFree(d_o));
}

// bench：单配置 warmup+iters 取均值（FLOPS = 4*B*H*N^2*D），两 kernel 各测一次
template <int kHeadDim, int Sk, int Sv>
static double bench_one(int H, int seqlen, int warmup, int iters, const char *tag,
                        bool use_tma) {
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
  BOOK_CUDA_CHECK(cudaMemcpy(d_q, hq.data(), count * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_k, hk.data(), count * sizeof(half), cudaMemcpyHostToDevice));
  BOOK_CUDA_CHECK(cudaMemcpy(d_v, hv.data(), count * sizeof(half), cudaMemcpyHostToDevice));

  using Traits = fa_cute::FFPAAttnSplitDCuTeTraits<kHeadDim>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  const size_t smem_bytes =
      (size_t(Sk) * (cosize(SmemLayoutQ{}) + cosize(SmemLayoutKV{})) +
       size_t(Sv) * cosize(SmemLayoutKV{})) * sizeof(cutlass::half_t);
  const int rows = B * H * seqlen;
  dim3 grid(seqlen / 64, B * H);

  cudaEvent_t beg, end;
  BOOK_CUDA_CHECK(cudaEventCreate(&beg));
  BOOK_CUDA_CHECK(cudaEventCreate(&end));
  double ms = 0.0;
  if (use_tma) {
    auto make_tma_q = [=]() {
      auto tensor = make_tensor(make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(d_q)),
                                make_shape(rows, Int<kHeadDim>{}),
                                make_stride(Int<kHeadDim>{}, _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, tensor, SmemLayoutQ{}, Shape<_64, _64>{}, _1{});
    };
    auto make_tma_kv = [=](half *pointer) {
      auto tensor = make_tensor(make_gmem_ptr(reinterpret_cast<cutlass::half_t *>(pointer)),
                                make_shape(rows, Int<kHeadDim>{}),
                                make_stride(Int<kHeadDim>{}, _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, tensor, SmemLayoutKV{}, Shape<_64, _64>{}, _1{});
    };
    auto tma_q = make_tma_q();
    auto tma_k = make_tma_kv(d_k);
    auto tma_v = make_tma_kv(d_v);
    auto fk = ffpa_attn_tma_mma_ws_split_d_cute<kHeadDim, decltype(tma_q), decltype(tma_k),
                                                decltype(tma_v), Sk, Sv>;
    BOOK_CUDA_CHECK(cudaFuncSetAttribute(
        fk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    for (int w = 0; w < warmup; ++w)
      fk<<<grid, 256, smem_bytes>>>(tma_q, tma_k, tma_v,
                                    reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
    BOOK_CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i)
      fk<<<grid, 256, smem_bytes>>>(tma_q, tma_k, tma_v,
                                    reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
    BOOK_CUDA_CHECK(cudaEventRecord(end));
  } else {
    auto fk = ffpa_split_d_cute<kHeadDim, Sk, Sv>;
    BOOK_CUDA_CHECK(cudaFuncSetAttribute(
        fk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    for (int w = 0; w < warmup; ++w)
      fk<<<grid, 128, smem_bytes>>>(reinterpret_cast<cutlass::half_t *>(d_q),
                                    reinterpret_cast<cutlass::half_t *>(d_k),
                                    reinterpret_cast<cutlass::half_t *>(d_v),
                                    reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
    BOOK_CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i)
      fk<<<grid, 128, smem_bytes>>>(reinterpret_cast<cutlass::half_t *>(d_q),
                                    reinterpret_cast<cutlass::half_t *>(d_k),
                                    reinterpret_cast<cutlass::half_t *>(d_v),
                                    reinterpret_cast<cutlass::half_t *>(d_o), rows, seqlen);
    BOOK_CUDA_CHECK(cudaEventRecord(end));
  }
  BOOK_CUDA_CHECK(cudaEventSynchronize(end));
  float elapsed = 0.0f;
  BOOK_CUDA_CHECK(cudaEventElapsedTime(&elapsed, beg, end));
  ms = double(elapsed) / iters;

  const double tflops = 4.0 * B * H * (double)seqlen * seqlen * kHeadDim /
                        (ms * 1e-3) / 1e12;
  printf("| %-34s | D=%-3d N=%-5d Sk=%d Sv=%d | %8.3f ms | %7.1f TFLOPS |\n", tag,
         kHeadDim, seqlen, Sk, Sv, ms, tflops);

  BOOK_CUDA_CHECK(cudaEventDestroy(beg));
  BOOK_CUDA_CHECK(cudaEventDestroy(end));
  BOOK_CUDA_CHECK(cudaFree(d_q));
  BOOK_CUDA_CHECK(cudaFree(d_k));
  BOOK_CUDA_CHECK(cudaFree(d_v));
  BOOK_CUDA_CHECK(cudaFree(d_o));
  return tflops;
}

static int bench() {
  const int H = 32, warmup = 2, iters = 5;
  cudaDeviceProp prop;
  BOOK_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d), B=1 H=%d, warmup=%d iters=%d (avg), FLOPS=4*B*H*N^2*D\n",
         prop.name, prop.major, prop.minor, H, warmup, iters);
  for (int n : {4096, 8192}) {
    if (n == 4096) bench_one<128, 2, 2>(H, n, warmup, iters, "cp.async", false);
    bench_one<512, 2, 2>(H, n, warmup, iters, "cp.async", false);
    if (n == 4096) bench_one<128, 2, 2>(H, n, warmup, iters, "tma ws", true);
    bench_one<512, 1, 1>(H, n, warmup, iters, "tma ws", true);
    bench_one<512, 2, 2>(H, n, warmup, iters, "tma ws", true);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "bench") return bench();
  int dev = 0, optin = 0;
  BOOK_CUDA_CHECK(cudaGetDevice(&dev));
  BOOK_CUDA_CHECK(
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  cudaDeviceProp prop;
  BOOK_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d), max shared memory per block (optin): %d B (%.1f KB)\n",
         prop.name, prop.major, prop.minor, optin, optin / 1024.0);
  run_case<64>(2, 256);   // 单 chunk 退化：kDChunks=1 < Sk=2
  run_case<128>(2, 256);  // 2 chunk：chunk 与 stage 整除
  run_case<192>(2, 192);  // 3 chunk：与 Sk=2 互质，相位交错
  run_case<512>(2, 128);  // 8 chunk：O 累加器寄存器极值
  return g_failures == 0 ? 0 : 1;
}
