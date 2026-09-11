// book/tests/ch01_roofline.cu — ch01 host 侧算术强度（AI）与 roofline 定位演示（RFC-C1）
// 无 GPU kernel（ch01 为原理章）：host 计算典型算子的 FLOPs/Bytes/AI，与解析期望
// 常数核对（PASS/FAIL），并打印本机 GPU 资源与理论带宽，供正文 roofline 定位引用。
#include "common_test.h"

// 访存口径（fp32，各数据只读写一次）：GEMM 读 A/B 写 C；GEMV 读 A/x 写 y；
// softmax/eadd/relu/dot 逐元素读+写，中间量（max/exp/part-sum）驻留寄存器。
static int check_ai(const char* name, double flops, double bytes, double expect,
                    double tol_rel) {
  double ai = flops / bytes;
  double rel = fabs(ai - expect) / expect;
  bool pass = rel <= tol_rel;
  printf("%s %-8s FLOPs=%.4e Bytes=%.4e AI=%9.4f (expect %9.4f, rel_err=%.1e)\n",
         pass ? "PASS" : "FAIL", name, flops, bytes, ai, expect, rel);
  if (!pass) g_failures++;
  return 0;
}

int main() {
  printf("==== ch01 roofline: AI analytic check ====\n");
  const double M = 4096.0, K = 4096.0, N = 4096.0;

  // GEMM: FLOPs=2MNK, Bytes=(MK+KN+MN)*4；M=N=K=4096 时 AI=2*4096/12=682.67
  check_ai("GEMM", 2.0 * M * N * K, (M * K + K * N + M * N) * 4.0, 682.6667, 1e-3);
  // GEMV: FLOPs=2MK, Bytes=(MK+K+M)*4；M=K=4096 时 AI≈0.5
  check_ai("GEMV", 2.0 * M * K, (M * K + K + M) * 4.0, 0.5, 1e-3);
  // softmax(N): FLOPs=5N, Bytes=2N*4 → AI=5/8=0.625（与 base.cuh L85 口径一致）
  check_ai("softmax", 5.0 * N, 2.0 * N * 4.0, 0.625, 1e-9);
  // elementwise add: FLOPs=N, Bytes=3N*4 → AI=1/12≈0.0833
  check_ai("eadd", 1.0 * N, 3.0 * N * 4.0, 1.0 / 12.0, 1e-9);
  // relu: FLOPs=N, Bytes=2N*4 → AI=1/8=0.125
  check_ai("relu", 1.0 * N, 2.0 * N * 4.0, 0.125, 1e-9);
  // dot: FLOPs=2N, Bytes=2N*4 → AI=2/8=0.25
  check_ai("dot", 2.0 * N, 2.0 * N * 4.0, 0.25, 1e-9);

  // ---- 本机 GPU 资源速览（roofline 两轴刻度来源；估算口径见正文）----
  // CUDA 13 的 cudaDeviceProp 已移除 clockRate/memoryClockRate 等字段，改用
  // cudaDeviceGetAttribute 查询
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int clk_khz = 0, mem_clk_khz = 0, bus_bits = 0;
  cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
  cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, 0);
  cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
  printf("---- device ----\n");
  printf("name=%s sm_%d%d SMs=%d  global_mem=%.1f GB\n", prop.name, prop.major,
         prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / 1e9);
  printf("regs/SM=%d  smem-per-block-optin=%zu KB  L2=%d MB\n", prop.regsPerBlock,
         prop.sharedMemPerBlockOptin / 1024, prop.l2CacheSize >> 20);
  if (mem_clk_khz > 0 && bus_bits > 0) {
    double bw_gbs = mem_clk_khz * 1e3 * (bus_bits / 8.0) * 2.0 / 1e9;
    printf("mem-theoretical-BW ~ %.0f GB/s (clk %.0f MHz x bus %d-bit x2/DDR)\n",
           bw_gbs, mem_clk_khz / 1e3, bus_bits);
    // FP32 峰值粗估：每 SM 每 clk 128 FMA → 256 FLOP/clk/SM（Blackwell 桌面口径）
    double peak_tflops =
        clk_khz > 0
            ? prop.multiProcessorCount * 256.0 * (clk_khz / 1e9)
            : 0.0;
    if (peak_tflops > 0.0)
      printf("fp32-peak-estimate ~ %.1f TFLOPS → ridge-point ~ %.1f FLOP/Byte\n",
             peak_tflops, peak_tflops * 1e12 / (bw_gbs * 1e9));
  } else {
    printf("mem clk/bus attrs unavailable (0) on this driver\n");
  }
  printf("ch01 roofline demo done (failures=%d)\n", g_failures);
  return g_failures;
}
