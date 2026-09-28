// base.cu — notes-v2.cu 拆分出的 base 模块 TU，承载基础原语测试
// （Phase 1-5 host 侧正确性测试 + swizzle v1/v2 等价性检查；kernel 实现在 base.cuh/common.cuh）
#include "base.cuh"

// 共享符号：定义在 utils.cu（notes-v2.cu 多 TU 拆分）
extern bool g_debug;
extern bool g_verbose;
extern int g_warmup;
extern int g_repeat;
extern bool g_bench_fa3_cute_only;
extern bool g_fa_skip_check;
void check(cudaError_t err, const char *msg);
bool check_smem_feasible(const void *kernel_func, size_t dyn_smem_bytes);
float bench_hgemm_tflops(int M, int N, int K, float time_ms);
float bench_fa_tflops(int B, int H, int N, int D, float time_ms);
size_t fp8_smem_optin_limit();
float bench_cublas_bf16_gemm_tflops(cublasHandle_t handle, int M, int N, int K,
                                    __nv_bfloat16 *d_a, __nv_bfloat16 *d_b,
                                    __nv_bfloat16 *d_c);


void test_block_reduce(int N) {

  srand(42);
  float *h_a = (float *)malloc((size_t)N * sizeof(float));
  for (int i = 0; i < N; i++)
    h_a[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  // CPU reference: sum of all elements
  double ref = 0.0;
  for (int i = 0; i < N; i++) ref += (double)h_a[i];

  float *d_a, *d_y;
  check(cudaMalloc(&d_a, (size_t)N * sizeof(float)), "blockreduce alloc A");
  check(cudaMalloc(&d_y, sizeof(float)), "blockreduce alloc Y");

  check(cudaMemcpy(d_a, h_a, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "blockreduce H2D A");
  check(cudaMemset(d_y, 0, sizeof(float)), "blockreduce zero Y");

  dim3 block(128);
  dim3 grid((N + 127) / 128);
  block_reduce_all<128><<<grid, block>>>(d_a, d_y, N);
  check(cudaGetLastError(), "blockreduce launch");
  check(cudaDeviceSynchronize(), "blockreduce sync");

  float result;
  check(cudaMemcpy(&result, d_y, sizeof(float), cudaMemcpyDeviceToHost), "blockreduce D2H");

  float err = fabsf(result - (float)ref);
  printf("| %-56s | %.3e |\n", "BlockReduce", err);

  free(h_a);
  cudaFree(d_a); cudaFree(d_y);
}


void test_dot(int N) {

  srand(42);
  float *h_a = (float *)malloc((size_t)N * sizeof(float));
  float *h_b = (float *)malloc((size_t)N * sizeof(float));
  for (int i = 0; i < N; i++) {
    h_a[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    h_b[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }

  // CPU reference
  double ref = 0.0;
  for (int i = 0; i < N; i++) ref += (double)h_a[i] * (double)h_b[i];

  float *d_a, *d_b, *d_y;
  check(cudaMalloc(&d_a, (size_t)N * sizeof(float)), "dot alloc A");
  check(cudaMalloc(&d_b, (size_t)N * sizeof(float)), "dot alloc B");
  check(cudaMalloc(&d_y, sizeof(float)), "dot alloc Y");

  check(cudaMemcpy(d_a, h_a, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "dot H2D A");
  check(cudaMemcpy(d_b, h_b, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "dot H2D B");
  check(cudaMemset(d_y, 0, sizeof(float)), "dot zero Y");

  dim3 block(128);
  dim3 grid((N + 127) / 128);
  dot<128><<<grid, block>>>(d_a, d_b, d_y, N);
  check(cudaGetLastError(), "dot launch");
  check(cudaDeviceSynchronize(), "dot sync");

  float result;
  check(cudaMemcpy(&result, d_y, sizeof(float), cudaMemcpyDeviceToHost), "dot D2H");

  float err = fabsf(result - (float)ref);
  printf("| %-56s | %.3e |\n", "Dot", err);

  // ---- Dot Vec4 ----
  check(cudaMemset(d_y, 0, sizeof(float)), "dot_vec4 zero Y");
  dim3 block_v4(32);
  dot_vec4<32><<<grid, block_v4>>>(d_a, d_b, d_y, N);
  check(cudaGetLastError(), "dot_vec4 launch");
  check(cudaDeviceSynchronize(), "dot_vec4 sync");

  check(cudaMemcpy(&result, d_y, sizeof(float), cudaMemcpyDeviceToHost), "dot_vec4 D2H");
  float err_v4 = fabsf(result - (float)ref);
  printf("| %-56s | %.3e |\n", "Dot-Vec4", err_v4);

  free(h_a); free(h_b);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
}


void test_relu(int N) {
  srand(42);
  float *h_x = (float *)malloc((size_t)N * sizeof(float));
  float *h_y = (float *)malloc((size_t)N * sizeof(float));
  for (int i = 0; i < N; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, (size_t)N * sizeof(float)), "relu alloc X");
  check(cudaMalloc(&d_y, (size_t)N * sizeof(float)), "relu alloc Y");
  check(cudaMemcpy(d_x, h_x, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "relu H2D");

  for (int i = 0; i < N; i++) h_y[i] = fmaxf(0.0f, h_x[i]);
  dim3 block256(256);
  dim3 grid256((N + 255) / 256);
  relu<<<grid256, block256>>>(d_x, d_y, N);
  check(cudaGetLastError(), "relu launch");
  check(cudaDeviceSynchronize(), "relu sync");
  check(cudaMemcpy(h_y, d_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "relu D2H");
  float max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float expected = fmaxf(0.0f, h_x[i]);
    float err = fabsf(h_y[i] - expected);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "ReLU", max_err);

  dim3 block64(64);
  relu_vec4<<<grid256, block64>>>(d_x, d_y, N);
  check(cudaGetLastError(), "relu_vec4 launch");
  check(cudaDeviceSynchronize(), "relu_vec4 sync");
  check(cudaMemcpy(h_y, d_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "relu_vec4 D2H");
  max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float expected = fmaxf(0.0f, h_x[i]);
    float err = fabsf(h_y[i] - expected);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "ReLU-Vec4", max_err);

  free(h_x); free(h_y);
  cudaFree(d_x); cudaFree(d_y);
}


void test_elementwise(int N) {
  srand(42);
  float *h_a = (float *)malloc((size_t)N * sizeof(float));
  float *h_b = (float *)malloc((size_t)N * sizeof(float));
  for (int i = 0; i < N; i++) {
    h_a[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    h_b[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }

  float *d_a, *d_b, *d_c;
  check(cudaMalloc(&d_a, (size_t)N * sizeof(float)), "eadd alloc A");
  check(cudaMalloc(&d_b, (size_t)N * sizeof(float)), "eadd alloc B");
  check(cudaMalloc(&d_c, (size_t)N * sizeof(float)), "eadd alloc C");
  check(cudaMemcpy(d_a, h_a, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "eadd H2D A");
  check(cudaMemcpy(d_b, h_b, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "eadd H2D B");

  dim3 block256(256);
  dim3 grid256((N + 255) / 256);
  elementwise_add<<<grid256, block256>>>(d_a, d_b, d_c, N);
  check(cudaGetLastError(), "eadd launch");
  check(cudaDeviceSynchronize(), "eadd sync");
  float *h_c = (float *)malloc((size_t)N * sizeof(float));
  check(cudaMemcpy(h_c, d_c, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "eadd D2H");
  float max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float err = fabsf(h_c[i] - (h_a[i] + h_b[i]));
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "ElemwiseAdd", max_err);

  dim3 block64(64);
  check(cudaMemset(d_c, 0, (size_t)N * sizeof(float)), "eadd_vec4 zero C");
  elementwise_add_vec4<<<grid256, block64>>>(d_a, d_b, d_c, N);
  check(cudaGetLastError(), "eadd_vec4 launch");
  check(cudaDeviceSynchronize(), "eadd_vec4 sync");
  check(cudaMemcpy(h_c, d_c, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "eadd_vec4 D2H");
  max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float err = fabsf(h_c[i] - (h_a[i] + h_b[i]));
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "ElemwiseAdd-Vec4", max_err);

  free(h_a); free(h_b); free(h_c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}


void test_histogram(int N) {
  srand(42);
  int BINS = 16;
  int *h_hist = (int *)calloc(BINS, sizeof(int));
  int *h_hist_ref = (int *)calloc(BINS, sizeof(int));
  int *h_idx = (int *)malloc((size_t)N * sizeof(int));
  for (int i = 0; i < N; i++) h_idx[i] = rand() % BINS;
  for (int i = 0; i < N; i++) h_hist_ref[h_idx[i]]++;

  int *d_idx, *d_hist;
  check(cudaMalloc(&d_idx, (size_t)N * sizeof(int)), "hist alloc idx");
  check(cudaMalloc(&d_hist, BINS * sizeof(int)), "hist alloc hist");
  check(cudaMemcpy(d_idx, h_idx, (size_t)N * sizeof(int), cudaMemcpyHostToDevice), "hist H2D idx");
  check(cudaMemset(d_hist, 0, BINS * sizeof(int)), "hist zero");

  dim3 block256(256);
  dim3 grid256((N + 255) / 256);
  histogram<<<grid256, block256>>>(d_idx, d_hist, N);
  check(cudaGetLastError(), "histogram launch");
  check(cudaDeviceSynchronize(), "histogram sync");
  check(cudaMemcpy(h_hist, d_hist, BINS * sizeof(int), cudaMemcpyDeviceToHost), "hist D2H");

  float max_err = 0.0f;
  for (int i = 0; i < BINS; i++) {
    float err = fabsf((float)(h_hist[i] - h_hist_ref[i]));
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "Histogram", max_err);

  free(h_hist); free(h_hist_ref); free(h_idx);
  cudaFree(d_idx); cudaFree(d_hist);
}


void test_merge_attn_states(int num_tokens, int num_heads,
                                   int head_size) {

  size_t out_size = (size_t)num_tokens * num_heads * head_size * sizeof(float);
  size_t lse_size = (size_t)num_heads * num_tokens * sizeof(float);

  float *h_prefix_out = (float *)malloc(out_size);
  float *h_suffix_out = (float *)malloc(out_size);
  float *h_prefix_lse = (float *)malloc(lse_size);
  float *h_suffix_lse = (float *)malloc(lse_size);
  float *h_output_ref = (float *)malloc(out_size);

  srand(42);
  for (int i = 0; i < num_tokens * num_heads * head_size; i++) {
    h_prefix_out[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    h_suffix_out[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
  for (int i = 0; i < num_heads * num_tokens; i++) {
    h_prefix_lse[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    h_suffix_lse[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
  }

  // CPU reference: 与 kernel 完全相同的浮点计算
  for (int t = 0; t < num_tokens; t++) {
    for (int h = 0; h < num_heads; h++) {
      float p_lse = h_prefix_lse[h * num_tokens + t];
      float s_lse = h_suffix_lse[h * num_tokens + t];
      p_lse = isinf(p_lse) ? -INFINITY : p_lse;
      s_lse = isinf(s_lse) ? -INFINITY : s_lse;

      float max_lse = fmaxf(p_lse, s_lse);
      p_lse -= max_lse;
      s_lse -= max_lse;
      float p_se = expf(p_lse);
      float s_se = expf(s_lse);
      float p_scale = p_se / (p_se + s_se);
      float s_scale = s_se / (p_se + s_se);

      int head_off = t * num_heads * head_size + h * head_size;
      for (int d = 0; d < head_size; d++) {
        h_output_ref[head_off + d] =
            h_prefix_out[head_off + d] * p_scale +
            h_suffix_out[head_off + d] * s_scale;
      }
    }
  }

  float *d_prefix_out, *d_suffix_out, *d_prefix_lse, *d_suffix_lse, *d_output;
  check(cudaMalloc(&d_prefix_out, out_size), "merge_attn alloc p_out");
  check(cudaMalloc(&d_suffix_out, out_size), "merge_attn alloc s_out");
  check(cudaMalloc(&d_prefix_lse, lse_size), "merge_attn alloc p_lse");
  check(cudaMalloc(&d_suffix_lse, lse_size), "merge_attn alloc s_lse");
  check(cudaMalloc(&d_output, out_size), "merge_attn alloc output");

  check(cudaMemcpy(d_prefix_out, h_prefix_out, out_size,
                   cudaMemcpyHostToDevice),
        "merge_attn H2D p_out");
  check(cudaMemcpy(d_suffix_out, h_suffix_out, out_size,
                   cudaMemcpyHostToDevice),
        "merge_attn H2D s_out");
  check(cudaMemcpy(d_prefix_lse, h_prefix_lse, lse_size,
                   cudaMemcpyHostToDevice),
        "merge_attn H2D p_lse");
  check(cudaMemcpy(d_suffix_lse, h_suffix_lse, lse_size,
                   cudaMemcpyHostToDevice),
        "merge_attn H2D s_lse");

  int threads_per_head = head_size / 4;
  int total_threads = num_tokens * num_heads * threads_per_head;
  dim3 block(128);
  dim3 grid((total_threads + 127) / 128);
  merge_attn_states<<<grid, block>>>(
      d_output, d_prefix_out, d_prefix_lse, d_suffix_out, d_suffix_lse,
      num_tokens, num_heads, head_size);
  check(cudaGetLastError(), "merge_attn launch");
  check(cudaDeviceSynchronize(), "merge_attn sync");

  float *h_output = (float *)malloc(out_size);
  check(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost),
        "merge_attn D2H");

  float max_err = 0.0f;
  for (int i = 0; i < num_tokens * num_heads * head_size; i++) {
    float err = fabsf(h_output[i] - h_output_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "MergeAttnStates", max_err);

  // 边界测试: +inf LSE → 权重退化为 0（空 attention 段）
  h_prefix_lse[0] = INFINITY; // head 0, token 0 的 LSE = +inf
  h_suffix_lse[0] = 0.0f;
  check(cudaMemcpy(d_prefix_lse, h_prefix_lse, lse_size,
                   cudaMemcpyHostToDevice),
        "merge_attn H2D inf lse");
  merge_attn_states<<<grid, block>>>(
      d_output, d_prefix_out, d_prefix_lse, d_suffix_out, d_suffix_lse,
      num_tokens, num_heads, head_size);
  check(cudaGetLastError(), "merge_attn inf launch");
  check(cudaDeviceSynchronize(), "merge_attn inf sync");
  check(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost),
        "merge_attn inf D2H");

  // token 0, head 0 的所有元素应等于 suffix_output（prefix 权重 α=0）
  float inf_err = 0.0f;
  for (int d = 0; d < head_size; d++) {
    float err = fabsf(h_output[d] - h_suffix_out[d]);
    if (err > inf_err) inf_err = err;
  }
  printf("| %-56s | %.3e |\n", "MergeAttnStates-inf", inf_err);

  free(h_prefix_out);
  free(h_suffix_out);
  free(h_prefix_lse);
  free(h_suffix_lse);
  free(h_output_ref);
  free(h_output);
  cudaFree(d_prefix_out);
  cudaFree(d_suffix_out);
  cudaFree(d_prefix_lse);
  cudaFree(d_suffix_lse);
  cudaFree(d_output);
}


void test_softmax(int N) {
  // Use N = blockDim.x so one block processes all elements as one token.
  constexpr int kNumThreads = 256;
  if (N != kNumThreads) {
    printf("  OnlineSafeSoftmax: N must be %d (got %d), skipping.\n", kNumThreads, N);
    return;
  }

  srand(42);
  float *h_x = (float *)malloc((size_t)N * sizeof(float));
  float *h_y_ref = (float *)malloc((size_t)N * sizeof(float));
  for (int i = 0; i < N; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;  // [-5, 5]

  // CPU reference: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
  float max_val = -FLT_MAX;
  for (int i = 0; i < N; i++) if (h_x[i] > max_val) max_val = h_x[i];
  double sum_exp = 0.0;
  for (int i = 0; i < N; i++) sum_exp += (double)expf(h_x[i] - max_val);
  for (int i = 0; i < N; i++)
    h_y_ref[i] = expf(h_x[i] - max_val) / (float)sum_exp;

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, (size_t)N * sizeof(float)), "softmax alloc X");
  check(cudaMalloc(&d_y, (size_t)N * sizeof(float)), "softmax alloc Y");

  check(cudaMemcpy(d_x, h_x, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "softmax H2D X");

  dim3 block(256);
  dim3 grid(1);  // one token covering all N elements
  online_safe_softmax_per_token<<<grid, block>>>(d_x, d_y, N);
  check(cudaGetLastError(), "softmax launch");
  check(cudaDeviceSynchronize(), "softmax sync");

  float *h_y = (float *)malloc((size_t)N * sizeof(float));
  check(cudaMemcpy(h_y, d_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "softmax D2H");

  float max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "OnlineSafeSoftmax", max_err);

  // ---- Safe Softmax ----
  safe_softmax_per_token<<<grid, block>>>(d_x, d_y, N);
  check(cudaGetLastError(), "safe_softmax launch");
  check(cudaDeviceSynchronize(), "safe_softmax sync");
  check(cudaMemcpy(h_y, d_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "safe_softmax D2H");
  max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "SafeSoftmax", max_err);

  // ---- Naive Softmax ----
  softmax_per_token<<<grid, block>>>(d_x, d_y, N);
  check(cudaGetLastError(), "naive_softmax launch");
  check(cudaDeviceSynchronize(), "naive_softmax sync");
  check(cudaMemcpy(h_y, d_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "naive_softmax D2H");
  max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "NaiveSoftmax", max_err);

  free(h_x); free(h_y); free(h_y_ref);
  cudaFree(d_x); cudaFree(d_y);
}


void test_rms_norm(int N, int K) {

  srand(42);
  float *h_x = (float *)malloc((size_t)N * K * sizeof(float));
  float *h_y_ref = (float *)malloc((size_t)N * K * sizeof(float));
  for (int i = 0; i < N * K; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  float g = 1.5f;  // gain

  // CPU reference: y = (x / rms(x)) * g
  float epsilon = 1e-5f;
  for (int n = 0; n < N; n++) {
    double sum_sq = 0.0;
    for (int k = 0; k < K; k++) sum_sq += (double)h_x[n * K + k] * (double)h_x[n * K + k];
    float rms = sqrtf((float)sum_sq / (float)K + epsilon);
    for (int k = 0; k < K; k++)
      h_y_ref[n * K + k] = (h_x[n * K + k] / rms) * g;
  }

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, (size_t)N * K * sizeof(float)), "rmsnorm alloc X");
  check(cudaMalloc(&d_y, (size_t)N * K * sizeof(float)), "rmsnorm alloc Y");
  check(cudaMemcpy(d_x, h_x, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice), "rmsnorm H2D X");

  dim3 block(128);
  dim3 grid(N);
  rms_norm<<<grid, block>>>(d_x, d_y, g, N, K);
  check(cudaGetLastError(), "rmsnorm launch");
  check(cudaDeviceSynchronize(), "rmsnorm sync");

  float *h_y = (float *)malloc((size_t)N * K * sizeof(float));
  check(cudaMemcpy(h_y, d_y, (size_t)N * K * sizeof(float), cudaMemcpyDeviceToHost), "rmsnorm D2H");

  float max_err = 0.0f;
  for (int i = 0; i < N * K; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "RMSNorm", max_err);

  // ---- RMS Norm Vec4 ----
  dim3 block_rv4(32);
  rms_norm_vec4<<<grid, block_rv4>>>(d_x, d_y, g, N, K);
  check(cudaGetLastError(), "rmsnorm_vec4 launch");
  check(cudaDeviceSynchronize(), "rmsnorm_vec4 sync");
  check(cudaMemcpy(h_y, d_y, (size_t)N * K * sizeof(float), cudaMemcpyDeviceToHost), "rmsnorm_vec4 D2H");
  max_err = 0.0f;
  for (int i = 0; i < N * K; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "RMSNorm-Vec4", max_err);

  free(h_x); free(h_y); free(h_y_ref);
  cudaFree(d_x); cudaFree(d_y);
}


void test_layer_norm(int N, int K) {

  srand(42);
  float *h_x = (float *)malloc((size_t)N * K * sizeof(float));
  float *h_y_ref = (float *)malloc((size_t)N * K * sizeof(float));
  for (int i = 0; i < N * K; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  float g = 1.5f, b = 0.3f;  // gain and bias

  // CPU reference: y = ((x - mean) / std) * g + b
  float epsilon = 1e-5f;
  for (int n = 0; n < N; n++) {
    double sum = 0.0;
    for (int k = 0; k < K; k++) sum += (double)h_x[n * K + k];
    float mean = (float)sum / (float)K;
    double sum_sq = 0.0;
    for (int k = 0; k < K; k++) {
      float diff = h_x[n * K + k] - mean;
      sum_sq += (double)diff * (double)diff;
    }
    float std = sqrtf((float)sum_sq / (float)K + epsilon);
    for (int k = 0; k < K; k++)
      h_y_ref[n * K + k] = ((h_x[n * K + k] - mean) / std) * g + b;
  }

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, (size_t)N * K * sizeof(float)), "layernorm alloc X");
  check(cudaMalloc(&d_y, (size_t)N * K * sizeof(float)), "layernorm alloc Y");
  check(cudaMemcpy(d_x, h_x, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice), "layernorm H2D X");

  dim3 block(128);
  dim3 grid(N);
  layer_norm<<<grid, block>>>(d_x, d_y, g, b, N, K);
  check(cudaGetLastError(), "layernorm launch");
  check(cudaDeviceSynchronize(), "layernorm sync");

  float *h_y = (float *)malloc((size_t)N * K * sizeof(float));
  check(cudaMemcpy(h_y, d_y, (size_t)N * K * sizeof(float), cudaMemcpyDeviceToHost), "layernorm D2H");

  float max_err = 0.0f;
  for (int i = 0; i < N * K; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "LayerNorm", max_err);

  // ---- Layer Norm Vec4 ----
  dim3 block_lv4(32);
  layer_norm_vec4<<<grid, block_lv4>>>(d_x, d_y, g, b, N, K);
  check(cudaGetLastError(), "layernorm_vec4 launch");
  check(cudaDeviceSynchronize(), "layernorm_vec4 sync");
  check(cudaMemcpy(h_y, d_y, (size_t)N * K * sizeof(float), cudaMemcpyDeviceToHost), "layernorm_vec4 D2H");
  max_err = 0.0f;
  for (int i = 0; i < N * K; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "LayerNorm-Vec4", max_err);

  free(h_x); free(h_y); free(h_y_ref);
  cudaFree(d_x); cudaFree(d_y);
}


void test_rope(int seq_len, int N) {

  int total_pairs = seq_len * N;
  int total_elems = total_pairs * 2;
  size_t size = (size_t)total_elems * sizeof(float);

  float *h_x = (float *)malloc(size);
  float *h_y_ref = (float *)malloc(size);

  srand(42);
  for (int i = 0; i < total_elems; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  // CPU reference: 2D rotation for each pair
  for (int idx = 0; idx < total_pairs; idx++) {
    int token_pos = idx / N;
    int token_idx = idx % N;
    float x1 = h_x[idx * 2];
    float x2 = h_x[idx * 2 + 1];
    float theta = 1.0f / powf(10000.0f, 2.0f * token_idx / (N * 2.0f));
    float angle = (float)token_pos * theta;
    float cos_v = cosf(angle);
    float sin_v = sinf(angle);
    h_y_ref[idx * 2] = x1 * cos_v - x2 * sin_v;
    h_y_ref[idx * 2 + 1] = x1 * sin_v + x2 * cos_v;
  }

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, size), "rope alloc X");
  check(cudaMalloc(&d_y, size), "rope alloc Y");
  check(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice), "rope H2D");

  dim3 block(256);
  dim3 grid((total_pairs + 255) / 256);
  rope<<<grid, block>>>(d_x, d_y, seq_len, N);
  check(cudaGetLastError(), "rope launch");
  check(cudaDeviceSynchronize(), "rope sync");

  float *h_y = (float *)malloc(size);
  check(cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost), "rope D2H");

  float max_err = 0.0f;
  for (int i = 0; i < total_elems; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "RoPE", max_err);

  free(h_x);
  free(h_y);
  free(h_y_ref);
  cudaFree(d_x);
  cudaFree(d_y);
}


void test_mat_transpose(int row, int col) {

  size_t size_in = (size_t)row * col * sizeof(float);
  size_t size_out = (size_t)col * row * sizeof(float);

  float *h_x = (float *)malloc(size_in);
  float *h_y_ref = (float *)malloc(size_out);

  srand(42);
  for (int i = 0; i < row * col; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  // CPU reference: y[j][i] = x[i][j] (row-major)
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      h_y_ref[j * row + i] = h_x[i * col + j];

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, size_in), "mattrans alloc X");
  check(cudaMalloc(&d_y, size_out), "mattrans alloc Y");
  check(cudaMemcpy(d_x, h_x, size_in, cudaMemcpyHostToDevice), "mattrans H2D");

  dim3 block(16, 16);
  dim3 grid((col + 15) / 16, (row + 15) / 16);
  mat_transpose<<<grid, block>>>(d_x, d_y, row, col);
  check(cudaGetLastError(), "mattrans launch");
  check(cudaDeviceSynchronize(), "mattrans sync");

  float *h_y = (float *)malloc(size_out);
  check(cudaMemcpy(h_y, d_y, size_out, cudaMemcpyDeviceToHost), "mattrans D2H");

  float max_err = 0.0f;
  for (int i = 0; i < col * row; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "MatTranspose", max_err);

  free(h_x);
  free(h_y);
  free(h_y_ref);
  cudaFree(d_x);
  cudaFree(d_y);
}


void test_mat_transpose_padded(int row, int col) {

  size_t size_in = (size_t)row * col * sizeof(float);
  size_t size_out = (size_t)col * row * sizeof(float);

  float *h_x = (float *)malloc(size_in);
  float *h_y_ref = (float *)malloc(size_out);

  srand(42);
  for (int i = 0; i < row * col; i++)
    h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  // CPU reference: y[j][i] = x[i][j] (row-major)
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      h_y_ref[j * row + i] = h_x[i * col + j];

  float *d_x, *d_y;
  check(cudaMalloc(&d_x, size_in), "mattrans_padded alloc X");
  check(cudaMalloc(&d_y, size_out), "mattrans_padded alloc Y");
  check(cudaMemcpy(d_x, h_x, size_in, cudaMemcpyHostToDevice), "mattrans_padded H2D");

  dim3 block(16, 16);
  dim3 grid((col + 15) / 16, (row + 63) / 64);
  mat_transpose_padded<<<grid, block>>>(d_x, d_y, row, col);
  check(cudaGetLastError(), "mattrans_padded launch");
  check(cudaDeviceSynchronize(), "mattrans_padded sync");

  float *h_y = (float *)malloc(size_out);
  check(cudaMemcpy(h_y, d_y, size_out, cudaMemcpyDeviceToHost), "mattrans_padded D2H");

  float max_err = 0.0f;
  for (int i = 0; i < col * row; i++) {
    float err = fabsf(h_y[i] - h_y_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("| %-56s | %.3e |\n", "MatTransposePadded", max_err);

  free(h_x);
  free(h_y);
  free(h_y_ref);
  cudaFree(d_x);
  cudaFree(d_y);
}


// Host 端 v1/v2 等价性测试: 遍历 kColStride/i/j, 断言 swizzle_v1_impl == swizzle_v2_impl.
// 用于在启用 NOTES_V2_ENABLE_SWIZZLE_V2 前后验证 v2 与 v1 bit-exact 等价.
// kColStride=8 在 v2 内回退 v1, 故 8 也应恒等.
template <int kColStride>
static void swizzle_equiv_check_one(long &total, long &fail) {
  for (int i = 0; i < 256; ++i) {
    for (int j = 0; j < kColStride; ++j) {
      int v1 = swizzle_v1_impl<kColStride>(i, j);
      int v2 = swizzle_v2_impl<kColStride>(i, j);
      ++total;
      if (v1 != v2) {
        ++fail;
        if (fail <= 10)
          printf("  MISMATCH cs=%d i=%d j=%d: v1=%d v2=%d\n", kColStride, i, j, v1, v2);
      }
    }
  }
}

void test_swizzle_equiv() {
  long total = 0, fail = 0;
  swizzle_equiv_check_one<8>(total, fail);
  swizzle_equiv_check_one<16>(total, fail);
  swizzle_equiv_check_one<32>(total, fail);
  swizzle_equiv_check_one<64>(total, fail);
  printf("| %-56s | %-9s |\n",
         "Swizzle v1/v2 equiv (host)", fail == 0 ? "ALL PASS" : "FAIL");
  printf("  total=%ld fail=%ld\n", total, fail);
}
