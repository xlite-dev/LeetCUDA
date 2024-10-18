Skip to content
Navigation Menu
DefTruth
/
CUDA-Learn-Notes

Type / to search
Code
Issues
4
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Files
Go to file
t
.github
cuda-slides
cutlass
dot-product
elementwise
embedding
flash-attn
gelu
hgemm
.gitignore
README.md
hgemm.cu
hgemm.py
hgemm_async.cu
hgemm_cublas.cu
hgemm_mma.cu
hgemm_mma_stage.cu
hgemm_wmma.cu
hgemm_wmma_stage.cu
prof.py
hgemv
histogram
layer-norm
mat_transpose
nms
nvidia-nsight
openai-triton
pytorch
reduce
relu
rms-norm
rope
sgemm
sgemv
sigmoid
softmax
swish
tensorrt
third-party
transformer
vllm-slides
.gitignore
.gitmodules
LICENSE
README.md
notes-v1.cu
Editing hgemm.py in CUDA-Learn-Notes
BreadcrumbsCUDA-Learn-Notes/hgemm
/
hgemm.py
in
main

Edit

Preview
Indent mode

Spaces
Indent size

2
Line wrap mode

No wrap
Editing hgemm.py file contents
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
            improve = round(improve, 2)
        else:
            improve = 0
        MAX_TFLOPS = TFLOPS
        print(f"{out_info:>40}: {out_val}, time:{mean_time}ms, "
              f"swizzle: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}(+{improve:.2f}%)")
    else:
        print(f"{out_info:>40}: {out_val}, time:{mean_time}ms, "
              f"swizzle: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}")
    if show_all: print(out)
    time.sleep(0.05)
    return out, mean_time


Ms = [4096, 8192, 16384]
Ns = [4096, 8192, 16384]
Ks = [2048, 4096, 8192]
MAX_M, MAX_N, MAX_K = 16384, 16384, 8192
# pre allocate for fast profiling.
A = torch.randn((MAX_M, MAX_K), dtype=torch.half).cuda()
B = torch.randn((MAX_K, MAX_N), dtype=torch.half).cuda()
C = torch.randn((MAX_M, MAX_N), dtype=torch.half).cuda()
torch.cuda.synchronize()

MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
for (M, N, K) in MNKs:
    MAX_TFLOPS = -1
    print("-" * 130)
    print(" " * 55 + f"M={M}, N={N}, K={K}")
    a = A[:M, :K].contiguous()
    b = B[:K, :N].contiguous()
    c = C[:M, :N].contiguous()
    torch.cuda.synchronize()

    # CUDA Cores FP16
    # run_benchmark(lib.hgemm_naive_f16, a, b, "f16(naive)",  c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x8_pack_bcf, a, b, "f16x8pack(t8x8+bcf)", c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf, a, b, "f16x8pack(t8x8+bcf+dbuf)", c)
    run_benchmark(lib.hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf, a, b, "f16x8pack(t8x8+k16+dbuf)", c)

    print("-" * 68 + "WMMA" + "-" * 58)
    # run_benchmark(lib.hgemm_wmma_m16n16k16_naive, a, b, "f16wmma(naive)", c)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2, a, b, "f16wmma(mma4x2)", c)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4, a, b, "f16wmma(mma4x2+warp2x4)", c)
    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_offset, a, b, "f16wmma(mma2x4+warp2x4+dbuf)", c)

    # Stages, dsmem
    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "f16wmma(mma2x4+warp2x4+stage4)", c, stages=4)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "f16wmma(mma2x4+warp2x4+stage3)", c, stages=3)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "f16wmma(mma2x4+warp2x4+stage2)", c, stages=2)

    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "f16wmma(mma2x4+...+stage4+dsmem)", c, stages=4)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "f16wmma(mma2x4+...+stage3+dsmem)", c, stages=3)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "f16wmma(mma2x4+...+stage2+dsmem)", c, stages=2)

    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "f16wmma(mma4x4+...+stage4+dsmem)", c, stages=4)
    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "f16wmma(mma4x4+...+stage3+dsmem)", c, stages=3)
    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "f16wmma(mma4x4+...+stage2+dsmem)", c, stages=2)
    
    # Thread block swizzle
    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "f16wmma(mma2x4+...+stage4+swizzle)", c, stages=4, swizzle=True)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "f16wmma(mma2x4+...+stage3+swizzle)", c, stages=3, swizzle=True)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, a, b, "f16wmma(mma2x4+...+stage2+swizzle)", c, stages=2, swizzle=True)

    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "f16wmma(...+stage4+dsmem+swizzle)", c, stages=4, swizzle=True)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "f16wmma(...+stage3+dsmem+swizzle)", c, stages=3, swizzle=True)
    run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, a, b, "f16wmma(...+stage2+dsmem+swizzle)", c, stages=2, swizzle=True)

    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "f16wmma(mma4x4+stage4+dsmem+swizzle)", c, stages=4, swizzle=True)
    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "f16wmma(mma4x4+stage3+dsmem+swizzle)", c, stages=3, swizzle=True)
    # run_benchmark(lib.hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, a, b, "f16wmma(mma4x4+stage2+dsmem+swizzle)", c, stages=2, swizzle=True)
    
    run_benchmark(lib.hgemm_cublas_tensor_op, a, b, "f16(cublas)", c)
    run_benchmark(partial(torch.matmul, out=c), a, b, "f16_th")
    torch.cuda.synchronize()
    print("-" * 130)

Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
Editing CUDA-Learn-Notes/hgemm/hgemm.py at main · DefTruth/CUDA-Learn-Notes 
