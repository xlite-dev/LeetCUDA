#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "notes-v2.fatbin.c"
extern __attribute__((visibility("hidden"))) void __device_stub__Z4reluPfS_i(float *, float *, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z9relu_vec4PfS_i(float *, float *, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z15elementwise_addPfS_S_i(float *, float *, float *, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z20elementwise_add_vec4PfS_S_i(float *, float *, float *, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z9histogramPiS_i(int *, int *, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z4ropePfS_ii(float *, float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z13mat_transposePfS_ii(float *, float *, const int, const int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z20mat_transpose_paddedPfS_ii(float *, float *, const int, const int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z9sgemv_k32PfS_S_ii(float *, float *, float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z10sgemv_k128PfS_S_ii(float *, float *, float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z5sgemmPfS_S_iii(float *, float *, float *, int, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z10sgemm_vec4PfS_S_iii(float *, float *, float *, int, int, int);
static void __device_stub__Z16block_reduce_allILi128EEvPfS0_i(float *, float *, int);
static void __device_stub__Z3dotILi128EEvPfS0_S0_i(float *, float *, float *, int);
static void __device_stub__Z8dot_vec4ILi32EEvPfS0_S0_i(float *, float *, float *, int);
static void __device_stub__Z29online_safe_softmax_per_tokenILi256EEvPKfPfi(const float *, float *, int);
static void __device_stub__Z22safe_softmax_per_tokenILi256EEvPfS0_i(float *, float *, int);
static void __device_stub__Z17softmax_per_tokenILi256EEvPfS0_i(float *, float *, int);
static void __device_stub__Z8rms_normILi128EEvPfS0_fii(float *, float *, float, int, int);
static void __device_stub__Z13rms_norm_vec4ILi32EEvPfS0_fii(float *, float *, float, int, int);
static void __device_stub__Z10layer_normILi128EEvPfS0_ffii(float *, float *, float, float, int, int);
static void __device_stub__Z15layer_norm_vec4ILi32EEvPfS0_ffii(float *, float *, float, float, int, int);
static void __device_stub__Z9sgemv_k16ILi2EEvPfS0_S0_ii(float *, float *, float *, int, int);
static void __device_stub__Z19hgemm_mma_stages_tnILi16ELi8ELi16ELi2ELi4ELi4ELi4ELi3ELb0EEvP6__halfS1_S1_iii(half *, half *, half *, int, int, int);
static void __device_stub__Z29flash_attn_mma_stages_split_qILi64ELi16ELi8ELi16ELi4ELi1ELi4ELi1ELi1ELi8ELi1ELi8ELi2ELi8EEvP6__halfS1_S1_S1_ii(half *, half *, half *, half *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
__attribute__((visibility("hidden"))) void __device_stub__Z4reluPfS_i(float *__par0, float *__par1, int __par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaLaunch(((char *)((void ( *)(float *, float *, int))relu)), 0U);}
# 353 "notes-v2.cu"
void relu( float *__cuda_0,float *__cuda_1,int __cuda_2)
# 353 "notes-v2.cu"
{__device_stub__Z4reluPfS_i( __cuda_0,__cuda_1,__cuda_2);



}
# 1 "notes-v2.compute_90.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z9relu_vec4PfS_i( float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int))relu_vec4)), 0U); }
# 365 "notes-v2.cu"
void relu_vec4( float *__cuda_0,float *__cuda_1,int __cuda_2)
# 365 "notes-v2.cu"
{__device_stub__Z9relu_vec4PfS_i( __cuda_0,__cuda_1,__cuda_2);
# 376 "notes-v2.cu"
}
# 1 "notes-v2.compute_90.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z15elementwise_addPfS_S_i( float *__par0,  float *__par1,  float *__par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, int))elementwise_add)), 0U); }
# 382 "notes-v2.cu"
void elementwise_add( float *__cuda_0,float *__cuda_1,float *__cuda_2,int __cuda_3)
# 382 "notes-v2.cu"
{__device_stub__Z15elementwise_addPfS_S_i( __cuda_0,__cuda_1,__cuda_2,__cuda_3);



}
# 1 "notes-v2.compute_90.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z20elementwise_add_vec4PfS_S_i( float *__par0,  float *__par1,  float *__par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, int))elementwise_add_vec4)), 0U); }
# 393 "notes-v2.cu"
void elementwise_add_vec4( float *__cuda_0,float *__cuda_1,float *__cuda_2,int __cuda_3)
# 393 "notes-v2.cu"
{__device_stub__Z20elementwise_add_vec4PfS_S_i( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 409 "notes-v2.cu"
}
# 1 "notes-v2.compute_90.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z9histogramPiS_i( int *__par0,  int *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(int *, int *, int))histogram)), 0U); }
# 416 "notes-v2.cu"
void histogram( int *__cuda_0,int *__cuda_1,int __cuda_2)
# 416 "notes-v2.cu"
{__device_stub__Z9histogramPiS_i( __cuda_0,__cuda_1,__cuda_2);



}
# 1 "notes-v2.compute_90.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z4ropePfS_ii( float *__par0,  float *__par1,  int __par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int, int))rope)), 0U); }
# 743 "notes-v2.cu"
void rope( float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 743 "notes-v2.cu"
{__device_stub__Z4ropePfS_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 760 "notes-v2.cu"
}
# 1 "notes-v2.compute_90.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z13mat_transposePfS_ii( float *__par0,  float *__par1,  const int __par2,  const int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(float *, float *, const int, const int))mat_transpose)), 0U); }
# 785 "notes-v2.cu"
void mat_transpose( float *__cuda_0,float *__cuda_1,const int __cuda_2,const int __cuda_3)
# 785 "notes-v2.cu"
{__device_stub__Z13mat_transposePfS_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 794 "notes-v2.cu"
}
# 1 "notes-v2.compute_90.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z20mat_transpose_paddedPfS_ii( float *__par0,  float *__par1,  const int __par2,  const int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(float *, float *, const int, const int))mat_transpose_padded)), 0U); }
# 802 "notes-v2.cu"
void mat_transpose_padded( float *__cuda_0,float *__cuda_1,const int __cuda_2,const int __cuda_3)
# 803 "notes-v2.cu"
{__device_stub__Z20mat_transpose_paddedPfS_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 846 "notes-v2.cu"
}
# 1 "notes-v2.compute_90.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z9sgemv_k32PfS_S_ii( float *__par0,  float *__par1,  float *__par2,  int __par3,  int __par4) {  __cudaLaunchPrologue(5); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, int, int))sgemv_k32)), 0U); }
# 869 "notes-v2.cu"
void sgemv_k32( float *__cuda_0,float *__cuda_1,float *__cuda_2,int __cuda_3,int __cuda_4)
# 869 "notes-v2.cu"
{__device_stub__Z9sgemv_k32PfS_S_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 891 "notes-v2.cu"
}
# 1 "notes-v2.compute_90.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z10sgemv_k128PfS_S_ii( float *__par0,  float *__par1,  float *__par2,  int __par3,  int __par4) {  __cudaLaunchPrologue(5); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, int, int))sgemv_k128)), 0U); }
# 899 "notes-v2.cu"
void sgemv_k128( float *__cuda_0,float *__cuda_1,float *__cuda_2,int __cuda_3,int __cuda_4)
# 899 "notes-v2.cu"
{__device_stub__Z10sgemv_k128PfS_S_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 922 "notes-v2.cu"
}
# 1 "notes-v2.compute_90.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z5sgemmPfS_S_iii( float *__par0,  float *__par1,  float *__par2,  int __par3,  int __par4,  int __par5) {  __cudaLaunchPrologue(6); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaSetupArgSimple(__par5, 32UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, int, int, int))sgemm)), 0U); }
# 977 "notes-v2.cu"
void sgemm( float *__cuda_0,float *__cuda_1,float *__cuda_2,int __cuda_3,int __cuda_4,int __cuda_5)
# 977 "notes-v2.cu"
{__device_stub__Z5sgemmPfS_S_iii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 1020 "notes-v2.cu"
}
# 1 "notes-v2.compute_90.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z10sgemm_vec4PfS_S_iii( float *__par0,  float *__par1,  float *__par2,  int __par3,  int __par4,  int __par5) {  __cudaLaunchPrologue(6); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaSetupArgSimple(__par5, 32UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, int, int, int))sgemm_vec4)), 0U); }
# 1049 "notes-v2.cu"
void sgemm_vec4( float *__cuda_0,float *__cuda_1,float *__cuda_2,int __cuda_3,int __cuda_4,int __cuda_5)
# 1049 "notes-v2.cu"
{__device_stub__Z10sgemm_vec4PfS_S_iii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 1120 "notes-v2.cu"
}
# 1 "notes-v2.compute_90.cudafe1.stub.c"
static void __device_stub__Z16block_reduce_allILi128EEvPfS0_i( float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int))block_reduce_all<(int)128> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_block_reduce_all<128>( float *&__cuda_0,float *&__cuda_1,int &__cuda_2){__device_stub__Z16block_reduce_allILi128EEvPfS0_i( (float *&)__cuda_0,(float *&)__cuda_1,(int &)__cuda_2);}
static void __device_stub__Z3dotILi128EEvPfS0_S0_i( float *__par0,  float *__par1,  float *__par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, int))dot<(int)128> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_dot<128>( float *&__cuda_0,float *&__cuda_1,float *&__cuda_2,int &__cuda_3){__device_stub__Z3dotILi128EEvPfS0_S0_i( (float *&)__cuda_0,(float *&)__cuda_1,(float *&)__cuda_2,(int &)__cuda_3);}
static void __device_stub__Z8dot_vec4ILi32EEvPfS0_S0_i( float *__par0,  float *__par1,  float *__par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, int))dot_vec4<(int)32> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_dot_vec4<32>( float *&__cuda_0,float *&__cuda_1,float *&__cuda_2,int &__cuda_3){__device_stub__Z8dot_vec4ILi32EEvPfS0_S0_i( (float *&)__cuda_0,(float *&)__cuda_1,(float *&)__cuda_2,(int &)__cuda_3);}
static void __device_stub__Z29online_safe_softmax_per_tokenILi256EEvPKfPfi( const float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(const float *, float *, int))online_safe_softmax_per_token<(int)256> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_online_safe_softmax_per_token<256>( const float *&__cuda_0,float *&__cuda_1,int &__cuda_2){__device_stub__Z29online_safe_softmax_per_tokenILi256EEvPKfPfi( (const float *&)__cuda_0,(float *&)__cuda_1,(int &)__cuda_2);}
static void __device_stub__Z22safe_softmax_per_tokenILi256EEvPfS0_i( float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int))safe_softmax_per_token<(int)256> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_safe_softmax_per_token<256>( float *&__cuda_0,float *&__cuda_1,int &__cuda_2){__device_stub__Z22safe_softmax_per_tokenILi256EEvPfS0_i( (float *&)__cuda_0,(float *&)__cuda_1,(int &)__cuda_2);}
static void __device_stub__Z17softmax_per_tokenILi256EEvPfS0_i( float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int))softmax_per_token<(int)256> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_softmax_per_token<256>( float *&__cuda_0,float *&__cuda_1,int &__cuda_2){__device_stub__Z17softmax_per_tokenILi256EEvPfS0_i( (float *&)__cuda_0,(float *&)__cuda_1,(int &)__cuda_2);}
static void __device_stub__Z8rms_normILi128EEvPfS0_fii( float *__par0,  float *__par1,  float __par2,  int __par3,  int __par4) {  __cudaLaunchPrologue(5); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float, int, int))rms_norm<(int)128> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_rms_norm<128>( float *&__cuda_0,float *&__cuda_1,float &__cuda_2,int &__cuda_3,int &__cuda_4){__device_stub__Z8rms_normILi128EEvPfS0_fii( (float *&)__cuda_0,(float *&)__cuda_1,(float &)__cuda_2,(int &)__cuda_3,(int &)__cuda_4);}
static void __device_stub__Z13rms_norm_vec4ILi32EEvPfS0_fii( float *__par0,  float *__par1,  float __par2,  int __par3,  int __par4) {  __cudaLaunchPrologue(5); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float, int, int))rms_norm_vec4<(int)32> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_rms_norm_vec4<32>( float *&__cuda_0,float *&__cuda_1,float &__cuda_2,int &__cuda_3,int &__cuda_4){__device_stub__Z13rms_norm_vec4ILi32EEvPfS0_fii( (float *&)__cuda_0,(float *&)__cuda_1,(float &)__cuda_2,(int &)__cuda_3,(int &)__cuda_4);}
static void __device_stub__Z10layer_normILi128EEvPfS0_ffii( float *__par0,  float *__par1,  float __par2,  float __par3,  int __par4,  int __par5) {  __cudaLaunchPrologue(6); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaSetupArgSimple(__par5, 28UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float, float, int, int))layer_norm<(int)128> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_layer_norm<128>( float *&__cuda_0,float *&__cuda_1,float &__cuda_2,float &__cuda_3,int &__cuda_4,int &__cuda_5){__device_stub__Z10layer_normILi128EEvPfS0_ffii( (float *&)__cuda_0,(float *&)__cuda_1,(float &)__cuda_2,(float &)__cuda_3,(int &)__cuda_4,(int &)__cuda_5);}
static void __device_stub__Z15layer_norm_vec4ILi32EEvPfS0_ffii( float *__par0,  float *__par1,  float __par2,  float __par3,  int __par4,  int __par5) {  __cudaLaunchPrologue(6); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaSetupArgSimple(__par5, 28UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float, float, int, int))layer_norm_vec4<(int)32> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_layer_norm_vec4<32>( float *&__cuda_0,float *&__cuda_1,float &__cuda_2,float &__cuda_3,int &__cuda_4,int &__cuda_5){__device_stub__Z15layer_norm_vec4ILi32EEvPfS0_ffii( (float *&)__cuda_0,(float *&)__cuda_1,(float &)__cuda_2,(float &)__cuda_3,(int &)__cuda_4,(int &)__cuda_5);}
static void __device_stub__Z9sgemv_k16ILi2EEvPfS0_S0_ii( float *__par0,  float *__par1,  float *__par2,  int __par3,  int __par4) {  __cudaLaunchPrologue(5); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *, int, int))sgemv_k16<(int)2> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_sgemv_k16<2>( float *&__cuda_0,float *&__cuda_1,float *&__cuda_2,int &__cuda_3,int &__cuda_4){__device_stub__Z9sgemv_k16ILi2EEvPfS0_S0_ii( (float *&)__cuda_0,(float *&)__cuda_1,(float *&)__cuda_2,(int &)__cuda_3,(int &)__cuda_4);}
static void __device_stub__Z19hgemm_mma_stages_tnILi16ELi8ELi16ELi2ELi4ELi4ELi4ELi3ELb0EEvP6__halfS1_S1_iii( half *__par0,  half *__par1,  half *__par2,  int __par3,  int __par4,  int __par5) {  __cudaLaunchPrologue(6); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaSetupArgSimple(__par5, 32UL); __cudaLaunch(((char *)((void ( *)(half *, half *, half *, int, int, int))hgemm_mma_stages_tn<(int)16, (int)8, (int)16, (int)2, (int)4, (int)4, (int)4, (int)3, (bool)0> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_hgemm_mma_stages_tn<16,8,16,2,4,4,4,3,false>( ::half *&__cuda_0,::half *&__cuda_1,::half *&__cuda_2,int &__cuda_3,int &__cuda_4,int &__cuda_5){__device_stub__Z19hgemm_mma_stages_tnILi16ELi8ELi16ELi2ELi4ELi4ELi4ELi3ELb0EEvP6__halfS1_S1_iii( (::half *&)__cuda_0,(::half *&)__cuda_1,(::half *&)__cuda_2,(int &)__cuda_3,(int &)__cuda_4,(int &)__cuda_5);}
static void __device_stub__Z29flash_attn_mma_stages_split_qILi64ELi16ELi8ELi16ELi4ELi1ELi4ELi1ELi1ELi8ELi1ELi8ELi2ELi8EEvP6__halfS1_S1_S1_ii( half *__par0,  half *__par1,  half *__par2,  half *__par3,  int __par4,  int __par5) {  __cudaLaunchPrologue(6); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaLaunch(((char *)((void ( *)(half *, half *, half *, half *, int, int))flash_attn_mma_stages_split_q<(int)64, (int)16, (int)8, (int)16, (int)4, (int)1, (int)4, (int)1, (int)1, (int)8, (int)1, (int)8, (int)2, (int)8> )), 0U); }
template<> __specialization_static void __wrapper__device_stub_flash_attn_mma_stages_split_q<64,16,8,16,4,1,4,1,1,8,1,8,2,8>( ::half *&__cuda_0,::half *&__cuda_1,::half *&__cuda_2,::half *&__cuda_3,int &__cuda_4,int &__cuda_5){__device_stub__Z29flash_attn_mma_stages_split_qILi64ELi16ELi8ELi16ELi4ELi1ELi4ELi1ELi1ELi8ELi1ELi8ELi2ELi8EEvP6__halfS1_S1_S1_ii( (::half *&)__cuda_0,(::half *&)__cuda_1,(::half *&)__cuda_2,(::half *&)__cuda_3,(int &)__cuda_4,(int &)__cuda_5);}
static void __nv_cudaEntityRegisterCallback( void **__T201) {  __nv_dummy_param_ref(__T201); __nv_save_fatbinhandle_for_managed_rt(__T201); __cudaRegisterEntry(__T201, ((void ( *)(half *, half *, half *, half *, int, int))flash_attn_mma_stages_split_q<(int)64, (int)16, (int)8, (int)16, (int)4, (int)1, (int)4, (int)1, (int)1, (int)8, (int)1, (int)8, (int)2, (int)8> ), _Z29flash_attn_mma_stages_split_qILi64ELi16ELi8ELi16ELi4ELi1ELi4ELi1ELi1ELi8ELi1ELi8ELi2ELi8EEvP6__halfS1_S1_S1_ii, 128); __cudaRegisterEntry(__T201, ((void ( *)(half *, half *, half *, int, int, int))hgemm_mma_stages_tn<(int)16, (int)8, (int)16, (int)2, (int)4, (int)4, (int)4, (int)3, (bool)0> ), _Z19hgemm_mma_stages_tnILi16ELi8ELi16ELi2ELi4ELi4ELi4ELi3ELb0EEvP6__halfS1_S1_iii, 256); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float *, int, int))sgemv_k16<(int)2> ), _Z9sgemv_k16ILi2EEvPfS0_S0_ii, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float, float, int, int))layer_norm_vec4<(int)32> ), _Z15layer_norm_vec4ILi32EEvPfS0_ffii, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float, float, int, int))layer_norm<(int)128> ), _Z10layer_normILi128EEvPfS0_ffii, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float, int, int))rms_norm_vec4<(int)32> ), _Z13rms_norm_vec4ILi32EEvPfS0_fii, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float, int, int))rms_norm<(int)128> ), _Z8rms_normILi128EEvPfS0_fii, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, int))softmax_per_token<(int)256> ), _Z17softmax_per_tokenILi256EEvPfS0_i, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, int))safe_softmax_per_token<(int)256> ), _Z22safe_softmax_per_tokenILi256EEvPfS0_i, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(const float *, float *, int))online_safe_softmax_per_token<(int)256> ), _Z29online_safe_softmax_per_tokenILi256EEvPKfPfi, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float *, int))dot_vec4<(int)32> ), _Z8dot_vec4ILi32EEvPfS0_S0_i, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float *, int))dot<(int)128> ), _Z3dotILi128EEvPfS0_S0_i, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, int))block_reduce_all<(int)128> ), _Z16block_reduce_allILi128EEvPfS0_i, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float *, int, int, int))sgemm_vec4), _Z10sgemm_vec4PfS_S_iii, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float *, int, int, int))sgemm), _Z5sgemmPfS_S_iii, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float *, int, int))sgemv_k128), _Z10sgemv_k128PfS_S_ii, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float *, int, int))sgemv_k32), _Z9sgemv_k32PfS_S_ii, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, const int, const int))mat_transpose_padded), _Z20mat_transpose_paddedPfS_ii, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, const int, const int))mat_transpose), _Z13mat_transposePfS_ii, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, int, int))rope), _Z4ropePfS_ii, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(int *, int *, int))histogram), _Z9histogramPiS_i, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float *, int))elementwise_add_vec4), _Z20elementwise_add_vec4PfS_S_i, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, float *, int))elementwise_add), _Z15elementwise_addPfS_S_i, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, int))relu_vec4), _Z9relu_vec4PfS_i, (-1)); __cudaRegisterEntry(__T201, ((void ( *)(float *, float *, int))relu), _Z4reluPfS_i, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
