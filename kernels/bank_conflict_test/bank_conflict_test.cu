#include <iostream>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#define WARP_SIZE 32

constexpr int S = 4096, K = 4096;
constexpr int N = S * K;
constexpr int B = 256;
float x[S][K];

template<const int NUM_THREADS>
__global__ void free_conflict_kernel(float *x, float *y, int N) {
    __shared__ float shmem[NUM_THREADS / WARP_SIZE][WARP_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (idx < N) {
        shmem[warp_id][lane_id] = x[idx];
    }
    __syncthreads();
    if (idx < N) {
        y[idx] = shmem[warp_id][lane_id];
    }

}

template<const int NUM_THREADS>
__global__ void full_conflict_kernel(float *x, float *y, int N) {
    __shared__ float shmem[WARP_SIZE][NUM_THREADS / WARP_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (idx < N) {
        shmem[lane_id][warp_id] = x[idx];
    }
    __syncthreads();
    if (idx < N) {
        y[idx] = shmem[lane_id][warp_id];
    }
}

template<const int NUM_THREADS>
__global__ void one_thread_save_shmem_kernel(float *x, float *y, int N) {
    __shared__ float shmem[NUM_THREADS / WARP_SIZE][WARP_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (lane_id == 0) {
        shmem[lane_id][warp_id] = x[idx];
    }
    __syncthreads();
    if (lane_id == 0) {
        y[idx] = shmem[warp_id][lane_id];
    }
}

// template<const int NUM_THREADS>
// __global__ void one_thread_load_shmem_kernel(float *x, float *y, int N) {
//     __shared__ float shmem[NUM_THREADS / WARP_SIZE][WARP_SIZE];
//     int tid = threadIdx.x;
//     int idx = blockIdx.x * NUM_THREADS + threadIdx.x;
//     int warp_id = tid / WARP_SIZE;
//     int lane_id = tid % WARP_SIZE;
    
//     if (lane_id == 0) {
//         x[idx] = shmem[lane_id][warp_id];
//     }
//     __syncthreads();
//     if (lane_id == 0) {
//         y[idx] = shmem[warp_id][lane_id];
//     }
// }

int main() {
    float *d_x_ptr;
    float *d_y_ptr;

    dim3 grid(N / B); 
    dim3 block(B);
    cudaMalloc(&d_x_ptr, sizeof(x));
    cudaMalloc(&d_y_ptr, sizeof(x));
    cudaMemcpy(d_x_ptr, x, sizeof(x), cudaMemcpyHostToDevice);
    nvtxRangePush("profiling");
    free_conflict_kernel<B><<<grid, block>>>(d_x_ptr, d_y_ptr, N);
    full_conflict_kernel<B><<<grid, block>>>(d_x_ptr, d_y_ptr, N);
    // one_thread_load_shmem_kernel<B><<<grid, block>>>(d_x_ptr, d_y_ptr, N);
    one_thread_save_shmem_kernel<B><<<grid, block>>>(d_x_ptr, d_y_ptr, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
    return 0;
}