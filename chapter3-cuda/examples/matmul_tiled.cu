// matmul_tiled.cu
// ═══════════════════════════════════════════════════════════════════
// HPC Course — Chapter 3: CUDA
// Benchmark: CPU vs Naive GPU vs Tiled GPU matrix multiplication
//
// Compile: nvcc -O3 -arch=sm_80 -o matmul matmul_tiled.cu
// Run:     ./matmul
// ═══════════════════════════════════════════════════════════════════

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

constexpr int TILE = 32;

__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main() {
    constexpr int M = 1024, N = 1024, K = 1024;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C_cpu = new float[M * N];
    float* h_C_gpu = new float[M * N];

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    // CPU baseline
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // GPU setup
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // Benchmark naive
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t s1, e1;
    CUDA_CHECK(cudaEventCreate(&s1));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(s1));
    for (int r = 0; r < 10; r++)
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float naive_ms;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, s1, e1));
    naive_ms /= 10;

    // Benchmark tiled
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t s2, e2;
    CUDA_CHECK(cudaEventCreate(&s2));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventRecord(s2));
    for (int r = 0; r < 10; r++)
        matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaEventSynchronize(e2));
    float tiled_ms;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, s2, e2));
    tiled_ms /= 10;

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes_C, cudaMemcpyDeviceToHost));

    float max_err = 0;
    for (int i = 0; i < M * N; i++)
        max_err = fmaxf(max_err, fabsf(h_C_gpu[i] - h_C_cpu[i]));

    double gflops = 2.0 * M * N * K / 1e9;
    std::cout << "=== Matrix Multiplication (" << M << "x" << K
              << " * " << K << "x" << N << ") ===" << std::endl;
    std::cout << "CPU:        " << cpu_ms << " ms  ("
              << gflops / (cpu_ms / 1000.0) << " GFLOP/s)" << std::endl;
    std::cout << "GPU Naive:  " << naive_ms << " ms  ("
              << gflops / (naive_ms / 1000.0) << " GFLOP/s)" << std::endl;
    std::cout << "GPU Tiled:  " << tiled_ms << " ms  ("
              << gflops / (tiled_ms / 1000.0) << " GFLOP/s)" << std::endl;
    std::cout << "Speedup (tiled vs CPU):   " << cpu_ms / tiled_ms << "x" << std::endl;
    std::cout << "Speedup (tiled vs naive): " << naive_ms / tiled_ms << "x" << std::endl;
    std::cout << "Max error: " << max_err << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    cudaEventDestroy(s1); cudaEventDestroy(e1);
    cudaEventDestroy(s2); cudaEventDestroy(e2);
    delete[] h_A; delete[] h_B; delete[] h_C_cpu; delete[] h_C_gpu;

    return 0;
}
