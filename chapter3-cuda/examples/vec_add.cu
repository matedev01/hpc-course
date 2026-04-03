// vec_add.cu
// ═══════════════════════════════════════════════════════════════════
// HPC Course — Chapter 3: CUDA Fundamentals
// Benchmark: CPU vs GPU vector addition
//
// Compile: nvcc -O3 -o vec_add vec_add.cu
// Run:     ./vec_add
// ═══════════════════════════════════════════════════════════════════

#include <iostream>
#include <chrono>
#include <cmath>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

__global__ void vec_add_kernel(const float* a, const float* b, float* c, int n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

void vec_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    constexpr int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    // Print device info
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("Device: %s\n", props.name);
    printf("SMs: %d, Max threads/block: %d\n\n",
           props.multiProcessorCount, props.maxThreadsPerBlock);

    // Host allocation
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c_gpu = new float[N];
    float* h_c_cpu = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = sinf(i * 0.001f);
        h_b[i] = cosf(i * 0.001f);
    }

    // CPU baseline
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vec_add_cpu(h_a, h_b, h_c_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // GPU version
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    // Warmup
    vec_add_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vec_add_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify
    float max_err = 0;
    for (int i = 0; i < N; i++) {
        max_err = fmaxf(max_err, fabsf(h_c_gpu[i] - h_c_cpu[i]));
    }

    std::cout << "=== Vector Addition (N = " << N << ") ===" << std::endl;
    std::cout << "CPU:      " << cpu_ms << " ms" << std::endl;
    std::cout << "GPU:      " << gpu_ms << " ms (kernel only)" << std::endl;
    std::cout << "Speedup:  " << cpu_ms / gpu_ms << "x" << std::endl;
    std::cout << "Max error: " << max_err << std::endl;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_gpu;
    delete[] h_c_cpu;

    return 0;
}
