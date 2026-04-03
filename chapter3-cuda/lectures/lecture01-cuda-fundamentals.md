# Lecture 1: CUDA — GPU Parallel Programming

## Chapter 3 | CUDA and SIMT (Main Focus)

**Duration:** ~3 hours
**Prerequisites:** Chapter 1 (SIMD), Chapter 2 (OpenMP), basic C/C++
**Objective:** Master GPU parallelism — thousands of threads, warp execution, memory hierarchy.

---

## Part I: The High School Explanation

### 1.1 The CPU vs GPU: Professors vs Students

Imagine you need to grade 10,000 multiple-choice exams.

**The CPU approach (professor):**
You have **4 brilliant professors**. Each professor can grade any type of exam — essays, math proofs, coding questions. They're smart and flexible. But there are only 4 of them. Grading 10,000 exams takes a long time.

**The GPU approach (students):**
You hire **10,000 high school students**. Each student can only do one simple thing: compare an answer to the answer key and mark it right or wrong. They're not brilliant, but there are 10,000 of them. They **all grade simultaneously**. Done in seconds.

> **CPU** = a few powerful, flexible cores (4–64)
> **GPU** = thousands of simple, specialized cores (thousands)

### 1.2 When to Use a GPU

**Good for GPU:**
- Millions of independent calculations
- Same formula applied to each data point
- Matrix math, image processing, physics simulation, AI training

**Bad for GPU:**
- Complex decision-making with lots of `if/else`
- Sequential algorithms where step N depends on step N-1
- Small problems (overhead of sending data to GPU > compute time)

### 1.3 The Bus Ride: Host and Device

Your computer has two separate processors:
- **Host** = CPU + system RAM
- **Device** = GPU + GPU memory (VRAM)

They're connected by a **bus** (PCIe), which is like a highway between two cities.

```
┌──────────────┐         PCIe Bus          ┌──────────────┐
│     CPU      │ ◄══════════════════════► │     GPU      │
│  (Host)      │    ~32 GB/s bandwidth      │  (Device)    │
│              │                            │              │
│  System RAM  │                            │  GPU Memory  │
│  (64+ GB)    │                            │  (8-80 GB)   │
└──────────────┘                            └──────────────┘
```

**The workflow:**
1. CPU prepares data in system RAM
2. CPU **copies data** to GPU memory (slow — crossing the highway)
3. CPU tells GPU: *"Run this function on the data"*
4. GPU runs the function with thousands of threads (fast!)
5. GPU **copies results** back to CPU (slow again)

> **Key insight:** The PCIe bus is the bottleneck. You want to minimize data transfers and maximize GPU compute time.

### 1.4 The Classroom Analogy: Grids, Blocks, and Threads

The GPU organizes work like a school:

- **School** = the GPU
- **Classrooms** = **blocks** (groups of students that can talk to each other)
- **Students** = **threads** (each one does one piece of work)
- **All classrooms together** = the **grid**

```
Grid (the whole school)
├── Block 0  (classroom 0)
│   ├── Thread 0
│   ├── Thread 1
│   ├── ...
│   └── Thread 255
├── Block 1  (classroom 1)
│   ├── Thread 0
│   ├── Thread 1
│   ├── ...
│   └── Thread 255
├── ...
└── Block N
```

**Rules:**
- Students in the **same classroom** can share a whiteboard (**shared memory**) and synchronize.
- Students in **different classrooms** cannot directly communicate.
- The school can have thousands of classrooms running simultaneously.

---

## Part II: Professional Deep Dive

### 2.1 CUDA Hardware Architecture

```
GPU (e.g., NVIDIA RTX 4090)
├── GPC (Graphics Processing Cluster) x N
│   ├── SM (Streaming Multiprocessor) x M
│   │   ├── CUDA Cores: 128 (FP32)
│   │   ├── Tensor Cores: 4 (matrix ops)
│   │   ├── Warp Schedulers: 4
│   │   ├── Register File: 256 KB
│   │   ├── Shared Memory / L1: 128 KB (configurable)
│   │   └── Load/Store Units, SFUs
│   └── ...
├── L2 Cache: 72 MB
├── Memory Controllers → GDDR6X: 24 GB @ 1 TB/s
└── PCIe 4.0 x16: ~32 GB/s to host
```

**Key numbers (RTX 4090):**
- 128 SMs, 16,384 CUDA cores
- 82.6 TFLOPS FP32
- 24 GB GDDR6X @ 1,008 GB/s bandwidth
- Max 1,536 threads per SM, 1,024 threads per block

### 2.2 SIMT Execution Model: Warps

The GPU doesn't execute individual threads. It executes in groups of **32 threads** called **warps**.

```
Block (256 threads)
├── Warp 0:  threads [  0.. 31]   ← All 32 execute the SAME instruction
├── Warp 1:  threads [ 32.. 63]
├── Warp 2:  threads [ 64.. 95]
├── Warp 3:  threads [ 96..127]
├── Warp 4:  threads [128..159]
├── Warp 5:  threads [160..191]
├── Warp 6:  threads [192..223]
└── Warp 7:  threads [224..255]
```

**SIMT (Single Instruction, Multiple Threads):**
- Like SIMD, but each "lane" is a full thread with its own registers and program counter.
- All 32 threads in a warp execute the **same instruction** at the same time.
- If threads diverge (`if/else`), both paths are executed serially → **warp divergence** → performance loss.

```cpp
// Warp divergence example
if (threadIdx.x < 16) {
    do_A();   // First: threads 0-15 execute, threads 16-31 are idle (masked)
} else {
    do_B();   // Then: threads 16-31 execute, threads 0-15 are idle (masked)
}
// Both paths executed sequentially → 2x slowdown for this warp
```

### 2.3 Memory Hierarchy

```
                        ┌───────────────┐
                        │   Registers   │  ← Per thread, fastest
                        │  (~255 regs)  │     1 cycle
                        └───────┬───────┘
                                ↓
                   ┌────────────────────────┐
                   │   Shared Memory / L1   │  ← Per SM, programmer-managed
                   │   (up to 100 KB)       │     ~20-30 cycles
                   └────────────┬───────────┘
                                ↓
                        ┌───────────────┐
                        │   L2 Cache    │  ← Device-wide
                        │  (up to 72 MB)│     ~200 cycles
                        └───────┬───────┘
                                ↓
                   ┌────────────────────────┐
                   │   Global Memory (VRAM) │  ← Device-wide, largest, slowest
                   │   (8-80 GB)            │     ~400-600 cycles
                   └────────────────────────┘
```

| Memory | Scope | Lifetime | Speed | Size |
|--------|-------|----------|-------|------|
| Registers | Thread | Thread | ~1 cycle | ~255 x 32-bit per thread |
| Shared Memory | Block | Block | ~20 cycles | Up to 100 KB per SM |
| L2 Cache | Device | Automatic | ~200 cycles | 6-72 MB |
| Global Memory | Device | Application | ~400 cycles | 8-80 GB |
| Constant Memory | Device | Application | ~1 cycle (cached) | 64 KB |

### 2.4 Your First CUDA Program

```cpp
// hello_cuda.cu
// Compile: nvcc -o hello hello_cuda.cu
// Run:     ./hello

#include <stdio.h>

// __global__ marks a function that runs on the GPU, called from the CPU
__global__ void hello_kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from thread %d (block %d, thread-in-block %d)\n",
           tid, blockIdx.x, threadIdx.x);
}

int main() {
    // Launch kernel: 2 blocks, 4 threads per block = 8 threads total
    //                <<<numBlocks, threadsPerBlock>>>
    hello_kernel<<<2, 4>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}
```

### 2.5 Vector Addition — The "Hello World" of CUDA

```cpp
// vec_add.cu
// Compile: nvcc -O3 -o vec_add vec_add.cu
// Run:     ./vec_add

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

// GPU kernel
__global__ void vec_add_kernel(const float* a, const float* b, float* c, int n) {
    // Grid-stride loop: handles arbitrary N with any grid size
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

// CPU version for comparison
void vec_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    constexpr int N = 1 << 24;  // ~16 million
    size_t bytes = N * sizeof(float);

    // ─── Allocate host memory ───
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c_gpu = new float[N];
    float* h_c_cpu = new float[N];

    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = sinf(i * 0.001f);
        h_b[i] = cosf(i * 0.001f);
    }

    // ─── CPU version ───
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vec_add_cpu(h_a, h_b, h_c_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ─── GPU version ───
    float *d_a, *d_b, *d_c;

    // Step 1: Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Step 2: Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Step 3: Launch kernel
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    // Warm up
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

    // Step 4: Copy results back
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    // ─── Verify ───
    float max_err = 0;
    for (int i = 0; i < N; i++) {
        max_err = fmaxf(max_err, fabsf(h_c_gpu[i] - h_c_cpu[i]));
    }

    std::cout << "=== Vector Addition (N = " << N << ") ===" << std::endl;
    std::cout << "CPU:      " << cpu_ms << " ms" << std::endl;
    std::cout << "GPU:      " << gpu_ms << " ms (kernel only)" << std::endl;
    std::cout << "Speedup:  " << cpu_ms / gpu_ms << "x" << std::endl;
    std::cout << "Max error: " << max_err << std::endl;

    // ─── Cleanup ───
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
```

### 2.6 Memory Coalescing — The #1 Optimization

When threads in a warp access **contiguous memory**, the hardware combines accesses into one transaction:

```
COALESCED (fast — one 128-byte transaction):
  Thread 0 reads a[0]
  Thread 1 reads a[1]
  Thread 2 reads a[2]
  ...
  Thread 31 reads a[31]
  → All 32 reads combine into ONE memory transaction

UNCOALESCED (slow — 32 separate transactions):
  Thread 0 reads a[0]
  Thread 1 reads a[1000]
  Thread 2 reads a[2000]
  ...
  Thread 31 reads a[31000]
  → Each read is a separate transaction → 32x slower!
```

```cpp
// GOOD: coalesced access
__global__ void good_kernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] * 2.0f;  // Threads access consecutive addresses
}

// BAD: strided access (uncoalesced)
__global__ void bad_kernel(float* data, int n, int stride) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx * stride] = data[idx * stride] * 2.0f;  // Gaps between accesses
}
```

### 2.7 Shared Memory — The Block's Scratchpad

Shared memory is a **fast, programmer-managed cache** within each SM:

```cpp
// Tiled matrix multiplication using shared memory
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    const int TILE = 32;

    __shared__ float As[TILE][TILE];  // Shared memory tile for A
    __shared__ float Bs[TILE][TILE];  // Shared memory tile for B

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Iterate over tiles
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load tile from global memory to shared memory
        if (row < M && (t * TILE + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * TILE + threadIdx.y) < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // Wait for all threads to finish loading

        // Compute partial dot product from shared memory (FAST!)
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();  // Wait before loading next tile
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Why this is faster:**

```
Without tiling:
  Each element of C reads K elements from A and K elements from B
  from GLOBAL memory (~400 cycles per read)

With tiling:
  Load tile once into shared memory (32x32 = 1024 reads from global)
  Reuse tile 32 times from shared memory (~20 cycles per read)
  → ~20x fewer global memory accesses
```

### 2.8 Warp-Level Primitives

Since CUDA 9.0, you can communicate directly within a warp without shared memory:

```cpp
// Warp-level reduction using shuffle
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;  // All threads in the warp now have the total sum
}

// Full block reduction
__global__ void reduce_kernel(const float* input, float* output, int n) {
    __shared__ float shared[32];  // One slot per warp

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val = (idx < n) ? input[idx] : 0.0f;

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Step 1: Reduce within each warp
    val = warp_reduce_sum(val);

    // Step 2: First thread of each warp writes to shared memory
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    // Step 3: First warp reduces the warp results
    if (warp_id == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }

    // Step 4: First thread writes the block result
    if (threadIdx.x == 0) {
        atomicAdd(output, val);
    }
}
```

### 2.9 Streams and Concurrency

By default, CUDA operations are **sequential**. Streams enable **overlap**:

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// These operations can overlap:
cudaMemcpyAsync(d_a, h_a, bytes/2, cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(d_b, h_b, bytes/2, cudaMemcpyHostToDevice, stream2);

kernel<<<grid, block, 0, stream1>>>(d_a, ...);
kernel<<<grid, block, 0, stream2>>>(d_b, ...);

cudaMemcpyAsync(h_c, d_c, bytes/2, cudaMemcpyDeviceToHost, stream1);
cudaMemcpyAsync(h_d, d_d, bytes/2, cudaMemcpyDeviceToHost, stream2);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

```
Without streams (sequential):
  [Copy H→D] [Compute] [Copy D→H]
  ──────────────────────────────────→ time

With streams (overlapped):
  Stream 1: [Copy H→D] [Compute] [Copy D→H]
  Stream 2:       [Copy H→D] [Compute] [Copy D→H]
  ──────────────────────────────────→ time (shorter!)
```

### 2.10 Error Handling — Always Check

**Always check CUDA errors.** Silent failures are the #1 debugging nightmare:

```cpp
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Usage:
CUDA_CHECK(cudaMalloc(&d_a, bytes));
CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

// Check kernel launch errors
my_kernel<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());          // Check launch error
CUDA_CHECK(cudaDeviceSynchronize());     // Check execution error
```

---

## Part III: Practical Example — Complete Matrix Multiplication

### 3.1 Full Compilable Example

```cpp
// matmul.cu
// Compile: nvcc -O3 -arch=sm_80 -o matmul matmul.cu
// Run:     ./matmul

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
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
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
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = sum;
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

    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

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
```

### 3.2 Expected Output

```
=== Matrix Multiplication (1024x1024 * 1024x1024) ===
CPU:        1823.45 ms  (1.18 GFLOP/s)
GPU Naive:    3.21 ms  (670.2 GFLOP/s)
GPU Tiled:    1.87 ms  (1150.5 GFLOP/s)
Speedup (tiled vs CPU):   974.6x
Speedup (tiled vs naive): 1.72x
Max error: 0.000488
```

---

## Part IV: Q&A — Questions Students Actually Ask

### Q1: "How do I choose the number of threads per block?"

**A:** Rules of thumb:
- **Must be a multiple of 32** (warp size). Wasted threads in a partial warp = wasted resources.
- **Common choices:** 128, 256, or 512.
- **Maximum:** 1,024 threads per block (hardware limit).
- **Occupancy:** More threads per SM = better latency hiding. Use `--ptxas-options=-v` to check register usage.

```bash
nvcc --ptxas-options=-v -o prog prog.cu
# Shows: registers per thread, shared memory per block
```

---

### Q2: "What's the difference between `cudaMalloc` and `cudaMallocManaged`?"

**A:**

| | `cudaMalloc` | `cudaMallocManaged` |
|---|---|---|
| Memory location | GPU only | CPU or GPU (migrates) |
| Access from CPU | No (must `cudaMemcpy`) | Yes (automatic migration) |
| Performance | Predictable, fast | Page faults on first access |
| Use case | Production code | Prototyping, convenience |

**Recommendation:** Start with `cudaMallocManaged` for prototyping. Switch to explicit `cudaMalloc` + `cudaMemcpy` for production.

---

### Q3: "Why does my GPU kernel run slower than the CPU for small inputs?"

**A:** GPU overhead:
1. **Kernel launch:** ~5-10 microseconds (fixed cost)
2. **Memory transfer:** PCIe latency + bandwidth
3. **Thread scheduling:** Block scheduling has overhead

**Rule:** GPUs win when N > ~10,000-100,000 elements. For small N, the overhead dominates.

---

### Q4: "What are bank conflicts in shared memory?"

**A:** Shared memory is divided into 32 **banks**. If two threads in a warp access the same bank (different addresses), they serialize:

```
NO conflict:  thread i reads bank i      → 1 cycle
2-way conflict: 2 threads read same bank → 2 cycles
32-way conflict: all threads same bank   → 32 cycles (serial!)
```

**Fix:** Pad shared memory arrays:
```cpp
__shared__ float tile[32][33];  // 33 instead of 32 → shifts bank alignment
```

---

### Q5: "How do I profile my CUDA code?"

**A:** Two essential tools:

```bash
# Nsight Systems — timeline view (where is time spent?)
nsys profile --stats=true ./my_program

# Nsight Compute — kernel analysis (why is this kernel slow?)
ncu --set full -o report ./my_program
```

**The profiling workflow:**
1. `nsys` first → find the **hottest kernel**
2. `ncu` on that kernel → find the **bottleneck** (compute or memory)
3. Optimize → repeat

---

### Q6: "What's the grid-stride loop and why should I always use it?"

**A:**

```cpp
// FRAGILE: assumes gridDim * blockDim >= N
__global__ void kernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] *= 2.0f;
}

// ROBUST: grid-stride loop — works for any N with any grid size
__global__ void kernel(float* data, int n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        data[i] *= 2.0f;
    }
}
```

Benefits:
1. Works for **any problem size** without adjusting grid dimensions
2. Enables **thread reuse** (fewer block scheduling overhead)
3. Easy to switch between debugging (1 thread) and production (millions)

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Host/Device | CPU prepares data, GPU computes, PCIe connects them |
| Grid/Block/Thread | Hierarchical thread organization |
| Warp (32 threads) | The fundamental execution unit |
| Coalescing | Consecutive threads should access consecutive memory |
| Shared Memory | Fast per-block scratchpad — use for data reuse (tiling) |
| Streams | Overlap transfers and compute for higher throughput |
| Grid-stride loop | Robust kernel pattern for arbitrary problem sizes |
| Error checking | Always use `CUDA_CHECK` — silent failures are the norm |

---

## Next Lecture

**Lecture 2:** Advanced CUDA — Reductions, Scan (Prefix Sum), and Cooperative Groups

---

*HPC Course — Chapter 3: CUDA Fundamentals*
*License: MIT*
