# Lecture 1: ROCm and HIP — AMD GPU Programming

## Chapter 4 | ROCm and HIP (AMD GPU Ecosystem)

**Duration:** ~2 hours
**Prerequisites:** Chapter 3 (CUDA) — this chapter assumes you can write CUDA code
**Objective:** Port CUDA knowledge to AMD hardware with minimal friction.

---

## Part I: The High School Explanation

### 1.1 The Translation Analogy

You already know how to write essays in English (CUDA). Now you need to write in Spanish (HIP).

Good news: Spanish and English share 80% of their structure. The sentences are almost the same. You just need to swap some words:

| English (CUDA) | Spanish (HIP) |
|----------------|---------------|
| "Hello" | "Hola" |
| `cudaMalloc` | `hipMalloc` |
| `cudaMemcpy` | `hipMemcpy` |
| `__syncthreads()` | `__syncthreads()` (same!) |

There's even an **automatic translator** (`hipify`) that converts CUDA code to HIP code for you.

> **HIP is CUDA with different names.** If you know CUDA, you already know 90% of HIP.

### 1.2 Why AMD?

- NVIDIA has the biggest ecosystem (CUDA), but AMD GPUs are **cheaper per TFLOP**
- The world's fastest supercomputers (Frontier, El Capitan) use **AMD GPUs**
- Cloud providers offer AMD Instinct GPUs at lower cost
- Competition is good — knowing both makes you more valuable

### 1.3 The One Real Difference: Warp Size

NVIDIA groups threads in **warps of 32**.
AMD groups threads in **wavefronts of 64**.

Imagine two marching bands:
- NVIDIA's band: rows of 32 musicians
- AMD's band: rows of 64 musicians

Same concept, different row width. This affects some optimizations, but the programming model is identical.

---

## Part II: Professional Deep Dive

### 2.1 The ROCm Software Stack

```
┌──────────────────────────────────────┐
│          Your Application            │
├──────────────────────────────────────┤
│     HIP (API Layer)                  │  ← You write code here
│     "CUDA-like syntax for AMD"       │
├──────────┬───────────────────────────┤
│  HIP/CUDA│  HIP/ROCm                │  ← HIP compiles to BOTH backends
│  (nvcc)  │  (hipcc → clang)         │
├──────────┴───────────────────────────┤
│     ROCr (Runtime)                   │  ← HSA-based runtime
├──────────────────────────────────────┤
│     ROCk (Kernel Driver)             │  ← Linux kernel module (amdgpu)
├──────────────────────────────────────┤
│     AMD GPU Hardware                 │  ← RDNA / CDNA architecture
└──────────────────────────────────────┘
```

**Key insight:** HIP is a **portability layer**. Code written in HIP can compile for **both** NVIDIA GPUs (via nvcc) and AMD GPUs (via hipcc). Write once, run on either vendor.

### 2.2 AMD Hardware Architecture: CDNA vs NVIDIA

```
AMD Instinct MI250X (CDNA2)            NVIDIA A100 (Ampere)
├── 110 Compute Units (CUs)             ├── 108 Streaming Multiprocessors (SMs)
│   ├── 64 Stream Processors each       │   ├── 64 FP32 cores each
│   ├── 4 SIMD units (16-wide)          │   ├── 4 Warp Schedulers
│   ├── VGPRs: 512 x 32-bit per SIMD   │   ├── Registers: 256 KB per SM
│   ├── SGPRs: 800 (scalar)             │   └── Shared Memory: 164 KB per SM
│   └── LDS: 64 KB per CU              │
├── L2 Cache: 8 MB                      ├── L2 Cache: 40 MB
├── HBM2e: 128 GB @ 3.2 TB/s           ├── HBM2e: 80 GB @ 2.0 TB/s
└── Peak FP32: 47.9 TFLOPS             └── Peak FP32: 19.5 TFLOPS
```

### 2.3 Critical Differences

| Concept | NVIDIA (CUDA) | AMD (HIP/ROCm) |
|---------|---------------|-----------------|
| **Thread group** | Warp (32 threads) | Wavefront (64 threads) |
| **Core unit** | SM (Streaming Multiprocessor) | CU (Compute Unit) |
| **Fast local memory** | Shared Memory | LDS (Local Data Share) |
| **Scalar unit** | N/A (all SIMT) | SGPR (Scalar General Purpose Registers) |
| **Compiler** | `nvcc` | `hipcc` (wraps clang) |
| **Profiler** | Nsight Systems / Compute | `rocprof` / `omniperf` |
| **Debug** | `cuda-gdb` | `roc-gdb` |
| **Max threads/block** | 1,024 | 1,024 |
| **Warp/wavefront size** | 32 | 64 (some RDNA GPUs: 32) |

### 2.4 The HIP API: A Direct Translation

Almost every CUDA function has a 1:1 HIP equivalent:

```cpp
// ─── CUDA ───                          // ─── HIP ───
#include <cuda_runtime.h>                 #include <hip/hip_runtime.h>

cudaMalloc(&d_ptr, size);                 hipMalloc(&d_ptr, size);
cudaMemcpy(dst, src, size,               hipMemcpy(dst, src, size,
    cudaMemcpyHostToDevice);                  hipMemcpyHostToDevice);
cudaFree(d_ptr);                          hipFree(d_ptr);

cudaDeviceSynchronize();                  hipDeviceSynchronize();

cudaEventCreate(&event);                  hipEventCreate(&event);
cudaEventRecord(event, stream);           hipEventRecord(event, stream);
cudaEventSynchronize(event);              hipEventSynchronize(event);
cudaEventElapsedTime(&ms, start, stop);   hipEventElapsedTime(&ms, start, stop);

cudaStreamCreate(&stream);                hipStreamCreate(&stream);
```

**Kernel launch syntax is identical:**
```cpp
kernel<<<grid, block, smem, stream>>>(args);
```

**Device code keywords are identical:**
```cpp
__global__   // Same in both
__device__   // Same in both
__shared__   // Same in both
__syncthreads()  // Same in both
threadIdx.x, blockIdx.x, blockDim.x, gridDim.x  // Same in both
```

### 2.5 Automatic Porting with `hipify`

ROCm provides tools to convert CUDA code automatically:

```bash
# Perl-based (quick, pattern matching)
hipify-perl my_cuda_code.cu > my_hip_code.cpp

# Clang-based (AST-aware, more accurate)
hipify-clang my_cuda_code.cu -o my_hip_code.cpp -- \
    -I/usr/local/cuda/include

# Batch convert a project
hipconvertinplace-perl.sh /path/to/cuda/project/
```

**What hipify handles:**
- API calls: `cudaMalloc` → `hipMalloc`
- Types: `cudaError_t` → `hipError_t`
- Enums: `cudaMemcpyHostToDevice` → `hipMemcpyHostToDevice`
- Kernel launch syntax (preserves `<<<>>>`)

**What hipify does NOT handle:**
- CUDA-specific libraries (`cuBLAS` → must switch to `rocBLAS`)
- PTX inline assembly
- Texture objects (different API on AMD)
- Performance tuning (warp→wavefront size changes)

### 2.6 Complete Example: Vector Addition on AMD

```cpp
// vec_add_hip.cpp
// Compile: hipcc -O3 -o vec_add vec_add_hip.cpp
// Run:     ./vec_add

#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err = call;                                           \
        if (err != hipSuccess) {                                         \
            fprintf(stderr, "HIP error at %s:%d — %s\n",                \
                    __FILE__, __LINE__, hipGetErrorString(err));          \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

__global__ void vec_add_kernel(const float* a, const float* b, float* c, int n) {
    // Identical to CUDA!
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    constexpr int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    // Print device info
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "Compute Units: " << props.multiProcessorCount << std::endl;
    std::cout << "Wavefront Size: " << props.warpSize << std::endl;  // 64 on AMD!
    std::cout << "Max Threads/Block: " << props.maxThreadsPerBlock << std::endl;

    // Host allocation
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = sinf(i * 0.001f);
        h_b[i] = cosf(i * 0.001f);
    }

    // Device allocation
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));

    // Launch
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warmup
    vec_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    vec_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

    HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));

    // Verify
    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        max_err = fmaxf(max_err, fabsf(h_c[i] - expected));
    }

    std::cout << "\n=== Vector Addition (N = " << N << ") ===" << std::endl;
    std::cout << "GPU Time:  " << ms << " ms" << std::endl;
    std::cout << "Bandwidth: " << (3.0 * bytes / 1e9) / (ms / 1000.0) << " GB/s" << std::endl;
    std::cout << "Max error: " << max_err << std::endl;

    // Cleanup
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    hipEventDestroy(start); hipEventDestroy(stop);
    delete[] h_a; delete[] h_b; delete[] h_c;

    return 0;
}
```

### 2.7 Wavefront vs Warp: Performance Implications

```
NVIDIA Warp (32 threads):
  ┌────────────────────────────────┐
  │ T0 T1 T2 T3 ... T30 T31       │  32 threads execute in lockstep
  └────────────────────────────────┘

AMD Wavefront (64 threads):
  ┌────────────────────────────────────────────────────────────────┐
  │ T0 T1 T2 T3 ... T62 T63                                       │  64 threads in lockstep
  └────────────────────────────────────────────────────────────────┘
```

**Impact on warp-level operations:**

```cpp
// CUDA: warp reduction — 5 steps (log2(32))
__device__ float warp_reduce_cuda(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// HIP: wavefront reduction — 6 steps (log2(64))
__device__ float wave_reduce_hip(float val) {
    for (int offset = 32; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);  // No sync mask needed on AMD
    return val;
}

// PORTABLE: use warpSize runtime query
__device__ float wave_reduce_portable(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}
```

### 2.8 LDS (Local Data Share) — AMD's Shared Memory

LDS is functionally identical to CUDA's shared memory, but with different bank layout:

```
NVIDIA Shared Memory:
  32 banks, 4 bytes each
  Conflict: 2+ threads in same warp access same bank (different address)

AMD LDS:
  32 banks, 4 bytes each (same structure)
  BUT: wavefront is 64 threads → more potential for bank conflicts
  Resolution: hardware splits 64-thread wavefront into two 32-thread halves
```

```cpp
// Tiled matrix multiply — works on both CUDA and HIP
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    constexpr int TILE = 32;
    __shared__ float As[TILE][TILE + 1];  // +1 padding avoids bank conflicts
    __shared__ float Bs[TILE][TILE + 1];  // Works on BOTH AMD and NVIDIA

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = sum;
}
```

### 2.9 Profiling with `rocprof`

```bash
# Basic kernel timing
rocprof --stats ./my_program
# Output: kernel_name, calls, total_ns, avg_ns, min_ns, max_ns

# Hardware counters
rocprof -i counters.txt ./my_program
# counters.txt:
#   pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SMEM FETCH_SIZE WRITE_SIZE

# Trace (timeline)
rocprof --sys-trace ./my_program
# View in Chrome tracing: chrome://tracing

# Omniperf (high-level analysis, like Nsight Compute)
omniperf profile -n my_analysis -- ./my_program
omniperf analyze -p workloads/my_analysis/
```

### 2.10 Building Portable HIP Code

```cmake
# CMakeLists.txt — builds for NVIDIA or AMD
cmake_minimum_required(VERSION 3.21)
project(hip_example LANGUAGES CXX)

find_package(hip REQUIRED)

add_executable(vec_add vec_add_hip.cpp)
target_link_libraries(vec_add hip::device)

# Target specific AMD GPU architecture
# gfx906 = MI50, gfx908 = MI100, gfx90a = MI250X, gfx942 = MI300X
set_target_properties(vec_add PROPERTIES
    HIP_ARCHITECTURES "gfx90a"
)
```

```bash
# Build for AMD
mkdir build && cd build
cmake .. -DCMAKE_HIP_COMPILER=hipcc
make

# Build for NVIDIA (HIP compiles to CUDA)
cmake .. -DCMAKE_HIP_PLATFORM=nvidia
make
```

### 2.11 Library Equivalents

| CUDA Library | ROCm Equivalent | Purpose |
|-------------|----------------|---------|
| cuBLAS | rocBLAS | Linear algebra |
| cuFFT | rocFFT | FFT |
| cuDNN | MIOpen | Deep learning primitives |
| cuRAND | rocRAND | Random numbers |
| cuSPARSE | rocSPARSE | Sparse matrices |
| Thrust | rocThrust | High-level algorithms |
| CUB | hipCUB | Block/warp primitives |
| NCCL | RCCL | Multi-GPU communication |

---

## Part III: Practical Example — Porting Workflow

### 3.1 Step-by-Step Porting

```bash
# Step 1: Analyze your CUDA code
hipexamine-perl.sh my_project/
# Output: info: converted 45 CUDA->HIP calls
#         warning: 3 calls require manual attention

# Step 2: Auto-convert
hipify-perl my_cuda_kernel.cu > my_hip_kernel.cpp

# Step 3: Fix manual items
# - cuBLAS → rocBLAS (different API)
# - Texture objects (different API)
# - Inline PTX assembly (replace with HIP builtins)

# Step 4: Compile and test
hipcc -O3 -o my_program my_hip_kernel.cpp -lrocblas

# Step 5: Profile and optimize for AMD
rocprof --stats ./my_program
```

---

## Part IV: Q&A — Questions Students Actually Ask

### Q1: "Should I write in CUDA or HIP?"

**A:** If you need to support both vendors, **write in HIP**. HIP compiles to both CUDA (via nvcc) and ROCm (via hipcc). You get portability for free.

If you only target NVIDIA, CUDA is fine — most mature ecosystem and tooling.

**Industry trend:** Most HPC centers are moving toward portability (HIP or SYCL).

---

### Q2: "Is AMD slower than NVIDIA?"

**A:** Not inherently. AMD MI250X has **higher raw FP32 TFLOPS and memory bandwidth** than A100. But:
- NVIDIA's software ecosystem (cuDNN, TensorRT, Nsight) is more mature
- Many libraries are CUDA-optimized first, AMD-optimized second
- Debugging tools on ROCm are improving but lag behind

**For HPC workloads** (scientific computing): AMD is competitive and often cheaper.
**For AI/ML:** NVIDIA still leads due to ecosystem.

---

### Q3: "What changes when warp size goes from 32 to 64?"

**A:** Watch for:
1. **Shared memory indexing:** `threadIdx.x % 32` → use `threadIdx.x % warpSize`
2. **Ballot operations:** Returns 32-bit on NVIDIA, 64-bit on AMD
3. **Divergence cost:** 64 threads affected instead of 32
4. **Occupancy:** Fewer wavefronts per CU for the same thread count
5. **Reduction steps:** 6 steps (log2(64)) vs 5 steps (log2(32))

**Best practice:** Use `warpSize` instead of hardcoded `32` everywhere.

---

### Q4: "Can I use CUDA libraries on AMD?"

**A:** No. CUDA libraries are NVIDIA-proprietary. Use ROCm equivalents:

```cpp
// CUDA
#include <cublas_v2.h>
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
            &alpha, d_A, M, d_B, K, &beta, d_C, M);

// ROCm
#include <rocblas/rocblas.h>
rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
```

The API is similar but not identical. `hipify` cannot auto-convert library calls.

---

### Q5: "What's the sync mask difference?"

**A:** On NVIDIA (CUDA 9+), warp-level operations require an **explicit mask**:
```cpp
// CUDA: must specify which threads participate
__shfl_xor_sync(0xffffffff, val, offset);
```

On AMD, **no mask is needed** (full wavefront always participates):
```cpp
// HIP on AMD: no sync mask
__shfl_xor(val, offset);
```

For portability, always provide the mask — it's ignored on AMD.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| HIP | CUDA-compatible API that runs on both AMD and NVIDIA |
| `hipify` | Automatic CUDA→HIP conversion tool |
| Wavefront | AMD's warp equivalent — 64 threads (vs 32) |
| LDS | AMD's shared memory — same concept, same syntax |
| ROCm stack | ROCr (runtime) + ROCk (driver) + hipcc (compiler) |
| Profiling | `rocprof` (counters/trace), `omniperf` (high-level) |
| Libraries | cuBLAS→rocBLAS, cuDNN→MIOpen, cuFFT→rocFFT |
| Portability | Write HIP → compile for AMD or NVIDIA |

---

## Next Lecture

**Lecture 2:** AMD-Specific Optimization — VGPR Pressure, Occupancy Tuning, and LDS Bank Conflicts

---

*HPC Course — Chapter 4: ROCm and HIP Fundamentals*
*License: MIT*
