# Lecture 1: SYCL and OpenCL — Portable Parallel Programming

## Chapter 5 | OpenCL and SYCL (Portable Parallelism)

**Duration:** ~2.5 hours
**Prerequisites:** Chapter 3 (CUDA), Chapter 4 (ROCm/HIP)
**Objective:** Write once, run on CPUs, NVIDIA GPUs, AMD GPUs, Intel GPUs, and FPGAs.

---

## Part I: The High School Explanation

### 1.1 The Universal Remote

You have three TVs at home:
- **Samsung** (NVIDIA GPU — speaks CUDA)
- **LG** (AMD GPU — speaks HIP/ROCm)
- **Sony** (Intel GPU — speaks Level Zero)

Each TV came with its own remote. Different buttons, different layouts. Annoying.

Now imagine a **universal remote** that works with all three TVs. You press "volume up" — it figures out which TV you're pointing at and sends the right signal.

> **That's SYCL.** One programming language that runs on any hardware. Write the code once, the runtime adapts it to whatever device is available.

**OpenCL** is the older universal remote — it works, but it has 50 buttons and requires you to read the manual for each TV. **SYCL** is the modern version — same 5 buttons, works everywhere, much simpler.

### 1.2 Why Portability Matters

Real-world scenario: Your company builds physics simulation software.

- Client A has NVIDIA A100 GPUs
- Client B has AMD MI250X GPUs
- Client C has Intel Data Center GPUs
- Client D runs on CPU clusters (no GPUs)

Without SYCL: you maintain **4 separate codebases**. Nightmare.
With SYCL: you maintain **1 codebase**. It compiles and runs on all four.

### 1.3 The Layer Cake

```
Your Code (one version)
        │
        ▼
┌──────────────┐
│    SYCL      │  ← Modern C++ API (you write this)
└──────┬───────┘
       │
       ├──→ Level Zero backend  → Intel GPUs
       ├──→ CUDA backend        → NVIDIA GPUs
       ├──→ ROCm backend        → AMD GPUs
       ├──→ OpenCL backend      → Any OpenCL device
       └──→ Host backend        → CPU (fallback)
```

---

## Part II: Professional Deep Dive

### 2.1 OpenCL — The Foundation

OpenCL (Open Computing Language) was the first cross-platform GPU programming standard (2009, Khronos Group). It's verbose but runs everywhere.

#### OpenCL Architecture

```
┌──────────────────────────────────────────────────┐
│                 Host Program (CPU)                │
│                                                  │
│  Platform → Device → Context → Command Queue     │
│                                                  │
│  Program (compile kernel source) → Kernel        │
│                                                  │
│  Buffer (device memory) → enqueue read/write     │
└──────────────────────────────────────────────────┘
```

**Concepts:**
- **Platform:** An OpenCL implementation (Intel, NVIDIA, AMD)
- **Device:** A compute device (GPU, CPU, FPGA)
- **Context:** Owns memory objects and command queues
- **Command Queue:** Ordered or out-of-order stream of operations
- **Buffer:** Device memory object
- **Program:** Compiled kernel source (SPIR-V or OpenCL C)
- **Kernel:** A function that runs on the device

#### OpenCL Vector Addition — The Full Boilerplate

```c
// vec_add_opencl.c
// Compile: gcc -O3 -o vec_add vec_add_opencl.c -lOpenCL
// This shows WHY people switched to SYCL...

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

const char* kernel_source =
    "__kernel void vec_add(__global const float* a,   \n"
    "                      __global const float* b,   \n"
    "                      __global float* c,         \n"
    "                      int n) {                   \n"
    "    int i = get_global_id(0);                    \n"
    "    if (i < n) c[i] = a[i] + b[i];              \n"
    "}                                                \n";

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 0.001f;
        h_b[i] = i * 0.002f;
    }

    // --- OpenCL setup (lots of boilerplate!) ---
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes, h_a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes, h_b, 0, NULL, NULL);

    // Compile kernel AT RUNTIME
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "vec_add", NULL);

    // Set arguments (one at a time!)
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    size_t global_size = N;
    size_t local_size = 256;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);

    printf("c[0] = %f, c[N-1] = %f\n", h_c[0], h_c[N - 1]);

    // Cleanup (6 releases!)
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_a); free(h_b); free(h_c);

    return 0;
}
```

**That's ~80 lines for vector addition.** CUDA does it in ~30. SYCL does it in ~20.

### 2.2 SYCL — The Modern Way

SYCL (pronounced "sickle") is a Khronos standard that provides a **single-source, modern C++ programming model** for heterogeneous computing.

#### SYCL Implementations

| Implementation | Vendor | Backends |
|---------------|--------|----------|
| **Intel oneAPI DPC++** | Intel | Level Zero, OpenCL, CUDA*, HIP* |
| **AdaptiveCpp (hipSYCL)** | Open source | CUDA, ROCm, OpenCL, Level Zero |
| **ComputeCpp** | Codeplay | OpenCL, SPIR-V |
| **triSYCL** | Xilinx | OpenCL, FPGA |

#### SYCL Vector Addition — Clean and Modern

```cpp
// vec_add_sycl.cpp
// Compile (Intel DPC++): icpx -fsycl -O3 -o vec_add vec_add_sycl.cpp
// Compile (AdaptiveCpp): acpp -O3 -o vec_add vec_add_sycl.cpp
// Run:                   ./vec_add

#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

int main() {
    constexpr int N = 1 << 24;

    // Select device (automatic — picks best available)
    sycl::queue q{sycl::default_selector_v};
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    // Allocate Unified Shared Memory (like cudaMallocManaged)
    float* a = sycl::malloc_shared<float>(N, q);
    float* b = sycl::malloc_shared<float>(N, q);
    float* c = sycl::malloc_shared<float>(N, q);

    // Initialize on host
    for (int i = 0; i < N; i++) {
        a[i] = std::sin(i * 0.001f);
        b[i] = std::cos(i * 0.001f);
    }

    // Warmup
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        c[i] = a[i] + b[i];
    }).wait();

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();

    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        c[i] = a[i] + b[i];
    }).wait();

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Verify
    float max_err = 0;
    for (int i = 0; i < N; i++) {
        max_err = std::max(max_err, std::abs(c[i] - (a[i] + b[i])));
    }

    std::cout << "Time:      " << ms << " ms" << std::endl;
    std::cout << "Max error: " << max_err << std::endl;

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);

    return 0;
}
```

**Compare:** 80 lines (OpenCL) vs 40 lines (SYCL) vs 50 lines (CUDA) for the same operation.

### 2.3 SYCL Memory Models

#### Approach 1: Buffer-Accessor Model (Automatic)

```cpp
{
    sycl::buffer<float, 1> buf_a(h_a, sycl::range<1>(N));
    sycl::buffer<float, 1> buf_b(h_b, sycl::range<1>(N));
    sycl::buffer<float, 1> buf_c(h_c, sycl::range<1>(N));

    q.submit([&](sycl::handler& h) {
        auto a = buf_a.get_access<sycl::access::mode::read>(h);
        auto b = buf_b.get_access<sycl::access::mode::read>(h);
        auto c = buf_c.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            c[i] = a[i] + b[i];
        });
    });
    // Buffer destructor waits and copies data back
}
// h_c now has the results
```

**Pros:** No explicit `memcpy`, SYCL optimizes data movement.
**Cons:** Less control, accessor syntax is verbose.

#### Approach 2: Unified Shared Memory (USM)

```cpp
float* a = sycl::malloc_shared<float>(N, q);  // Host AND device
float* b = sycl::malloc_device<float>(N, q);  // Device only

q.memcpy(b, host_b, N * sizeof(float)).wait();
q.parallel_for(N, [=](auto i) { a[i] += b[i]; }).wait();

sycl::free(a, q);
sycl::free(b, q);
```

| USM Type | Host Access | Device Access | Transfer |
|----------|------------|---------------|----------|
| `malloc_shared` | Yes | Yes | Automatic (page migration) |
| `malloc_device` | No | Yes | Explicit `memcpy` |
| `malloc_host` | Yes | Yes (slow) | Via PCIe on each access |

### 2.4 SYCL `nd_range` — Work-Groups and Local Memory

```cpp
constexpr int TILE = 256;

q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> local_data(sycl::range<1>(TILE), h);

    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(TILE)),
        [=](sycl::nd_item<1> item) {
            int global_id = item.get_global_id(0);
            int local_id  = item.get_local_id(0);

            local_data[local_id] = a[global_id];
            item.barrier(sycl::access::fence_space::local_space);

            c[global_id] = local_data[local_id] + b[global_id];
        }
    );
});
```

**CUDA → SYCL Mapping:**

| CUDA | SYCL |
|------|------|
| `blockIdx.x` | `item.get_group(0)` |
| `threadIdx.x` | `item.get_local_id(0)` |
| `blockDim.x` | `item.get_local_range(0)` |
| `gridDim.x` | `item.get_group_range(0)` |
| `__shared__` | `sycl::local_accessor` |
| `__syncthreads()` | `item.barrier()` |
| `atomicAdd` | `sycl::atomic_ref` |

### 2.5 SYCL Reduction

```cpp
float total = 0.0f;
{
    sycl::buffer<float> sum_buf(&total, 1);

    q.submit([&](sycl::handler& h) {
        auto sum_reduction = sycl::reduction(sum_buf, h, sycl::plus<float>());

        h.parallel_for(sycl::range<1>(N), sum_reduction,
            [=](sycl::id<1> i, auto& sum) {
                sum += a[i];
            }
        );
    });
}
std::cout << "Sum = " << total << std::endl;
```

### 2.6 Device Selection

```cpp
// Automatic (picks best available)
sycl::queue q{sycl::default_selector_v};

// Force GPU
sycl::queue q{sycl::gpu_selector_v};

// Force CPU
sycl::queue q{sycl::cpu_selector_v};

// Custom selector (e.g., pick NVIDIA GPU)
auto selector = [](const sycl::device& d) {
    std::string name = d.get_info<sycl::info::device::name>();
    if (name.find("NVIDIA") != std::string::npos) return 1;
    return -1;
};
sycl::queue q{selector};

// List all available devices
for (auto& platform : sycl::platform::get_platforms()) {
    std::cout << "Platform: "
              << platform.get_info<sycl::info::platform::name>() << std::endl;
    for (auto& device : platform.get_devices()) {
        std::cout << "  Device: "
                  << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "  Max CUs: "
                  << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    }
}
```

### 2.7 Tiled Matrix Multiplication in SYCL

```cpp
// matmul_sycl.cpp
// Compile: icpx -fsycl -O3 -o matmul matmul_sycl.cpp

#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

constexpr int TILE = 16;

void matmul_sycl(sycl::queue& q, const float* A, const float* B, float* C,
                  int M, int N, int K) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> As(sycl::range<2>(TILE, TILE), h);
        sycl::local_accessor<float, 2> Bs(sycl::range<2>(TILE, TILE), h);

        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(M, N),
                sycl::range<2>(TILE, TILE)
            ),
            [=](sycl::nd_item<2> item) {
                int row = item.get_global_id(0);
                int col = item.get_global_id(1);
                int ly = item.get_local_id(0);
                int lx = item.get_local_id(1);

                float sum = 0.0f;

                for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
                    int a_col = t * TILE + lx;
                    int b_row = t * TILE + ly;

                    As[ly][lx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
                    Bs[ly][lx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

                    item.barrier(sycl::access::fence_space::local_space);

                    for (int k = 0; k < TILE; k++)
                        sum += As[ly][k] * Bs[k][lx];

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (row < M && col < N)
                    C[row * N + col] = sum;
            }
        );
    }).wait();
}

int main() {
    constexpr int M = 1024, N = 1024, K = 1024;

    sycl::queue q{sycl::gpu_selector_v};
    std::cout << "Device: "
              << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    float* A = sycl::malloc_shared<float>(M * K, q);
    float* B = sycl::malloc_shared<float>(K * N, q);
    float* C = sycl::malloc_shared<float>(M * N, q);

    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 100) / 100.0f;

    matmul_sycl(q, A, B, C, M, N, K);  // warmup

    auto start = std::chrono::high_resolution_clock::now();
    matmul_sycl(q, A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    double gflops = 2.0 * M * N * K / 1e9;
    std::cout << "Time:    " << ms << " ms" << std::endl;
    std::cout << "GFLOP/s: " << gflops / (ms / 1000.0) << std::endl;

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);

    return 0;
}
```

### 2.8 Cross-Platform Compilation

```bash
# Intel GPU (default)
icpx -fsycl -O3 -o matmul matmul_sycl.cpp

# NVIDIA GPU (DPC++ with CUDA plugin)
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 -o matmul matmul_sycl.cpp

# AMD GPU (AdaptiveCpp)
acpp --acpp-targets=hip:gfx90a -O3 -o matmul matmul_sycl.cpp

# CPU fallback (AdaptiveCpp)
acpp --acpp-targets=omp -O3 -o matmul matmul_sycl.cpp
```

---

## Part III: Comparison — The Same Kernel in Every Framework

```
// c[i] = a[i] + b[i]  — How each framework expresses it:

CUDA:     __global__ void add(float* a, float* b, float* c, int n) {
              int i = threadIdx.x + blockIdx.x * blockDim.x;
              if (i < n) c[i] = a[i] + b[i];
          }
          add<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

HIP:      // Identical syntax to CUDA (compile with hipcc instead of nvcc)

OpenCL:   "__kernel void add(__global float* a, __global float* b,
                              __global float* c, int n) {"
          "    int i = get_global_id(0);"
          "    if (i < n) c[i] = a[i] + b[i];"
          "}"
          // + 50 lines of setup code

SYCL:     q.parallel_for(range<1>(N), [=](id<1> i) {
              c[i] = a[i] + b[i];
          });

OpenMP:   #pragma omp target teams distribute parallel for
          for (int i = 0; i < N; i++) c[i] = a[i] + b[i];
```

---

## Part IV: Q&A — Questions Students Actually Ask

### Q1: "Should I learn SYCL or CUDA?"

**A:** Both, but context matters:

| Goal | Recommendation |
|------|---------------|
| Job at NVIDIA | CUDA |
| HPC / national labs | SYCL or HIP |
| AI inference startup | CUDA (ecosystem) |
| Long-term career | SYCL (portability trend) |
| Academic research | CUDA first, then SYCL |

**Practical advice:** Learn CUDA deeply first (Chapter 3). The concepts transfer directly. SYCL becomes a syntax exercise, not a conceptual one.

---

### Q2: "Is SYCL slower than native CUDA?"

**A:** Typically 0-15% overhead for well-written code. The gap is closing.

- **Compute-bound kernels:** Near parity.
- **Memory-bound kernels:** USM shared can add overhead. Use `malloc_device` + explicit copies for critical paths.
- **Library calls:** Native libraries (cuBLAS) may outperform oneMKL wrappers.

**Where SYCL wins:** Developer productivity. 1 codebase vs 3 saves months of engineering.

---

### Q3: "What's the difference between SYCL and OpenCL?"

**A:**

| | OpenCL | SYCL |
|---|---|---|
| **Language** | C99 kernel + C/C++ host (separate) | **Single-source** C++ |
| **Compilation** | Runtime kernel compilation | Ahead-of-time + JIT |
| **Type safety** | Weak (strings for kernel names) | Strong (C++ templates) |
| **Memory** | Manual buffers | Buffers + USM |
| **Verbosity** | ~80 lines for vec_add | ~20 lines |

SYCL is the modern, ergonomic evolution of OpenCL.

---

### Q4: "Can I use SYCL for AI/ML?"

**A:** Increasingly:
- **oneDNN** has SYCL backends
- **PyTorch** targets Intel GPUs via IPEX
- **oneDAL** provides SYCL-accelerated ML algorithms

But CUDA + cuDNN + TensorRT remains dominant for AI. SYCL for AI is growing.

---

### Q5: "How do I handle devices with different capabilities?"

**A:** Query at runtime:

```cpp
auto device = q.get_device();
size_t global_mem = device.get_info<sycl::info::device::global_mem_size>();
size_t local_mem  = device.get_info<sycl::info::device::local_mem_size>();
int max_wg        = device.get_info<sycl::info::device::max_work_group_size>();

// Adapt tile size based on device
int tile = std::min(max_wg, 256);
```

---

### Q6: "What about Vulkan Compute?"

**A:**
- **Vulkan:** Low-level graphics API with compute shaders. Extremely verbose. Best when you need both graphics and compute.
- **SYCL:** Pure compute abstraction. Much simpler.

Physics simulation for a game engine → Vulkan Compute.
HPC / scientific computing → SYCL.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| OpenCL | Cross-platform but verbose — the foundation |
| SYCL | Modern C++ single-source — the future of portable HPC |
| Buffer-Accessor | SYCL manages data movement automatically |
| USM | Explicit or shared memory — like CUDA managed memory |
| `parallel_for` | Flat parallelism — no work-groups needed |
| `nd_range` | Work-groups + local memory — like CUDA blocks + shared memory |
| Device Selection | `gpu_selector_v`, `cpu_selector_v`, or custom |
| Backends | Level Zero (Intel), CUDA (NVIDIA), ROCm (AMD), OpenCL (all) |
| Portability | Write once → compile for any target with different flags |

---

## Next Lecture

**Lecture 2:** Advanced SYCL — Sub-groups, Joint Matrix, and Multi-Device Programming

---

*HPC Course — Chapter 5: SYCL and OpenCL Fundamentals*
*License: MIT*
