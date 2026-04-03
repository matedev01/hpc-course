# Lecture 1: SIMD — Single Instruction, Multiple Data

## Chapter 1 | C++ and SIMD (Vectorization)

**Duration:** ~2 hours
**Prerequisites:** Basic C/C++ (variables, loops, arrays, pointers)
**Objective:** Understand how a single CPU core processes multiple data points in one instruction.

---

## Part I: The High School Explanation

### 1.1 The Cashier Analogy

Imagine you work at a grocery store. Your job is to scan items.

**The slow way (Scalar processing):**
You pick up **one item**, scan it, put it down. Pick up the next item, scan it, put it down. One at a time.

If you have 8 items, you scan 8 times. Simple, but slow.

**The fast way (SIMD processing):**
Now imagine you have a **magic scanner** that can scan **4 items at once**. You grab 4 items in one hand, pass them all through the scanner in a single beep, and they're all done.

8 items? That's only **2 scans** instead of 8.

> **That's SIMD.** One instruction. Multiple data. Same operation applied to a batch at once.

### 1.2 Why Does This Matter?

Your CPU already has this "magic scanner" built in. It's been there since the late 1990s. Most programs just don't use it.

Think about it:
- A video has millions of pixels. Each pixel needs the same brightness adjustment.
- A game calculates physics for thousands of particles. Same formula, different numbers.
- Audio processing applies the same filter to millions of samples.

**Same operation + lots of data = perfect for SIMD.**

### 1.3 Visual Model

```
SCALAR (one at a time):
  Step 1:  a[0] + b[0] = c[0]
  Step 2:  a[1] + b[1] = c[1]
  Step 3:  a[2] + b[2] = c[2]
  Step 4:  a[3] + b[3] = c[3]
  → 4 instructions

SIMD (all at once):
  Step 1:  [a[0], a[1], a[2], a[3]]  +  [b[0], b[1], b[2], b[3]]  =  [c[0], c[1], c[2], c[3]]
  → 1 instruction
```

One CPU cycle does the work of four. **4x speedup**, theoretically, from a *single core*.

### 1.4 Real-World Analogy: The Assembly Line

In a car factory:
- **Scalar** = one worker builds an entire car, start to finish.
- **SIMD** = one worker tightens 4 bolts simultaneously with a multi-head wrench.

The worker is the same. The tool is different. SIMD is the better tool.

---

## Part II: Professional Deep Dive

### 2.1 What Is SIMD, Precisely?

**SIMD (Single Instruction, Multiple Data)** is a class of parallel processing described in Flynn's taxonomy (1966). A single instruction operates on multiple data elements packed into wide registers.

| Taxonomy | Instruction Stream | Data Stream |
|----------|-------------------|-------------|
| SISD     | Single            | Single      |
| **SIMD** | **Single**        | **Multiple**|
| MIMD     | Multiple          | Multiple    |

Modern CPUs implement SIMD through **vector extensions**:

| Extension  | Register Width | Floats per Register | Year |
|-----------|---------------|---------------------|------|
| SSE       | 128-bit       | 4 x float32        | 1999 |
| AVX       | 256-bit       | 8 x float32        | 2011 |
| AVX-512   | 512-bit       | 16 x float32       | 2017 |

### 2.2 The Hardware: Vector Registers

Your CPU has **special wide registers** dedicated to SIMD:

```
┌──────────────────────────────────────────────────────────┐
│                    256-bit YMM Register (AVX)            │
├──────────┬──────────┬──────────┬──────────┬──────────────┤
│ float[0] │ float[1] │ float[2] │ float[3] │ float[4]... │
│  32-bit  │  32-bit  │  32-bit  │  32-bit  │   32-bit    │
└──────────┴──────────┴──────────┴──────────┴──────────────┘
```

- **SSE registers:** `XMM0`–`XMM15` (128-bit each)
- **AVX registers:** `YMM0`–`YMM15` (256-bit each, XMM is the lower half)
- **AVX-512 registers:** `ZMM0`–`ZMM31` (512-bit each)

### 2.3 Memory Alignment

SIMD loads are fastest when data is **aligned** to the register width.

```
Aligned (16-byte boundary for SSE):
  Address: 0x00, 0x10, 0x20, 0x30  → fast _mm_load_ps

Unaligned:
  Address: 0x03, 0x13, 0x23        → slower _mm_loadu_ps (penalty on older CPUs)
```

To align data in C++:

```cpp
// C++11 and later
alignas(32) float data[1024];  // 32-byte aligned for AVX

// Or use platform-specific allocation
#include <cstdlib>
float* data = (float*)aligned_alloc(32, 1024 * sizeof(float));
```

### 2.4 Three Levels of SIMD Usage

#### Level 1: Compiler Auto-Vectorization (Easiest)

The compiler tries to vectorize your loops automatically.

```cpp
// auto_vec.cpp
void add_arrays(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // Compiler may auto-vectorize this
    }
}
```

Compile with optimization flags:
```bash
g++ -O3 -march=native -ftree-vectorize -fopt-info-vec-optimized auto_vec.cpp
#                                        ^^^ this flag shows what got vectorized
```

**Why auto-vectorization fails:**
```cpp
// FAILS: pointer aliasing — compiler can't prove a and c don't overlap
void add_arrays(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// FIX: use __restrict__ to promise no aliasing
void add_arrays(float* __restrict__ a, float* __restrict__ b,
                float* __restrict__ c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // Now the compiler can vectorize
    }
}
```

Common reasons auto-vectorization fails:
1. **Pointer aliasing** — use `__restrict__`
2. **Loop-carried dependencies** — `a[i] = a[i-1] + 1` can't be parallelized
3. **Function calls inside loops** — compiler can't inline/vectorize external calls
4. **Complex control flow** — `if/else` inside tight loops
5. **Non-contiguous memory access** — scatter/gather patterns

#### Level 2: Compiler Pragmas (Guided)

```cpp
#include <cstddef>

void add_arrays(float* a, float* b, float* c, size_t n) {
    #pragma omp simd                     // "I promise this loop is safe to vectorize"
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

```bash
g++ -O2 -fopenmp-simd -march=native add_pragma.cpp
```

#### Level 3: Manual Intrinsics (Full Control)

When you need **absolute control** over what the CPU does:

```cpp
#include <immintrin.h>  // AVX/SSE intrinsics

void add_arrays_avx(float* a, float* b, float* c, int n) {
    int i = 0;

    // Process 8 floats at a time with AVX (256-bit)
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);   // Load 8 floats from a
        __m256 vb = _mm256_load_ps(&b[i]);   // Load 8 floats from b
        __m256 vc = _mm256_add_ps(va, vb);   // Add all 8 pairs at once
        _mm256_store_ps(&c[i], vc);          // Store 8 results to c
    }

    // Handle remaining elements (tail loop)
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

**Naming convention for intrinsics:**
```
_mm256_add_ps
  │     │   │
  │     │   └── ps = packed single-precision (float)
  │     │       pd = packed double-precision (double)
  │     │       epi32 = packed 32-bit integers
  │     │       si256 = 256-bit integer
  │     │
  │     └── operation (add, sub, mul, fmadd, load, store, set...)
  │
  └── register width: _mm = 128-bit (SSE), _mm256 = 256-bit (AVX), _mm512 = 512-bit (AVX-512)
```

### 2.5 Fused Multiply-Add (FMA)

One of the most powerful SIMD instructions. Computes `a * b + c` in a **single instruction** with **one rounding** (more accurate than separate mul + add):

```cpp
#include <immintrin.h>

// Dot product: sum of a[i] * b[i]
float dot_product_fma(const float* a, const float* b, int n) {
    __m256 sum = _mm256_setzero_ps();  // sum = [0, 0, 0, 0, 0, 0, 0, 0]

    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);  // sum += a[i..i+7] * b[i..i+7]
    }

    // Horizontal sum: reduce 8 floats to 1
    __m128 hi = _mm256_extractf128_ps(sum, 1);       // [s4, s5, s6, s7]
    __m128 lo = _mm256_castps256_ps128(sum);          // [s0, s1, s2, s3]
    __m128 sum128 = _mm_add_ps(lo, hi);              // [s0+s4, s1+s5, s2+s6, s3+s7]
    sum128 = _mm_hadd_ps(sum128, sum128);             // pairwise horizontal add
    sum128 = _mm_hadd_ps(sum128, sum128);             // final horizontal add

    float result;
    _mm_store_ss(&result, sum128);

    // Add remaining elements
    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}
```

### 2.6 C++26 `std::simd` — The Portable Future

Manual intrinsics are powerful but **not portable** (SSE won't run on ARM, AVX-512 won't run on older CPUs). C++26 introduces `std::simd`:

```cpp
// Requires: GCC 11+ with -std=c++23 and <experimental/simd>
// Full std::simd in C++26

#include <experimental/simd>
namespace stdx = std::experimental;

void add_arrays_stdsimd(float* a, float* b, float* c, int n) {
    using V = stdx::native_simd<float>;  // Compiler picks best width
    constexpr int lanes = V::size();     // e.g., 8 for AVX, 16 for AVX-512

    int i = 0;
    for (; i + lanes - 1 < n; i += lanes) {
        V va(&a[i], stdx::element_aligned);
        V vb(&b[i], stdx::element_aligned);
        V vc = va + vb;                     // Natural C++ syntax!
        vc.copy_to(&c[i], stdx::element_aligned);
    }

    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

**Why this matters:**
- Same code compiles to **SSE on old machines**, **AVX on modern x86**, **NEON on ARM**
- No `#ifdef` hell
- Natural C++ operators (`+`, `*`, `<`) instead of cryptic intrinsics

### 2.7 Cache Efficiency and Data Layout

SIMD is useless if you're waiting for data from memory. **Cache locality is king.**

```
                ┌─────────┐
                │ Register│  < 1 cycle
                └────┬────┘
                     ↓
                ┌─────────┐
                │ L1 Cache│  ~4 cycles    (32-64 KB)
                └────┬────┘
                     ↓
                ┌─────────┐
                │ L2 Cache│  ~12 cycles   (256 KB - 1 MB)
                └────┬────┘
                     ↓
                ┌─────────┐
                │ L3 Cache│  ~40 cycles   (8-64 MB)
                └────┬────┘
                     ↓
                ┌─────────┐
                │  DRAM   │  ~200 cycles  (GBs)
                └─────────┘
```

**Array of Structures (AoS) vs Structure of Arrays (SoA):**

```cpp
// AoS — BAD for SIMD
// Memory layout: [x0,y0,z0, x1,y1,z1, x2,y2,z2, ...]
// Loading all x values requires gather (slow)
struct Particle_AoS {
    float x, y, z;
    float vx, vy, vz;
};
Particle_AoS particles[1000];

// SoA — GOOD for SIMD
// Memory layout: [x0,x1,x2,..., y0,y1,y2,..., z0,z1,z2,...]
// Loading x values is contiguous (fast SIMD load)
struct Particles_SoA {
    float x[1000];
    float y[1000];
    float z[1000];
    float vx[1000];
    float vy[1000];
    float vz[1000];
};
Particles_SoA particles;
```

**Rule of thumb:** If you process one field across many objects, use SoA. If you process all fields of one object, use AoS.

---

## Part III: Practical Example — Benchmarked Dot Product

### 3.1 Full Compilable Example

```cpp
// dot_product_benchmark.cpp
// Compile: g++ -O3 -mavx2 -mfma -std=c++17 -o dot_bench dot_product_benchmark.cpp
// Run:     ./dot_bench

#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

constexpr int N = 1 << 24;  // ~16 million elements
constexpr int RUNS = 100;

// ─── Scalar version ───
float dot_scalar(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ─── AVX2 + FMA version ───
float dot_avx2(const float* a, const float* b, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();  // Two accumulators for ILP

    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 a0 = _mm256_load_ps(&a[i]);
        __m256 b0 = _mm256_load_ps(&b[i]);
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);

        __m256 a1 = _mm256_load_ps(&a[i + 8]);
        __m256 b1 = _mm256_load_ps(&b[i + 8]);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
    }

    sum0 = _mm256_add_ps(sum0, sum1);  // Merge accumulators

    // Handle remaining
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        sum0 = _mm256_fmadd_ps(va, vb, sum0);
    }

    // Horizontal reduction
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 r = _mm_add_ps(lo, hi);
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);

    float result;
    _mm_store_ss(&result, r);

    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}

// ─── Benchmark helper ───
template<typename Func>
double benchmark(Func fn, const float* a, const float* b, int n, float& result) {
    // Warmup
    result = fn(a, b, n);

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) {
        result = fn(a, b, n);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / RUNS;
    return ms;
}

int main() {
    // Allocate aligned memory
    float* a = (float*)aligned_alloc(32, N * sizeof(float));
    float* b = (float*)aligned_alloc(32, N * sizeof(float));

    // Initialize with small values to avoid overflow
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(i % 100) * 0.01f;
        b[i] = static_cast<float>(i % 100) * 0.01f;
    }

    float result_scalar, result_avx;

    double ms_scalar = benchmark(dot_scalar, a, b, N, result_scalar);
    double ms_avx    = benchmark(dot_avx2,   a, b, N, result_avx);

    std::cout << "=== Dot Product Benchmark (N = " << N << ") ===" << std::endl;
    std::cout << "Scalar:  " << ms_scalar << " ms  (result: " << result_scalar << ")" << std::endl;
    std::cout << "AVX2:    " << ms_avx    << " ms  (result: " << result_avx    << ")" << std::endl;
    std::cout << "Speedup: " << ms_scalar / ms_avx << "x" << std::endl;

    // Verify correctness
    float rel_error = std::fabs(result_scalar - result_avx) / std::fabs(result_scalar);
    std::cout << "Relative error: " << rel_error << std::endl;

    free(a);
    free(b);
    return 0;
}
```

### 3.2 Expected Output

```
=== Dot Product Benchmark (N = 16777216) ===
Scalar:  12.34 ms  (result: 5.49442e+06)
AVX2:     2.15 ms  (result: 5.49442e+06)
Speedup: 5.74x
Relative error: 1.23e-07
```

> **Note:** Speedup exceeds 4x because of FMA (fused multiply-add counts as 2 FLOP) and dual accumulators (instruction-level parallelism hides latency).

### 3.3 Why Two Accumulators?

```
Without ILP (1 accumulator):
  Cycle 1: load a0, b0 → FMA → sum0    (FMA latency: ~4 cycles)
  Cycle 5: load a1, b1 → FMA → sum0    (must wait for sum0)
  → Pipeline stalls waiting for previous FMA to finish

With ILP (2 accumulators):
  Cycle 1: load a0, b0 → FMA → sum0
  Cycle 2: load a1, b1 → FMA → sum1    (independent! runs in parallel)
  Cycle 3: load a2, b2 → FMA → sum0    (sum0 from cycle 1 is now ready)
  → Pipeline stays full, throughput doubles
```

---

## Part IV: Q&A — Questions Students Actually Ask

### Q1: "If SIMD is so fast, why doesn't the compiler always use it?"

**A:** The compiler is conservative. It must **prove** the transformation is safe. If it can't guarantee that pointers don't overlap (aliasing), that there are no loop-carried dependencies, or that the loop count is large enough to justify SIMD overhead — it won't vectorize.

**Practical tip:** Always check the vectorization report:
```bash
g++ -O3 -march=native -fopt-info-vec-missed source.cpp
```

---

### Q2: "What's the difference between SIMD and multithreading?"

**A:** They're orthogonal:

| | SIMD | Multithreading |
|---|---|---|
| **Level** | Instruction-level | Thread-level |
| **Cores used** | 1 core | Multiple cores |
| **Parallelism** | Data parallelism (same op, different data) | Task parallelism (different ops) |
| **Max speedup** | Width-limited (4x–16x per core) | Core-limited (e.g., 8x on 8 cores) |

**Best performance = SIMD + multithreading.** Each core uses SIMD. Eight cores, each doing 8-wide AVX = 64x potential throughput vs scalar single-core.

---

### Q3: "When should I NOT use SIMD?"

**A:**
1. **Branchy code** — SIMD processes all lanes. If each element needs different logic, you waste lanes.
2. **Tiny data** — SIMD has overhead (wider loads, horizontal reductions). Below ~64 elements, scalar may win.
3. **Memory-bound workloads** — If you're waiting on DRAM, faster arithmetic doesn't help. Optimize memory access first.
4. **Non-contiguous data** — Gather/scatter operations exist but are slow. Restructure data (SoA) first.

---

### Q4: "Does AVX-512 always beat AVX2?"

**A:** Not always. On some Intel CPUs, the core **downclocks** when executing AVX-512 instructions (thermal throttling). For short bursts of AVX-512 surrounded by scalar code, the frequency transition overhead can erase the gains.

**Rule:** Profile, don't assume. Use `perf stat` to check actual clock frequencies:
```bash
perf stat -e cycles,instructions,cpu-clock ./my_program
```

---

### Q5: "How do I know if my loop was vectorized?"

**A:** Three ways:
1. **Compiler reports:** `-fopt-info-vec` (GCC), `-Rpass=loop-vectorize` (Clang)
2. **Disassembly:** Look for `vaddps`, `vmulps`, `vfmadd` (AVX) vs `addss`, `mulss` (scalar)
   ```bash
   objdump -d -M intel my_program | grep -E "vadd|vmul|vfma"
   ```
3. **Performance:** If your loop processes floats and doesn't get ~4x speedup with `-O3 -mavx2`, something went wrong.

---

### Q6: "What about ARM? I'm on a Mac with Apple Silicon."

**A:** ARM uses **NEON** (128-bit, always available on ARMv8) and **SVE/SVE2** (variable width, 128–2048 bit, server chips).

The intrinsics are different (`vaddq_f32` instead of `_mm_add_ps`), which is exactly why `std::simd` (C++26) matters — write once, run optimally on both x86 and ARM.

On Apple M-series: NEON is 128-bit, the compiler auto-vectorizes well with `-O3`. Apple's Accelerate framework wraps NEON for common operations.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| SIMD | One instruction operates on multiple data elements simultaneously |
| Registers | XMM (128-bit), YMM (256-bit), ZMM (512-bit) |
| Alignment | Align data to register width for fastest loads |
| Auto-vectorization | Compiler does it, but fragile — check reports |
| Intrinsics | Full control, not portable — use when auto-vec fails |
| `std::simd` | C++26 portable solution — use when available |
| ILP | Multiple accumulators hide FMA latency |
| SoA | Structure of Arrays layout enables contiguous SIMD loads |

---

## Next Lecture

**Lecture 2:** Masking, Shuffles, and Real-World Patterns (image processing, filtering, horizontal operations)

---

*HPC Course — Chapter 1: SIMD Fundamentals*
*License: MIT*
