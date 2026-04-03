// dot_product_benchmark.cpp
// ═══════════════════════════════════════════════════════════════════
// HPC Course — Chapter 1: SIMD Fundamentals
// Benchmark: Scalar vs AVX2+FMA dot product
//
// Compile: g++ -O3 -mavx2 -mfma -std=c++17 -o dot_bench dot_product_benchmark.cpp
// Run:     ./dot_bench
// ═══════════════════════════════════════════════════════════════════

#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

constexpr int N = 1 << 24;  // ~16 million elements
constexpr int RUNS = 100;

// ─── Scalar version ───────────────────────────────────────────────
float dot_scalar(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ─── AVX2 + FMA version (dual accumulator for ILP) ───────────────
float dot_avx2(const float* a, const float* b, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 a0 = _mm256_load_ps(&a[i]);
        __m256 b0 = _mm256_load_ps(&b[i]);
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);

        __m256 a1 = _mm256_load_ps(&a[i + 8]);
        __m256 b1 = _mm256_load_ps(&b[i + 8]);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
    }

    sum0 = _mm256_add_ps(sum0, sum1);

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

// ─── Benchmark helper ─────────────────────────────────────────────
template<typename Func>
double benchmark(Func fn, const float* a, const float* b, int n, float& result) {
    result = fn(a, b, n);  // warmup

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; r++) {
        result = fn(a, b, n);
    }
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / RUNS;
}

int main() {
    float* a = (float*)aligned_alloc(32, N * sizeof(float));
    float* b = (float*)aligned_alloc(32, N * sizeof(float));

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

    float rel_error = std::fabs(result_scalar - result_avx) / std::fabs(result_scalar);
    std::cout << "Relative error: " << rel_error << std::endl;

    free(a);
    free(b);
    return 0;
}
