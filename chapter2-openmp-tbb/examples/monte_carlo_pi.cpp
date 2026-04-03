// monte_carlo_pi.cpp
// ═══════════════════════════════════════════════════════════════════
// HPC Course — Chapter 2: OpenMP Fundamentals
// Benchmark: Serial vs Parallel Monte Carlo Pi estimation
//
// Compile: g++ -O3 -fopenmp -std=c++17 -o mc_pi monte_carlo_pi.cpp
// Run:     OMP_NUM_THREADS=8 ./mc_pi
// ═══════════════════════════════════════════════════════════════════

#include <omp.h>
#include <iostream>
#include <random>
#include <iomanip>

int main() {
    constexpr long TOTAL_SAMPLES = 500'000'000L;
    const int num_threads = omp_get_max_threads();

    std::cout << "Monte Carlo Pi Estimation" << std::endl;
    std::cout << "Samples: " << TOTAL_SAMPLES << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "=========================" << std::endl;

    // ─── Serial version ───────────────────────────────────────────
    double serial_pi, serial_ms;
    {
        double start = omp_get_wtime();
        long count = 0;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (long i = 0; i < TOTAL_SAMPLES; i++) {
            float x = dist(rng);
            float y = dist(rng);
            if (x * x + y * y <= 1.0f) count++;
        }

        serial_ms = (omp_get_wtime() - start) * 1000.0;
        serial_pi = 4.0 * count / TOTAL_SAMPLES;
        std::cout << "Serial:   Pi = " << std::fixed << std::setprecision(8)
                  << serial_pi << "  Time: " << serial_ms << " ms" << std::endl;
    }

    // ─── Parallel version (OpenMP) ────────────────────────────────
    double parallel_pi, parallel_ms;
    {
        double start = omp_get_wtime();
        long count = 0;

        #pragma omp parallel reduction(+:count)
        {
            int tid = omp_get_thread_num();
            std::mt19937 rng(tid * 99991 + 42);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            #pragma omp for schedule(static)
            for (long i = 0; i < TOTAL_SAMPLES; i++) {
                float x = dist(rng);
                float y = dist(rng);
                if (x * x + y * y <= 1.0f) count++;
            }
        }

        parallel_ms = (omp_get_wtime() - start) * 1000.0;
        parallel_pi = 4.0 * count / TOTAL_SAMPLES;
        std::cout << "Parallel: Pi = " << std::fixed << std::setprecision(8)
                  << parallel_pi << "  Time: " << parallel_ms << " ms" << std::endl;
    }

    std::cout << "Speedup:  " << std::fixed << std::setprecision(2)
              << serial_ms / parallel_ms << "x" << std::endl;

    return 0;
}
