# Lecture 1: OpenMP — Shared-Memory Parallel Programming

## Chapter 2 | OpenMP and OneTBB (CPU Multithreading)

**Duration:** ~2.5 hours
**Prerequisites:** Chapter 1 (SIMD), basic understanding of processes and threads
**Objective:** Distribute work across all CPU cores using OpenMP directives.

---

## Part I: The High School Explanation

### 1.1 The Pizza Kitchen Analogy

You're running a pizza shop. One order comes in: **make 100 pizzas**.

**Single-threaded (one cook):**
Cook #1 makes pizza 1, then pizza 2, then pizza 3... all the way to 100.
Takes forever. The other cooks sit idle, scrolling their phones.

**Multi-threaded (four cooks, OpenMP):**
You shout: *"Everyone! Each of you take 25 pizzas. Go!"*

- Cook #1: pizzas 1–25
- Cook #2: pizzas 26–50
- Cook #3: pizzas 51–75
- Cook #4: pizzas 76–100

Same 100 pizzas. **4x faster** (approximately). All you did was give one instruction to the team.

> **That's OpenMP.** You write a hint above your loop — `#pragma omp parallel for` — and the compiler splits the work across all CPU cores. It's that simple.

### 1.2 But There's a Catch: The Shared Sink

All four cooks share **one sink** to wash their hands. If cook #1 and cook #3 both rush to the sink at the same time, they bump into each other. One has to wait.

In computing, the "sink" is **shared memory** — a variable that multiple threads read and write. If two threads modify the same variable simultaneously, the result is garbage. This is called a **race condition**.

```
Thread 1:  reads counter (= 5)
Thread 2:  reads counter (= 5)
Thread 1:  writes counter = 5 + 1 = 6
Thread 2:  writes counter = 5 + 1 = 6   ← WRONG! Should be 7
```

OpenMP gives you tools to handle this: **locks** (only one cook at the sink), **reductions** (each cook counts privately, then combines), and **atomic operations** (the sink has a traffic light).

### 1.3 When Parallelism Helps (and When It Doesn't)

**Helps:**
- Processing 10 million data points independently
- Rendering different pixels of an image
- Running independent simulations

**Doesn't help:**
- Reading a single file sequentially
- Tasks where step N depends on step N-1
- Tiny workloads (overhead of creating threads > time saved)

> **Amdahl's Law (high school version):**
> If 90% of your program can run in parallel and 10% must be serial,
> then even with infinite cores, you can only get a **10x speedup** (the serial 10% becomes the bottleneck).

---

## Part II: Professional Deep Dive

### 2.1 OpenMP Execution Model

OpenMP uses the **fork-join** model:

```
Main Thread (master)
    │
    ├──── #pragma omp parallel ────┐
    │                               │
    │  ┌─────┬─────┬─────┬─────┐  │
    │  │ T0  │ T1  │ T2  │ T3  │  │   ← FORK: team of threads
    │  │work │work │work │work │  │
    │  └──┬──┴──┬──┴──┬──┴──┬──┘  │
    │     └─────┴─────┴─────┘     │
    │              │               │
    ├──────────────┘               │   ← JOIN: threads synchronize
    │                               │
    │  (single thread continues)   │
    ▼
```

Key properties:
- Threads share the **same address space** (heap, globals)
- Each thread has its own **stack** (local variables are private by default)
- An **implicit barrier** exists at the end of every parallel region

### 2.2 Core Directives

#### The Basics: `parallel for`

```cpp
// parallel_for_demo.cpp
// Compile: g++ -O3 -fopenmp -o par_demo parallel_for_demo.cpp
// Run:     OMP_NUM_THREADS=8 ./par_demo

#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    constexpr int N = 100'000'000;
    std::vector<float> a(N), b(N), c(N);

    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = std::sin(i * 0.001f);
        b[i] = std::cos(i * 0.001f);
    }

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        c[i] = a[i] * a[i] + b[i] * b[i];  // Independent iterations
    }

    double end = omp_get_wtime();
    std::cout << "Time: " << (end - start) * 1000.0 << " ms" << std::endl;
    std::cout << "Threads used: " << omp_get_max_threads() << std::endl;

    return 0;
}
```

#### Data Sharing Clauses

```cpp
int shared_var = 42;
int private_var = 0;

#pragma omp parallel private(private_var) shared(shared_var)
{
    // private_var: each thread gets its OWN uninitialized copy
    // shared_var:  all threads see the SAME variable

    private_var = omp_get_thread_num();  // Safe: each has own copy
    // shared_var++;                     // DANGEROUS: race condition!
}
```

| Clause | Behavior |
|--------|----------|
| `shared(x)` | All threads share one `x`. Must protect writes. |
| `private(x)` | Each thread gets its own `x`, **uninitialized**. |
| `firstprivate(x)` | Each thread gets its own `x`, **initialized** from original. |
| `lastprivate(x)` | Like `private`, but the last iteration's value is copied back. |
| `reduction(+:x)` | Each thread gets private `x`, then all are **combined** with `+`. |

#### Reduction — Safe Accumulation

```cpp
// Monte Carlo estimation of Pi
#include <omp.h>
#include <random>
#include <iostream>

int main() {
    constexpr long N = 100'000'000;
    long count = 0;

    #pragma omp parallel reduction(+:count)
    {
        // Each thread gets its own random generator (important!)
        std::mt19937 rng(omp_get_thread_num() * 12345 + 42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        #pragma omp for
        for (long i = 0; i < N; i++) {
            float x = dist(rng);
            float y = dist(rng);
            if (x * x + y * y <= 1.0f) {
                count++;  // Safe! Each thread has private count
            }
        }
    }
    // After the parallel region, all private counts are summed into 'count'

    double pi = 4.0 * count / N;
    std::cout << "Pi ≈ " << pi << std::endl;

    return 0;
}
```

### 2.3 Scheduling Strategies

When iterations have **unequal work**, scheduling matters:

```cpp
#pragma omp parallel for schedule(static)    // Equal chunks, no overhead
#pragma omp parallel for schedule(dynamic)   // Threads grab chunks as they finish
#pragma omp parallel for schedule(guided)    // Chunks shrink over time
#pragma omp parallel for schedule(auto)      // Let runtime decide
```

```
STATIC (default):
  Thread 0: [  0.. 24]
  Thread 1: [ 25.. 49]    ← If work is uniform, this is optimal (zero overhead)
  Thread 2: [ 50.. 74]
  Thread 3: [ 75.. 99]

DYNAMIC:
  Thread 0 grabs [0..9],   finishes, grabs [40..49] ...
  Thread 1 grabs [10..19], finishes, grabs [50..59] ...
  Thread 2 grabs [20..29], finishes, grabs [60..69] ...
  ← Great when some iterations take longer (load balancing)
  ← Has overhead: threads must synchronize to grab work
```

**When to use what:**

| Scenario | Schedule |
|----------|----------|
| All iterations equal work | `static` |
| Work varies per iteration (e.g., variable-length inner loop) | `dynamic` |
| Early iterations are heavy, later ones light | `guided` |
| Don't know | `auto` or experiment |

### 2.4 Synchronization Primitives

```cpp
// atomic — hardware-level single operation (fastest)
#pragma omp atomic
shared_counter++;

// critical — mutual exclusion block (slower, flexible)
#pragma omp critical
{
    shared_vector.push_back(result);  // Arbitrary code, one thread at a time
}

// barrier — all threads wait here until everyone arrives
#pragma omp barrier

// single — only one thread executes this block
#pragma omp single
{
    std::cout << "Initializing..." << std::endl;
}
```

### 2.5 NUMA Awareness

On multi-socket servers, memory is **physically closer** to some cores than others:

```
┌────────────────────┐    ┌────────────────────┐
│   Socket 0         │    │   Socket 1         │
│  ┌──────────────┐  │    │  ┌──────────────┐  │
│  │ Cores 0-7    │  │    │  │ Cores 8-15   │  │
│  └──────┬───────┘  │    │  └──────┬───────┘  │
│         │          │    │         │          │
│  ┌──────┴───────┐  │    │  ┌──────┴───────┐  │
│  │ LOCAL DRAM   │  │←──→│  │ LOCAL DRAM   │  │
│  │ (fast access)│  │ QPI │  │ (fast access)│  │
│  └──────────────┘  │    │  └──────────────┘  │
└────────────────────┘    └────────────────────┘
```

Core 0 accessing Socket 1's memory = **remote access** (2-3x slower).

**First-touch policy:** Memory is allocated on the NUMA node of the thread that first writes to it.

```cpp
// BAD: main thread initializes everything → all on Socket 0
for (int i = 0; i < N; i++) a[i] = 0;

// GOOD: parallel initialization → distributed across NUMA nodes
#pragma omp parallel for
for (int i = 0; i < N; i++) a[i] = 0;

// Control thread-to-core binding
// OMP_PROC_BIND=close     → threads stick near each other
// OMP_PROC_BIND=spread    → threads spread across sockets
// OMP_PLACES=cores        → one thread per physical core
```

### 2.6 OneTBB — Task-Based Parallelism

While OpenMP is **loop-centric**, Intel OneTBB is **task-centric**:

```cpp
// tbb_parallel_for.cpp
// Compile: g++ -O3 -std=c++17 -ltbb -o tbb_demo tbb_parallel_for.cpp

#include <tbb/tbb.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>

int main() {
    constexpr int N = 100'000'000;
    std::vector<float> a(N), b(N), c(N);

    // Initialize
    tbb::parallel_for(0, N, [&](int i) {
        a[i] = std::sin(i * 0.001f);
        b[i] = std::cos(i * 0.001f);
    });

    auto start = std::chrono::high_resolution_clock::now();

    // TBB automatically determines grain size and scheduling
    tbb::parallel_for(
        tbb::blocked_range<int>(0, N),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                c[i] = a[i] * a[i] + b[i] * b[i];
            }
        }
    );

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "TBB Time: " << ms << " ms" << std::endl;

    return 0;
}
```

#### TBB `parallel_reduce`

```cpp
// Parallel sum with TBB
float total = tbb::parallel_reduce(
    tbb::blocked_range<int>(0, N),
    0.0f,                                              // Identity value
    [&](const tbb::blocked_range<int>& r, float sum) { // Body
        for (int i = r.begin(); i < r.end(); i++) {
            sum += a[i];
        }
        return sum;
    },
    std::plus<float>()                                  // Reduction operator
);
```

#### TBB Flow Graph — Pipeline Parallelism

```cpp
// Image processing pipeline: Read → Filter → Compress → Write
#include <tbb/flow_graph.h>

tbb::flow::graph g;

// Each node processes different stages IN PARALLEL
tbb::flow::function_node<ImageFrame, ImageFrame> read_node(g, 1, read_frame);
tbb::flow::function_node<ImageFrame, ImageFrame> filter_node(g, tbb::flow::unlimited, apply_filter);
tbb::flow::function_node<ImageFrame, ImageFrame> compress_node(g, 4, compress_frame);
tbb::flow::function_node<ImageFrame> write_node(g, 1, write_frame);

tbb::flow::make_edge(read_node, filter_node);
tbb::flow::make_edge(filter_node, compress_node);
tbb::flow::make_edge(compress_node, write_node);

// Feed frames
for (int i = 0; i < num_frames; i++) {
    read_node.try_put(frames[i]);
}
g.wait_for_all();
```

### 2.7 OpenMP vs TBB — When to Use Which

| Feature | OpenMP | OneTBB |
|---------|--------|--------|
| **Ease of use** | Pragmas, minimal code change | API-based, more verbose |
| **Best for** | Regular loops, HPC | Irregular tasks, pipelines |
| **Scheduling** | You choose (static/dynamic/guided) | Automatic work-stealing |
| **Composability** | Nested parallelism is tricky | Naturally composable |
| **GPU offload** | Yes (`target` directive, OpenMP 5.0+) | No (CPU only) |
| **Language** | C, C++, Fortran | C++ only |
| **Vendor** | Standard (all compilers) | Intel (open-source) |

---

## Part III: Practical Example — Parallel Monte Carlo Pi

### 3.1 Full Compilable Example

```cpp
// monte_carlo_pi.cpp
// Compile: g++ -O3 -fopenmp -std=c++17 -o mc_pi monte_carlo_pi.cpp
// Run:     OMP_NUM_THREADS=8 ./mc_pi

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

    // ─── Serial version ───
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

        double elapsed = omp_get_wtime() - start;
        double pi = 4.0 * count / TOTAL_SAMPLES;
        std::cout << "Serial:   Pi = " << std::fixed << std::setprecision(8)
                  << pi << "  Time: " << elapsed * 1000 << " ms" << std::endl;
    }

    // ─── Parallel version (OpenMP) ───
    {
        double start = omp_get_wtime();
        long count = 0;

        #pragma omp parallel reduction(+:count)
        {
            int tid = omp_get_thread_num();
            std::mt19937 rng(tid * 99991 + 42);  // Different seed per thread!
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            #pragma omp for schedule(static)
            for (long i = 0; i < TOTAL_SAMPLES; i++) {
                float x = dist(rng);
                float y = dist(rng);
                if (x * x + y * y <= 1.0f) count++;
            }
        }

        double elapsed = omp_get_wtime() - start;
        double pi = 4.0 * count / TOTAL_SAMPLES;
        std::cout << "Parallel: Pi = " << std::fixed << std::setprecision(8)
                  << pi << "  Time: " << elapsed * 1000 << " ms" << std::endl;
    }

    return 0;
}
```

### 3.2 Expected Output

```
Monte Carlo Pi Estimation
Samples: 500000000
Threads: 8
=========================
Serial:   Pi = 3.14161280  Time: 4523.12 ms
Parallel: Pi = 3.14158984  Time:  612.45 ms
```

Speedup: ~7.4x on 8 cores (not exactly 8x due to overhead and memory bandwidth).

---

## Part IV: Q&A — Questions Students Actually Ask

### Q1: "Why not just use `std::thread`?"

**A:** You can! But OpenMP handles:
- Thread creation/destruction (thread pool, no repeated fork cost)
- Work distribution (automatic chunk splitting)
- Reductions (safe accumulation)
- Synchronization (barriers, critical sections)

Writing equivalent `std::thread` code for a parallel for-loop requires ~40 lines vs 1 `#pragma` line.

**Use `std::thread`** when you need fine-grained control over individual threads (e.g., producer-consumer patterns). **Use OpenMP** when you want to parallelize compute-heavy loops quickly.

---

### Q2: "What's a race condition and how do I find them?"

**A:** A race condition occurs when two threads access the same memory location, at least one writes, and there's no synchronization.

Detection tools:
```bash
# ThreadSanitizer (best tool for this)
g++ -fsanitize=thread -g -fopenmp -o prog prog.cpp
./prog
# Output: "WARNING: ThreadSanitizer: data race on address 0x..."

# Helgrind (Valgrind-based)
valgrind --tool=helgrind ./prog
```

---

### Q3: "What is false sharing and why does it kill performance?"

**A:** False sharing occurs when threads write to different variables that happen to be on the **same cache line** (typically 64 bytes).

```cpp
// BAD: false sharing — all counters on same cache line
int counters[8];  // 8 ints = 32 bytes → fits in one 64-byte cache line

#pragma omp parallel
{
    int tid = omp_get_thread_num();
    for (int i = 0; i < 1000000; i++) {
        counters[tid]++;  // Each thread writes its own element
        // BUT: the hardware invalidates the entire cache line for all cores
    }
}

// GOOD: pad to avoid false sharing
struct alignas(64) PaddedCounter {
    int value;
};
PaddedCounter counters[8];  // Each counter on its own cache line
```

False sharing can cause a 10-50x slowdown because the CPU constantly invalidates and reloads cache lines across cores.

---

### Q4: "How many threads should I use?"

**A:** Rules of thumb:
- **Compute-bound:** `num_threads = num_physical_cores` (not hyperthreads)
- **Memory-bound:** Fewer threads may be better (memory bandwidth saturates)
- **I/O-bound:** More threads than cores can help (threads wait on I/O)

Check your system:
```bash
lscpu | grep -E "^CPU\(s\)|Thread|Core|Socket"
# Example output:
# CPU(s):              16
# Thread(s) per core:  2     ← Hyperthreading
# Core(s) per socket:  8     ← Physical cores per socket
# Socket(s):           1
# → Use OMP_NUM_THREADS=8 for compute-bound work
```

---

### Q5: "Can I combine OpenMP with SIMD from Chapter 1?"

**A:** Yes! This is the ideal setup:

```cpp
#pragma omp parallel for schedule(static)  // Distribute across cores
for (int i = 0; i < N; i += 8) {
    __m256 va = _mm256_load_ps(&a[i]);     // SIMD within each core
    __m256 vb = _mm256_load_ps(&b[i]);
    __m256 vc = _mm256_add_ps(va, vb);
    _mm256_store_ps(&c[i], vc);
}

// Or let the compiler do both:
#pragma omp parallel for simd
for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];  // OpenMP threads + auto-vectorization
}
```

**Peak throughput** = (cores) x (SIMD width) x (clock frequency) x (FLOPs per instruction)
Example: 8 cores x 8-wide AVX x 3.5 GHz x 2 FMA = **448 GFLOP/s** (single precision)

---

### Q6: "What is Amdahl's Law, precisely?"

**A:**

```
Speedup(N) = 1 / ( s + (1 - s) / N )

Where:
  s = fraction of program that is serial (cannot be parallelized)
  N = number of processors
```

| Serial fraction (s) | Max speedup (N→inf) |
|---------------------|---------------------|
| 50% | 2x |
| 10% | 10x |
| 5%  | 20x |
| 1%  | 100x |

**Gustafson's Law** (alternative view): As you add cores, you solve **bigger problems**, not the same problem faster. The serial fraction shrinks relative to total work.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Fork-Join Model | Master thread forks, team works, threads join |
| `parallel for` | Simplest way to parallelize a loop |
| Data Sharing | `shared` = one copy, `private` = per-thread copy |
| Reduction | Safe accumulation: `reduction(+:sum)` |
| Scheduling | `static` for uniform, `dynamic` for variable work |
| NUMA | Initialize data in parallel for correct placement |
| False Sharing | Pad shared arrays to cache-line boundaries |
| OneTBB | Task-based, work-stealing, great for pipelines |
| Amdahl's Law | Serial fraction limits maximum speedup |

---

## Next Lecture

**Lecture 2:** Advanced OpenMP — Tasks, Nested Parallelism, and GPU Offloading (`target` directive)

---

*HPC Course — Chapter 2: OpenMP Fundamentals*
*License: MIT*
