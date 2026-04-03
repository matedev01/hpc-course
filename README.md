# 🚀 High-Performance Computing (HPC) Mastery

### SIMD → Multithreading → CUDA → ROCm → SYCL

![HPC](https://img.shields.io/badge/Focus-High%20Performance%20Computing-blue)
![C++](https://img.shields.io/badge/Language-C%2B%2B-00599C?logo=c%2B%2B)
![CUDA](https://img.shields.io/badge/GPU-CUDA-76B900?logo=nvidia)
![ROCm](https://img.shields.io/badge/GPU-ROCm-E01F27)
![SYCL](https://img.shields.io/badge/Portable-SYCL-purple)
![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-green)
![TBB](https://img.shields.io/badge/Intel-OneTBB-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ⚡ Build Real High-Performance Systems (Not Just Theory)

This repository is a **practical, engineering-focused course** on modern high-performance computing.
It teaches you how to **extract maximum performance from hardware**—from CPU vector units to GPUs.

> 💡 Focus: *fast MVP → production-grade optimized systems*

---

## 🧠 What Makes This Different?

Most courses stop at *“parallel programming basics.”*
This one goes deeper:

* 🔬 **Hardware-level understanding** (SIMD lanes, warps, memory hierarchy)
* ⚙️ **Real optimization techniques** (coalescing, tiling, cache locality)
* 📊 **Profiling-driven development** (Nsight, VTune, perf, rocprof)
* 🏭 **Production mindset** (scalability, bottleneck removal, efficiency)

---

## 📸 Architecture Overview

```
        ┌───────────────┐
        │   SIMD (CPU)  │  → Vectorization (AVX, SSE)
        └──────┬────────┘
               ↓
        ┌───────────────┐
        │ Multi-Core CPU│  → OpenMP / OneTBB
        └──────┬────────┘
               ↓
        ┌───────────────┐
        │   GPU (CUDA)  │  → Thousands of threads (SIMT)
        └──────┬────────┘
               ↓
        ┌───────────────┐
        │ AMD GPU (HIP) │  → ROCm ecosystem
        └──────┬────────┘
               ↓
        ┌───────────────┐
        │ Portable HPC  │  → SYCL / OpenCL
        └───────────────┘
```

---

## 🎯 What You Will Learn

### 🧩 CPU Optimization

* SIMD (SSE, AVX, AVX-512)
* Auto-vectorization vs manual intrinsics
* Cache optimization & memory alignment

### 🧵 Parallel CPU Programming

* OpenMP (pragmas, scheduling, NUMA)
* OneTBB (task-based pipelines, flow graphs)

### 🔥 GPU Programming (Core Focus)

* CUDA architecture (warps, SMs, occupancy)
* Memory hierarchy (global, shared, registers)
* Kernel optimization (coalescing, tiling, ILP)
* Warp-level programming & reductions

### 🔴 AMD GPU (ROCm / HIP)

* CUDA → HIP porting
* Wavefront vs warp differences
* LDS optimization & profiling

### 🌍 Portable Parallelism

* OpenCL fundamentals
* SYCL modern C++ abstraction
* Cross-platform GPU execution

---

## 🛠️ Tech Stack

* **Languages:** C, C++ (Modern C++20/23/26 features)
* **CPU:** SIMD intrinsics, OpenMP, OneTBB
* **GPU:** CUDA, ROCm/HIP
* **Portable:** SYCL, OpenCL
* **Profiling:** Nsight, VTune, perf, rocprof

---

## 🧪 Hands-On Projects

| Chapter | Project                           |
| ------- | --------------------------------- |
| SIMD    | Vectorized dot product vs scalar  |
| OpenMP  | Parallel Monte Carlo simulation   |
| OneTBB  | Image processing pipeline         |
| CUDA    | Tiled matrix multiplication       |
| CUDA    | Reduction & prefix sum (scan)     |
| CUDA    | Convolution + shared memory       |
| ROCm    | CUDA → HIP port                   |
| SYCL    | Cross-platform kernel (CPU + GPU) |

---

## 📂 Repository Structure

```
hpc-course/
│
├── chapter1-simd/
├── chapter2-openmp-tbb/
├── chapter3-cuda/
├── chapter4-rocm-hip/
├── chapter5-sycl-opencl/
│
├── common/              # Utilities & shared code
├── benchmarks/          # Performance comparisons
├── profiling/           # Profiling configs/scripts
└── README.md
```

---

## 🧭 Learning Path

```
C++ Basics
   ↓
SIMD (Single Core Performance)
   ↓
Multithreading (CPU Scaling)
   ↓
CUDA (Massive Parallelism)
   ↓
ROCm (Cross-Vendor GPU)
   ↓
SYCL (Portable Future)
```

---

## 🧑‍💻 Who This Is For

* Embedded / firmware engineers moving into **high-performance systems**
* C++ developers wanting **low-level optimization skills**
* AI / robotics engineers needing **real-time performance**
* Engineers preparing for **HPC / GPU / systems roles**

---

## 📈 Outcomes

By completing this course, you will:

* Think in terms of **hardware execution**
* Optimize for **cache, memory, and compute**
* Build systems that scale from:

  * 🧠 Single core → 🧵 Multi-core → 🔥 GPU
* Deliver **cost-optimized, production-ready performance**

---

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/hpc-course.git
cd hpc-course
```

Each chapter contains:

* 📘 Theory
* 🧪 Labs
* ⚡ Optimization tasks
* 📊 Profiling exercises

---

## ⭐ Contributing

Contributions are welcome!
Feel free to open issues, suggest optimizations, or add new benchmarks.

---

## 📜 License

MIT License

---

## 💡 Future Additions

* FPGA acceleration (HLS, Vitis)
* AI accelerators (TensorRT, OpenVINO)
* Distributed computing (MPI)
