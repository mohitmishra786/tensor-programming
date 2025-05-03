# Tensor Programming in C: Building High-Performance Numerical Systems from Scratch

A hands-on, practical guide to implementing tensor operations, optimizations, and real-world applications in C. This book series takes you from basic tensor structures to high-performance neural networks and embedded deployments.

## GitHub Pages

This repository is set up with GitHub Pages to provide an easy-to-read format for the book. You can access the book online at: https://YOUR_USERNAME.github.io/tensor-programming/

## Chapters

Each chapter of the book is available in the `chapters` directory as Markdown files:

1. [Setting Up Your Tensor Programming Toolkit](chapters/chapter-01-setting-up-your-tensor-programming-toolkit.md)
2. [Implementing Core Tensor Operations from Scratch](chapters/chapter-02-implementing-core-tensor-operations-from-scratch.md)
3. [Mastering Memory Layouts for Speed](chapters/chapter-03-mastering-memory-layouts-for-speed.md)
4. [Parallelizing Tensor Workloads with OpenMP](chapters/chapter-04-parallelizing-tensor-workloads-with-openmp.md)
5. [Vectorizing Code with SIMD Intrinsics](chapters/chapter-05-vectorizing-code-with-simd-intrinsics.md)
6. [Integrating BLAS for Production-Grade Performance](chapters/chapter-06-integrating-blas-for-production-grade-performance.md)
7. [Debugging Memory Corruption in Tensor Programs](chapters/chapter-07-debugging-memory-corruption-in-tensor-programs.md)
8. [Profiling and Optimizing Hotspots](chapters/chapter-08-profiling-and-optimizing-hotspots.md)
9. [Building a Neural Network Layer with Tensors](chapters/chapter-09-building-a-neural-network-layer-with-tensors.md)
10. [Deploying Tensor Applications on Embedded Systems](chapters/chapter-10-deploying-tensor-applications-on-embedded-systems.md)

## About This Book Series

I wrote this series after years of struggling with the gap between theoretical tensor mathematics and practical, high-performance C implementations. Too many resources either focus on abstract concepts without addressing real-world implementation challenges, or they provide code snippets that fall apart in production environments.

This book series is different. It's the guide I wish I had when I started building tensor systems for embedded devices and HPC clusters. Each chapter tackles concrete problems you'll face when implementing tensor operations in C, with complete code examples that prioritize:

- **Hands-on learning**: Every concept is accompanied by working C code you can compile and run
- **Performance optimization**: Techniques for squeezing maximum performance from your hardware
- **Memory management**: Proper allocation, tracking, and freeing of resources
- **Debugging strategies**: Real-world approaches to finding and fixing common bugs
- **Production readiness**: Code that's robust enough for industrial applications

## Key Features

- **Problem-driven approach**: Each chapter solves specific challenges rather than presenting abstract theory
- **C-centric implementation**: All code uses manual memory management, pointers, and structs—no hidden abstractions
- **Toolchain mastery**: Learn to use GDB, Valgrind, perf, and compiler flags as essential parts of your workflow
- **Bare-metal focus**: Designed for developers working on embedded systems, HPC, or custom ML runtimes
- **Progressive complexity**: Start with basic tensor structures and advance to neural networks and embedded deployment

## Who This Book Is For

This series is ideal for C programmers who need to implement high-performance numerical systems from scratch. You might be:

- An embedded systems developer implementing ML algorithms on resource-constrained hardware
- A performance engineer optimizing numerical code for HPC applications
- A machine learning practitioner who wants to understand what happens beneath high-level frameworks
- A computer science student looking to bridge the gap between theory and practical implementation

Basic knowledge of C programming and linear algebra concepts is assumed, but each topic is explained from first principles with a focus on practical implementation.

---

## Running the Jekyll Site Locally

To run this site locally:

1. Install Ruby and Bundler
2. Clone this repository
3. Run `bundle install --path vendor/bundle` to install dependencies
4. Run `bundle exec jekyll serve` to start the local server
5. Visit `http://localhost:4000/tensor-programming` in your browser

## Contributing

If you find any errors or have suggestions for improvements, please open an issue or submit a pull request.