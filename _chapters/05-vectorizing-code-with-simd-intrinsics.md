---
layout: chapter
title: Vectorizing Code with SIMD Intrinsics
number: 5
description: Unlock the full potential of modern CPUs by leveraging SIMD instructions. Learn to use vector units for dramatic performance improvements in tensor operations.
---

## You Will Learn To...

- Leverage SIMD (Single Instruction, Multiple Data) instructions to process multiple tensor elements simultaneously
- Implement vectorized tensor operations using platform-specific intrinsics (SSE, AVX, AVX-512)
- Align memory properly for optimal SIMD performance
- Handle edge cases when tensor dimensions aren't multiples of vector width
- Write portable code that adapts to different SIMD capabilities

## 5.1 Understanding SIMD Vectorization

After parallelizing our tensor operations across multiple cores with OpenMP, the next step to maximize performance is to exploit the SIMD capabilities of modern CPUs. SIMD instructions allow us to perform the same operation on multiple data elements simultaneously within a single CPU core.

### The Basics of SIMD

Modern CPUs include vector processing units that can operate on multiple values at once. For example:

- SSE (Streaming SIMD Extensions): 128-bit registers, processing 4 floats or 2 doubles at once
- AVX (Advanced Vector Extensions): 256-bit registers, processing 8 floats or 4 doubles at once
- AVX-512: 512-bit registers, processing 16 floats or 8 doubles at once

Let's visualize how SIMD operations work:

```
Scalar Addition:       Vector Addition (4-wide):

a = 1.0                a = [1.0, 2.0, 3.0, 4.0]
b = 2.0                b = [5.0, 6.0, 7.0, 8.0]
c = a + b              c = a + b = [6.0, 8.0, 10.0, 12.0]
                                    (all additions happen simultaneously)
```

### Detecting SIMD Support

Before using SIMD instructions, we need to detect which instruction sets are supported by the CPU. Here's a simple function to detect SIMD support:

```c
/* Add to tensor.h */
#include <cpuid.h>

typedef enum {
    SIMD_NONE = 0,
    SIMD_SSE = 1,
    SIMD_SSE2 = 2,
    SIMD_SSE3 = 3,
    SIMD_SSSE3 = 4,
    SIMD_SSE4_1 = 5,
    SIMD_SSE4_2 = 6,
    SIMD_AVX = 7,
    SIMD_AVX2 = 8,
    SIMD_AVX512F = 9
} simd_support_t;

/* Add to tensor.c */
simd_support_t detect_simd_support() {
    unsigned int eax, ebx, ecx, edx;
    
    /* Check if CPUID is supported */
    if (!__get_cpuid_max(0, NULL)) {
        return SIMD_NONE;
    }
    
    /* Check SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2 */
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    
    if (!(edx & bit_SSE)) {
        return SIMD_NONE;
    }
    if (!(edx & bit_SSE2)) {
        return SIMD_SSE;
    }
    if (!(ecx & bit_SSE3)) {
        return SIMD_SSE2;
    }
    if (!(ecx & bit_SSSE3)) {
        return SIMD_SSE3;
    }
    if (!(ecx & bit_SSE4_1)) {
        return SIMD_SSSE3;
    }
    if (!(ecx & bit_SSE4_2)) {
        return SIMD_SSE4_1;
    }
    
    /* Check AVX */
    if (!(ecx & bit_AVX)) {
        return SIMD_SSE4_2;
    }
    
    /* Check AVX2 */
    __get_cpuid(7, &eax, &ebx, &ecx, &edx);
    if (!(ebx & bit_AVX2)) {
        return SIMD_AVX;
    }
    
    /* Check AVX-512F */
    if (!(ebx & bit_AVX512F)) {
        return SIMD_AVX2;
    }
    
    return SIMD_AVX512F;
}
```

Note that this function requires the `cpuid.h` header, which is available in GCC and Clang. For other compilers, you might need to use platform-specific methods.

### Memory Alignment Requirements

SIMD instructions often require aligned memory access for optimal performance. Let's update our tensor creation function to ensure proper alignment:

```c
/* Add to tensor.h */
#define TENSOR_ALIGNMENT 32  /* For AVX (256-bit) alignment */

/* Add to tensor.c */
int tensor_create_aligned(tensor_t *t, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Allocate aligned memory for data */
    tensor_elem_t *data;
    int status = posix_memalign((void**)&data, TENSOR_ALIGNMENT, 
                              rows * cols * sizeof(tensor_elem_t));
    
    if (status != 0 || data == NULL) {
        return TENSOR_ALLOCATION_FAILED;
    }
    
    /* Initialize metadata */
    t->rows = rows;
    t->cols = cols;
    t->row_stride = cols;
    t->col_stride = 1;
    t->layout = TENSOR_ROW_MAJOR;
    t->owner = 1;
    t->data = data;
    
    /* Initialize data to zero */
    #pragma omp parallel for collapse(2) if(rows * cols > 1000)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            t->data[i * t->row_stride + j * t->col_stride] = 0.0;
        }
    }
    
    return TENSOR_SUCCESS;
}
```

The `posix_memalign` function ensures that the memory is aligned to the specified boundary (32 bytes for AVX). This alignment is crucial for efficient SIMD operations.

## 5.2 Implementing Basic SIMD Operations

Now let's implement some basic tensor operations using SIMD intrinsics. We'll start with element-wise addition.

### SIMD Headers and Macros

First, let's include the necessary headers and define some helper macros:

```c
/* Add to tensor_simd.h */
#ifndef TENSOR_SIMD_H
#define TENSOR_SIMD_H

#include "tensor.h"

/* SIMD headers */
#if defined(__AVX512F__)
    #include <immintrin.h>
    #define SIMD_WIDTH 16  /* 16 floats in a 512-bit register */
    #define SIMD_ALIGNMENT 64
    typedef __m512 simd_vector;
    #define simd_load _mm512_load_ps
    #define simd_loadu _mm512_loadu_ps
    #define simd_store _mm512_store_ps
    #define simd_add _mm512_add_ps
    #define simd_sub _mm512_sub_ps
    #define simd_mul _mm512_mul_ps
    #define simd_div _mm512_div_ps
#elif defined(__AVX__)
    #include <immintrin.h>
    #define SIMD_WIDTH 8   /* 8 floats in a 256-bit register */
    #define SIMD_ALIGNMENT 32
    typedef __m256 simd_vector;
    #define simd_load _mm256_load_ps
    #define simd_loadu _mm256_loadu_ps
    #define simd_store _mm256_store_ps
    #define simd_add _mm256_add_ps
    #define simd_sub _mm256_sub_ps
    #define simd_mul _mm256_mul_ps
    #define simd_div _mm256_div_ps
#elif defined(__SSE__)
    #include <xmmintrin.h>
    #define SIMD_WIDTH 4   /* 4 floats in a 128-bit register */
    #define SIMD_ALIGNMENT 16
    typedef __m128 simd_vector;
    #define simd_load _mm_load_ps
    #define simd_loadu _mm_loadu_ps
    #define simd_store _mm_store_ps
    #define simd_add _mm_add_ps
    #define simd_sub _mm_sub_ps
    #define simd_mul _mm_mul_ps
    #define simd_div _mm_div_ps
#else
    #define SIMD_WIDTH 1   /* Fallback to scalar */
    #define SIMD_ALIGNMENT 8
    typedef float simd_vector;
    #define simd_load(x) (*(x))
    #define simd_loadu(x) (*(x))
    #define simd_store(x, v) (*(x) = (v))
    #define simd_add(a, b) ((a) + (b))
    #define simd_sub(a, b) ((a) - (b))
    #define simd_mul(a, b) ((a) * (b))
    #define simd_div(a, b) ((a) / (b))
#endif

/* Function declarations */
int tensor_add_simd(tensor_t *result, const tensor_t *a, const tensor_t *b);
int tensor_sub_simd(tensor_t *result, const tensor_t *a, const tensor_t *b);
int tensor_mul_simd(tensor_t *result, const tensor_t *a, const tensor_t *b);
int tensor_div_simd(tensor_t *result, const tensor_t *a, const tensor_t *b);
int tensor_scale_simd(tensor_t *result, const tensor_t *a, tensor_elem_t scalar);

#endif /* TENSOR_SIMD_H */
```

This header provides a unified interface for different SIMD instruction sets, allowing us to write code that adapts to the available hardware.

### Implementing Element-wise Addition with SIMD

Now let's implement element-wise addition using SIMD intrinsics:

```c
/* Add to tensor_simd.c */
#include "tensor_simd.h"

int tensor_add_simd(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if shapes match */
    if (a->rows != b->rows || a->cols != b->cols) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_aligned(result, a->rows, a->cols);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Calculate total number of elements */
    size_t total_elements = a->rows * a->cols;
    
    /* Process elements in SIMD_WIDTH chunks */
    size_t simd_limit = total_elements - (total_elements % SIMD_WIDTH);
    
    #pragma omp parallel for if(total_elements > 1000)
    for (size_t i = 0; i < simd_limit; i += SIMD_WIDTH) {
        /* Load data into SIMD registers */
        simd_vector va = simd_loadu(&a->data[i]);
        simd_vector vb = simd_loadu(&b->data[i]);
        
        /* Perform addition */
        simd_vector vresult = simd_add(va, vb);
        
        /* Store result */
        simd_store(&result->data[i], vresult);
    }
    
    /* Handle remaining elements */
    for (size_t i = simd_limit; i < total_elements; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    
    return TENSOR_SUCCESS;
}
```

This implementation processes elements in chunks of `SIMD_WIDTH` (4 for SSE, 8 for AVX, 16 for AVX-512), with a scalar fallback for any remaining elements.

### Implementing Other Element-wise Operations

Let's implement other element-wise operations using the same pattern:

```c
/* Add to tensor_simd.c */
int tensor_sub_simd(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if shapes match */
    if (a->rows != b->rows || a->cols != b->cols) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_aligned(result, a->rows, a->cols);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Calculate total number of elements */
    size_t total_elements = a->rows * a->cols;
    
    /* Process elements in SIMD_WIDTH chunks */
    size_t simd_limit = total_elements - (total_elements % SIMD_WIDTH);
    
    #pragma omp parallel for if(total_elements > 1000)
    for (size_t i = 0; i < simd_limit; i += SIMD_WIDTH) {
        /* Load data into SIMD registers */
        simd_vector va = simd_loadu(&a->data[i]);
        simd_vector vb = simd_loadu(&b->data[i]);
        
        /* Perform subtraction */
        simd_vector vresult = simd_sub(va, vb);
        
        /* Store result */
        simd_store(&result->data[i], vresult);
    }
    
    /* Handle remaining elements */
    for (size_t i = simd_limit; i < total_elements; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    
    return TENSOR_SUCCESS;
}

int tensor_mul_simd(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if shapes match */
    if (a->rows != b->rows || a->cols != b->cols) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_aligned(result, a->rows, a->cols);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Calculate total number of elements */
    size_t total_elements = a->rows * a->cols;
    
    /* Process elements in SIMD_WIDTH chunks */
    size_t simd_limit = total_elements - (total_elements % SIMD_WIDTH);
    
    #pragma omp parallel for if(total_elements > 1000)
    for (size_t i = 0; i < simd_limit; i += SIMD_WIDTH) {
        /* Load data into SIMD registers */
        simd_vector va = simd_loadu(&a->data[i]);
        simd_vector vb = simd_loadu(&b->data[i]);
        
        /* Perform multiplication */
        simd_vector vresult = simd_mul(va, vb);
        
        /* Store result */
        simd_store(&result->data[i], vresult);
    }
    
    /* Handle remaining elements */
    for (size_t i = simd_limit; i < total_elements; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    
    return TENSOR_SUCCESS;
}

int tensor_scale_simd(tensor_t *result, const tensor_t *a, tensor_elem_t scalar) {
    /* Create or resize result tensor */
    int status = tensor_create_aligned(result, a->rows, a->cols);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Calculate total number of elements */
    size_t total_elements = a->rows * a->cols;
    
    /* Create a SIMD vector with the scalar value replicated */
    #if defined(__AVX512F__)
        simd_vector vscalar = _mm512_set1_ps(scalar);
    #elif defined(__AVX__)
        simd_vector vscalar = _mm256_set1_ps(scalar);
    #elif defined(__SSE__)
        simd_vector vscalar = _mm_set1_ps(scalar);
    #else
        simd_vector vscalar = scalar;
    #endif
    
    /* Process elements in SIMD_WIDTH chunks */
    size_t simd_limit = total_elements - (total_elements % SIMD_WIDTH);
    
    #pragma omp parallel for if(total_elements > 1000)
    for (size_t i = 0; i < simd_limit; i += SIMD_WIDTH) {
        /* Load data into SIMD register */
        simd_vector va = simd_loadu(&a->data[i]);
        
        /* Perform scaling */
        simd_vector vresult = simd_mul(va, vscalar);
        
        /* Store result */
        simd_store(&result->data[i], vresult);
    }
    
    /* Handle remaining elements */
    for (size_t i = simd_limit; i < total_elements; i++) {
        result->data[i] = a->data[i] * scalar;
    }
    
    return TENSOR_SUCCESS;
}
```

## 5.3 Vectorizing Matrix Multiplication

Matrix multiplication is more complex but can benefit significantly from SIMD vectorization. Let's implement a vectorized version of matrix multiplication:

```c
/* Add to tensor_simd.h */
int tensor_matmul_simd(tensor_t *result, const tensor_t *a, const tensor_t *b);

/* Add to tensor_simd.c */
int tensor_matmul_simd(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if dimensions are compatible for matrix multiplication */
    if (a->cols != b->rows) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_aligned(result, a->rows, b->cols);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Zero out the result tensor */
    memset(result->data, 0, result->rows * result->cols * sizeof(tensor_elem_t));
    
    /* For each row of A */
    #pragma omp parallel for if(a->rows * b->cols > 100)
    for (size_t i = 0; i < a->rows; i++) {
        /* For each column of B */
        for (size_t j = 0; j < b->cols; j++) {
            size_t idx_result = i * result->row_stride + j * result->col_stride;
            
            /* Process k in SIMD_WIDTH chunks */
            size_t k_simd_limit = a->cols - (a->cols % SIMD_WIDTH);
            
            /* Initialize accumulator */
            #if defined(__AVX512F__)
                simd_vector vsum = _mm512_setzero_ps();
            #elif defined(__AVX__)
                simd_vector vsum = _mm256_setzero_ps();
            #elif defined(__SSE__)
                simd_vector vsum = _mm_setzero_ps();
            #else
                tensor_elem_t sum = 0.0f;
            #endif
            
            /* Process in SIMD chunks */
            for (size_t k = 0; k < k_simd_limit; k += SIMD_WIDTH) {
                /* Load SIMD_WIDTH elements from row i of A */
                simd_vector va = simd_loadu(&a->data[i * a->row_stride + k]);
                
                /* Load SIMD_WIDTH elements from column j of B */
                /* This is trickier because we need to gather elements with stride */
                #if defined(__AVX512F__)
                    float b_elements[SIMD_WIDTH];
                    for (size_t l = 0; l < SIMD_WIDTH; l++) {
                        b_elements[l] = b->data[(k + l) * b->row_stride + j];
                    }
                    simd_vector vb = _mm512_loadu_ps(b_elements);
                #elif defined(__AVX__)
                    float b_elements[SIMD_WIDTH];
                    for (size_t l = 0; l < SIMD_WIDTH; l++) {
                        b_elements[l] = b->data[(k + l) * b->row_stride + j];
                    }
                    simd_vector vb = _mm256_loadu_ps(b_elements);
                #elif defined(__SSE__)
                    float b_elements[SIMD_WIDTH];
                    for (size_t l = 0; l < SIMD_WIDTH; l++) {
                        b_elements[l] = b->data[(k + l) * b->row_stride + j];
                    }
                    simd_vector vb = _mm_loadu_ps(b_elements);
                #else
                    float b_element = b->data[k * b->row_stride + j];
                #endif
                
                /* Multiply and accumulate */
                #if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
                    vsum = simd_add(vsum, simd_mul(va, vb));
                #else
                    sum += a->data[i * a->row_stride + k] * b_element;
                #endif
            }
            
            /* Reduce SIMD vector to scalar sum */
            #if defined(__AVX512F__)
                float temp[SIMD_WIDTH];
                _mm512_store_ps(temp, vsum);
                tensor_elem_t sum = 0.0f;
                for (size_t l = 0; l < SIMD_WIDTH; l++) {
                    sum += temp[l];
                }
            #elif defined(__AVX__)
                float temp[SIMD_WIDTH];
                _mm256_store_ps(temp, vsum);
                tensor_elem_t sum = 0.0f;
                for (size_t l = 0; l < SIMD_WIDTH; l++) {
                    sum += temp[l];
                }
            #elif defined(__SSE__)
                float temp[SIMD_WIDTH];
                _mm_store_ps(temp, vsum);
                tensor_elem_t sum = 0.0f;
                for (size_t l = 0; l < SIMD_WIDTH; l++) {
                    sum += temp[l];
                }
            #endif
            
            /* Handle remaining elements */
            for (size_t k = k_simd_limit; k < a->cols; k++) {
                sum += a->data[i * a->row_stride + k] * b->data[k * b->row_stride + j];
            }
            
            /* Store result */
            result->data[idx_result] = sum;
        }
    }
    
    return TENSOR_SUCCESS;
}
```

This implementation vectorizes the inner loop of matrix multiplication, processing multiple elements of the dot product simultaneously.

## 5.4 Advanced SIMD Techniques

Let's explore some advanced SIMD techniques to further optimize our tensor operations.

### Handling Non-Aligned Memory

Sometimes we can't guarantee that memory is aligned, especially when working with tensor views or slices. Let's implement a function that handles both aligned and unaligned memory:

```c
/* Add to tensor_simd.h */
int tensor_add_simd_unaligned(tensor_t *result, const tensor_t *a, const tensor_t *b);

/* Add to tensor_simd.c */
int tensor_add_simd_unaligned(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if shapes match */
    if (a->rows != b->rows || a->cols != b->cols) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_result(result, a);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Calculate total number of elements */
    size_t total_elements = a->rows * a->cols;
    
    /* Process elements in SIMD_WIDTH chunks */
    size_t simd_limit = total_elements - (total_elements % SIMD_WIDTH);
    
    #pragma omp parallel for if(total_elements > 1000)
    for (size_t i = 0; i < simd_limit; i += SIMD_WIDTH) {
        /* Load data into SIMD registers (using unaligned loads) */
        simd_vector va = simd_loadu(&a->data[i]);
        simd_vector vb = simd_loadu(&b->data[i]);
        
        /* Perform addition */
        simd_vector vresult = simd_add(va, vb);
        
        /* Store result (using unaligned store) */
        #if defined(__AVX512F__)
            _mm512_storeu_ps(&result->data[i], vresult);
        #elif defined(__AVX__)
            _mm256_storeu_ps(&result->data[i], vresult);
        #elif defined(__SSE__)
            _mm_storeu_ps(&result->data[i], vresult);
        #else
            result->data[i] = vresult;
        #endif
    }
    
    /* Handle remaining elements */
    for (size_t i = simd_limit; i < total_elements; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    
    return TENSOR_SUCCESS;
}
```

This function uses unaligned loads and stores, which are slightly slower but work with any memory address.

### Fused Multiply-Add (FMA) Operations

Modern CPUs support Fused Multiply-Add (FMA) operations, which perform a multiplication and addition in a single instruction with higher precision. Let's implement a function that uses FMA for tensor operations:

```c
/* Add to tensor_simd.h */
int tensor_fma_simd(tensor_t *result, const tensor_t *a, const tensor_t *b, const tensor_t *c);

/* Add to tensor_simd.c */
int tensor_fma_simd(tensor_t *result, const tensor_t *a, const tensor_t *b, const tensor_t *c) {
    /* Check if shapes match */
    if (a->rows != b->rows || a->rows != c->rows || a->cols != b->cols || a->cols != c->cols) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_aligned(result, a->rows, a->cols);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Calculate total number of elements */
    size_t total_elements = a->rows * a->cols;
    
    /* Process elements in SIMD_WIDTH chunks */
    size_t simd_limit = total_elements - (total_elements % SIMD_WIDTH);
    
    #pragma omp parallel for if(total_elements > 1000)
    for (size_t i = 0; i < simd_limit; i += SIMD_WIDTH) {
        /* Load data into SIMD registers */
        simd_vector va = simd_loadu(&a->data[i]);
        simd_vector vb = simd_loadu(&b->data[i]);
        simd_vector vc = simd_loadu(&c->data[i]);
        
        /* Perform fused multiply-add: result = a * b + c */
        #if defined(__AVX512F__) && defined(__FMA__)
            simd_vector vresult = _mm512_fmadd_ps(va, vb, vc);
        #elif defined(__AVX2__) && defined(__FMA__)
            simd_vector vresult = _mm256_fmadd_ps(va, vb, vc);
        #elif defined(__SSE__) && defined(__FMA__)
            simd_vector vresult = _mm_fmadd_ps(va, vb, vc);
        #else
            /* Fallback for processors without FMA */
            simd_vector vresult = simd_add(simd_mul(va, vb), vc);
        #endif
        
        /* Store result */
        simd_store(&result->data[i], vresult);
    }
    
    /* Handle remaining elements */
    for (size_t i = simd_limit; i < total_elements; i++) {
        result->data[i] = a->data[i] * b->data[i] + c->data[i];
    }
    
    return TENSOR_SUCCESS;
}
```

This function uses FMA instructions when available, falling back to separate multiply and add operations otherwise.

### Horizontal Operations

Some tensor operations require horizontal operations across vector lanes. Let's implement a vectorized sum reduction:

```c
/* Add to tensor_simd.h */
tensor_elem_t tensor_sum_simd(const tensor_t *t);

/* Add to tensor_simd.c */
tensor_elem_t tensor_sum_simd(const tensor_t *t) {
    /* Calculate total number of elements */
    size_t total_elements = t->rows * t->cols;
    
    /* Initialize sum accumulators */
    #if defined(__AVX512F__)
        simd_vector vsum = _mm512_setzero_ps();
    #elif defined(__AVX__)
        simd_vector vsum = _mm256_setzero_ps();
    #elif defined(__SSE__)
        simd_vector vsum = _mm_setzero_ps();
    #else
        tensor_elem_t sum = 0.0f;
    #endif
    
    /* Process elements in SIMD_WIDTH chunks */
    size_t simd_limit = total_elements - (total_elements % SIMD_WIDTH);
    
    for (size_t i = 0; i < simd_limit; i += SIMD_WIDTH) {
        /* Load data into SIMD register */
        simd_vector v = simd_loadu(&t->data[i]);
        
        /* Accumulate sum */
        #if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
            vsum = simd_add(vsum, v);
        #else
            sum += t->data[i];
        #endif
    }
    
    /* Reduce SIMD vector to scalar sum */
    #if defined(__AVX512F__)
        float temp[SIMD_WIDTH];
        _mm512_store_ps(temp, vsum);
        tensor_elem_t sum = 0.0f;
        for (size_t i = 0; i < SIMD_WIDTH; i++) {
            sum += temp[i];
        }
    #elif defined(__AVX__)
        /* Horizontal sum for AVX */
        __m128 vlow = _mm256_castps256_ps128(vsum);
        __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        __m128 shuf = _mm_movehdup_ps(vlow);
        __m128 sums = _mm_add_ps(vlow, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        tensor_elem_t sum = _mm_cvtss_f32(sums);
    #elif defined(__SSE3__)
        /* Horizontal sum for SSE3 */
        __m128 shuf = _mm_movehdup_ps(vsum);
        __m128 sums = _mm_add_ps(vsum, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        tensor_elem_t sum = _mm_cvtss_f32(sums);
    #elif defined(__SSE__)
        /* Horizontal sum for SSE */
        float temp[SIMD_WIDTH];
        _mm_store_ps(temp, vsum);
        tensor_elem_t sum = 0.0f;
        for (size_t i = 0; i < SIMD_WIDTH; i++) {
            sum += temp[i];
        }
    #endif
    
    /* Handle remaining elements */
    for (size_t i = simd_limit; i < total_elements; i++) {
        sum += t->data[i];
    }
    
    return sum;
}
```

This function uses different horizontal sum techniques depending on the available instruction set.

## 5.5 Benchmarking and Performance Analysis

Let's create a benchmark to measure the performance improvement from SIMD vectorization:

```c
/* benchmark_simd.c */
#include "tensor.h"
#include "tensor_ops.h"
#include "tensor_simd.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>

/* Benchmark function for element-wise addition */
double benchmark_add(const tensor_t *a, const tensor_t *b, tensor_t *result, int use_simd) {
    /* Warm up the cache */
    if (use_simd) {
        tensor_add_simd(result, a, b);
    } else {
        tensor_add(result, a, b);
    }
    
    /* Measure performance */
    double start_time = omp_get_wtime();
    
    /* Perform addition */
    if (use_simd) {
        tensor_add_simd(result, a, b);
    } else {
        tensor_add(result, a, b);
    }
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

/* Benchmark function for matrix multiplication */
double benchmark_matmul(const tensor_t *a, const tensor_t *b, tensor_t *result, int use_simd) {
    /* Warm up the cache */
    if (use_simd) {
        tensor_matmul_simd(result, a, b);
    } else {
        tensor_matmul(result, a, b);
    }
    
    /* Measure performance */
    double start_time = omp_get_wtime();
    
    /* Perform matrix multiplication */
    if (use_simd) {
        tensor_matmul_simd(result, a, b);
    } else {
        tensor_matmul(result, a, b);
    }
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main() {
    tensor_t a, b, result;
    size_t size = 1000;  /* Large enough to see the difference */
    
    /* Create tensors */
    tensor_create_aligned(&a, size, size);
    tensor_create_aligned(&b, size, size);
    tensor_create_aligned(&result, size, size);
    
    /* Initialize with some values */
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            size_t idx_a = i * a.row_stride + j * a.col_stride;
            size_t idx_b = i * b.row_stride + j * b.col_stride;
            a.data[idx_a] = (tensor_elem_t)(i * 0.01 + j * 0.01);
            b.data[idx_b] = (tensor_elem_t)(i * 0.01 - j * 0.01);
        }
    }
    
    /* Detect SIMD support */
    simd_support_t simd_level = detect_simd_support();
    printf("Detected SIMD support level: %d\n", simd_level);
    printf("SIMD width: %d elements\n", SIMD_WIDTH);
    
    /* Benchmark element-wise addition */
    printf("\nElement-wise Addition:\n");
    printf("Method | Time (seconds) | Speedup\n");
    printf("-------|----------------|--------\n");
    
    double scalar_add_time = benchmark_add(&a, &b, &result, 0);
    printf("Scalar | %14.6f | %7.2f\n", scalar_add_time, 1.0);
    
    double simd_add_time = benchmark_add(&a, &b, &result, 1);
    printf("SIMD   | %14.6f | %7.2f\n", simd_add_time, scalar_add_time / simd_add_time);
    
    /* Benchmark matrix multiplication */
    printf("\nMatrix Multiplication:\n");
    printf("Method | Time (seconds) | Speedup\n");
    printf("-------|----------------|--------\n");
    
    double scalar_matmul_time = benchmark_matmul(&a, &b, &result, 0);
    printf("Scalar | %14.6f | %7.2f\n", scalar_matmul_time, 1.0);
    
    double simd_matmul_time = benchmark_matmul(&a, &b, &result, 1);
    printf("SIMD   | %14.6f | %7.2f\n", simd_matmul_time, scalar_matmul_time / simd_matmul_time);
    
    /* Clean up */
    tensor_free(&a);
    tensor_free(&b);
    tensor_free(&result);
    
    return 0;
}
```

This benchmark compares the performance of scalar and SIMD implementations for element-wise addition and matrix multiplication.

## 5.6 Handling Edge Cases and Portability

Let's address some common edge cases and portability concerns when using SIMD intrinsics.

### Runtime SIMD Selection

To handle different CPU capabilities at runtime, we can implement a dispatcher that selects the appropriate implementation:

```c
/* Add to tensor_simd.h */
typedef int (*tensor_add_func_t)(tensor_t *result, const tensor_t *a, const tensor_t *b);
tensor_add_func_t get_optimal_tensor_add();

/* Add to tensor_simd.c */
tensor_add_func_t get_optimal_tensor_add() {
    simd_support_t simd_level = detect_simd_support();
    
    if (simd_level >= SIMD_AVX512F) {
        return tensor_add_simd;  /* AVX-512 implementation */
    } else if (simd_level >= SIMD_AVX) {
        return tensor_add_simd;  /* AVX implementation */
    } else if (simd_level >= SIMD_SSE) {
        return tensor_add_simd;  /* SSE implementation */
    } else {
        return tensor_add;       /* Scalar fallback */
    }
}
```

This function returns the optimal implementation based on the CPU's capabilities.

### Masking for Partial Vector Operations

When processing the last few elements of a tensor, we might need to use masking to handle partial vectors:

```c
/* Add to tensor_simd.c */
int tensor_add_simd_masked(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* ... (setup code) ... */
    
    /* Process elements in SIMD_WIDTH chunks */
    size_t simd_limit = total_elements - (total_elements % SIMD_WIDTH);
    
    /* ... (main SIMD loop) ... */
    
    /* Handle remaining elements with masking */
    if (simd_limit < total_elements) {
        size_t remaining = total_elements - simd_limit;
        
        #if defined(__AVX512F__)
            /* Create mask for remaining elements */
            __mmask16 mask = (1ULL << remaining) - 1;
            
            /* Load data with masking */
            simd_vector va = _mm512_maskz_loadu_ps(mask, &a->data[simd_limit]);
            simd_vector vb = _mm512_maskz_loadu_ps(mask, &b->data[simd_limit]);
            
            /* Perform addition */
            simd_vector vresult = _mm512_add_ps(va, vb);
            
            /* Store result with masking */
            _mm512_mask_storeu_ps(&result->data[simd_limit], mask, vresult);
        #else
            /* Fallback to scalar for non-AVX512 */
            for (size_t i = simd_limit; i < total_elements; i++) {
                result->data[i] = a->data[i] + b->data[i];
            }
        #endif
    }
    
    return TENSOR_SUCCESS;
}
```

This function uses AVX-512 masking to handle partial vectors, with a scalar fallback for other instruction sets.

### Cross-Platform Considerations

To ensure our code works across different platforms, we need to handle compiler-specific intrinsics and fallbacks:

```c
/* Add to tensor_simd.h */
#if defined(_MSC_VER)
    /* MSVC-specific headers and definitions */
    #include <intrin.h>
    #define ALIGN32 __declspec(align(32))
#elif defined(__GNUC__)
    /* GCC/Clang-specific headers and definitions */
    #include <x86intrin.h>
    #define ALIGN32 __attribute__((aligned(32)))
#else
    /* Fallback for other compilers */
    #define ALIGN32
#endif
```

This ensures that our code compiles correctly on different platforms and compilers.

## Visualizing SIMD Operations

Let's visualize how SIMD operations work to better understand the concepts.

### SIMD Register Layout

```
AVX 256-bit Register (8 floats):

+------+------+------+------+------+------+------+------+
| f[0] | f[1] | f[2] | f[3] | f[4] | f[5] | f[6] | f[7] |
+------+------+------+------+------+------+------+------+
```

### SIMD Addition

```
Vector Addition with AVX (8-wide):

Register A:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Register B:  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
             |    |    |    |    |    |    |    |    |
             v    v    v    v    v    v    v    v    v
Result:      [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
```

### Memory Alignment

```
Memory Alignment for AVX (32-byte boundary):

0       8       16      24      32      40      48
|       |       |       |       |       |       |
V       V       V       V       V       V       V
+-------+-------+-------+-------+-------+-------+
|  Aligned to 32-byte boundary   |       |       |
+-------+-------+-------+-------+-------+-------+
        ^                       ^
        |                       |
    Aligned load           Unaligned load
```

## Common Pitfalls and Debugging

Let's discuss some common issues you might encounter when using SIMD intrinsics.

### Alignment Issues

Misaligned memory access can cause performance degradation or crashes. Symptoms include:

1. Segmentation faults when using aligned load/store instructions with unaligned data
2. Performance that's worse than expected

To diagnose alignment issues, you can use tools like `gdb` to check memory addresses:

```bash
printf "0x%lx\n", (unsigned long)(&tensor->data[0]) % 32
```

This should print 0 for properly aligned memory.

Solutions include:

1. Using `posix_memalign` or `aligned_alloc` for memory allocation
2. Using unaligned load/store instructions when alignment can't be guaranteed
3. Padding arrays to ensure alignment

### Compiler Optimization Flags

To get the best performance from SIMD intrinsics, you need to use appropriate compiler flags:

```bash
gcc -O3 -march=native -mavx2 -mfma your_program.c -o your_program
```

The `-march=native` flag tells the compiler to use all available instruction sets on the current CPU, while `-mavx2` and `-mfma` explicitly enable AVX2 and FMA instructions.

### Debugging SIMD Code

Debugging SIMD code can be challenging because you're dealing with multiple values at once. Here are some tips:

1. Use temporary arrays to inspect SIMD register contents:

```c
simd_vector v = /* some computation */;
float temp[SIMD_WIDTH];
simd_store(temp, v);

for (int i = 0; i < SIMD_WIDTH; i++) {
    printf("v[%d] = %f\n", i, temp[i]);
}
```

2. Start with a small, known input dataset and verify each step of the computation.

3. Compare results with a scalar implementation to identify discrepancies.

### Performance Pitfalls

Common performance pitfalls when using SIMD include:

1. **Memory Access Patterns**: Non-contiguous memory access can negate SIMD benefits.
2. **Branch Mispredictions**: Conditional branches inside SIMD loops can hurt performance.
3. **Mixed Data Types**: Converting between different data types can add overhead.
4. **Insufficient Work**: SIMD has overhead, so very small operations might be faster with scalar code.

## Exercises

### Exercise 1: Implement a Vectorized Softmax Function

Implement a function `tensor_softmax_simd` that computes the softmax of a tensor using SIMD intrinsics.

Hint: Softmax involves computing exp(x_i) / sum(exp(x_j)) for each element.

Partial solution:

```c
int tensor_softmax_simd(tensor_t *result, const tensor_t *input) {
    /* Create or resize result tensor */
    int status = tensor_create_aligned(result, input->rows, input->cols);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Process each row separately */
    #pragma omp parallel for if(input->rows > 10)
    for (size_t i = 0; i < input->rows; i++) {
        /* Find maximum value in the row (for numerical stability) */
        tensor_elem_t max_val = -INFINITY;
        for (size_t j = 0; j < input->cols; j++) {
            size_t idx = i * input->row_stride + j * input->col_stride;
            if (input->data[idx] > max_val) {
                max_val = input->data[idx];
            }
        }
        
        /* Create a vector with the max value */
        #if defined(__AVX512F__)
            simd_vector vmax = _mm512_set1_ps(max_val);
        #elif defined(__AVX__)
            simd_vector vmax = _mm256_set1_ps(max_val);
        #elif defined(__SSE__)
            simd_vector vmax = _mm_set1_ps(max_val);
        #else
            tensor_elem_t vmax = max_val;
        #endif
        
        /* Compute exp(x - max) for each element */
        /* ... (implement this part) ... */
        
        /* Compute sum of exp values */
        /* ... (implement this part) ... */
        
        /* Normalize by dividing each exp value by the sum */
        /* ... (implement this part) ... */
    }
    
    return TENSOR_SUCCESS;
}
```

### Exercise 2: Implement a Vectorized Convolution Function

Implement a function `tensor_convolve_simd` that performs 2D convolution of a tensor with a kernel, using SIMD intrinsics for optimization.

Hint: Convolution involves sliding a kernel over the input tensor and computing the sum of element-wise products.

### Exercise 3: Benchmark Different Memory Layouts for SIMD Performance

Implement and benchmark different memory layouts (row-major, column-major, blocked) for tensor operations with SIMD, and analyze which layout provides the best performance for different operations.

Partial solution:

```c
/* Create tensors with different layouts */
tensor_t row_major, col_major, blocked;
tensor_create_aligned(&row_major, size, size);
tensor_create_aligned(&col_major, size, size);
tensor_create_aligned(&blocked, size, size);

/* Set layouts */
row_major.layout = TENSOR_ROW_MAJOR;
row_major.row_stride = size;
row_major.col_stride = 1;

col_major.layout = TENSOR_COL_MAJOR;
col_major.row_stride = 1;
col_major.col_stride = size;

/* Initialize with the same values */
/* ... (initialization code) ... */

/* Benchmark operations with different layouts */
/* ... (benchmarking code) ... */
```

## Summary and Key Takeaways

In this chapter, we've explored vectorizing tensor code with SIMD intrinsics:

- We implemented basic tensor operations using platform-specific SIMD intrinsics (SSE, AVX, AVX-512).
- We addressed memory alignment requirements for optimal SIMD performance.
- We developed techniques for handling edge cases when tensor dimensions aren't multiples of vector width.
- We created portable code that adapts to different SIMD capabilities at runtime.

Key takeaways:

1. **SIMD Vectorization**: SIMD instructions can process multiple data elements simultaneously, providing significant speedups for tensor operations.
2. **Memory Alignment**: Proper memory alignment is crucial for optimal SIMD performance.
3. **Portability**: Using preprocessor directives and runtime detection allows your code to work efficiently across different platforms and CPU generations.
4. **Edge Cases**: Handling partial vectors and non-aligned memory requires special techniques like masking and unaligned loads/stores.
5. **Compiler Flags**: Using appropriate compiler flags is essential for getting the best performance from SIMD intrinsics.

By effectively vectorizing tensor operations, you can achieve 4-16x speedups (depending on the SIMD width) within a single CPU core, complementing the multi-core parallelism we explored in the previous chapter.

In the next chapter, we'll explore integrating BLAS libraries for production-grade performance, combining our custom SIMD code with highly optimized external libraries.

## Further Reading

1. Intel's Intrinsics Guide (https://software.intel.com/sites/landingpage/IntrinsicsGuide/) - A comprehensive reference for x86 SIMD intrinsics.

2. "SIMD Programming Manual for Linux and Windows" by Aart J.C. Bik - A practical guide to SIMD programming techniques.

3. "The Software Vectorization Handbook" by Aart J.C. Bik - Explores advanced vectorization techniques and optimization strategies.