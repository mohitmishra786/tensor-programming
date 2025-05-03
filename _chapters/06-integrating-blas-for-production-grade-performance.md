---
layout: chapter
title: Integrating BLAS for Production-Grade Performance
number: 6
description: Connect your tensor library to battle-tested, highly optimized BLAS implementations. Learn when and how to leverage external libraries for maximum performance.
---

## You Will Learn To...

- Leverage industry-standard BLAS libraries for maximum tensor operation performance
- Link your C tensor library with various BLAS implementations (OpenBLAS, Intel MKL, ATLAS)
- Create wrapper functions that seamlessly integrate BLAS with your tensor infrastructure
- Implement fallback mechanisms for operations not supported by BLAS
- Benchmark and compare custom implementations against BLAS routines

## 6.1 Understanding BLAS and Its Importance

After implementing our own tensor operations and optimizing them with OpenMP and SIMD intrinsics, it's time to take our library to production-grade performance by integrating with BLAS (Basic Linear Algebra Subprograms).

### What is BLAS?

BLAS is a specification for a set of low-level routines that perform common linear algebra operations such as vector addition, scalar multiplication, dot products, linear combinations, and matrix multiplication. The specification has been implemented by various vendors and organizations, each with different performance characteristics:

- **OpenBLAS**: An open-source implementation with good performance across various platforms
- **Intel MKL (Math Kernel Library)**: Highly optimized for Intel processors
- **ATLAS (Automatically Tuned Linear Algebra Software)**: Self-optimizing implementation
- **Apple Accelerate**: Optimized for Apple hardware
- **cuBLAS**: NVIDIA's implementation for GPUs

### Why Use BLAS Instead of Our Custom Code?

While our custom implementations with OpenMP and SIMD are educational and provide good performance, BLAS libraries offer several advantages:

1. **Decades of Optimization**: BLAS implementations have been refined over decades by teams of experts
2. **Hardware-Specific Tuning**: Vendor implementations like MKL contain processor-specific optimizations
3. **Automatic Adaptation**: Many BLAS libraries auto-tune for the specific hardware they run on
4. **Industry Standard**: Using BLAS makes your code compatible with the broader scientific computing ecosystem

Let's visualize the performance difference:

```
Matrix Multiplication Performance (1000x1000 matrices):

+------------------+----------+
| Implementation   | Time (s) |
+------------------+----------+
| Naive C          | 2.500    |
| OpenMP           | 0.350    |
| SIMD + OpenMP    | 0.120    |
| OpenBLAS         | 0.045    |
| Intel MKL        | 0.030    |
+------------------+----------+
```

As you can see, even our optimized implementations can't match the performance of specialized BLAS libraries.

## 6.2 Setting Up BLAS for Your Project

Before we can use BLAS, we need to install the libraries and set up our build system to link with them.

### Installing BLAS Libraries

On most Linux distributions, you can install OpenBLAS using the package manager:

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Fedora/RHEL/CentOS
sudo dnf install openblas-devel
```

For Intel MKL, you can download it from Intel's website or use package managers on some distributions:

```bash
# Ubuntu/Debian
sudo apt-get install intel-mkl
```

### Linking with BLAS in Your Build System

Let's update our Makefile to support different BLAS implementations:

```makefile
# Makefile
CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra

# Default to OpenBLAS if not specified
BLAS ?= OPENBLAS

ifeq ($(BLAS), MKL)
    BLAS_CFLAGS = -DUSE_MKL
    BLAS_LDFLAGS = -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
    BLAS_INCLUDE = -I${MKLROOT}/include
else ifeq ($(BLAS), ATLAS)
    BLAS_CFLAGS = -DUSE_ATLAS
    BLAS_LDFLAGS = -latlas -lcblas
    BLAS_INCLUDE =
else # Default to OpenBLAS
    BLAS_CFLAGS = -DUSE_OPENBLAS
    BLAS_LDFLAGS = -lopenblas
    BLAS_INCLUDE =
endif

CFLAGS += $(BLAS_CFLAGS) $(BLAS_INCLUDE)
LDFLAGS = $(BLAS_LDFLAGS)

SRCS = tensor.c tensor_ops.c tensor_simd.c tensor_blas.c
OBJS = $(SRCS:.c=.o)

all: libtensor.a test_tensor benchmark_blas

libtensor.a: $(OBJS)
	ar rcs $@ $^

test_tensor: test_tensor.c libtensor.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

benchmark_blas: benchmark_blas.c libtensor.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o *.a test_tensor benchmark_blas

.PHONY: all clean
```

This Makefile allows you to select the BLAS implementation using the `BLAS` variable:

```bash
make BLAS=MKL  # Build with Intel MKL
make BLAS=ATLAS  # Build with ATLAS
make  # Default to OpenBLAS
```

### Creating BLAS Header Wrappers

Different BLAS implementations have slightly different headers and function names. Let's create a wrapper to abstract these differences:

```c
/* tensor_blas.h */
#ifndef TENSOR_BLAS_H
#define TENSOR_BLAS_H

#include "tensor.h"

/* Include appropriate BLAS headers based on build configuration */
#if defined(USE_MKL)
    #include <mkl.h>
    #include <mkl_cblas.h>
    #define BLAS_IMPLEMENTATION "Intel MKL"
#elif defined(USE_ATLAS)
    #include <cblas.h>
    #define BLAS_IMPLEMENTATION "ATLAS"
#elif defined(USE_OPENBLAS)
    #include <cblas.h>
    #define BLAS_IMPLEMENTATION "OpenBLAS"
#else
    #error "No BLAS implementation specified. Define USE_MKL, USE_ATLAS, or USE_OPENBLAS."
#endif

/* Function declarations */
const char* tensor_blas_implementation();
int tensor_matmul_blas(tensor_t *result, const tensor_t *a, const tensor_t *b);
int tensor_axpy_blas(tensor_t *result, tensor_elem_t alpha, const tensor_t *x, const tensor_t *y);
int tensor_gemv_blas(tensor_t *result, tensor_elem_t alpha, const tensor_t *a, const tensor_t *x, tensor_elem_t beta, const tensor_t *y);

#endif /* TENSOR_BLAS_H */
```

And the implementation file:

```c
/* tensor_blas.c */
#include "tensor_blas.h"
#include <string.h>

const char* tensor_blas_implementation() {
    return BLAS_IMPLEMENTATION;
}
```

## 6.3 Implementing Matrix Multiplication with BLAS

Let's start by implementing matrix multiplication using BLAS's GEMM (General Matrix Multiplication) routine.

### Understanding GEMM

GEMM performs the operation: C = alpha * A * B + beta * C, where A, B, and C are matrices, and alpha and beta are scalars.

The CBLAS interface for GEMM looks like this:

```c
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);
```

Where:
- `Order`: Row-major or column-major order
- `TransA`, `TransB`: Whether to transpose A or B
- `M`, `N`, `K`: Matrix dimensions (M rows of A/C, N columns of B/C, K columns of A/rows of B)
- `alpha`, `beta`: Scalar multipliers
- `A`, `B`, `C`: Pointers to matrices
- `lda`, `ldb`, `ldc`: Leading dimensions (stride between consecutive rows/columns)

### Implementing GEMM Wrapper

Let's implement a wrapper for GEMM that works with our tensor structure:

```c
/* Add to tensor_blas.c */
int tensor_matmul_blas(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if dimensions are compatible for matrix multiplication */
    if (a->cols != b->rows) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status;
    if (result->data != NULL && result->owner) {
        if (result->rows != a->rows || result->cols != b->cols) {
            tensor_free(result);
            status = tensor_create_aligned(result, a->rows, b->cols);
            if (status != TENSOR_SUCCESS) {
                return status;
            }
        }
    } else if (result->data == NULL) {
        status = tensor_create_aligned(result, a->rows, b->cols);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* Determine if we need to handle non-contiguous memory */
    int a_contiguous = (a->layout == TENSOR_ROW_MAJOR && a->col_stride == 1) ||
                      (a->layout == TENSOR_COL_MAJOR && a->row_stride == 1);
    int b_contiguous = (b->layout == TENSOR_ROW_MAJOR && b->col_stride == 1) ||
                      (b->layout == TENSOR_COL_MAJOR && b->row_stride == 1);
    
    /* Create temporary contiguous copies if needed */
    tensor_t a_copy, b_copy;
    const tensor_t *a_ptr = a;
    const tensor_t *b_ptr = b;
    
    if (!a_contiguous) {
        tensor_create_aligned(&a_copy, a->rows, a->cols);
        tensor_copy(&a_copy, a);
        a_ptr = &a_copy;
    }
    
    if (!b_contiguous) {
        tensor_create_aligned(&b_copy, b->rows, b->cols);
        tensor_copy(&b_copy, b);
        b_ptr = &b_copy;
    }
    
    /* Determine CBLAS order and parameters based on tensor layout */
    enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE trans_a, trans_b;
    int m, n, k, lda, ldb, ldc;
    
    if (a_ptr->layout == TENSOR_ROW_MAJOR) {
        order = CblasRowMajor;
        trans_a = CblasNoTrans;
        trans_b = CblasNoTrans;
        m = a_ptr->rows;
        n = b_ptr->cols;
        k = a_ptr->cols;
        lda = a_ptr->row_stride;
        ldb = b_ptr->row_stride;
        ldc = result->row_stride;
    } else { /* TENSOR_COL_MAJOR */
        order = CblasColMajor;
        trans_a = CblasNoTrans;
        trans_b = CblasNoTrans;
        m = a_ptr->rows;
        n = b_ptr->cols;
        k = a_ptr->cols;
        lda = a_ptr->col_stride;
        ldb = b_ptr->col_stride;
        ldc = result->col_stride;
    }
    
    /* Call BLAS GEMM */
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cblas_sgemm(order, trans_a, trans_b, m, n, k, alpha, a_ptr->data, lda, 
                b_ptr->data, ldb, beta, result->data, ldc);
    
    /* Clean up temporary tensors if created */
    if (!a_contiguous) {
        tensor_free(&a_copy);
    }
    if (!b_contiguous) {
        tensor_free(&b_copy);
    }
    
    return TENSOR_SUCCESS;
}
```

This wrapper handles the complexities of different tensor layouts and non-contiguous memory, creating temporary copies when necessary.

### Handling Transposed Matrices

BLAS can efficiently handle transposed matrices without creating copies. Let's add a function for matrix multiplication with transposition:

```c
/* Add to tensor_blas.h */
int tensor_matmul_transposed_blas(tensor_t *result, const tensor_t *a, int transpose_a,
                                 const tensor_t *b, int transpose_b);

/* Add to tensor_blas.c */
int tensor_matmul_transposed_blas(tensor_t *result, const tensor_t *a, int transpose_a,
                                 const tensor_t *b, int transpose_b) {
    /* Check if dimensions are compatible for matrix multiplication */
    size_t a_rows = transpose_a ? a->cols : a->rows;
    size_t a_cols = transpose_a ? a->rows : a->cols;
    size_t b_rows = transpose_b ? b->cols : b->rows;
    size_t b_cols = transpose_b ? b->rows : b->cols;
    
    if (a_cols != b_rows) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status;
    if (result->data != NULL && result->owner) {
        if (result->rows != a_rows || result->cols != b_cols) {
            tensor_free(result);
            status = tensor_create_aligned(result, a_rows, b_cols);
            if (status != TENSOR_SUCCESS) {
                return status;
            }
        }
    } else if (result->data == NULL) {
        status = tensor_create_aligned(result, a_rows, b_cols);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* Determine CBLAS order and parameters based on tensor layout */
    enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE trans_a, trans_b;
    int m, n, k, lda, ldb, ldc;
    
    if (a->layout == TENSOR_ROW_MAJOR) {
        order = CblasRowMajor;
        trans_a = transpose_a ? CblasTrans : CblasNoTrans;
        trans_b = transpose_b ? CblasTrans : CblasNoTrans;
        m = a_rows;
        n = b_cols;
        k = a_cols;
        lda = a->row_stride;
        ldb = b->row_stride;
        ldc = result->row_stride;
    } else { /* TENSOR_COL_MAJOR */
        order = CblasColMajor;
        trans_a = transpose_a ? CblasTrans : CblasNoTrans;
        trans_b = transpose_b ? CblasTrans : CblasNoTrans;
        m = a_rows;
        n = b_cols;
        k = a_cols;
        lda = a->col_stride;
        ldb = b->col_stride;
        ldc = result->col_stride;
    }
    
    /* Call BLAS GEMM */
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cblas_sgemm(order, trans_a, trans_b, m, n, k, alpha, a->data, lda, 
                b->data, ldb, beta, result->data, ldc);
    
    return TENSOR_SUCCESS;
}
```

This function allows you to multiply matrices with optional transposition, which is very useful for operations like computing covariance matrices (A^T * A).

## 6.4 Implementing Vector Operations with BLAS

BLAS provides optimized routines for vector operations as well. Let's implement some common ones.

### AXPY: y = alpha*x + y

AXPY ("A times X Plus Y") is a fundamental BLAS Level 1 operation:

```c
/* Add to tensor_blas.c */
int tensor_axpy_blas(tensor_t *result, tensor_elem_t alpha, const tensor_t *x, const tensor_t *y) {
    /* Check if shapes match */
    if (x->rows != y->rows || x->cols != y->cols) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* If result is not y, we need to copy y to result first */
    if (result != y) {
        int status = tensor_copy(result, y);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* Calculate total number of elements */
    size_t n = x->rows * x->cols;
    
    /* Determine if we need to handle non-contiguous memory */
    int x_contiguous = (x->layout == TENSOR_ROW_MAJOR && x->col_stride == 1) ||
                      (x->layout == TENSOR_COL_MAJOR && x->row_stride == 1);
    int result_contiguous = (result->layout == TENSOR_ROW_MAJOR && result->col_stride == 1) ||
                           (result->layout == TENSOR_COL_MAJOR && result->row_stride == 1);
    
    if (x_contiguous && result_contiguous) {
        /* Both tensors are contiguous, use BLAS directly */
        int incx = 1;
        int incy = 1;
        
        cblas_saxpy(n, alpha, x->data, incx, result->data, incy);
    } else {
        /* Handle non-contiguous memory */
        for (size_t i = 0; i < x->rows; i++) {
            for (size_t j = 0; j < x->cols; j++) {
                size_t idx_x = i * x->row_stride + j * x->col_stride;
                size_t idx_result = i * result->row_stride + j * result->col_stride;
                
                result->data[idx_result] += alpha * x->data[idx_x];
            }
        }
    }
    
    return TENSOR_SUCCESS;
}
```

### GEMV: y = alpha*A*x + beta*y

GEMV (General Matrix-Vector Multiplication) is a BLAS Level 2 operation:

```c
/* Add to tensor_blas.c */
int tensor_gemv_blas(tensor_t *result, tensor_elem_t alpha, const tensor_t *a, 
                    const tensor_t *x, tensor_elem_t beta, const tensor_t *y) {
    /* Check if dimensions are compatible */
    if (a->cols != x->rows || (x->cols != 1 && x->rows != 1)) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    if (y != NULL && (y->rows != a->rows || (y->cols != 1 && y->rows != 1))) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status;
    if (result != y) {
        if (result->data != NULL && result->owner) {
            if (result->rows != a->rows || result->cols != 1) {
                tensor_free(result);
                status = tensor_create_aligned(result, a->rows, 1);
                if (status != TENSOR_SUCCESS) {
                    return status;
                }
            }
        } else if (result->data == NULL) {
            status = tensor_create_aligned(result, a->rows, 1);
            if (status != TENSOR_SUCCESS) {
                return status;
            }
        }
        
        /* If y is provided and result is not y, copy y to result and scale by beta */
        if (y != NULL) {
            tensor_copy(result, y);
            if (beta != 1.0f) {
                for (size_t i = 0; i < result->rows; i++) {
                    result->data[i * result->row_stride] *= beta;
                }
            }
        } else {
            /* Initialize result to zero */
            for (size_t i = 0; i < result->rows; i++) {
                result->data[i * result->row_stride] = 0.0f;
            }
            beta = 0.0f;
        }
    } else {
        /* result is y, scale it by beta */
        if (beta != 1.0f) {
            for (size_t i = 0; i < result->rows; i++) {
                result->data[i * result->row_stride] *= beta;
            }
        }
    }
    
    /* Determine if we need to handle non-contiguous memory */
    int a_contiguous = (a->layout == TENSOR_ROW_MAJOR && a->col_stride == 1) ||
                      (a->layout == TENSOR_COL_MAJOR && a->row_stride == 1);
    int x_contiguous = (x->layout == TENSOR_ROW_MAJOR && x->col_stride == 1) ||
                      (x->layout == TENSOR_COL_MAJOR && x->row_stride == 1);
    int result_contiguous = (result->layout == TENSOR_ROW_MAJOR && result->col_stride == 1) ||
                           (result->layout == TENSOR_COL_MAJOR && result->row_stride == 1);
    
    /* Create temporary contiguous copies if needed */
    tensor_t a_copy, x_copy;
    const tensor_t *a_ptr = a;
    const tensor_t *x_ptr = x;
    
    if (!a_contiguous) {
        tensor_create_aligned(&a_copy, a->rows, a->cols);
        tensor_copy(&a_copy, a);
        a_ptr = &a_copy;
    }
    
    if (!x_contiguous) {
        tensor_create_aligned(&x_copy, x->rows, x->cols);
        tensor_copy(&x_copy, x);
        x_ptr = &x_copy;
    }
    
    /* Determine CBLAS order and parameters based on tensor layout */
    enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE trans;
    int m, n, lda, incx, incy;
    
    if (a_ptr->layout == TENSOR_ROW_MAJOR) {
        order = CblasRowMajor;
        trans = CblasNoTrans;
        m = a_ptr->rows;
        n = a_ptr->cols;
        lda = a_ptr->row_stride;
    } else { /* TENSOR_COL_MAJOR */
        order = CblasColMajor;
        trans = CblasNoTrans;
        m = a_ptr->rows;
        n = a_ptr->cols;
        lda = a_ptr->col_stride;
    }
    
    incx = (x_ptr->cols == 1) ? x_ptr->row_stride : x_ptr->col_stride;
    incy = (result->cols == 1) ? result->row_stride : result->col_stride;
    
    /* Call BLAS GEMV */
    cblas_sgemv(order, trans, m, n, alpha, a_ptr->data, lda, 
                x_ptr->data, incx, beta, result->data, incy);
    
    /* Clean up temporary tensors if created */
    if (!a_contiguous) {
        tensor_free(&a_copy);
    }
    if (!x_contiguous) {
        tensor_free(&x_copy);
    }
    
    return TENSOR_SUCCESS;
}
```

This function handles matrix-vector multiplication, which is a common operation in many tensor algorithms.

## 6.5 Implementing Tensor Contractions with BLAS

Tensor contraction is a generalization of matrix multiplication to higher-dimensional tensors. While BLAS doesn't directly support tensor contractions, we can implement them using BLAS matrix operations.

### Batched Matrix Multiplication

Let's implement batched matrix multiplication, which is a common tensor operation:

```c
/* Add to tensor_blas.h */
int tensor_batched_matmul_blas(tensor_t *result, const tensor_t *a, const tensor_t *b, size_t batch_size);

/* Add to tensor_blas.c */
int tensor_batched_matmul_blas(tensor_t *result, const tensor_t *a, const tensor_t *b, size_t batch_size) {
    /* Check if dimensions are compatible */
    if (a->rows % batch_size != 0 || b->rows % batch_size != 0) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    size_t a_batch_rows = a->rows / batch_size;
    size_t b_batch_rows = b->rows / batch_size;
    
    if (a->cols != b_batch_rows) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status;
    if (result->data != NULL && result->owner) {
        if (result->rows != a_batch_rows * batch_size || result->cols != b->cols) {
            tensor_free(result);
            status = tensor_create_aligned(result, a_batch_rows * batch_size, b->cols);
            if (status != TENSOR_SUCCESS) {
                return status;
            }
        }
    } else if (result->data == NULL) {
        status = tensor_create_aligned(result, a_batch_rows * batch_size, b->cols);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* Process each batch separately */
    #pragma omp parallel for if(batch_size > 1)
    for (size_t batch = 0; batch < batch_size; batch++) {
        /* Create tensor views for this batch */
        tensor_t a_batch, b_batch, result_batch;
        
        tensor_view(&a_batch, a, batch * a_batch_rows, 0, a_batch_rows, a->cols);
        tensor_view(&b_batch, b, batch * b_batch_rows, 0, b_batch_rows, b->cols);
        tensor_view(&result_batch, result, batch * a_batch_rows, 0, a_batch_rows, b->cols);
        
        /* Perform matrix multiplication for this batch */
        tensor_matmul_blas(&result_batch, &a_batch, &b_batch);
    }
    
    return TENSOR_SUCCESS;
}
```

This function performs matrix multiplication for multiple batches in parallel, which is useful for operations like batch processing in neural networks.

## 6.6 Fallback Mechanisms for Unsupported Operations

Not all tensor operations are directly supported by BLAS. Let's implement a fallback mechanism that uses BLAS when possible and falls back to our custom implementations otherwise.

### Creating a Unified Interface

Let's create a unified interface that automatically selects the best implementation:

```c
/* Add to tensor_ops.h */
/* Flags to control which implementation to use */
typedef enum {
    TENSOR_IMPL_AUTO = 0,    /* Automatically select best implementation */
    TENSOR_IMPL_NAIVE = 1,    /* Use naive implementation */
    TENSOR_IMPL_OPENMP = 2,   /* Use OpenMP implementation */
    TENSOR_IMPL_SIMD = 3,     /* Use SIMD implementation */
    TENSOR_IMPL_BLAS = 4      /* Use BLAS implementation */
} tensor_impl_t;

/* Set global implementation preference */
void tensor_set_default_impl(tensor_impl_t impl);

/* Get current implementation preference */
tensor_impl_t tensor_get_default_impl();

/* Unified interface for matrix multiplication */
int tensor_matmul_auto(tensor_t *result, const tensor_t *a, const tensor_t *b, tensor_impl_t impl);

/* Add to tensor_ops.c */
static tensor_impl_t default_impl = TENSOR_IMPL_AUTO;

void tensor_set_default_impl(tensor_impl_t impl) {
    default_impl = impl;
}

tensor_impl_t tensor_get_default_impl() {
    return default_impl;
}

int tensor_matmul_auto(tensor_t *result, const tensor_t *a, const tensor_t *b, tensor_impl_t impl) {
    /* If implementation not specified, use default */
    if (impl == TENSOR_IMPL_AUTO) {
        impl = default_impl;
    }
    
    /* If still auto, select based on tensor size and available implementations */
    if (impl == TENSOR_IMPL_AUTO) {
        size_t total_elements = a->rows * b->cols;
        
        #if defined(USE_MKL) || defined(USE_OPENBLAS) || defined(USE_ATLAS)
            /* Use BLAS for large matrices */
            if (total_elements > 10000) {
                impl = TENSOR_IMPL_BLAS;
            }
            /* Use SIMD for medium matrices */
            else if (total_elements > 1000) {
                impl = TENSOR_IMPL_SIMD;
            }
            /* Use OpenMP for small matrices */
            else if (total_elements > 100) {
                impl = TENSOR_IMPL_OPENMP;
            }
            /* Use naive for tiny matrices */
            else {
                impl = TENSOR_IMPL_NAIVE;
            }
        #elif defined(__AVX__) || defined(__SSE__)
            /* No BLAS, but SIMD available */
            if (total_elements > 1000) {
                impl = TENSOR_IMPL_SIMD;
            }
            else if (total_elements > 100) {
                impl = TENSOR_IMPL_OPENMP;
            }
            else {
                impl = TENSOR_IMPL_NAIVE;
            }
        #elif defined(_OPENMP)
            /* No BLAS or SIMD, but OpenMP available */
            if (total_elements > 100) {
                impl = TENSOR_IMPL_OPENMP;
            }
            else {
                impl = TENSOR_IMPL_NAIVE;
            }
        #else
            /* Only naive implementation available */
            impl = TENSOR_IMPL_NAIVE;
        #endif
    }
    
    /* Call appropriate implementation */
    switch (impl) {
        case TENSOR_IMPL_BLAS:
            #if defined(USE_MKL) || defined(USE_OPENBLAS) || defined(USE_ATLAS)
                return tensor_matmul_blas(result, a, b);
            #else
                /* Fall through to SIMD if BLAS not available */
            #endif
            
        case TENSOR_IMPL_SIMD:
            #if defined(__AVX__) || defined(__SSE__)
                return tensor_matmul_simd(result, a, b);
            #else
                /* Fall through to OpenMP if SIMD not available */
            #endif
            
        case TENSOR_IMPL_OPENMP:
            #if defined(_OPENMP)
                return tensor_matmul(result, a, b);  /* Our OpenMP implementation */
            #else
                /* Fall through to naive if OpenMP not available */
            #endif
            
        case TENSOR_IMPL_NAIVE:
        default:
            return tensor_matmul_naive(result, a, b);  /* Our naive implementation */
    }
}
```

This unified interface automatically selects the best available implementation based on tensor size and available libraries.

## 6.7 Benchmarking BLAS vs. Custom Implementations

Let's create a benchmark to compare the performance of different implementations:

```c
/* benchmark_blas.c */
#include "tensor.h"
#include "tensor_ops.h"
#include "tensor_simd.h"
#include "tensor_blas.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>

/* Benchmark function for matrix multiplication */
double benchmark_matmul(const tensor_t *a, const tensor_t *b, tensor_t *result, tensor_impl_t impl) {
    /* Warm up the cache */
    tensor_matmul_auto(result, a, b, impl);
    
    /* Measure performance */
    double start_time = omp_get_wtime();
    
    /* Perform matrix multiplication */
    tensor_matmul_auto(result, a, b, impl);
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main() {
    tensor_t a, b, result;
    size_t sizes[] = {100, 500, 1000, 2000};
    
    printf("BLAS Implementation: %s\n\n", tensor_blas_implementation());
    
    printf("Matrix Multiplication Performance:\n");
    printf("Size  | Naive (s) | OpenMP (s) | SIMD (s) | BLAS (s) | BLAS Speedup\n");
    printf("------|-----------|------------|----------|----------|-------------\n");
    
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        size_t size = sizes[i];
        
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
        
        /* Benchmark different implementations */
        double naive_time = benchmark_matmul(&a, &b, &result, TENSOR_IMPL_NAIVE);
        double openmp_time = benchmark_matmul(&a, &b, &result, TENSOR_IMPL_OPENMP);
        double simd_time = benchmark_matmul(&a, &b, &result, TENSOR_IMPL_SIMD);
        double blas_time = benchmark_matmul(&a, &b, &result, TENSOR_IMPL_BLAS);
        
        /* Calculate speedup */
        double blas_speedup = naive_time / blas_time;
        
        printf("%5zu | %9.6f | %10.6f | %8.6f | %8.6f | %11.2fx\n",
               size, naive_time, openmp_time, simd_time, blas_time, blas_speedup);
        
        /* Clean up */
        tensor_free(&a);
        tensor_free(&b);
        tensor_free(&result);
    }
    
    return 0;
}
```

This benchmark compares the performance of naive, OpenMP, SIMD, and BLAS implementations for different matrix sizes.

## Visualizing BLAS Integration

Let's visualize how BLAS integrates with our tensor library:

```
Tensor Library Architecture with BLAS Integration:

+---------------------+
| User Application    |
+---------------------+
          |
          v
+---------------------+
| Unified Tensor API  |
| (tensor_ops.h)      |
+---------------------+
          |
          v
+---------------------+     +---------------------+
| Implementation      |---->| BLAS Implementation |
| Selector            |     | (tensor_blas.h)     |
+---------------------+     +---------------------+
          |                           |
          |                           v
          |                 +---------------------+
          |                 | BLAS Library        |
          |                 | (MKL/OpenBLAS/ATLAS)|
          |                 +---------------------+
          |
          v
+---------------------+     +---------------------+
| Custom              |---->| SIMD Implementation |
| Implementations     |     | (tensor_simd.h)     |
+---------------------+     +---------------------+
          |
          v
+---------------------+
| OpenMP Implementation|
| (tensor_ops.c)      |
+---------------------+
          |
          v
+---------------------+
| Naive Implementation |
| (tensor_ops.c)      |
+---------------------+
```

## Common Pitfalls and Debugging

Let's discuss some common issues you might encounter when integrating BLAS libraries.

### Linking Issues

One of the most common problems is incorrect linking. Here are some typical errors and solutions:

1. **Undefined symbols**: If you see errors like "undefined reference to `cblas_sgemm`", it means the linker can't find the BLAS library.

   Solution: Check your linking flags. For OpenBLAS, ensure you're using `-lopenblas`. For MKL, the linking is more complex and requires multiple libraries.

2. **Wrong BLAS implementation**: If your program links but produces incorrect results, you might be linking against the wrong BLAS implementation.

   Solution: Use `dlopen` to check which library is actually being loaded:

   ```c
   #include <dlfcn.h>
   
   void check_blas_library() {
       void *handle = dlopen(NULL, RTLD_NOW);
       void *sym = dlsym(handle, "cblas_sgemm");
       Dl_info info;
       dladdr(sym, &info);
       printf("BLAS library: %s\n", info.dli_fname);
       dlclose(handle);
   }
   ```

3. **Thread safety issues**: Some BLAS implementations have thread safety issues when used with OpenMP.

   Solution: Set the number of BLAS threads to 1 when using OpenMP:

   ```c
   /* For OpenBLAS */
   openblas_set_num_threads(1);
   
   /* For MKL */
   mkl_set_num_threads(1);
   ```

### Performance Issues

If you're not seeing the expected performance improvements, consider these issues:

1. **Small matrices**: BLAS has overhead that makes it less efficient for very small matrices.

   Solution: Use the unified interface that selects the appropriate implementation based on matrix size.

2. **Non-contiguous memory**: BLAS expects contiguous memory layouts.

   Solution: Create temporary contiguous copies for non-contiguous tensors, as shown in our implementations.

3. **Excessive copying**: Creating temporary copies can negate performance benefits.

   Solution: Design your algorithms to work with contiguous memory when possible.

### Debugging BLAS Integration

Here are some tips for debugging BLAS integration:

1. **Verify correctness**: Always compare BLAS results with your custom implementation for small test cases.

   ```c
   /* Verify BLAS implementation */
   tensor_t a_small, b_small, result_blas, result_custom;
   tensor_create_aligned(&a_small, 3, 3);
   tensor_create_aligned(&b_small, 3, 3);
   tensor_create_aligned(&result_blas, 3, 3);
   tensor_create_aligned(&result_custom, 3, 3);
   
   /* Initialize with known values */
   /* ... */
   
   /* Compute using both implementations */
   tensor_matmul_blas(&result_blas, &a_small, &b_small);
   tensor_matmul(&result_custom, &a_small, &b_small);
   
   /* Compare results */
   for (size_t i = 0; i < 3; i++) {
       for (size_t j = 0; j < 3; j++) {
           float diff = fabs(result_blas.data[i * 3 + j] - result_custom.data[i * 3 + j]);
           if (diff > 1e-5) {
               printf("Mismatch at (%zu, %zu): BLAS = %f, Custom = %f\n",
                      i, j, result_blas.data[i * 3 + j], result_custom.data[i * 3 + j]);
           }
       }
   }
   ```

2. **Check BLAS parameters**: Incorrect parameters to BLAS functions can cause subtle bugs.

   Solution: Add debug prints for all BLAS parameters:

   ```c
   printf("GEMM params: order=%d, transA=%d, transB=%d, M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d\n",
          order, trans_a, trans_b, m, n, k, lda, ldb, ldc);
   ```

3. **Memory alignment**: Some BLAS implementations require aligned memory.

   Solution: Use `posix_memalign` or `aligned_alloc` for memory allocation, as shown in our implementations.

## Exercises

### Exercise 1: Implement a BLAS-Accelerated Neural Network Layer

Implement a fully connected neural network layer using BLAS for both forward and backward passes.

Hint: Forward pass uses GEMM, backward pass uses GEMM with transposed matrices.

Partial solution:

```c
/* Forward pass: output = activation(weights * input + bias) */
int nn_forward_blas(tensor_t *output, const tensor_t *weights, const tensor_t *input, const tensor_t *bias) {
    /* Check dimensions */
    if (weights->cols != input->rows) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create temporary tensor for weights * input */
    tensor_t temp;
    tensor_create_aligned(&temp, weights->rows, input->cols);
    
    /* Compute weights * input using BLAS */
    tensor_matmul_blas(&temp, weights, input);
    
    /* Add bias */
    if (bias != NULL) {
        /* ... (implement bias addition) ... */
    }
    
    /* Apply activation function (e.g., ReLU) */
    tensor_create_aligned(output, temp.rows, temp.cols);
    
    for (size_t i = 0; i < temp.rows; i++) {
        for (size_t j = 0; j < temp.cols; j++) {
            size_t idx = i * temp.row_stride + j * temp.col_stride;
            /* ReLU activation: max(0, x) */
            output->data[idx] = temp.data[idx] > 0 ? temp.data[idx] : 0;
        }
    }
    
    /* Clean up */
    tensor_free(&temp);
    
    return TENSOR_SUCCESS;
}
```

### Exercise 2: Implement a BLAS-Accelerated Convolutional Layer

Implement a convolutional layer using BLAS by converting convolution to matrix multiplication (im2col technique).

Hint: Rearrange input patches into columns of a matrix, then use GEMM.

### Exercise 3: Create a Benchmark Suite for Different BLAS Implementations

Create a comprehensive benchmark suite that compares different BLAS implementations (OpenBLAS, MKL, ATLAS) across various tensor operations and sizes.

Partial solution:

```c
/* Benchmark different BLAS implementations */
void benchmark_blas_implementations() {
    /* Define tensor sizes to benchmark */
    size_t sizes[] = {100, 500, 1000, 2000};
    
    /* Define operations to benchmark */
    const char *operations[] = {"GEMM", "GEMV", "AXPY"};
    
    /* Define BLAS implementations to benchmark */
    const char *implementations[] = {"OpenBLAS", "MKL", "ATLAS"};
    
    /* For each operation and size */
    for (size_t op = 0; op < sizeof(operations) / sizeof(operations[0]); op++) {
        printf("\nBenchmarking %s:\n", operations[op]);
        printf("Size  | OpenBLAS (s) | MKL (s) | ATLAS (s)\n");
        printf("------|--------------|---------|----------\n");
        
        for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
            size_t size = sizes[i];
            printf("%5zu | ", size);
            
            /* Create tensors */
            tensor_t a, b, c, result;
            tensor_create_aligned(&a, size, size);
            tensor_create_aligned(&b, size, size);
            tensor_create_aligned(&c, size, 1);
            tensor_create_aligned(&result, size, size);
            
            /* Initialize with random values */
            /* ... (initialization code) ... */
            
            /* Benchmark each implementation */
            /* ... (benchmarking code) ... */
            
            /* Clean up */
            tensor_free(&a);
            tensor_free(&b);
            tensor_free(&c);
            tensor_free(&result);
        }
    }
}
```

## Summary and Key Takeaways

In this chapter, we've explored integrating BLAS libraries for production-grade tensor performance:

- We implemented wrappers for BLAS routines that work seamlessly with our tensor infrastructure.
- We created a unified interface that automatically selects the best implementation based on tensor size and available libraries.
- We addressed common issues like non-contiguous memory and different tensor layouts.
- We benchmarked BLAS against our custom implementations to quantify the performance benefits.

Key takeaways:

1. **BLAS Integration**: BLAS libraries provide highly optimized implementations of common linear algebra operations, offering significant performance improvements over custom code.
2. **Abstraction Layers**: Creating proper abstraction layers allows you to switch between different BLAS implementations without changing your application code.
3. **Memory Layout Considerations**: BLAS expects specific memory layouts, so you need to handle non-contiguous tensors appropriately.
4. **Implementation Selection**: Different implementations are optimal for different tensor sizes and operations, so a unified interface that selects the best implementation is valuable.
5. **Fallback Mechanisms**: Not all operations are supported by BLAS, so you need fallback mechanisms to your custom implementations.

By effectively integrating BLAS libraries, you can achieve production-grade performance for your tensor operations, often 10-100x faster than naive implementations.

In the next chapter, we'll explore debugging memory corruption in tensor programs, a critical skill for building robust tensor systems.

## Further Reading

1. "BLAS (Basic Linear Algebra Subprograms)" - Official documentation at http://www.netlib.org/blas/

2. "Intel Math Kernel Library Developer Reference" - Comprehensive guide to Intel MKL functions and usage

3. "OpenBLAS: An optimized BLAS library" - GitHub repository and documentation at https://github.com/xianyi/OpenBLAS