---
layout: chapter
title: Setting Up Your Tensor Programming Toolkit
number: 1
description: Start your journey by building a solid foundation for tensor programming in C. You'll set up a development environment, implement a basic tensor structure, and learn essential debugging techniques.
---

## You Will Learn To...

- Configure a complete C development environment optimized for numerical computing
- Implement a foundational tensor struct with proper memory management
- Link your C programs with high-performance BLAS/LAPACK libraries
- Create validation tools to ensure tensor operations work correctly
- Debug common memory and initialization issues in tensor code

## 1.1 Installing Compilers and Tools

Before diving into tensor programming, we need a proper development environment. I've wasted countless hours tracking down bugs that stemmed from compiler inconsistencies or missing tools, so let's get this right from the start.

### GCC and Essential Build Tools

On Debian/Ubuntu systems, you'll need the build-essential package:

```bash
sudo apt update
sudo apt install build-essential gdb make cmake
```

On Fedora/RHEL:

```bash
sudo dnf groupinstall "Development Tools"
sudo dnf install gdb cmake
```

I prefer GCC for tensor programming because of its mature optimization capabilities and excellent support for OpenMP, which we'll use later for parallelization. Clang is a good alternative, especially if you're working on macOS.

Make sure your GCC is recent enough (at least version 9) to support modern C standards and optimizations:

```bash
gcc --version
```

### Setting Up Debugging Tools

Debugging memory issues is critical when working with tensors, as we'll be allocating large blocks and performing complex pointer arithmetic. Valgrind is indispensable here:

```bash
sudo apt install valgrind  # Debian/Ubuntu
sudo dnf install valgrind   # Fedora/RHEL
```

I also recommend installing `perf` for performance analysis:

```bash
sudo apt install linux-tools-common linux-tools-generic  # Debian/Ubuntu
sudo dnf install perf                                     # Fedora/RHEL
```

A quick test to ensure Valgrind is working properly:

```bash
cat > memtest.c << EOF
#include <stdlib.h>

int main() {
    int *x = malloc(10 * sizeof(int));
    x[10] = 0;  // Buffer overflow
    return 0;
}
EOF

gcc -g memtest.c -o memtest
valgrind ./memtest
```

You should see Valgrind report an "Invalid write" error, which confirms it's working correctly. If you don't see this error, something's wrong with your Valgrind installation.

## 1.2 Configuring BLAS/LAPACK for C Linkage

While we'll implement tensor operations from scratch, we'll eventually want to leverage optimized BLAS (Basic Linear Algebra Subprograms) libraries for production code. Let's set these up now.

### Installing BLAS/LAPACK Development Libraries

For development, OpenBLAS provides excellent performance across different architectures:

```bash
sudo apt install libopenblas-dev liblapack-dev  # Debian/Ubuntu
sudo dnf install openblas-devel lapack-devel    # Fedora/RHEL
```

If you're working on a system with Intel processors, you might consider Intel's Math Kernel Library (MKL) instead, which is free for most uses but requires registration:

```bash
# After downloading and extracting MKL
source /path/to/mkl/bin/mklvars.sh intel64
```

### Testing BLAS Linkage

Let's verify our BLAS installation with a simple matrix multiplication example:

```c
/* blas_test.c */
#include <stdio.h>
#include <stdlib.h>

/* BLAS function prototype */
extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
                 double *alpha, double *a, int *lda, double *b, int *ldb,
                 double *beta, double *c, int *ldc);

int main() {
    int m = 2, n = 2, k = 2;
    int lda = 2, ldb = 2, ldc = 2;
    double alpha = 1.0, beta = 0.0;
    char trans = 'N';
    
    /* Allocate matrices */
    double *A = malloc(m * k * sizeof(double));
    double *B = malloc(k * n * sizeof(double));
    double *C = malloc(m * n * sizeof(double));
    
    /* Initialize matrices */
    A[0] = 1.0; A[1] = 2.0; A[2] = 3.0; A[3] = 4.0;
    B[0] = 5.0; B[1] = 6.0; B[2] = 7.0; B[3] = 8.0;
    
    /* Call BLAS dgemm (matrix multiplication) */
    dgemm_(&trans, &trans, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    
    /* Print result */
    printf("Result matrix C:\n");
    printf("%f %f\n%f %f\n", C[0], C[1], C[2], C[3]);
    
    /* Free memory */
    free(A);
    free(B);
    free(C);
    
    return 0;
}
```

Compile with:

```bash
gcc -o blas_test blas_test.c -lopenblas -O3
```

If you get linker errors like `undefined reference to 'dgemm_'`, try these troubleshooting steps:

1. Check if the library is installed correctly:
   ```bash
   ls -l /usr/lib/libopenblas.so
   ```

2. Try specifying the library path explicitly:
   ```bash
   gcc -o blas_test blas_test.c -L/usr/lib -lopenblas -O3
   ```

3. On some systems, you might need to link against LAPACK as well:
   ```bash
   gcc -o blas_test blas_test.c -lopenblas -llapack -O3
   ```

One common issue I've encountered is that some systems use different naming conventions for BLAS functions. If you're still having trouble, try removing the trailing underscore from the function name in your code.

## 1.3 Creating Your First Tensor Struct: Memory Allocation and Metadata

Now let's implement our core tensor data structure. A tensor is essentially a multi-dimensional array with associated metadata. For simplicity, we'll start with a 2D tensor (matrix) and extend it later.

### Designing the Tensor Struct

```c
/* tensor.h */
#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Error codes */
#define TENSOR_SUCCESS 0
#define TENSOR_DIMENSION_MISMATCH 1
#define TENSOR_ALLOCATION_FAILED 2
#define TENSOR_INDEX_OUT_OF_BOUNDS 3

/* Tensor data type - we'll use double for now */
typedef double tensor_elem_t;

/* Tensor structure */
typedef struct {
    size_t rows;        /* Number of rows */
    size_t cols;        /* Number of columns */
    size_t stride;      /* Row stride (for non-contiguous storage) */
    tensor_elem_t *data; /* Pointer to the actual data */
    int owner;          /* Whether this tensor owns its data (for memory management) */
} tensor_t;

/* Function prototypes */
int tensor_create(tensor_t *t, size_t rows, size_t cols);
void tensor_free(tensor_t *t);
int tensor_set(tensor_t *t, size_t i, size_t j, tensor_elem_t value);
int tensor_get(const tensor_t *t, size_t i, size_t j, tensor_elem_t *value);
void tensor_print(const tensor_t *t);

#endif /* TENSOR_H */
```

Now let's implement these functions:

```c
/* tensor.c */
#include "tensor.h"

/* Create a new tensor with specified dimensions */
int tensor_create(tensor_t *t, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Allocate memory for data */
    t->data = (tensor_elem_t *)malloc(rows * cols * sizeof(tensor_elem_t));
    if (t->data == NULL) {
        return TENSOR_ALLOCATION_FAILED;
    }
    
    /* Initialize metadata */
    t->rows = rows;
    t->cols = cols;
    t->stride = cols;  /* For row-major storage, stride equals columns */
    t->owner = 1;      /* This tensor owns its data */
    
    /* Initialize data to zero */
    memset(t->data, 0, rows * cols * sizeof(tensor_elem_t));
    
    return TENSOR_SUCCESS;
}

/* Free tensor resources */
void tensor_free(tensor_t *t) {
    if (t->data != NULL && t->owner) {
        free(t->data);
        t->data = NULL;
    }
    t->rows = 0;
    t->cols = 0;
    t->stride = 0;
}

/* Set value at specified position */
int tensor_set(tensor_t *t, size_t i, size_t j, tensor_elem_t value) {
    if (i >= t->rows || j >= t->cols) {
        return TENSOR_INDEX_OUT_OF_BOUNDS;
    }
    
    t->data[i * t->stride + j] = value;
    return TENSOR_SUCCESS;
}

/* Get value at specified position */
int tensor_get(const tensor_t *t, size_t i, size_t j, tensor_elem_t *value) {
    if (i >= t->rows || j >= t->cols) {
        return TENSOR_INDEX_OUT_OF_BOUNDS;
    }
    
    *value = t->data[i * t->stride + j];
    return TENSOR_SUCCESS;
}

/* Print tensor contents */
void tensor_print(const tensor_t *t) {
    printf("Tensor %zux%zu:\n", t->rows, t->cols);
    for (size_t i = 0; i < t->rows; i++) {
        for (size_t j = 0; j < t->cols; j++) {
            tensor_elem_t val;
            tensor_get(t, i, j, &val);
            printf("%8.4f ", val);
        }
        printf("\n");
    }
}
```

Let's test our tensor implementation:

```c
/* tensor_test.c */
#include "tensor.h"

int main() {
    tensor_t t;
    int status;
    
    /* Create a 3x3 tensor */
    status = tensor_create(&t, 3, 3);
    if (status != TENSOR_SUCCESS) {
        fprintf(stderr, "Failed to create tensor: error code %d\n", status);
        return 1;
    }
    
    /* Set some values */
    tensor_set(&t, 0, 0, 1.0);
    tensor_set(&t, 0, 1, 2.0);
    tensor_set(&t, 0, 2, 3.0);
    tensor_set(&t, 1, 0, 4.0);
    tensor_set(&t, 1, 1, 5.0);
    tensor_set(&t, 1, 2, 6.0);
    tensor_set(&t, 2, 0, 7.0);
    tensor_set(&t, 2, 1, 8.0);
    tensor_set(&t, 2, 2, 9.0);
    
    /* Print the tensor */
    tensor_print(&t);
    
    /* Test out-of-bounds access */
    status = tensor_set(&t, 3, 3, 10.0);
    if (status == TENSOR_INDEX_OUT_OF_BOUNDS) {
        printf("Successfully caught out-of-bounds access\n");
    }
    
    /* Free the tensor */
    tensor_free(&t);
    
    return 0;
}
```

Compile and run:

```bash
gcc -c tensor.c -o tensor.o -O3 -Wall
gcc tensor_test.c tensor.o -o tensor_test -O3 -Wall
./tensor_test
```

### Memory Layout Considerations

Our tensor implementation uses row-major storage, which is the C convention. This means that elements in the same row are stored contiguously in memory. This layout affects performance, especially for operations that access elements column-wise.

Here's a visual representation of our tensor's memory layout:

```
Memory Layout (Row-Major):

+-------------------+
| rows, cols, etc.  |  Tensor struct (metadata)
+-------------------+
| data pointer      | -------+
+-------------------+        |
                             |
                             v
+---+---+---+---+---+---+---+---+---+
|0,0|0,1|0,2|1,0|1,1|1,2|2,0|2,1|2,2|  Contiguous data in memory
+---+---+---+---+---+---+---+---+---+
```

The `stride` field in our struct allows for non-contiguous storage, which will be useful later when we implement operations like transposition without copying data.

## 1.4 Writing Validation Tests for Tensor Integrity

Proper testing is crucial for tensor operations, as errors can be subtle and hard to detect. Let's create a simple testing framework.

### A Minimal Testing Framework

```c
/* tensor_test_framework.h */
#ifndef TENSOR_TEST_FRAMEWORK_H
#define TENSOR_TEST_FRAMEWORK_H

#include <stdio.h>

#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "ASSERTION FAILED: %s\n", message); \
            fprintf(stderr, "  at %s:%d\n", __FILE__, __LINE__); \
            return 0; \
        } \
    } while (0)

#define RUN_TEST(test_function) \
    do { \
        printf("Running %s... ", #test_function); \
        if (test_function()) { \
            printf("PASSED\n"); \
        } else { \
            printf("FAILED\n"); \
            test_failures++; \
        } \
        test_count++; \
    } while (0)

#endif /* TENSOR_TEST_FRAMEWORK_H */
```

Now let's write some tests for our tensor implementation:

```c
/* tensor_tests.c */
#include "tensor.h"
#include "tensor_test_framework.h"
#include <math.h>

/* Test tensor creation and initialization */
int test_tensor_create() {
    tensor_t t;
    int status = tensor_create(&t, 2, 3);
    
    TEST_ASSERT(status == TENSOR_SUCCESS, "tensor_create should succeed");
    TEST_ASSERT(t.rows == 2, "tensor should have 2 rows");
    TEST_ASSERT(t.cols == 3, "tensor should have 3 columns");
    TEST_ASSERT(t.data != NULL, "tensor data should not be NULL");
    
    /* Check that all elements are initialized to zero */
    for (size_t i = 0; i < t.rows; i++) {
        for (size_t j = 0; j < t.cols; j++) {
            tensor_elem_t val;
            tensor_get(&t, i, j, &val);
            TEST_ASSERT(val == 0.0, "tensor elements should be initialized to zero");
        }
    }
    
    tensor_free(&t);
    return 1;
}

/* Test setting and getting tensor values */
int test_tensor_set_get() {
    tensor_t t;
    tensor_create(&t, 3, 3);
    
    /* Set some values */
    tensor_set(&t, 0, 0, 1.5);
    tensor_set(&t, 1, 1, 2.5);
    tensor_set(&t, 2, 2, 3.5);
    
    /* Check the values */
    tensor_elem_t val;
    tensor_get(&t, 0, 0, &val);
    TEST_ASSERT(fabs(val - 1.5) < 1e-6, "tensor_get should return the value set");
    
    tensor_get(&t, 1, 1, &val);
    TEST_ASSERT(fabs(val - 2.5) < 1e-6, "tensor_get should return the value set");
    
    tensor_get(&t, 2, 2, &val);
    TEST_ASSERT(fabs(val - 3.5) < 1e-6, "tensor_get should return the value set");
    
    /* Check that other elements are still zero */
    tensor_get(&t, 0, 1, &val);
    TEST_ASSERT(val == 0.0, "unset tensor elements should remain zero");
    
    tensor_free(&t);
    return 1;
}

/* Test out-of-bounds access */
int test_tensor_bounds_checking() {
    tensor_t t;
    tensor_create(&t, 2, 2);
    
    /* Try to access out-of-bounds elements */
    int status = tensor_set(&t, 2, 0, 1.0);
    TEST_ASSERT(status == TENSOR_INDEX_OUT_OF_BOUNDS, "out-of-bounds row access should be detected");
    
    status = tensor_set(&t, 0, 2, 1.0);
    TEST_ASSERT(status == TENSOR_INDEX_OUT_OF_BOUNDS, "out-of-bounds column access should be detected");
    
    tensor_elem_t val;
    status = tensor_get(&t, 2, 0, &val);
    TEST_ASSERT(status == TENSOR_INDEX_OUT_OF_BOUNDS, "out-of-bounds row access should be detected");
    
    tensor_free(&t);
    return 1;
}

/* Test memory management */
int test_tensor_memory_management() {
    tensor_t t;
    tensor_create(&t, 100, 100);  /* Allocate a reasonably large tensor */
    
    /* Set some values */
    for (size_t i = 0; i < t.rows; i++) {
        for (size_t j = 0; j < t.cols; j++) {
            tensor_set(&t, i, j, (tensor_elem_t)(i + j));
        }
    }
    
    /* Free the tensor */
    tensor_free(&t);
    
    /* Verify that data pointer is NULL after freeing */
    TEST_ASSERT(t.data == NULL, "tensor_free should set data pointer to NULL");
    TEST_ASSERT(t.rows == 0 && t.cols == 0, "tensor_free should reset dimensions");
    
    return 1;
}

int main() {
    int test_count = 0;
    int test_failures = 0;
    
    RUN_TEST(test_tensor_create);
    RUN_TEST(test_tensor_set_get);
    RUN_TEST(test_tensor_bounds_checking);
    RUN_TEST(test_tensor_memory_management);
    
    printf("\n%d tests run, %d passed, %d failed\n", 
           test_count, test_count - test_failures, test_failures);
    
    return test_failures > 0 ? 1 : 0;
}
```

Compile and run the tests:

```bash
gcc -c tensor.c -o tensor.o -O3 -Wall
gcc tensor_tests.c tensor.o -o tensor_tests -O3 -Wall -lm
./tensor_tests
```

## Common Pitfalls and Debugging

Let's discuss some common issues you might encounter when working with tensor code.

### Memory Leaks in Tensor Operations

One of the most common issues is forgetting to free tensor memory, especially in complex operations that create temporary tensors. Let's see how to detect and fix this using Valgrind.

Consider this flawed function that creates a temporary tensor but doesn't free it:

```c
/* Bad function with memory leak */
int tensor_add_leaky(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create a temporary tensor (never freed!) */
    tensor_t temp;
    tensor_create(&temp, a->rows, a->cols);
    
    /* Perform addition */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            tensor_elem_t val_a, val_b;
            tensor_get(a, i, j, &val_a);
            tensor_get(b, i, j, &val_b);
            tensor_set(&temp, i, j, val_a + val_b);
        }
    }
    
    /* Copy to result (but forget to free temp!) */
    *result = temp;
    
    return TENSOR_SUCCESS;
}
```

To detect this leak, we can use Valgrind:

```bash
valgrind --leak-check=full ./tensor_leak_test
```

Valgrind would report something like:

```
==12345== HEAP SUMMARY:
==12345==     in use at exit: 72 bytes in 1 blocks
==12345==   total heap usage: 3 allocs, 2 frees, 1,096 bytes allocated
==12345== 
==12345== 72 bytes in 1 blocks are definitely lost in loss record 1 of 1
==12345==    at 0x4C2AB80: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x400B2D: tensor_create (tensor.c:15)
==12345==    by 0x400C9F: tensor_add_leaky (tensor_operations.c:7)
==12345==    by 0x400D3A: main (tensor_leak_test.c:12)
```

The correct implementation would free the temporary tensor after copying its data:

```c
/* Corrected function */
int tensor_add(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create a temporary tensor */
    tensor_t temp;
    int status = tensor_create(&temp, a->rows, a->cols);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Perform addition */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            tensor_elem_t val_a, val_b;
            tensor_get(a, i, j, &val_a);
            tensor_get(b, i, j, &val_b);
            tensor_set(&temp, i, j, val_a + val_b);
        }
    }
    
    /* Free existing result data if needed */
    if (result->data != NULL && result->owner) {
        tensor_free(result);
    }
    
    /* Copy data to result */
    result->rows = temp.rows;
    result->cols = temp.cols;
    result->stride = temp.stride;
    result->data = temp.data;
    result->owner = temp.owner;
    
    /* Prevent double-free by marking temp as non-owner */
    temp.owner = 0;
    temp.data = NULL;
    
    return TENSOR_SUCCESS;
}
```

### Alignment Issues for SIMD Operations

When we start using SIMD instructions (in later chapters), memory alignment becomes crucial. Modern CPUs perform best when data is aligned to cache line boundaries (typically 64 bytes).

Here's how we could modify our tensor creation to ensure proper alignment:

```c
#include <stdlib.h>

/* For aligned allocation */
#define TENSOR_ALIGNMENT 64

/* Aligned allocation function */
void* tensor_aligned_malloc(size_t size) {
    void* ptr;
    int result;
    
#ifdef _WIN32
    ptr = _aligned_malloc(size, TENSOR_ALIGNMENT);
    result = (ptr == NULL);
#else
    result = posix_memalign(&ptr, TENSOR_ALIGNMENT, size);
#endif
    
    if (result != 0) {
        return NULL;
    }
    
    return ptr;
}

/* Aligned free function */
void tensor_aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* Modified tensor_create function */
int tensor_create(tensor_t *t, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Allocate aligned memory for data */
    t->data = (tensor_elem_t *)tensor_aligned_malloc(rows * cols * sizeof(tensor_elem_t));
    if (t->data == NULL) {
        return TENSOR_ALLOCATION_FAILED;
    }
    
    /* Initialize metadata */
    t->rows = rows;
    t->cols = cols;
    t->stride = cols;  /* For row-major storage, stride equals columns */
    t->owner = 1;      /* This tensor owns its data */
    
    /* Initialize data to zero */
    memset(t->data, 0, rows * cols * sizeof(tensor_elem_t));
    
    return TENSOR_SUCCESS;
}

/* Modified tensor_free function */
void tensor_free(tensor_t *t) {
    if (t->data != NULL && t->owner) {
        tensor_aligned_free(t->data);
        t->data = NULL;
    }
    t->rows = 0;
    t->cols = 0;
    t->stride = 0;
}
```

### Incorrect BLAS Linkage

When using BLAS functions, a common issue is incorrect function signatures. BLAS functions follow Fortran calling conventions, which can be tricky in C. Here's a debugging example:

```c
/* Incorrect BLAS call */
void incorrect_blas_example() {
    double A[4] = {1.0, 2.0, 3.0, 4.0};
    double B[4] = {5.0, 6.0, 7.0, 8.0};
    double C[4] = {0.0, 0.0, 0.0, 0.0};
    
    int m = 2, n = 2, k = 2;
    double alpha = 1.0, beta = 0.0;
    char trans = 'N';
    
    /* Incorrect: passing values directly */
    dgemm_(trans, trans, m, n, k, alpha, A, m, B, k, beta, C, m);
}
```

The correct way is to pass pointers to all arguments, even scalars and characters:

```c
/* Correct BLAS call */
void correct_blas_example() {
    double A[4] = {1.0, 2.0, 3.0, 4.0};
    double B[4] = {5.0, 6.0, 7.0, 8.0};
    double C[4] = {0.0, 0.0, 0.0, 0.0};
    
    int m = 2, n = 2, k = 2;
    int lda = 2, ldb = 2, ldc = 2;
    double alpha = 1.0, beta = 0.0;
    char trans = 'N';
    
    /* Correct: passing pointers to all arguments */
    dgemm_(&trans, &trans, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}
```

## Building a Complete Tensor Library

Let's create a Makefile to build our tensor library:

```makefile
# Makefile for tensor library

CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -g
LDFLAGS = -lm

# Add OpenBLAS if available
ifneq ($(shell pkg-config --exists openblas && echo yes),)
    CFLAGS += $(shell pkg-config --cflags openblas)
    LDFLAGS += $(shell pkg-config --libs openblas)
else
    LDFLAGS += -lopenblas
endif

# Source files
SRC = tensor.c
TEST_SRC = tensor_tests.c

# Object files
OBJ = $(SRC:.c=.o)
TEST_OBJ = $(TEST_SRC:.c=.o)

# Targets
all: tensor_tests

tensor.o: tensor.c tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

tensor_tests: $(OBJ) $(TEST_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f *.o tensor_tests

.PHONY: all clean
```

## Exercises

### Exercise 1: Extend the Tensor Struct to Support N-Dimensional Tensors

Modify the tensor struct and functions to support tensors of arbitrary dimensions, not just 2D matrices.

Hint: You'll need to replace `rows` and `cols` with an array of dimensions and a `ndim` field:

```c
typedef struct {
    size_t ndim;           /* Number of dimensions */
    size_t *dims;          /* Array of dimension sizes */
    size_t *strides;       /* Array of strides for each dimension */
    tensor_elem_t *data;   /* Pointer to the actual data */
    int owner;             /* Whether this tensor owns its data */
} tensor_t;
```

### Exercise 2: Implement Basic Tensor Operations

Implement the following tensor operations:
- `tensor_add`: Element-wise addition of two tensors
- `tensor_multiply`: Element-wise multiplication of two tensors
- `tensor_scale`: Multiply all elements by a scalar

Ensure proper error handling for dimension mismatches and memory allocation failures.

### Exercise 3: Create a View Function for Tensors

Implement a `tensor_view` function that creates a new tensor that shares data with an existing tensor but may have different dimensions or strides. This is useful for operations like transposition without copying data.

Partial solution:

```c
int tensor_view(tensor_t *view, const tensor_t *original, 
                size_t row_start, size_t row_end,
                size_t col_start, size_t col_end) {
    /* Check bounds */
    if (row_start >= row_end || row_end > original->rows ||
        col_start >= col_end || col_end > original->cols) {
        return TENSOR_INDEX_OUT_OF_BOUNDS;
    }
    
    /* Set up view metadata */
    view->rows = row_end - row_start;
    view->cols = col_end - col_start;
    view->stride = original->stride;
    
    /* Point to the original data with offset */
    view->data = &original->data[row_start * original->stride + col_start];
    
    /* View doesn't own the data */
    view->owner = 0;
    
    return TENSOR_SUCCESS;
}
```

## Summary and Key Takeaways

In this chapter, we've laid the groundwork for tensor programming in C:

- We set up a complete development environment with GCC, debugging tools, and BLAS libraries.
- We implemented a basic tensor struct with proper memory management and bounds checking.
- We created a testing framework to validate tensor operations.
- We explored common pitfalls like memory leaks and alignment issues.
- We learned how to link with BLAS for high-performance operations.

These fundamentals will serve as the foundation for more advanced tensor operations in the coming chapters. The key insight is that proper memory management and careful design of data structures are crucial for building reliable tensor systems.

In the next chapter, we'll implement core tensor operations from scratch, focusing on efficiency and numerical stability.

## Further Reading

1. BLAS Technical Forum Standard: [http://www.netlib.org/blas/blast-forum/](http://www.netlib.org/blas/blast-forum/) - The definitive reference for BLAS function signatures and behavior.

2. Valgrind User Manual: [http://valgrind.org/docs/manual/manual.html](http://valgrind.org/docs/manual/manual.html) - Comprehensive guide to memory debugging with Valgrind.

3. "Optimizing Software in C++" by Agner Fog: [https://www.agner.org/optimize/optimizing_cpp.pdf](https://www.agner.org/optimize/optimizing_cpp.pdf) - Excellent resource on low-level optimization techniques relevant to tensor programming.