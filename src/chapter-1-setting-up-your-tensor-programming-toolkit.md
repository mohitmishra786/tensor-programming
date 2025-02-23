# Chapter 1: Setting Up Your Tensor Programming Toolkit

## You Will Learn To:
- Configure a complete C development environment for high-performance tensor operations
- Implement a flexible tensor structure supporting n-dimensional arrays
- Link and validate external numerical libraries (BLAS/LAPACK)
- Write robust validation tests for tensor operations

## 1.1 Setting Up Your Development Environment

### Essential Tools Installation

On Debian/Ubuntu systems, install the core toolchain:

```bash
sudo apt-get update
sudo apt-get install build-essential gdb valgrind
sudo apt-get install libopenblas-dev liblapack-dev
```

For macOS users with Homebrew:

```bash
brew install gcc llvm open-mpi openblas lapack
```

Verify your GCC installation and ensure it supports OpenMP:

```bash
gcc --version
echo "#include <omp.h>" | gcc -fopenmp -E - > /dev/null
```

### Compiler Flags for Tensor Operations

Create a base Makefile for tensor projects:

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -fopenmp
LDFLAGS = -lopenblas -llapack

# Debug build with sanitizers
debug: CFLAGS += -g -fsanitize=address -fsanitize=undefined
debug: LDFLAGS += -fsanitize=address -fsanitize=undefined
debug: all

# Release build
release: CFLAGS += -DNDEBUG
release: all

all: tensor_test

tensor_test: tensor.o test_tensor.o
    $(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c tensor.h
    $(CC) $(CFLAGS) -c $<

clean:
    rm -f *.o tensor_test
```

The flags we've chosen deserve explanation:
- `-O3`: Aggressive optimization, crucial for tensor operations
- `-march=native`: Generate CPU-specific instructions
- `-fopenmp`: Enable OpenMP for parallel processing
- `-fsanitize=address,undefined`: Catch memory and undefined behavior errors in debug builds

## 1.2 Implementing the Tensor Structure

Let's create a flexible tensor structure that will serve as our foundation:

```c
// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdbool.h>

typedef float dtype;  // Easily change precision as needed

typedef struct {
    dtype* data;        // Contiguous data buffer
    size_t* dims;       // Array of dimension sizes
    size_t* strides;    // Stride for each dimension
    size_t ndims;       // Number of dimensions
    size_t size;        // Total number of elements
    bool owns_data;     // Whether we should free data
} Tensor;

// Core tensor operations
Tensor* tensor_create(size_t ndims, const size_t* dims);
void tensor_free(Tensor* t);
Tensor* tensor_view(Tensor* t, size_t ndims, const size_t* dims);
dtype tensor_get(const Tensor* t, const size_t* indices);
void tensor_set(Tensor* t, const size_t* indices, dtype value);

// Validation
bool tensor_validate(const Tensor* t);
const char* tensor_validation_error(void);

#endif // TENSOR_H
```

Now let's implement these functions:

```c
// tensor.c
#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static char validation_error[256];

Tensor* tensor_create(size_t ndims, const size_t* dims) {
    if (!dims || ndims == 0) {
        return NULL;
    }

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    // Allocate dimension and stride arrays
    t->dims = (size_t*)malloc(ndims * sizeof(size_t));
    t->strides = (size_t*)malloc(ndims * sizeof(size_t));
    if (!t->dims || !t->strides) {
        free(t->dims);
        free(t->strides);
        free(t);
        return NULL;
    }

    // Calculate total size and copy dimensions
    t->ndims = ndims;
    t->size = 1;
    memcpy(t->dims, dims, ndims * sizeof(size_t));

    // Calculate strides (row-major order)
    for (int i = ndims - 1; i >= 0; i--) {
        t->strides[i] = t->size;
        t->size *= dims[i];
    }

    // Allocate data buffer with alignment for SIMD
    t->data = (dtype*)aligned_alloc(32, t->size * sizeof(dtype));
    if (!t->data) {
        free(t->dims);
        free(t->strides);
        free(t);
        return NULL;
    }

    t->owns_data = true;
    return t;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    
    if (t->owns_data && t->data) {
        free(t->data);
    }
    free(t->dims);
    free(t->strides);
    free(t);
}

dtype tensor_get(const Tensor* t, const size_t* indices) {
    if (!tensor_validate(t) || !indices) {
        return 0.0f;  // Error value
    }

    size_t offset = 0;
    for (size_t i = 0; i < t->ndims; i++) {
        if (indices[i] >= t->dims[i]) {
            return 0.0f;  // Index out of bounds
        }
        offset += indices[i] * t->strides[i];
    }
    return t->data[offset];
}

void tensor_set(Tensor* t, const size_t* indices, dtype value) {
    if (!tensor_validate(t) || !indices) {
        return;
    }

    size_t offset = 0;
    for (size_t i = 0; i < t->ndims; i++) {
        if (indices[i] >= t->dims[i]) {
            return;  // Index out of bounds
        }
        offset += indices[i] * t->strides[i];
    }
    t->data[offset] = value;
}

bool tensor_validate(const Tensor* t) {
    if (!t) {
        snprintf(validation_error, sizeof(validation_error),
                "Null tensor pointer");
        return false;
    }

    if (!t->data) {
        snprintf(validation_error, sizeof(validation_error),
                "Null data pointer");
        return false;
    }

    if (!t->dims || !t->strides) {
        snprintf(validation_error, sizeof(validation_error),
                "Null dims or strides pointer");
        return false;
    }

    if (t->ndims == 0) {
        snprintf(validation_error, sizeof(validation_error),
                "Zero dimensions");
        return false;
    }

    size_t computed_size = 1;
    for (size_t i = 0; i < t->ndims; i++) {
        if (t->dims[i] == 0) {
            snprintf(validation_error, sizeof(validation_error),
                    "Zero-length dimension at index %zu", i);
            return false;
        }
        computed_size *= t->dims[i];
    }

    if (computed_size != t->size) {
        snprintf(validation_error, sizeof(validation_error),
                "Size mismatch: stored=%zu, computed=%zu",
                t->size, computed_size);
        return false;
    }

    return true;
}

const char* tensor_validation_error(void) {
    return validation_error;
}
```

## 1.3 Writing Validation Tests

Let's create a simple testing framework:

```c
// test_tensor.c
#include "tensor.h"
#include <stdio.h>
#include <assert.h>

#define TEST(name) static void test_##name(void)
#define RUN_TEST(name) do { \
    printf("Running %s...\n", #name); \
    test_##name(); \
    printf("PASSED\n"); \
} while(0)

TEST(creation) {
    size_t dims[] = {2, 3, 4};
    Tensor* t = tensor_create(3, dims);
    assert(t != NULL);
    assert(tensor_validate(t));
    assert(t->size == 24);
    assert(t->strides[0] == 12);
    assert(t->strides[1] == 4);
    assert(t->strides[2] == 1);
    tensor_free(t);
}

TEST(access) {
    size_t dims[] = {2, 3};
    Tensor* t = tensor_create(2, dims);
    assert(t != NULL);
    
    size_t idx[] = {1, 2};
    tensor_set(t, idx, 42.0f);
    assert(tensor_get(t, idx) == 42.0f);
    
    tensor_free(t);
}

int main(void) {
    RUN_TEST(creation);
    RUN_TEST(access);
    printf("All tests passed!\n");
    return 0;
}
```

## 1.4 Linking with BLAS/LAPACK

Create a helper header for BLAS operations:

```c
// tensor_blas.h
#ifndef TENSOR_BLAS_H
#define TENSOR_BLAS_H

#include "tensor.h"

// BLAS headers might differ by system
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

// Verify BLAS linkage at runtime
bool verify_blas_linkage(void);

// Matrix multiplication: C = alpha*A*B + beta*C
bool tensor_matmul(Tensor* C, const Tensor* A, const Tensor* B,
                  float alpha, float beta);

#endif // TENSOR_BLAS_H
```

Implementation:

```c
// tensor_blas.c
#include "tensor_blas.h"
#include <math.h>

bool verify_blas_linkage(void) {
    // Create small vectors for BLAS test
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {4.0f, 5.0f, 6.0f};
    
    // Compute dot product using BLAS
    float result = cblas_sdot(3, x, 1, y, 1);
    
    // Verify result (should be 32.0)
    return fabsf(result - 32.0f) < 1e-6;
}

bool tensor_matmul(Tensor* C, const Tensor* A, const Tensor* B,
                  float alpha, float beta) {
    if (!tensor_validate(A) || !tensor_validate(B) || !tensor_validate(C)) {
        return false;
    }
    
    // Verify matrix dimensions
    if (A->ndims != 2 || B->ndims != 2 || C->ndims != 2) {
        return false;
    }
    
    size_t M = A->dims[0];
    size_t K = A->dims[1];
    size_t N = B->dims[1];
    
    if (B->dims[0] != K || C->dims[0] != M || C->dims[1] != N) {
        return false;
    }
    
    // Call BLAS matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A->data, K, B->data, N,
                beta, C->data, N);
    
    return true;
}
```

## Debugging Common Issues

### Memory Leaks Detection

Run your tensor programs under Valgrind:

```bash
valgrind --leak-check=full --show-leak-kinds=all ./tensor_test
```

Common issues and solutions:

1. **Double Free Errors**
```text
==12345== Invalid free() / delete / delete[]
==12345==    at 0x4C3123B: free (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x108B41: tensor_free (tensor.c:45)
```
Solution: Check tensor ownership with `owns_data` flag before freeing.

2. **Memory Leaks**
```text
==12345== 24 bytes in 1 blocks are definitely lost
==12345==    at 0x4C31B25: malloc (vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x108A41: tensor_create (tensor.c:23)
```
Solution: Ensure all allocation paths have matching deallocation in error cases.

### Using GDB for Debugging

Set up a .gdbinit file for tensor debugging:

```
# .gdbinit
define pt
    print *((Tensor*)$arg0)
    print *((Tensor*)$arg0)->dims@((Tensor*)$arg0)->ndims
end

break tensor_validate
commands
    silent
    pt $rdi
    continue
end
```

### Custom Assert Macros  
```c
#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "Test failed: %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void test_tensor_allocation() {
    size_t dims[] = {2, 2};
    Tensor t = tensor_create(2, dims);
    TEST_ASSERT(t.data != NULL);
    TEST_ASSERT(t.dims[0] == 2);
    tensor_free(&t);
}
``` 

## Exercises

1. **Extended Tensor Views**
   Implement `tensor_view()` to create a new tensor that shares memory with an existing tensor but has different dimensions. This is useful for reshaping without copying data.

2. **Dimension Broadcasting**
   Add support for broadcasting in `tensor_get` and `tensor_set` when accessing tensors with compatible but different shapes.

3. **BLAS Performance Comparison**
   Write a benchmark comparing your naive matrix multiplication with BLAS. Use different matrix sizes and plot the performance difference.

## Key Takeaways

- A well-designed tensor structure balances flexibility with performance
- Proper memory alignment is crucial for SIMD operations
- Validation routines catch errors early and provide meaningful feedback
- Integration with BLAS provides production-grade performance
- Testing and debugging tools are essential for robust tensor implementations
- Always pair `malloc`/`calloc` with `free` in ownership-aware structs
- Use `-lopenblas` **after** object files during linking
- Valgrind’s `--leak-check=full` is indispensable for memory debugging
- Compiler flags like `-O3` require rigorous validation via unit tests  
