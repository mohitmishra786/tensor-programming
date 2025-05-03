---
layout: chapter
title: Implementing Core Tensor Operations from Scratch
number: 2
description: Build the fundamental operations that form the backbone of any tensor library. This chapter focuses on creating efficient implementations without external dependencies.
---

## You Will Learn To...

- Implement fundamental tensor operations using raw C loops and pointers
- Build efficient element-wise operations with proper memory access patterns
- Create tensor contraction algorithms including matrix multiplication
- Develop broadcasting mechanics to handle tensors of different shapes
- Test and validate numerical stability in floating-point operations

## 2.1 Element-Wise Operations: Loops, Pointers, and In-Place Modifications

Element-wise operations form the foundation of tensor programming. These operations apply the same function to each corresponding pair of elements from two tensors (or to each element of a single tensor). Let's start by implementing the most common ones.

### Direct Memory Access vs. Accessor Functions

In Chapter 1, we created accessor functions `tensor_get` and `tensor_set` for safety. While these are great for debugging and initial development, they introduce function call overhead. For performance-critical operations, we'll use direct memory access when appropriate, but always with careful bounds checking.

Here's our first element-wise operation, addition:

```c
/* tensor_ops.h */
#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.h"

/* Element-wise operations */
int tensor_add(tensor_t *result, const tensor_t *a, const tensor_t *b);
int tensor_subtract(tensor_t *result, const tensor_t *a, const tensor_t *b);
int tensor_multiply(tensor_t *result, const tensor_t *a, const tensor_t *b);
int tensor_divide(tensor_t *result, const tensor_t *a, const tensor_t *b);

/* Scalar operations */
int tensor_scale(tensor_t *result, const tensor_t *a, tensor_elem_t scalar);
int tensor_add_scalar(tensor_t *result, const tensor_t *a, tensor_elem_t scalar);

/* In-place variants */
int tensor_add_inplace(tensor_t *a, const tensor_t *b);
int tensor_scale_inplace(tensor_t *a, tensor_elem_t scalar);

#endif /* TENSOR_OPS_H */
```

Now let's implement these operations:

```c
/* tensor_ops.c */
#include "tensor_ops.h"
#include <string.h> /* for memcpy */

/* Helper function to check if two tensors have the same shape */
static int tensor_shapes_match(const tensor_t *a, const tensor_t *b) {
    return (a->rows == b->rows && a->cols == b->cols);
}

/* Helper function to create a result tensor with the same shape as inputs */
static int tensor_create_result(tensor_t *result, const tensor_t *a) {
    /* If result already exists and has the right shape, reuse it */
    if (result->data != NULL && result->rows == a->rows && result->cols == a->cols) {
        return TENSOR_SUCCESS;
    }
    
    /* If result exists but has wrong shape, free it first */
    if (result->data != NULL && result->owner) {
        tensor_free(result);
    }
    
    /* Create new result tensor */
    return tensor_create(result, a->rows, a->cols);
}

/* Element-wise addition */
int tensor_add(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if shapes match */
    if (!tensor_shapes_match(a, b)) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_result(result, a);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Perform element-wise addition */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            size_t idx = i * a->stride + j;
            result->data[idx] = a->data[idx] + b->data[i * b->stride + j];
        }
    }
    
    return TENSOR_SUCCESS;
}

/* Element-wise subtraction */
int tensor_subtract(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if shapes match */
    if (!tensor_shapes_match(a, b)) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_result(result, a);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Perform element-wise subtraction */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            size_t idx = i * a->stride + j;
            result->data[idx] = a->data[idx] - b->data[i * b->stride + j];
        }
    }
    
    return TENSOR_SUCCESS;
}

/* Element-wise multiplication */
int tensor_multiply(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if shapes match */
    if (!tensor_shapes_match(a, b)) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_result(result, a);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Perform element-wise multiplication */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            size_t idx = i * a->stride + j;
            result->data[idx] = a->data[idx] * b->data[i * b->stride + j];
        }
    }
    
    return TENSOR_SUCCESS;
}

/* Element-wise division */
int tensor_divide(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if shapes match */
    if (!tensor_shapes_match(a, b)) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_result(result, a);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Perform element-wise division with check for division by zero */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            size_t idx_a = i * a->stride + j;
            size_t idx_b = i * b->stride + j;
            
            /* Check for division by zero */
            if (b->data[idx_b] == 0.0) {
                /* Set to infinity or handle as error */
                result->data[idx_a] = INFINITY;
            } else {
                result->data[idx_a] = a->data[idx_a] / b->data[idx_b];
            }
        }
    }
    
    return TENSOR_SUCCESS;
}

/* Scalar multiplication */
int tensor_scale(tensor_t *result, const tensor_t *a, tensor_elem_t scalar) {
    /* Create or resize result tensor */
    int status = tensor_create_result(result, a);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Perform scalar multiplication */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            size_t idx = i * a->stride + j;
            result->data[idx] = a->data[idx] * scalar;
        }
    }
    
    return TENSOR_SUCCESS;
}

/* Add scalar to all elements */
int tensor_add_scalar(tensor_t *result, const tensor_t *a, tensor_elem_t scalar) {
    /* Create or resize result tensor */
    int status = tensor_create_result(result, a);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Perform scalar addition */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            size_t idx = i * a->stride + j;
            result->data[idx] = a->data[idx] + scalar;
        }
    }
    
    return TENSOR_SUCCESS;
}

/* In-place addition */
int tensor_add_inplace(tensor_t *a, const tensor_t *b) {
    /* Check if shapes match */
    if (!tensor_shapes_match(a, b)) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Perform in-place addition */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            size_t idx_a = i * a->stride + j;
            size_t idx_b = i * b->stride + j;
            a->data[idx_a] += b->data[idx_b];
        }
    }
    
    return TENSOR_SUCCESS;
}

/* In-place scalar multiplication */
int tensor_scale_inplace(tensor_t *a, tensor_elem_t scalar) {
    /* Perform in-place scalar multiplication */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            size_t idx = i * a->stride + j;
            a->data[idx] *= scalar;
        }
    }
    
    return TENSOR_SUCCESS;
}
```

### Optimizing Memory Access Patterns

The nested loops in our implementations follow row-major order, which matches our tensor's memory layout. This is crucial for performance, as it maximizes cache utilization. Let's see how we can further optimize our code by using pointer arithmetic directly:

```c
/* Optimized element-wise addition using pointer arithmetic */
int tensor_add_optimized(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if shapes match */
    if (!tensor_shapes_match(a, b)) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_result(result, a);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Special case: if all tensors have contiguous memory (stride == cols),
       we can process the entire data as a single block */
    if (a->stride == a->cols && b->stride == b->cols && result->stride == result->cols) {
        size_t total_elements = a->rows * a->cols;
        tensor_elem_t *result_ptr = result->data;
        const tensor_elem_t *a_ptr = a->data;
        const tensor_elem_t *b_ptr = b->data;
        
        /* Process elements in a single loop */
        for (size_t i = 0; i < total_elements; i++) {
            *result_ptr++ = *a_ptr++ + *b_ptr++;
        }
    } else {
        /* General case: handle each row separately due to strides */
        for (size_t i = 0; i < a->rows; i++) {
            tensor_elem_t *result_row = result->data + i * result->stride;
            const tensor_elem_t *a_row = a->data + i * a->stride;
            const tensor_elem_t *b_row = b->data + i * b->stride;
            
            for (size_t j = 0; j < a->cols; j++) {
                result_row[j] = a_row[j] + b_row[j];
            }
        }
    }
    
    return TENSOR_SUCCESS;
}
```

This optimized version checks if the tensors have contiguous memory (stride equals columns), which allows us to process the entire data as a single block. This reduces loop overhead and may enable better compiler optimizations.

### Testing Element-Wise Operations

Let's write tests for our element-wise operations:

```c
/* tensor_ops_tests.c */
#include "tensor.h"
#include "tensor_ops.h"
#include "tensor_test_framework.h"
#include <math.h>

/* Test element-wise addition */
int test_tensor_add() {
    tensor_t a, b, result;
    
    /* Create test tensors */
    tensor_create(&a, 2, 3);
    tensor_create(&b, 2, 3);
    
    /* Initialize with test values */
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            tensor_set(&a, i, j, (tensor_elem_t)(i + j));
            tensor_set(&b, i, j, (tensor_elem_t)(i * j + 1));
        }
    }
    
    /* Perform addition */
    int status = tensor_add(&result, &a, &b);
    
    /* Check status */
    TEST_ASSERT(status == TENSOR_SUCCESS, "tensor_add should succeed");
    
    /* Check result values */
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            tensor_elem_t expected = (i + j) + (i * j + 1);
            tensor_elem_t actual;
            tensor_get(&result, i, j, &actual);
            TEST_ASSERT(fabs(actual - expected) < 1e-6, "tensor_add should compute correct values");
        }
    }
    
    /* Test dimension mismatch */
    tensor_t c;
    tensor_create(&c, 3, 2);  /* Different dimensions from a and b */
    status = tensor_add(&result, &a, &c);
    TEST_ASSERT(status == TENSOR_DIMENSION_MISMATCH, "tensor_add should detect dimension mismatch");
    
    /* Clean up */
    tensor_free(&a);
    tensor_free(&b);
    tensor_free(&c);
    tensor_free(&result);
    
    return 1;
}

/* Test in-place addition */
int test_tensor_add_inplace() {
    tensor_t a, b;
    
    /* Create test tensors */
    tensor_create(&a, 2, 2);
    tensor_create(&b, 2, 2);
    
    /* Initialize with test values */
    tensor_set(&a, 0, 0, 1.0);
    tensor_set(&a, 0, 1, 2.0);
    tensor_set(&a, 1, 0, 3.0);
    tensor_set(&a, 1, 1, 4.0);
    
    tensor_set(&b, 0, 0, 5.0);
    tensor_set(&b, 0, 1, 6.0);
    tensor_set(&b, 1, 0, 7.0);
    tensor_set(&b, 1, 1, 8.0);
    
    /* Save original values for comparison */
    tensor_elem_t a_orig[4];
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            tensor_get(&a, i, j, &a_orig[i*2+j]);
        }
    }
    
    /* Perform in-place addition */
    int status = tensor_add_inplace(&a, &b);
    
    /* Check status */
    TEST_ASSERT(status == TENSOR_SUCCESS, "tensor_add_inplace should succeed");
    
    /* Check result values */
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            tensor_elem_t expected = a_orig[i*2+j] + b.data[i*b.stride+j];
            tensor_elem_t actual;
            tensor_get(&a, i, j, &actual);
            TEST_ASSERT(fabs(actual - expected) < 1e-6, "tensor_add_inplace should compute correct values");
        }
    }
    
    /* Clean up */
    tensor_free(&a);
    tensor_free(&b);
    
    return 1;
}

/* Test scalar multiplication */
int test_tensor_scale() {
    tensor_t a, result;
    tensor_elem_t scalar = 2.5;
    
    /* Create test tensor */
    tensor_create(&a, 2, 3);
    
    /* Initialize with test values */
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            tensor_set(&a, i, j, (tensor_elem_t)(i + j));
        }
    }
    
    /* Perform scalar multiplication */
    int status = tensor_scale(&result, &a, scalar);
    
    /* Check status */
    TEST_ASSERT(status == TENSOR_SUCCESS, "tensor_scale should succeed");
    
    /* Check result values */
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            tensor_elem_t expected = (i + j) * scalar;
            tensor_elem_t actual;
            tensor_get(&result, i, j, &actual);
            TEST_ASSERT(fabs(actual - expected) < 1e-6, "tensor_scale should compute correct values");
        }
    }
    
    /* Clean up */
    tensor_free(&a);
    tensor_free(&result);
    
    return 1;
}

int main() {
    int test_count = 0;
    int test_failures = 0;
    
    RUN_TEST(test_tensor_add);
    RUN_TEST(test_tensor_add_inplace);
    RUN_TEST(test_tensor_scale);
    
    printf("\n%d tests run, %d passed, %d failed\n", 
           test_count, test_count - test_failures, test_failures);
    
    return test_failures > 0 ? 1 : 0;
}
```

## 2.2 Tensor Contraction: Nested Loops for Matrix Multiplication

Tensor contraction is a generalization of matrix multiplication to higher-dimensional tensors. For now, we'll focus on the most common case: matrix multiplication, which is a contraction of two 2D tensors.

### Basic Matrix Multiplication

Let's implement matrix multiplication using nested loops:

```c
/* Add to tensor_ops.h */
int tensor_matmul(tensor_t *result, const tensor_t *a, const tensor_t *b);

/* Implementation in tensor_ops.c */
int tensor_matmul(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if dimensions are compatible for matrix multiplication */
    if (a->cols != b->rows) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status;
    if (result->data != NULL && result->owner) {
        if (result->rows != a->rows || result->cols != b->cols) {
            tensor_free(result);
            status = tensor_create(result, a->rows, b->cols);
            if (status != TENSOR_SUCCESS) {
                return status;
            }
        }
    } else {
        status = tensor_create(result, a->rows, b->cols);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* Perform matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j] */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            tensor_elem_t sum = 0.0;
            for (size_t k = 0; k < a->cols; k++) {
                sum += a->data[i * a->stride + k] * b->data[k * b->stride + j];
            }
            result->data[i * result->stride + j] = sum;
        }
    }
    
    return TENSOR_SUCCESS;
}
```

This implementation follows the standard algorithm for matrix multiplication, with three nested loops. However, it's not very efficient due to poor cache utilization. Let's improve it.

### Cache-Friendly Matrix Multiplication

The key to efficient matrix multiplication is maximizing cache utilization. We can achieve this by using a technique called "blocking" or "tiling":

```c
/* Cache-friendly matrix multiplication */
int tensor_matmul_blocked(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if dimensions are compatible for matrix multiplication */
    if (a->cols != b->rows) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status;
    if (result->data != NULL && result->owner) {
        if (result->rows != a->rows || result->cols != b->cols) {
            tensor_free(result);
            status = tensor_create(result, a->rows, b->cols);
            if (status != TENSOR_SUCCESS) {
                return status;
            }
        } else {
            /* Zero out the result tensor */
            memset(result->data, 0, result->rows * result->cols * sizeof(tensor_elem_t));
        }
    } else {
        status = tensor_create(result, a->rows, b->cols);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* Block size for cache-friendly multiplication */
    const size_t block_size = 32;  /* Adjust based on your CPU's cache size */
    
    /* Initialize result to zero */
    for (size_t i = 0; i < result->rows * result->cols; i++) {
        result->data[i] = 0.0;
    }
    
    /* Blocked matrix multiplication */
    for (size_t i0 = 0; i0 < a->rows; i0 += block_size) {
        size_t i_end = (i0 + block_size < a->rows) ? i0 + block_size : a->rows;
        
        for (size_t j0 = 0; j0 < b->cols; j0 += block_size) {
            size_t j_end = (j0 + block_size < b->cols) ? j0 + block_size : b->cols;
            
            for (size_t k0 = 0; k0 < a->cols; k0 += block_size) {
                size_t k_end = (k0 + block_size < a->cols) ? k0 + block_size : a->cols;
                
                /* Process current block */
                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        tensor_elem_t sum = result->data[i * result->stride + j];
                        
                        for (size_t k = k0; k < k_end; k++) {
                            sum += a->data[i * a->stride + k] * b->data[k * b->stride + j];
                        }
                        
                        result->data[i * result->stride + j] = sum;
                    }
                }
            }
        }
    }
    
    return TENSOR_SUCCESS;
}
```

This blocked implementation divides the matrices into smaller blocks that fit in the CPU cache, which significantly improves performance for large matrices. The block size should be tuned based on your CPU's cache size.

### Testing Matrix Multiplication

Let's write a test for our matrix multiplication function:

```c
/* Add to tensor_ops_tests.c */
int test_tensor_matmul() {
    tensor_t a, b, result;
    
    /* Create test matrices */
    tensor_create(&a, 2, 3);  /* 2x3 matrix */
    tensor_create(&b, 3, 2);  /* 3x2 matrix */
    
    /* Initialize matrices */
    /* Matrix A:
     * 1 2 3
     * 4 5 6
     */
    tensor_set(&a, 0, 0, 1.0);
    tensor_set(&a, 0, 1, 2.0);
    tensor_set(&a, 0, 2, 3.0);
    tensor_set(&a, 1, 0, 4.0);
    tensor_set(&a, 1, 1, 5.0);
    tensor_set(&a, 1, 2, 6.0);
    
    /* Matrix B:
     * 7  8
     * 9  10
     * 11 12
     */
    tensor_set(&b, 0, 0, 7.0);
    tensor_set(&b, 0, 1, 8.0);
    tensor_set(&b, 1, 0, 9.0);
    tensor_set(&b, 1, 1, 10.0);
    tensor_set(&b, 2, 0, 11.0);
    tensor_set(&b, 2, 1, 12.0);
    
    /* Perform matrix multiplication */
    int status = tensor_matmul(&result, &a, &b);
    
    /* Check status */
    TEST_ASSERT(status == TENSOR_SUCCESS, "tensor_matmul should succeed");
    
    /* Check dimensions of result */
    TEST_ASSERT(result.rows == 2 && result.cols == 2, "Result should be a 2x2 matrix");
    
    /* Expected result:
     * 1*7 + 2*9 + 3*11 = 58    1*8 + 2*10 + 3*12 = 64
     * 4*7 + 5*9 + 6*11 = 139   4*8 + 5*10 + 6*12 = 154
     */
    tensor_elem_t expected[4] = {58.0, 64.0, 139.0, 154.0};
    
    /* Check result values */
    for (size_t i = 0; i < result.rows; i++) {
        for (size_t j = 0; j < result.cols; j++) {
            tensor_elem_t actual;
            tensor_get(&result, i, j, &actual);
            TEST_ASSERT(fabs(actual - expected[i*2+j]) < 1e-6, 
                      "tensor_matmul should compute correct values");
        }
    }
    
    /* Test dimension mismatch */
    tensor_t c;
    tensor_create(&c, 2, 2);  /* Incompatible dimensions for multiplication with a */
    status = tensor_matmul(&result, &a, &c);
    TEST_ASSERT(status == TENSOR_DIMENSION_MISMATCH, "tensor_matmul should detect dimension mismatch");
    
    /* Clean up */
    tensor_free(&a);
    tensor_free(&b);
    tensor_free(&c);
    tensor_free(&result);
    
    return 1;
}
```

## 2.3 Broadcasting Mechanics: Handling Mismatched Dimensions

Broadcasting allows operations between tensors of different shapes. The basic idea is to "stretch" the smaller tensor to match the shape of the larger one, without actually copying data. This is particularly useful for operations like adding a vector to each row of a matrix.

### Implementing Broadcasting for Addition

Let's implement a function that adds a row vector to each row of a matrix:

```c
/* Add to tensor_ops.h */
int tensor_add_row_vector(tensor_t *result, const tensor_t *matrix, const tensor_t *vector);

/* Implementation in tensor_ops.c */
int tensor_add_row_vector(tensor_t *result, const tensor_t *matrix, const tensor_t *vector) {
    /* Check if dimensions are compatible for broadcasting */
    if (matrix->cols != vector->cols || vector->rows != 1) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_result(result, matrix);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Perform broadcasting addition */
    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->cols; j++) {
            size_t idx_matrix = i * matrix->stride + j;
            size_t idx_vector = j;  /* Vector is a row vector */
            result->data[idx_matrix] = matrix->data[idx_matrix] + vector->data[idx_vector];
        }
    }
    
    return TENSOR_SUCCESS;
}
```

Similarly, we can implement a function to add a column vector to each column of a matrix:

```c
/* Add to tensor_ops.h */
int tensor_add_column_vector(tensor_t *result, const tensor_t *matrix, const tensor_t *vector);

/* Implementation in tensor_ops.c */
int tensor_add_column_vector(tensor_t *result, const tensor_t *matrix, const tensor_t *vector) {
    /* Check if dimensions are compatible for broadcasting */
    if (matrix->rows != vector->rows || vector->cols != 1) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_result(result, matrix);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Perform broadcasting addition */
    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->cols; j++) {
            size_t idx_matrix = i * matrix->stride + j;
            size_t idx_vector = i;  /* Vector is a column vector */
            result->data[idx_matrix] = matrix->data[idx_matrix] + vector->data[idx_vector];
        }
    }
    
    return TENSOR_SUCCESS;
}
```

### General Broadcasting Rules

For more general broadcasting, we need to define rules for how dimensions are aligned. In NumPy and other tensor libraries, broadcasting follows these rules:

1. If the tensors have different ranks, prepend 1s to the shape of the lower-rank tensor.
2. Two dimensions are compatible if they are equal or one of them is 1.
3. In the output, each dimension is the maximum of the corresponding dimensions in the inputs.

Implementing general broadcasting is complex and beyond the scope of this chapter, but the row and column vector broadcasting we've implemented covers many common use cases.

### Testing Broadcasting Operations

Let's write tests for our broadcasting functions:

```c
/* Add to tensor_ops_tests.c */
int test_tensor_add_row_vector() {
    tensor_t matrix, vector, result;
    
    /* Create test tensors */
    tensor_create(&matrix, 3, 2);  /* 3x2 matrix */
    tensor_create(&vector, 1, 2);  /* 1x2 row vector */
    
    /* Initialize matrix:
     * 1 2
     * 3 4
     * 5 6
     */
    tensor_set(&matrix, 0, 0, 1.0);
    tensor_set(&matrix, 0, 1, 2.0);
    tensor_set(&matrix, 1, 0, 3.0);
    tensor_set(&matrix, 1, 1, 4.0);
    tensor_set(&matrix, 2, 0, 5.0);
    tensor_set(&matrix, 2, 1, 6.0);
    
    /* Initialize vector: [10, 20] */
    tensor_set(&vector, 0, 0, 10.0);
    tensor_set(&vector, 0, 1, 20.0);
    
    /* Perform broadcasting addition */
    int status = tensor_add_row_vector(&result, &matrix, &vector);
    
    /* Check status */
    TEST_ASSERT(status == TENSOR_SUCCESS, "tensor_add_row_vector should succeed");
    
    /* Expected result:
     * 1+10 2+20
     * 3+10 4+20
     * 5+10 6+20
     */
    tensor_elem_t expected[6] = {11.0, 22.0, 13.0, 24.0, 15.0, 26.0};
    
    /* Check result values */
    for (size_t i = 0; i < result.rows; i++) {
        for (size_t j = 0; j < result.cols; j++) {
            tensor_elem_t actual;
            tensor_get(&result, i, j, &actual);
            TEST_ASSERT(fabs(actual - expected[i*2+j]) < 1e-6, 
                      "tensor_add_row_vector should compute correct values");
        }
    }
    
    /* Clean up */
    tensor_free(&matrix);
    tensor_free(&vector);
    tensor_free(&result);
    
    return 1;
}
```

## 2.4 Testing Numerical Stability: Floating-Point Error Analysis

Floating-point arithmetic is inherently imprecise due to the finite representation of real numbers. When implementing tensor operations, we need to be aware of these limitations and test for numerical stability.

### Common Numerical Issues

1. **Catastrophic Cancellation**: Subtraction of nearly equal numbers can lead to significant loss of precision.
2. **Accumulation Errors**: Small errors can accumulate in long sequences of operations.
3. **Overflow and Underflow**: Values can become too large or too small to represent.

Let's implement a function to test the numerical stability of our matrix multiplication:

```c
/* Add to tensor_ops_tests.c */
int test_numerical_stability() {
    tensor_t a, b, c, temp;
    
    /* Create large matrices to amplify numerical errors */
    size_t n = 100;
    tensor_create(&a, n, n);
    tensor_create(&b, n, n);
    tensor_create(&c, n, n);
    
    /* Initialize matrices with values that can cause numerical issues */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            /* A is the identity matrix */
            tensor_set(&a, i, j, (i == j) ? 1.0 : 0.0);
            
            /* B has values with large differences in magnitude */
            tensor_set(&b, i, j, (i == j) ? 1e10 : 1e-10);
            
            /* C is also the identity matrix */
            tensor_set(&c, i, j, (i == j) ? 1.0 : 0.0);
        }
    }
    
    /* Compute A * B */
    tensor_matmul(&temp, &a, &b);
    
    /* Compute (A * B) * C, which should equal A * B */
    tensor_t result;
    tensor_matmul(&result, &temp, &c);
    
    /* Check if result approximately equals B */
    double max_error = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            tensor_elem_t expected, actual;
            tensor_get(&b, i, j, &expected);
            tensor_get(&result, i, j, &actual);
            
            double rel_error = fabs(actual - expected) / (fabs(expected) + 1e-10);
            if (rel_error > max_error) {
                max_error = rel_error;
            }
        }
    }
    
    printf("Maximum relative error: %e\n", max_error);
    TEST_ASSERT(max_error < 1e-6, "Matrix multiplication should be numerically stable");
    
    /* Clean up */
    tensor_free(&a);
    tensor_free(&b);
    tensor_free(&c);
    tensor_free(&temp);
    tensor_free(&result);
    
    return 1;
}
```

### Improving Numerical Stability

There are several techniques to improve numerical stability:

1. **Use Kahan Summation**: This algorithm reduces accumulation errors in sums.
2. **Avoid Catastrophic Cancellation**: Rearrange calculations to avoid subtracting nearly equal numbers.
3. **Use Higher Precision**: For critical calculations, consider using double-double arithmetic or arbitrary precision libraries.

Let's implement a more numerically stable version of matrix multiplication using Kahan summation:

```c
/* Numerically stable matrix multiplication using Kahan summation */
int tensor_matmul_stable(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if dimensions are compatible for matrix multiplication */
    if (a->cols != b->rows) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status;
    if (result->data != NULL && result->owner) {
        if (result->rows != a->rows || result->cols != b->cols) {
            tensor_free(result);
            status = tensor_create(result, a->rows, b->cols);
            if (status != TENSOR_SUCCESS) {
                return status;
            }
        }
    } else {
        status = tensor_create(result, a->rows, b->cols);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* Perform matrix multiplication with Kahan summation */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            tensor_elem_t sum = 0.0;
            tensor_elem_t c = 0.0;  /* Compensation term for Kahan summation */
            
            for (size_t k = 0; k < a->cols; k++) {
                tensor_elem_t product = a->data[i * a->stride + k] * b->data[k * b->stride + j];
                tensor_elem_t y = product - c;
                tensor_elem_t t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            
            result->data[i * result->stride + j] = sum;
        }
    }
    
    return TENSOR_SUCCESS;
}
```

Kahan summation compensates for the loss of precision that occurs when adding numbers of different magnitudes. It keeps track of the rounding error and adds it back in the next iteration.

## Visualizing Tensor Operations

Let's visualize some of the operations we've implemented to better understand how they work.

### Element-Wise Addition

```
Tensor A:       Tensor B:       Result (A + B):
+---+---+       +---+---+       +---+---+
| 1 | 2 |       | 5 | 6 |       | 6 | 8 |
+---+---+       +---+---+       +---+---+
| 3 | 4 |       | 7 | 8 |       | 10| 12|
+---+---+       +---+---+       +---+---+
```

### Matrix Multiplication

```
Tensor A:       Tensor B:       Result (A * B):
+---+---+---+   +---+---+       +---+---+
| 1 | 2 | 3 |   | 7 | 8 |       | 58| 64|
+---+---+---+   +---+---+       +---+---+
| 4 | 5 | 6 |   | 9 | 10|       |139|154|
+---+---+---+   +---+---+       +---+---+
                | 11| 12|
                +---+---+
```

### Broadcasting (Row Vector Addition)

```
Matrix:         Row Vector:     Result:
+---+---+       +---+---+       +---+---+
| 1 | 2 |       | 10| 20|       | 11| 22|
+---+---+       +---+---+       +---+---+
| 3 | 4 |                       | 13| 24|
+---+---+                       +---+---+
| 5 | 6 |                       | 15| 26|
+---+---+                       +---+---+
```

## Common Pitfalls and Debugging

Let's discuss some common issues you might encounter when implementing tensor operations.

### Memory Management Issues

One of the most common issues is improper memory management, especially when creating result tensors. Always ensure that:

1. You free existing result tensors before creating new ones.
2. You handle the case where the result tensor is the same as one of the input tensors (in-place operations).
3. You check for allocation failures and return appropriate error codes.

Here's an example of a problematic function and its fix:

```c
/* Problematic function with memory leak */
int tensor_add_problematic(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Create a new result tensor without checking if one already exists */
    tensor_create(result, a->rows, a->cols);  /* Memory leak if result already has data */
    
    /* ... rest of the function ... */
}

/* Fixed function */
int tensor_add_fixed(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if result already exists and has the right shape */
    if (result->data != NULL) {
        if (result->rows != a->rows || result->cols != a->cols) {
            /* Free existing data before allocating new */
            if (result->owner) {
                tensor_free(result);
            }
            tensor_create(result, a->rows, a->cols);
        }
        /* else reuse existing tensor */
    } else {
        /* Create new tensor */
        tensor_create(result, a->rows, a->cols);
    }
    
    /* ... rest of the function ... */
}
```

### Dimension Mismatch Errors

Another common issue is failing to check for dimension compatibility before performing operations. Always validate dimensions at the beginning of your functions.

### Numerical Precision Issues

When comparing floating-point results, never use exact equality. Instead, check if the difference is within a small epsilon:

```c
/* Bad: exact equality check */
if (result == expected) {  /* This might fail due to floating-point precision */
    /* ... */
}

/* Good: approximate equality check */
if (fabs(result - expected) < 1e-6) {  /* Allow for small differences */
    /* ... */
}
```

## Exercises

### Exercise 1: Implement Tensor Transposition

Implement a function `tensor_transpose` that transposes a 2D tensor (swaps rows and columns).

Hint: You'll need to handle the case where the input and output tensors are the same (in-place transposition), which is tricky because you can't simply swap elements.

Partial solution:

```c
int tensor_transpose(tensor_t *result, const tensor_t *a) {
    /* Check if we can do in-place transposition */
    if (result == a) {
        /* In-place transposition only works for square matrices */
        if (a->rows != a->cols) {
            return TENSOR_DIMENSION_MISMATCH;
        }
        
        /* Swap elements across the diagonal */
        for (size_t i = 0; i < a->rows; i++) {
            for (size_t j = i + 1; j < a->cols; j++) {
                tensor_elem_t temp = a->data[i * a->stride + j];
                a->data[i * a->stride + j] = a->data[j * a->stride + i];
                a->data[j * a->stride + i] = temp;
            }
        }
    } else {
        /* Create result tensor with swapped dimensions */
        int status = tensor_create(result, a->cols, a->rows);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
        
        /* Copy elements with transposed indices */
        for (size_t i = 0; i < a->rows; i++) {
            for (size_t j = 0; j < a->cols; j++) {
                result->data[j * result->stride + i] = a->data[i * a->stride + j];
            }
        }
    }
    
    return TENSOR_SUCCESS;
}
```

### Exercise 2: Implement Tensor Reduction Operations

Implement functions to compute the sum, mean, maximum, and minimum values along a specified axis (rows or columns) of a tensor.

Hint: You'll need to create a result tensor with the appropriate shape (e.g., a row vector for column-wise reduction).

### Exercise 3: Implement a Function for Element-Wise Application of a Custom Function

Implement a function `tensor_apply` that applies a user-provided function to each element of a tensor.

Partial solution:

```c
/* Function pointer type for element-wise operations */
typedef tensor_elem_t (*tensor_elem_func_t)(tensor_elem_t);

/* Apply a function to each element */
int tensor_apply(tensor_t *result, const tensor_t *a, tensor_elem_func_t func) {
    /* Create or resize result tensor */
    int status = tensor_create_result(result, a);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Apply function to each element */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            size_t idx = i * a->stride + j;
            result->data[idx] = func(a->data[idx]);
        }
    }
    
    return TENSOR_SUCCESS;
}

/* Example usage: */
tensor_elem_t square(tensor_elem_t x) {
    return x * x;
}

/* In main function: */
tensor_apply(&result, &a, square);
```

## Summary and Key Takeaways

In this chapter, we've implemented core tensor operations from scratch in C:

- We built element-wise operations like addition, subtraction, multiplication, and division.
- We implemented matrix multiplication with both basic and cache-friendly algorithms.
- We created broadcasting functions to handle operations between tensors of different shapes.
- We explored numerical stability issues and techniques to mitigate them.

Key takeaways:

1. **Memory Management**: Proper handling of tensor memory is crucial for preventing leaks and crashes.
2. **Performance Optimization**: Cache-friendly algorithms can significantly improve performance for large tensors.
3. **Numerical Stability**: Floating-point arithmetic requires careful handling to maintain precision.
4. **Error Handling**: Robust error checking prevents subtle bugs and makes debugging easier.

In the next chapter, we'll explore memory layouts in more depth and learn how to optimize tensor operations for different hardware architectures.

## Further Reading

1. "Numerical Recipes in C: The Art of Scientific Computing" by Press, Teukolsky, Vetterling, and Flannery - A comprehensive reference for numerical algorithms in C.

2. "What Every Computer Scientist Should Know About Floating-Point Arithmetic" by David Goldberg - Essential reading for understanding floating-point precision issues.

3. "Optimizing Matrix Multiplication" by Kazushige Goto and Robert A. van de Geijn - Insights into high-performance matrix multiplication algorithms.