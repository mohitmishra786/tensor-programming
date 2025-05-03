---
layout: chapter
title: Mastering Memory Layouts for Speed
number: 3
description: Dive into the critical relationship between memory layout and performance. Learn how proper data organization can dramatically speed up tensor operations.
---

## You Will Learn To...

- Understand how memory layout affects performance in tensor operations
- Implement and optimize row-major and column-major storage formats
- Design strided tensor access for efficient non-contiguous operations
- Create transposition strategies that avoid unnecessary data copying
- Implement cache-aware blocking techniques for large-scale tensor operations

## 3.1 Row-Major vs. Column-Major: Impact on Iteration Patterns

The way we organize tensor data in memory has a profound impact on performance. The two most common memory layouts are row-major (used in C/C++) and column-major (used in Fortran and some libraries like MATLAB).

### Understanding Memory Layout Fundamentals

In a 2D tensor with dimensions (rows × cols), we need to map a 2D coordinate (i, j) to a 1D memory offset. The mapping formulas are:

- **Row-major**: offset = i × cols + j
- **Column-major**: offset = i + j × rows

Let's visualize these layouts for a 3×4 matrix:

```
Logical Matrix:
+----+----+----+----+
| 0,0| 0,1| 0,2| 0,3|
+----+----+----+----+
| 1,0| 1,1| 1,2| 1,3|
+----+----+----+----+
| 2,0| 2,1| 2,2| 2,3|
+----+----+----+----+

Row-Major Layout in Memory:
+----+----+----+----+----+----+----+----+----+----+----+----+
| 0,0| 0,1| 0,2| 0,3| 1,0| 1,1| 1,2| 1,3| 2,0| 2,1| 2,2| 2,3|
+----+----+----+----+----+----+----+----+----+----+----+----+

Column-Major Layout in Memory:
+----+----+----+----+----+----+----+----+----+----+----+----+
| 0,0| 1,0| 2,0| 0,1| 1,1| 2,1| 0,2| 1,2| 2,2| 0,3| 1,3| 2,3|
+----+----+----+----+----+----+----+----+----+----+----+----+
```

In our tensor implementation from previous chapters, we've been using row-major layout. Let's extend our tensor struct to support both layouts:

```c
/* Add to tensor.h */
/* Memory layout types */
typedef enum {
    TENSOR_ROW_MAJOR = 0,
    TENSOR_COL_MAJOR = 1
} tensor_layout_t;

/* Extended tensor structure */
typedef struct {
    size_t rows;        /* Number of rows */
    size_t cols;        /* Number of columns */
    size_t row_stride;  /* Row stride (for row-major: cols, for col-major: 1) */
    size_t col_stride;  /* Column stride (for row-major: 1, for col-major: rows) */
    tensor_elem_t *data; /* Pointer to the actual data */
    int owner;          /* Whether this tensor owns its data */
    tensor_layout_t layout; /* Memory layout */
} tensor_t;

/* Function to create a tensor with specified layout */
int tensor_create_with_layout(tensor_t *t, size_t rows, size_t cols, tensor_layout_t layout);
```

Now let's implement the creation function and update our accessor functions:

```c
/* Add to tensor.c */
int tensor_create_with_layout(tensor_t *t, size_t rows, size_t cols, tensor_layout_t layout) {
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
    t->layout = layout;
    
    /* Set strides based on layout */
    if (layout == TENSOR_ROW_MAJOR) {
        t->row_stride = cols;
        t->col_stride = 1;
    } else { /* TENSOR_COL_MAJOR */
        t->row_stride = 1;
        t->col_stride = rows;
    }
    
    t->owner = 1;  /* This tensor owns its data */
    
    /* Initialize data to zero */
    memset(t->data, 0, rows * cols * sizeof(tensor_elem_t));
    
    return TENSOR_SUCCESS;
}

/* Update tensor_create to use row-major by default */
int tensor_create(tensor_t *t, size_t rows, size_t cols) {
    return tensor_create_with_layout(t, rows, cols, TENSOR_ROW_MAJOR);
}

/* Update tensor_set to use strides */
int tensor_set(tensor_t *t, size_t i, size_t j, tensor_elem_t value) {
    if (i >= t->rows || j >= t->cols) {
        return TENSOR_INDEX_OUT_OF_BOUNDS;
    }
    
    t->data[i * t->row_stride + j * t->col_stride] = value;
    return TENSOR_SUCCESS;
}

/* Update tensor_get to use strides */
int tensor_get(const tensor_t *t, size_t i, size_t j, tensor_elem_t *value) {
    if (i >= t->rows || j >= t->cols) {
        return TENSOR_INDEX_OUT_OF_BOUNDS;
    }
    
    *value = t->data[i * t->row_stride + j * t->col_stride];
    return TENSOR_SUCCESS;
}
```

### Performance Implications of Memory Layout

The choice of memory layout significantly affects performance due to how modern CPUs access memory. CPUs fetch data in cache lines (typically 64 bytes), so accessing contiguous memory is much faster than random access.

Let's benchmark row-major vs. column-major access patterns:

```c
/* benchmark_layouts.c */
#include "tensor.h"
#include <stdio.h>
#include <time.h>

/* Benchmark function for row-wise iteration */
double benchmark_row_iteration(tensor_t *t) {
    clock_t start = clock();
    tensor_elem_t sum = 0.0;
    
    for (size_t i = 0; i < t->rows; i++) {
        for (size_t j = 0; j < t->cols; j++) {
            tensor_elem_t val;
            tensor_get(t, i, j, &val);
            sum += val;
        }
    }
    
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

/* Benchmark function for column-wise iteration */
double benchmark_col_iteration(tensor_t *t) {
    clock_t start = clock();
    tensor_elem_t sum = 0.0;
    
    for (size_t j = 0; j < t->cols; j++) {
        for (size_t i = 0; i < t->rows; i++) {
            tensor_elem_t val;
            tensor_get(t, i, j, &val);
            sum += val;
        }
    }
    
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

int main() {
    tensor_t row_major, col_major;
    size_t size = 2000;  /* Large enough to see the difference */
    
    /* Create tensors with different layouts */
    tensor_create_with_layout(&row_major, size, size, TENSOR_ROW_MAJOR);
    tensor_create_with_layout(&col_major, size, size, TENSOR_COL_MAJOR);
    
    /* Initialize with some values */
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            tensor_elem_t val = (tensor_elem_t)(i * size + j);
            tensor_set(&row_major, i, j, val);
            tensor_set(&col_major, i, j, val);
        }
    }
    
    /* Benchmark row-wise iteration */
    printf("Row-wise iteration on row-major: %.6f seconds\n", 
           benchmark_row_iteration(&row_major));
    printf("Row-wise iteration on column-major: %.6f seconds\n", 
           benchmark_row_iteration(&col_major));
    
    /* Benchmark column-wise iteration */
    printf("Column-wise iteration on row-major: %.6f seconds\n", 
           benchmark_col_iteration(&row_major));
    printf("Column-wise iteration on column-major: %.6f seconds\n", 
           benchmark_col_iteration(&col_major));
    
    /* Clean up */
    tensor_free(&row_major);
    tensor_free(&col_major);
    
    return 0;
}
```

When you run this benchmark, you'll likely see that:

1. Row-wise iteration is faster on row-major tensors
2. Column-wise iteration is faster on column-major tensors

This is because each access pattern follows the memory layout, maximizing cache utilization. The performance difference can be substantial, often 2-10x depending on the tensor size and hardware.

### Choosing the Right Layout for Your Operations

The optimal layout depends on your most common operations:

- Use row-major if you frequently access elements row by row (e.g., image processing)
- Use column-major if you frequently access elements column by column (e.g., some linear algebra operations)
- Consider the layout of libraries you interface with (e.g., BLAS typically uses column-major)

In practice, I've found that row-major is generally more intuitive for C programmers and works well for most tensor operations. However, when interfacing with BLAS or Fortran code, you may need to use column-major or convert between layouts.

## 3.2 Designing Strided Tensors for Non-Contiguous Data

Strided tensors allow us to work with non-contiguous slices of data without copying. This is particularly useful for operations like extracting sub-tensors, transposing, or working with padded data.

### Understanding Tensor Strides

A stride is the number of elements to skip in memory to move to the next element in a particular dimension. In our updated tensor struct, we have:

- `row_stride`: Elements to skip to move to the next row
- `col_stride`: Elements to skip to move to the next column

For standard layouts, these are fixed values, but for strided tensors, they can be arbitrary.

### Creating Tensor Views

Let's implement a function to create a view of a sub-tensor without copying data:

```c
/* Add to tensor.h */
int tensor_view(tensor_t *view, const tensor_t *t, 
                size_t row_start, size_t row_end,
                size_t col_start, size_t col_end);

/* Add to tensor.c */
int tensor_view(tensor_t *view, const tensor_t *t, 
                size_t row_start, size_t row_end,
                size_t col_start, size_t col_end) {
    /* Check bounds */
    if (row_start >= t->rows || row_end > t->rows || row_start >= row_end ||
        col_start >= t->cols || col_end > t->cols || col_start >= col_end) {
        return TENSOR_INDEX_OUT_OF_BOUNDS;
    }
    
    /* Set up view metadata */
    view->rows = row_end - row_start;
    view->cols = col_end - col_start;
    view->row_stride = t->row_stride;
    view->col_stride = t->col_stride;
    view->layout = t->layout;
    
    /* Calculate data pointer offset */
    view->data = &t->data[row_start * t->row_stride + col_start * t->col_stride];
    
    /* View doesn't own the data */
    view->owner = 0;
    
    return TENSOR_SUCCESS;
}
```

This function creates a view that points to a subset of the original tensor's data. The view has the same strides as the original tensor, which means it maintains the same memory layout.

### Implementing Strided Operations

Now let's update our tensor operations to work with strided tensors. Here's an example with element-wise addition:

```c
/* Update tensor_add in tensor_ops.c */
int tensor_add(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if shapes match */
    if (a->rows != b->rows || a->cols != b->cols) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status;
    if (result->data != NULL && result->owner) {
        if (result->rows != a->rows || result->cols != a->cols) {
            tensor_free(result);
            status = tensor_create(result, a->rows, a->cols);
            if (status != TENSOR_SUCCESS) {
                return status;
            }
        }
    } else if (result->data == NULL) {
        status = tensor_create(result, a->rows, a->cols);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* Perform element-wise addition with strides */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            size_t idx_a = i * a->row_stride + j * a->col_stride;
            size_t idx_b = i * b->row_stride + j * b->col_stride;
            size_t idx_result = i * result->row_stride + j * result->col_stride;
            
            result->data[idx_result] = a->data[idx_a] + b->data[idx_b];
        }
    }
    
    return TENSOR_SUCCESS;
}
```

### Testing Strided Tensor Operations

Let's write a test for our strided tensor view and operations:

```c
/* Add to tensor_tests.c */
int test_tensor_view() {
    tensor_t t, view;
    
    /* Create a 4x4 tensor */
    tensor_create(&t, 4, 4);
    
    /* Initialize with values 0-15 */
    for (size_t i = 0; i < t.rows; i++) {
        for (size_t j = 0; j < t.cols; j++) {
            tensor_set(&t, i, j, (tensor_elem_t)(i * t.cols + j));
        }
    }
    
    /* Create a view of the central 2x2 region */
    int status = tensor_view(&view, &t, 1, 3, 1, 3);
    
    /* Check status */
    TEST_ASSERT(status == TENSOR_SUCCESS, "tensor_view should succeed");
    
    /* Check view dimensions */
    TEST_ASSERT(view.rows == 2 && view.cols == 2, "View should be 2x2");
    
    /* Check view values */
    tensor_elem_t expected[4] = {5.0, 6.0, 9.0, 10.0};  /* Values at positions (1,1), (1,2), (2,1), (2,2) */
    
    for (size_t i = 0; i < view.rows; i++) {
        for (size_t j = 0; j < view.cols; j++) {
            tensor_elem_t val;
            tensor_get(&view, i, j, &val);
            TEST_ASSERT(fabs(val - expected[i*2+j]) < 1e-6, "View should contain correct values");
        }
    }
    
    /* Modify the view and check that the original tensor is updated */
    tensor_set(&view, 0, 0, 99.0);
    
    tensor_elem_t val;
    tensor_get(&t, 1, 1, &val);
    TEST_ASSERT(fabs(val - 99.0) < 1e-6, "Modifying view should update original tensor");
    
    /* Clean up */
    tensor_free(&t);  /* No need to free view as it doesn't own data */
    
    return 1;
}

int test_strided_operations() {
    tensor_t t, view1, view2, result;
    
    /* Create a 4x4 tensor */
    tensor_create(&t, 4, 4);
    
    /* Initialize with values */
    for (size_t i = 0; i < t.rows; i++) {
        for (size_t j = 0; j < t.cols; j++) {
            tensor_set(&t, i, j, (tensor_elem_t)(i + j));
        }
    }
    
    /* Create two overlapping views */
    tensor_view(&view1, &t, 0, 2, 0, 2);  /* Top-left 2x2 */
    tensor_view(&view2, &t, 1, 3, 1, 3);  /* Center 2x2 */
    
    /* Add the views */
    tensor_add(&result, &view1, &view2);
    
    /* Check result dimensions */
    TEST_ASSERT(result.rows == 2 && result.cols == 2, "Result should be 2x2");
    
    /* Expected values:
     * view1: [0,1; 1,2]
     * view2: [2,3; 3,4]
     * result: [2,4; 4,6]
     */
    tensor_elem_t expected[4] = {2.0, 4.0, 4.0, 6.0};
    
    for (size_t i = 0; i < result.rows; i++) {
        for (size_t j = 0; j < result.cols; j++) {
            tensor_elem_t val;
            tensor_get(&result, i, j, &val);
            TEST_ASSERT(fabs(val - expected[i*2+j]) < 1e-6, "Result should contain correct values");
        }
    }
    
    /* Clean up */
    tensor_free(&t);
    tensor_free(&result);
    
    return 1;
}
```

## 3.3 Transposition Strategies: Avoiding Costly Data Copies

Transposition is a common operation in tensor programming, but naive implementations can be inefficient due to unnecessary data copying. Let's explore strategies to optimize transposition.

### In-Place Transposition for Square Matrices

For square matrices, we can perform transposition in-place by swapping elements across the diagonal:

```c
/* Add to tensor_ops.h */
int tensor_transpose_inplace(tensor_t *t);

/* Add to tensor_ops.c */
int tensor_transpose_inplace(tensor_t *t) {
    /* In-place transposition only works for square matrices */
    if (t->rows != t->cols) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Swap elements across the diagonal */
    for (size_t i = 0; i < t->rows; i++) {
        for (size_t j = i + 1; j < t->cols; j++) {
            size_t idx1 = i * t->row_stride + j * t->col_stride;
            size_t idx2 = j * t->row_stride + i * t->col_stride;
            
            tensor_elem_t temp = t->data[idx1];
            t->data[idx1] = t->data[idx2];
            t->data[idx2] = temp;
        }
    }
    
    return TENSOR_SUCCESS;
}
```

### Transposition Using Stride Manipulation

For non-square matrices, we can create a transposed view without copying data by swapping the strides:

```c
/* Add to tensor_ops.h */
int tensor_transpose_view(tensor_t *result, const tensor_t *t);

/* Add to tensor_ops.c */
int tensor_transpose_view(tensor_t *result, const tensor_t *t) {
    /* Set up transposed view */
    result->rows = t->cols;
    result->cols = t->rows;
    result->row_stride = t->col_stride;
    result->col_stride = t->row_stride;
    result->data = t->data;
    result->owner = 0;  /* View doesn't own the data */
    result->layout = (t->layout == TENSOR_ROW_MAJOR) ? TENSOR_COL_MAJOR : TENSOR_ROW_MAJOR;
    
    return TENSOR_SUCCESS;
}
```

This approach is extremely efficient as it involves no data copying. However, subsequent operations on the transposed view may be less cache-efficient due to non-contiguous memory access.

### Explicit Transposition with Data Copy

For cases where we need a contiguous transposed tensor, we can perform an explicit copy:

```c
/* Add to tensor_ops.h */
int tensor_transpose(tensor_t *result, const tensor_t *t);

/* Add to tensor_ops.c */
int tensor_transpose(tensor_t *result, const tensor_t *t) {
    /* Create result tensor with swapped dimensions */
    int status;
    if (result->data != NULL && result->owner) {
        if (result->rows != t->cols || result->cols != t->rows) {
            tensor_free(result);
            status = tensor_create(result, t->cols, t->rows);
            if (status != TENSOR_SUCCESS) {
                return status;
            }
        }
    } else if (result->data == NULL) {
        status = tensor_create(result, t->cols, t->rows);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* Copy elements with transposed indices */
    for (size_t i = 0; i < t->rows; i++) {
        for (size_t j = 0; j < t->cols; j++) {
            size_t src_idx = i * t->row_stride + j * t->col_stride;
            size_t dst_idx = j * result->row_stride + i * result->col_stride;
            
            result->data[dst_idx] = t->data[src_idx];
        }
    }
    
    return TENSOR_SUCCESS;
}
```

### Benchmarking Transposition Strategies

Let's benchmark these different transposition strategies:

```c
/* benchmark_transpose.c */
#include "tensor.h"
#include "tensor_ops.h"
#include <stdio.h>
#include <time.h>

/* Benchmark function for matrix-vector multiplication */
double benchmark_matmul(const tensor_t *a, const tensor_t *b, tensor_t *result) {
    clock_t start = clock();
    
    /* Perform matrix multiplication */
    tensor_matmul(result, a, b);
    
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

int main() {
    tensor_t a, b, b_trans_view, b_trans_copy, result1, result2;
    size_t size = 1000;  /* Large enough to see the difference */
    
    /* Create tensors */
    tensor_create(&a, size, size);
    tensor_create(&b, size, size);
    tensor_create(&result1, size, size);
    tensor_create(&result2, size, size);
    
    /* Initialize with some values */
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            tensor_set(&a, i, j, (tensor_elem_t)(i * 0.01 + j * 0.01));
            tensor_set(&b, i, j, (tensor_elem_t)(i * 0.01 - j * 0.01));
        }
    }
    
    /* Create transposed view of b */
    tensor_transpose_view(&b_trans_view, &b);
    
    /* Create transposed copy of b */
    tensor_transpose(&b_trans_copy, &b);
    
    /* Benchmark matrix multiplication with transposed view */
    double time_view = benchmark_matmul(&a, &b_trans_view, &result1);
    printf("Matrix multiplication with transposed view: %.6f seconds\n", time_view);
    
    /* Benchmark matrix multiplication with transposed copy */
    double time_copy = benchmark_matmul(&a, &b_trans_copy, &result2);
    printf("Matrix multiplication with transposed copy: %.6f seconds\n", time_copy);
    
    /* Clean up */
    tensor_free(&a);
    tensor_free(&b);
    tensor_free(&b_trans_copy);
    tensor_free(&result1);
    tensor_free(&result2);
    
    return 0;
}
```

The results will likely show that using a transposed copy is faster for matrix multiplication, despite the initial copying overhead. This is because the copy creates a contiguous memory layout that's more cache-friendly for subsequent operations.

### When to Use Each Strategy

Choose your transposition strategy based on your specific needs:

1. **In-place transposition**: Use for square matrices when memory is limited
2. **Transposed view**: Use when you need to transpose a tensor temporarily and memory is a concern
3. **Explicit transposition**: Use when you'll perform multiple operations on the transposed tensor

In my experience, the explicit copy is often worth it for large tensors that will be used in multiple operations, as the improved cache efficiency outweighs the copying cost.

## 3.4 Cache-Aware Blocking for Large-Scale Tensors

For large tensors, cache efficiency becomes critical. Blocking (or tiling) is a technique that divides large tensors into smaller blocks that fit in the CPU cache.

### Understanding CPU Cache Hierarchy

Modern CPUs have a hierarchy of caches:

1. L1 cache: Smallest but fastest (typically 32-64 KB per core)
2. L2 cache: Larger but slower (typically 256 KB - 1 MB per core)
3. L3 cache: Shared among cores (typically 3-32 MB)

When data isn't in cache, the CPU must fetch it from main memory, which is much slower (100-300 cycles vs. 3-10 cycles for cache).

### Implementing Blocked Matrix Multiplication

Let's implement a cache-aware blocked matrix multiplication:

```c
/* Add to tensor_ops.h */
int tensor_matmul_blocked(tensor_t *result, const tensor_t *a, const tensor_t *b, size_t block_size);

/* Add to tensor_ops.c */
int tensor_matmul_blocked(tensor_t *result, const tensor_t *a, const tensor_t *b, size_t block_size) {
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
    } else if (result->data == NULL) {
        status = tensor_create(result, a->rows, b->cols);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* If block_size is 0, use a default based on cache size */
    if (block_size == 0) {
        /* Assuming L1 cache is 32 KB and each element is 8 bytes (double) */
        block_size = 64;  /* sqrt(32 KB / 8 bytes) ≈ 64 */
    }
    
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
                        tensor_elem_t sum = result->data[i * result->row_stride + j * result->col_stride];
                        
                        for (size_t k = k0; k < k_end; k++) {
                            sum += a->data[i * a->row_stride + k * a->col_stride] * 
                                   b->data[k * b->row_stride + j * b->col_stride];
                        }
                        
                        result->data[i * result->row_stride + j * result->col_stride] = sum;
                    }
                }
            }
        }
    }
    
    return TENSOR_SUCCESS;
}
```

### Tuning Block Size for Optimal Performance

The optimal block size depends on your CPU's cache size and the tensor element size. A good starting point is to choose a block size such that three blocks (one each from A, B, and C) fit in the L1 cache:

```
block_size = sqrt(L1_cache_size / (3 * sizeof(tensor_elem_t)))
```

For example, with a 32 KB L1 cache and 8-byte doubles:

```
block_size = sqrt(32 * 1024 / (3 * 8)) ≈ 64
```

Let's write a function to find the optimal block size empirically:

```c
/* benchmark_blocking.c */
#include "tensor.h"
#include "tensor_ops.h"
#include <stdio.h>
#include <time.h>

/* Benchmark function for blocked matrix multiplication */
double benchmark_blocked_matmul(const tensor_t *a, const tensor_t *b, tensor_t *result, size_t block_size) {
    clock_t start = clock();
    
    /* Perform blocked matrix multiplication */
    tensor_matmul_blocked(result, a, b, block_size);
    
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

int main() {
    tensor_t a, b, result;
    size_t size = 1000;  /* Large enough to see the difference */
    
    /* Create tensors */
    tensor_create(&a, size, size);
    tensor_create(&b, size, size);
    tensor_create(&result, size, size);
    
    /* Initialize with some values */
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            tensor_set(&a, i, j, (tensor_elem_t)(i * 0.01 + j * 0.01));
            tensor_set(&b, i, j, (tensor_elem_t)(i * 0.01 - j * 0.01));
        }
    }
    
    /* Try different block sizes */
    size_t block_sizes[] = {16, 32, 64, 128, 256};
    size_t num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    printf("Block Size | Time (seconds)\n");
    printf("-----------+---------------\n");
    
    for (size_t i = 0; i < num_block_sizes; i++) {
        double time = benchmark_blocked_matmul(&a, &b, &result, block_sizes[i]);
        printf("%10zu | %15.6f\n", block_sizes[i], time);
    }
    
    /* Clean up */
    tensor_free(&a);
    tensor_free(&b);
    tensor_free(&result);
    
    return 0;
}
```

### Applying Blocking to Other Operations

Blocking can be applied to other tensor operations as well. For example, here's a blocked implementation of tensor transposition:

```c
/* Add to tensor_ops.c */
int tensor_transpose_blocked(tensor_t *result, const tensor_t *t, size_t block_size) {
    /* Create result tensor with swapped dimensions */
    int status;
    if (result->data != NULL && result->owner) {
        if (result->rows != t->cols || result->cols != t->rows) {
            tensor_free(result);
            status = tensor_create(result, t->cols, t->rows);
            if (status != TENSOR_SUCCESS) {
                return status;
            }
        }
    } else if (result->data == NULL) {
        status = tensor_create(result, t->cols, t->rows);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* If block_size is 0, use a default */
    if (block_size == 0) {
        block_size = 64;
    }
    
    /* Blocked transposition */
    for (size_t i0 = 0; i0 < t->rows; i0 += block_size) {
        size_t i_end = (i0 + block_size < t->rows) ? i0 + block_size : t->rows;
        
        for (size_t j0 = 0; j0 < t->cols; j0 += block_size) {
            size_t j_end = (j0 + block_size < t->cols) ? j0 + block_size : t->cols;
            
            /* Process current block */
            for (size_t i = i0; i < i_end; i++) {
                for (size_t j = j0; j < j_end; j++) {
                    size_t src_idx = i * t->row_stride + j * t->col_stride;
                    size_t dst_idx = j * result->row_stride + i * result->col_stride;
                    
                    result->data[dst_idx] = t->data[src_idx];
                }
            }
        }
    }
    
    return TENSOR_SUCCESS;
}
```

## Memory Layout Visualization

Let's visualize the different memory layouts and access patterns we've discussed.

### Row-Major vs. Column-Major Layout

```
2x3 Matrix Logical View:
+---+---+---+
| A | B | C |
+---+---+---+
| D | E | F |
+---+---+---+

Row-Major Memory Layout:
+---+---+---+---+---+---+
| A | B | C | D | E | F |
+---+---+---+---+---+---+

Column-Major Memory Layout:
+---+---+---+---+---+---+
| A | D | B | E | C | F |
+---+---+---+---+---+---+
```

### Strided Tensor View

```
Original 4x4 Matrix:
+---+---+---+---+
| A | B | C | D |
+---+---+---+---+
| E | F | G | H |
+---+---+---+---+
| I | J | K | L |
+---+---+---+---+
| M | N | O | P |
+---+---+---+---+

2x2 View (starting at F):
+---+---+
| F | G |
+---+---+
| J | K |
+---+---+

Memory Layout (Row-Major):
+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
| A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P |
+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
                      ^       ^       ^       ^
                      |       |       |       |
                    View elements (non-contiguous)
```

### Blocked Matrix Multiplication

```
Blocked Matrix Multiplication (2x2 blocks):

Matrix A:                 Matrix B:
+---+---+---+---+         +---+---+---+---+
| A1  |  A2  |           | B1  |  B2  |
+---+---+---+---+         +---+---+---+---+
| A3  |  A4  |           | B3  |  B4  |
+---+---+---+---+         +---+---+---+---+

Result = A1*B1 + A2*B3 | A1*B2 + A2*B4
         --------------|---------------
         A3*B1 + A4*B3 | A3*B2 + A4*B4
```

## Common Pitfalls and Debugging

Let's discuss some common issues you might encounter when working with different memory layouts.

### Mixing Row-Major and Column-Major Code

One of the most common pitfalls is mixing row-major and column-major code without proper conversion. This often happens when interfacing with libraries that use a different convention.

For example, if you're using a BLAS library (which typically uses column-major) with your row-major tensors, you need to transpose your matrices or adjust your indexing:

```c
/* Incorrect: Passing row-major matrix to column-major BLAS function */
dgemm_('N', 'N', &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

/* Correct: Transpose the operation to account for layout difference */
dgemm_('N', 'N', &n, &m, &k, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
```

### Cache Thrashing

Cache thrashing occurs when your access pattern repeatedly evicts cache lines before they can be fully utilized. This can happen with large tensors or when your block size is poorly chosen.

Symptoms of cache thrashing include:

1. Performance that's much worse than expected
2. Performance that degrades dramatically as tensor size increases
3. CPU performance counters showing high cache miss rates

To diagnose cache thrashing, you can use tools like `perf` on Linux:

```bash
perf stat -e cache-misses,cache-references ./your_program
```

### Memory Alignment Issues

Modern CPUs perform best when data is aligned to cache line boundaries (typically 64 bytes). Misaligned memory access can cause performance degradation.

To ensure proper alignment:

1. Use aligned memory allocation (e.g., `posix_memalign` or `aligned_alloc`)
2. Pad your tensor dimensions to multiples of the alignment size

Here's an example of aligned tensor creation:

```c
/* Add to tensor.c */
int tensor_create_aligned(tensor_t *t, size_t rows, size_t cols, size_t alignment) {
    if (rows == 0 || cols == 0) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Ensure alignment is a power of 2 */
    if (alignment & (alignment - 1)) {
        return TENSOR_INVALID_PARAMETER;
    }
    
    /* Allocate aligned memory */
    void *ptr;
    int result = posix_memalign(&ptr, alignment, rows * cols * sizeof(tensor_elem_t));
    if (result != 0) {
        return TENSOR_ALLOCATION_FAILED;
    }
    
    t->data = (tensor_elem_t *)ptr;
    t->rows = rows;
    t->cols = cols;
    t->row_stride = cols;
    t->col_stride = 1;
    t->owner = 1;
    t->layout = TENSOR_ROW_MAJOR;
    
    /* Initialize data to zero */
    memset(t->data, 0, rows * cols * sizeof(tensor_elem_t));
    
    return TENSOR_SUCCESS;
}
```

## Exercises

### Exercise 1: Implement a Function to Convert Between Row-Major and Column-Major Layouts

Implement a function `tensor_convert_layout` that converts a tensor from one layout to another.

Hint: You'll need to create a new tensor with the desired layout and copy the data with appropriate index mapping.

Partial solution:

```c
int tensor_convert_layout(tensor_t *result, const tensor_t *t, tensor_layout_t new_layout) {
    /* If the tensor already has the desired layout, just create a copy */
    if (t->layout == new_layout) {
        return tensor_copy(result, t);
    }
    
    /* Create a new tensor with the desired layout */
    int status = tensor_create_with_layout(result, t->rows, t->cols, new_layout);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Copy data with appropriate index mapping */
    for (size_t i = 0; i < t->rows; i++) {
        for (size_t j = 0; j < t->cols; j++) {
            tensor_elem_t val;
            tensor_get(t, i, j, &val);
            tensor_set(result, i, j, val);
        }
    }
    
    return TENSOR_SUCCESS;
}
```

### Exercise 2: Implement a Cache-Oblivious Matrix Transposition Algorithm

Implement a recursive, cache-oblivious matrix transposition algorithm that works well across different cache sizes without explicit tuning.

Hint: Divide the matrix into four quadrants and recursively transpose each quadrant.

### Exercise 3: Benchmark Different Memory Access Patterns

Create a benchmark program that compares the performance of different memory access patterns (row-wise, column-wise, diagonal, random) on tensors with different layouts and sizes.

Partial solution:

```c
/* Types of access patterns */
typedef enum {
    ACCESS_ROW_WISE,
    ACCESS_COLUMN_WISE,
    ACCESS_DIAGONAL,
    ACCESS_RANDOM
} access_pattern_t;

/* Benchmark function */
double benchmark_access_pattern(tensor_t *t, access_pattern_t pattern) {
    clock_t start = clock();
    tensor_elem_t sum = 0.0;
    size_t iterations = t->rows * t->cols;
    
    switch (pattern) {
        case ACCESS_ROW_WISE:
            for (size_t i = 0; i < t->rows; i++) {
                for (size_t j = 0; j < t->cols; j++) {
                    tensor_elem_t val;
                    tensor_get(t, i, j, &val);
                    sum += val;
                }
            }
            break;
            
        case ACCESS_COLUMN_WISE:
            for (size_t j = 0; j < t->cols; j++) {
                for (size_t i = 0; i < t->rows; i++) {
                    tensor_elem_t val;
                    tensor_get(t, i, j, &val);
                    sum += val;
                }
            }
            break;
            
        /* Implement other patterns... */
    }
    
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}
```

## Summary and Key Takeaways

In this chapter, we've explored memory layouts and their impact on tensor performance:

- We learned about row-major and column-major layouts and how they affect iteration performance.
- We implemented strided tensors for efficient non-contiguous data access.
- We developed transposition strategies that avoid unnecessary data copying.
- We applied cache-aware blocking techniques to improve performance for large tensors.

Key takeaways:

1. **Memory layout matters**: Choose the right layout for your most common operations.
2. **Stride manipulation**: Use strides to create views and avoid copying data when possible.
3. **Cache awareness**: Design algorithms with the CPU cache hierarchy in mind.
4. **Benchmark and tune**: Optimal parameters depend on your specific hardware and workload.

By mastering memory layouts, you can significantly improve the performance of your tensor operations, often by an order of magnitude or more.

In the next chapter, we'll explore parallelizing tensor workloads with OpenMP to take advantage of multi-core processors.

## Further Reading

1. "What Every Programmer Should Know About Memory" by Ulrich Drepper - A comprehensive guide to memory hierarchies and their performance implications.

2. "Optimizing Matrix Transpose: A case study in memory hierarchy and tiling" by Siddhartha Chatterjee et al. - Research paper on optimizing matrix transposition.

3. "Auto-Tuning Matrix Transpose for Arbitrary Matrices" by Matteo Frigo et al. - Discusses cache-oblivious algorithms for matrix transposition.