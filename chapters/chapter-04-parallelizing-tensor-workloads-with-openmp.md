# Chapter 4: Parallelizing Tensor Workloads with OpenMP

*Tensor Programming in C: Building High-Performance Numerical Systems from Scratch*

## You Will Learn To...

- Harness multi-core CPUs for concurrent tensor processing
- Implement thread-safe tensor allocation and initialization
- Parallelize tensor operations using OpenMP directives
- Identify and eliminate race conditions in accumulation operations
- Balance workloads across heterogeneous cores for optimal performance

## 4.1 Thread-Safe Tensor Allocation and Initialization

Before diving into parallel tensor operations, we need to ensure our tensor infrastructure is thread-safe. When multiple threads create, modify, or free tensors simultaneously, we need to prevent race conditions and memory corruption.

### Understanding Thread Safety Issues

Let's first identify potential thread safety issues in our tensor implementation:

1. **Memory allocation**: `malloc` and `free` are typically thread-safe, but we need to ensure our wrapper functions maintain this property.
2. **Error handling**: Global error states can cause race conditions.
3. **Initialization**: Multiple threads initializing the same tensor can lead to data corruption.

### Thread-Safe Tensor Creation

Let's modify our tensor creation function to be thread-safe:

```c
/* Add to tensor.h */
#include <omp.h>  /* For OpenMP support */

/* Thread-safe tensor creation */
int tensor_create_ts(tensor_t *t, size_t rows, size_t cols);

/* Add to tensor.c */
int tensor_create_ts(tensor_t *t, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Allocate memory for data - malloc is thread-safe */
    tensor_elem_t *data = (tensor_elem_t *)malloc(rows * cols * sizeof(tensor_elem_t));
    if (data == NULL) {
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
    
    /* Initialize data to zero - use OpenMP for parallel initialization */
    #pragma omp parallel for collapse(2) if(rows * cols > 1000)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            t->data[i * t->row_stride + j * t->col_stride] = 0.0;
        }
    }
    
    return TENSOR_SUCCESS;
}
```

The key changes are:

1. We use a local variable `data` for allocation before assigning to the tensor struct, ensuring atomic updates.
2. We use OpenMP to parallelize the initialization loop for large tensors.
3. The `collapse(2)` directive flattens the nested loops, creating a larger parallel workload.
4. The `if(rows * cols > 1000)` clause only parallelizes initialization for tensors with more than 1000 elements, avoiding overhead for small tensors.

### Thread-Safe Error Handling

Instead of using global error states, we return error codes from each function. This approach is already thread-safe, as each thread receives its own copy of the return value.

### Thread-Local Storage for Temporary Tensors

In complex tensor operations, we often need temporary tensors. Let's create a thread-local tensor pool to avoid repeated allocations:

```c
/* Add to tensor.h */
/* Maximum number of temporary tensors per thread */
#define MAX_TEMP_TENSORS 5

/* Thread-local tensor pool */
tensor_t *tensor_get_temp(size_t rows, size_t cols);
void tensor_release_temp(tensor_t *t);

/* Add to tensor.c */
/* Thread-local storage for temporary tensors */
#pragma omp threadprivate(temp_tensors, temp_tensor_count)
static tensor_t temp_tensors[MAX_TEMP_TENSORS];
static int temp_tensor_count = 0;

/* Get a temporary tensor from the pool */
tensor_t *tensor_get_temp(size_t rows, size_t cols) {
    /* Find an available tensor in the pool */
    for (int i = 0; i < temp_tensor_count; i++) {
        if (temp_tensors[i].rows == rows && temp_tensors[i].cols == cols) {
            return &temp_tensors[i];
        }
    }
    
    /* If no suitable tensor found and pool not full, create a new one */
    if (temp_tensor_count < MAX_TEMP_TENSORS) {
        tensor_create_ts(&temp_tensors[temp_tensor_count], rows, cols);
        return &temp_tensors[temp_tensor_count++];
    }
    
    /* If pool is full, reuse the first tensor (could use LRU instead) */
    tensor_free(&temp_tensors[0]);
    tensor_create_ts(&temp_tensors[0], rows, cols);
    return &temp_tensors[0];
}

/* Release a temporary tensor back to the pool */
void tensor_release_temp(tensor_t *t) {
    /* No need to do anything, as the tensor remains in the pool */
    /* Just reset the data to zero for safety */
    #pragma omp parallel for collapse(2) if(t->rows * t->cols > 1000)
    for (size_t i = 0; i < t->rows; i++) {
        for (size_t j = 0; j < t->cols; j++) {
            t->data[i * t->row_stride + j * t->col_stride] = 0.0;
        }
    }
}
```

The `#pragma omp threadprivate` directive ensures that each thread has its own copy of the tensor pool, preventing race conditions.

### Testing Thread Safety

Let's write a test to verify that our tensor creation is thread-safe:

```c
/* Add to tensor_tests.c */
int test_tensor_thread_safety() {
    const int num_threads = 8;
    const int tensors_per_thread = 10;
    tensor_t tensors[num_threads][tensors_per_thread];
    int success = 1;
    
    /* Create tensors in parallel */
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        
        for (int i = 0; i < tensors_per_thread; i++) {
            /* Create tensors of different sizes */
            size_t rows = 10 + thread_id;
            size_t cols = 10 + i;
            
            int status = tensor_create_ts(&tensors[thread_id][i], rows, cols);
            
            if (status != TENSOR_SUCCESS) {
                #pragma omp critical
                {
                    printf("Thread %d failed to create tensor %d\n", thread_id, i);
                    success = 0;
                }
            }
            
            /* Verify tensor dimensions and initialization */
            if (tensors[thread_id][i].rows != rows || tensors[thread_id][i].cols != cols) {
                #pragma omp critical
                {
                    printf("Thread %d: tensor %d has wrong dimensions\n", thread_id, i);
                    success = 0;
                }
            }
            
            /* Check that all elements are initialized to zero */
            for (size_t r = 0; r < rows; r++) {
                for (size_t c = 0; c < cols; c++) {
                    if (tensors[thread_id][i].data[r * cols + c] != 0.0) {
                        #pragma omp critical
                        {
                            printf("Thread %d: tensor %d not initialized to zero\n", thread_id, i);
                            success = 0;
                        }
                        break;
                    }
                }
            }
        }
    }
    
    /* Clean up */
    for (int t = 0; t < num_threads; t++) {
        for (int i = 0; i < tensors_per_thread; i++) {
            tensor_free(&tensors[t][i]);
        }
    }
    
    TEST_ASSERT(success, "Thread-safe tensor creation should work correctly");
    return 1;
}
```

This test creates multiple tensors in parallel and verifies that they are correctly initialized.

## 4.2 Parallelizing Loops with `#pragma omp` Directives

Now that our tensor infrastructure is thread-safe, let's parallelize our core tensor operations using OpenMP directives.

### Understanding OpenMP Basics

OpenMP is a set of compiler directives, library routines, and environment variables that enable shared-memory parallelism in C, C++, and Fortran. The basic syntax for parallelizing a loop is:

```c
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    // Loop body executed in parallel
}
```

OpenMP automatically divides the iterations among available threads. Let's apply this to our tensor operations.

### Parallelizing Element-Wise Operations

Element-wise operations are embarrassingly parallel, making them ideal candidates for parallelization:

```c
/* Update tensor_add in tensor_ops.c */
int tensor_add(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* Check if shapes match */
    if (a->rows != b->rows || a->cols != b->cols) {
        return TENSOR_DIMENSION_MISMATCH;
    }
    
    /* Create or resize result tensor */
    int status = tensor_create_result(result, a);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Perform element-wise addition in parallel */
    #pragma omp parallel for collapse(2) if(a->rows * a->cols > 1000)
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

The `collapse(2)` directive flattens the nested loops, creating a larger parallel workload. The `if` clause only parallelizes operations on tensors with more than 1000 elements, avoiding overhead for small tensors.

### Parallelizing Matrix Multiplication

Matrix multiplication is more complex but can still be effectively parallelized:

```c
/* Update tensor_matmul in tensor_ops.c */
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
            status = tensor_create_ts(result, a->rows, b->cols);
            if (status != TENSOR_SUCCESS) {
                return status;
            }
        } else {
            /* Zero out the result tensor */
            #pragma omp parallel for collapse(2) if(result->rows * result->cols > 1000)
            for (size_t i = 0; i < result->rows; i++) {
                for (size_t j = 0; j < result->cols; j++) {
                    size_t idx = i * result->row_stride + j * result->col_stride;
                    result->data[idx] = 0.0;
                }
            }
        }
    } else if (result->data == NULL) {
        status = tensor_create_ts(result, a->rows, b->cols);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* Perform matrix multiplication in parallel */
    #pragma omp parallel for collapse(2) if(a->rows * b->cols > 100)
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            tensor_elem_t sum = 0.0;
            
            for (size_t k = 0; k < a->cols; k++) {
                size_t idx_a = i * a->row_stride + k * a->col_stride;
                size_t idx_b = k * b->row_stride + j * b->col_stride;
                sum += a->data[idx_a] * b->data[idx_b];
            }
            
            size_t idx_result = i * result->row_stride + j * result->col_stride;
            result->data[idx_result] = sum;
        }
    }
    
    return TENSOR_SUCCESS;
}
```

Here, we parallelize the outer two loops, with each thread computing one element of the result matrix.

### Parallelizing Blocked Matrix Multiplication

For large matrices, blocked multiplication with parallelization provides even better performance:

```c
/* Update tensor_matmul_blocked in tensor_ops.c */
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
            status = tensor_create_ts(result, a->rows, b->cols);
            if (status != TENSOR_SUCCESS) {
                return status;
            }
        } else {
            /* Zero out the result tensor */
            #pragma omp parallel for collapse(2) if(result->rows * result->cols > 1000)
            for (size_t i = 0; i < result->rows; i++) {
                for (size_t j = 0; j < result->cols; j++) {
                    size_t idx = i * result->row_stride + j * result->col_stride;
                    result->data[idx] = 0.0;
                }
            }
        }
    } else if (result->data == NULL) {
        status = tensor_create_ts(result, a->rows, b->cols);
        if (status != TENSOR_SUCCESS) {
            return status;
        }
    }
    
    /* If block_size is 0, use a default based on cache size */
    if (block_size == 0) {
        block_size = 64;  /* sqrt(32 KB / 8 bytes) u2248 64 */
    }
    
    /* Calculate number of blocks */
    size_t num_row_blocks = (a->rows + block_size - 1) / block_size;
    size_t num_col_blocks = (b->cols + block_size - 1) / block_size;
    size_t num_k_blocks = (a->cols + block_size - 1) / block_size;
    
    /* Parallelize at the block level */
    #pragma omp parallel for collapse(2) if(num_row_blocks * num_col_blocks > 4)
    for (size_t i0 = 0; i0 < a->rows; i0 += block_size) {
        for (size_t j0 = 0; j0 < b->cols; j0 += block_size) {
            size_t i_end = (i0 + block_size < a->rows) ? i0 + block_size : a->rows;
            size_t j_end = (j0 + block_size < b->cols) ? j0 + block_size : b->cols;
            
            /* Process all k blocks for this (i,j) block */
            for (size_t k0 = 0; k0 < a->cols; k0 += block_size) {
                size_t k_end = (k0 + block_size < a->cols) ? k0 + block_size : a->cols;
                
                /* Process current block */
                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        size_t idx_result = i * result->row_stride + j * result->col_stride;
                        tensor_elem_t sum = result->data[idx_result];
                        
                        for (size_t k = k0; k < k_end; k++) {
                            size_t idx_a = i * a->row_stride + k * a->col_stride;
                            size_t idx_b = k * b->row_stride + j * b->col_stride;
                            sum += a->data[idx_a] * b->data[idx_b];
                        }
                        
                        result->data[idx_result] = sum;
                    }
                }
            }
        }
    }
    
    return TENSOR_SUCCESS;
}
```

In this implementation, we parallelize at the block level rather than the element level. This approach reduces thread management overhead and improves cache utilization.

### Benchmarking Parallel Performance

Let's create a benchmark to measure the speedup from parallelization:

```c
/* benchmark_parallel.c */
#include "tensor.h"
#include "tensor_ops.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>

/* Benchmark function for matrix multiplication */
double benchmark_matmul(const tensor_t *a, const tensor_t *b, tensor_t *result, int num_threads) {
    /* Set number of threads */
    omp_set_num_threads(num_threads);
    
    /* Warm up the cache */
    tensor_matmul(result, a, b);
    
    /* Measure performance */
    double start_time = omp_get_wtime();
    
    /* Perform matrix multiplication */
    tensor_matmul(result, a, b);
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main() {
    tensor_t a, b, result;
    size_t size = 1000;  /* Large enough to see the difference */
    
    /* Create tensors */
    tensor_create_ts(&a, size, size);
    tensor_create_ts(&b, size, size);
    tensor_create_ts(&result, size, size);
    
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
    
    /* Try different numbers of threads */
    printf("Threads | Time (seconds) | Speedup\n");
    printf("--------|----------------|--------\n");
    
    /* Baseline: single-threaded */
    double baseline_time = benchmark_matmul(&a, &b, &result, 1);
    printf("%7d | %14.6f | %7.2f\n", 1, baseline_time, 1.0);
    
    /* Test with increasing thread counts */
    int max_threads = omp_get_max_threads();
    for (int threads = 2; threads <= max_threads; threads *= 2) {
        double time = benchmark_matmul(&a, &b, &result, threads);
        double speedup = baseline_time / time;
        printf("%7d | %14.6f | %7.2f\n", threads, time, speedup);
    }
    
    /* Clean up */
    tensor_free(&a);
    tensor_free(&b);
    tensor_free(&result);
    
    return 0;
}
```

This benchmark measures the performance of matrix multiplication with different numbers of threads, showing the speedup relative to single-threaded execution.

## 4.3 Reducing Race Conditions in Accumulation Operations

Accumulation operations, where multiple threads update the same memory location, can lead to race conditions. Let's explore techniques to prevent these issues.

### Understanding Race Conditions

A race condition occurs when multiple threads access and modify the same data concurrently, leading to unpredictable results. Consider this naive parallel implementation of tensor reduction:

```c
/* Naive implementation with race condition */
tensor_elem_t tensor_sum_naive(const tensor_t *t) {
    tensor_elem_t sum = 0.0;
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < t->rows; i++) {
        for (size_t j = 0; j < t->cols; j++) {
            size_t idx = i * t->row_stride + j * t->col_stride;
            sum += t->data[idx];  /* RACE CONDITION: multiple threads update sum */
        }
    }
    
    return sum;
}
```

In this code, multiple threads try to update `sum` simultaneously, leading to lost updates.

### Using OpenMP Reduction Clause

The simplest solution is to use OpenMP's `reduction` clause:

```c
/* Add to tensor_ops.h */
tensor_elem_t tensor_sum(const tensor_t *t);

/* Add to tensor_ops.c */
tensor_elem_t tensor_sum(const tensor_t *t) {
    tensor_elem_t sum = 0.0;
    
    #pragma omp parallel for collapse(2) reduction(+:sum) if(t->rows * t->cols > 1000)
    for (size_t i = 0; i < t->rows; i++) {
        for (size_t j = 0; j < t->cols; j++) {
            size_t idx = i * t->row_stride + j * t->col_stride;
            sum += t->data[idx];
        }
    }
    
    return sum;
}
```

The `reduction(+:sum)` clause tells OpenMP to create a private copy of `sum` for each thread, accumulate values into these private copies, and then combine them at the end.

### Implementing Custom Reductions

For more complex reductions, we can implement our own thread-local accumulation:

```c
/* Add to tensor_ops.h */
tensor_elem_t tensor_max(const tensor_t *t);

/* Add to tensor_ops.c */
tensor_elem_t tensor_max(const tensor_t *t) {
    tensor_elem_t max_val = -INFINITY;
    
    #pragma omp parallel if(t->rows * t->cols > 1000)
    {
        tensor_elem_t thread_max = -INFINITY;
        
        #pragma omp for collapse(2) nowait
        for (size_t i = 0; i < t->rows; i++) {
            for (size_t j = 0; j < t->cols; j++) {
                size_t idx = i * t->row_stride + j * t->col_stride;
                if (t->data[idx] > thread_max) {
                    thread_max = t->data[idx];
                }
            }
        }
        
        /* Combine thread-local results */
        #pragma omp critical
        {
            if (thread_max > max_val) {
                max_val = thread_max;
            }
        }
    }
    
    return max_val;
}
```

In this implementation, each thread maintains its own `thread_max` variable, and we only use a critical section to combine the final results.

### Atomic Operations

For simple updates, we can use atomic operations instead of critical sections:

```c
/* Count non-zero elements */
size_t tensor_count_nonzero(const tensor_t *t) {
    size_t count = 0;
    
    #pragma omp parallel if(t->rows * t->cols > 1000)
    {
        size_t thread_count = 0;
        
        #pragma omp for collapse(2) nowait
        for (size_t i = 0; i < t->rows; i++) {
            for (size_t j = 0; j < t->cols; j++) {
                size_t idx = i * t->row_stride + j * t->col_stride;
                if (t->data[idx] != 0.0) {
                    thread_count++;
                }
            }
        }
        
        /* Atomic update of the global count */
        #pragma omp atomic
        count += thread_count;
    }
    
    return count;
}
```

The `#pragma omp atomic` directive ensures that the update to `count` is performed atomically, preventing race conditions.

### Testing Parallel Reductions

Let's write a test to verify that our parallel reductions work correctly:

```c
/* Add to tensor_tests.c */
int test_tensor_parallel_reduction() {
    tensor_t t;
    size_t size = 1000;
    
    /* Create and initialize tensor */
    tensor_create_ts(&t, size, size);
    
    /* Initialize with known values */
    tensor_elem_t expected_sum = 0.0;
    tensor_elem_t expected_max = -INFINITY;
    
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            tensor_elem_t val = (tensor_elem_t)(i * size + j);
            tensor_set(&t, i, j, val);
            expected_sum += val;
            if (val > expected_max) {
                expected_max = val;
            }
        }
    }
    
    /* Test sum reduction */
    tensor_elem_t sum = tensor_sum(&t);
    TEST_ASSERT(fabs(sum - expected_sum) < 1e-6, "Parallel sum should compute correct value");
    
    /* Test max reduction */
    tensor_elem_t max_val = tensor_max(&t);
    TEST_ASSERT(fabs(max_val - expected_max) < 1e-6, "Parallel max should compute correct value");
    
    /* Clean up */
    tensor_free(&t);
    
    return 1;
}
```

## 4.4 Balancing Workloads Across Heterogeneous Cores

Modern CPUs often have heterogeneous cores (e.g., big.LITTLE architecture) or varying performance due to thermal throttling. Let's explore techniques to balance workloads effectively.

### Understanding Workload Imbalance

The default OpenMP scheduling divides iterations equally among threads, which can lead to imbalance if:

1. Some iterations take longer than others
2. Some cores are faster than others
3. The system is running other processes that compete for resources

### Dynamic Scheduling

OpenMP provides different scheduling strategies. Dynamic scheduling can help balance workloads:

```c
/* Update tensor_matmul in tensor_ops.c */
int tensor_matmul(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* ... (previous code) ... */
    
    /* Perform matrix multiplication in parallel with dynamic scheduling */
    #pragma omp parallel for collapse(2) schedule(dynamic, 16) if(a->rows * b->cols > 100)
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            /* ... (computation) ... */
        }
    }
    
    return TENSOR_SUCCESS;
}
```

The `schedule(dynamic, 16)` clause assigns chunks of 16 iterations at a time to threads as they become available. This helps balance the workload dynamically.

### Guided Scheduling

For workloads with decreasing computation time, guided scheduling can be more efficient:

```c
/* Triangular matrix multiplication (more work at the beginning) */
int tensor_triangular_matmul(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* ... (setup code) ... */
    
    /* Use guided scheduling for decreasing workload */
    #pragma omp parallel for schedule(guided) if(a->rows * b->cols > 100)
    for (size_t i = 0; i < a->rows; i++) {
        /* Triangular matrix: only process up to i */
        for (size_t j = 0; j <= i; j++) {
            /* ... (computation) ... */
        }
    }
    
    return TENSOR_SUCCESS;
}
```

Guided scheduling starts with large chunks and gradually reduces chunk size, which works well for triangular or other non-uniform workloads.

### Manual Load Balancing

For complex workloads, we can implement manual load balancing:

```c
/* Sparse matrix multiplication with manual load balancing */
int tensor_sparse_matmul(tensor_t *result, const sparse_tensor_t *a, const tensor_t *b) {
    /* ... (setup code) ... */
    
    /* Count non-zeros in each row to estimate workload */
    size_t *row_nnz = (size_t *)malloc(a->rows * sizeof(size_t));
    for (size_t i = 0; i < a->rows; i++) {
        row_nnz[i] = 0;
        for (size_t j = 0; j < a->cols; j++) {
            if (sparse_tensor_get(a, i, j) != 0.0) {
                row_nnz[i]++;
            }
        }
    }
    
    /* Sort rows by workload */
    size_t *row_indices = (size_t *)malloc(a->rows * sizeof(size_t));
    for (size_t i = 0; i < a->rows; i++) {
        row_indices[i] = i;
    }
    
    /* Sort in descending order of non-zeros */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = i + 1; j < a->rows; j++) {
            if (row_nnz[row_indices[i]] < row_nnz[row_indices[j]]) {
                size_t temp = row_indices[i];
                row_indices[i] = row_indices[j];
                row_indices[j] = temp;
            }
        }
    }
    
    /* Process rows in order of decreasing workload */
    #pragma omp parallel for schedule(dynamic, 1) if(a->rows > 10)
    for (size_t idx = 0; idx < a->rows; idx++) {
        size_t i = row_indices[idx];
        /* ... (process row i) ... */
    }
    
    /* Clean up */
    free(row_nnz);
    free(row_indices);
    
    return TENSOR_SUCCESS;
}
```

This approach estimates the workload for each row based on the number of non-zero elements, sorts rows by workload, and processes them in decreasing order of workload with dynamic scheduling.

### Nested Parallelism

For hierarchical workloads, we can use nested parallelism:

```c
/* Enable nested parallelism */
omp_set_nested(1);

/* Hierarchical tensor operation */
int tensor_hierarchical_op(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    /* ... (setup code) ... */
    
    /* Outer parallelism for blocks */
    #pragma omp parallel for if(num_blocks > 4)
    for (size_t block = 0; block < num_blocks; block++) {
        /* ... (block setup) ... */
        
        /* Inner parallelism for elements within a block */
        #pragma omp parallel for collapse(2) if(block_size > 32)
        for (size_t i = 0; i < block_size; i++) {
            for (size_t j = 0; j < block_size; j++) {
                /* ... (element computation) ... */
            }
        }
    }
    
    return TENSOR_SUCCESS;
}
```

Nested parallelism can be useful for complex workloads, but it adds overhead and should be used judiciously.

### Benchmarking Different Scheduling Strategies

Let's benchmark different scheduling strategies to find the optimal approach:

```c
/* benchmark_scheduling.c */
#include "tensor.h"
#include "tensor_ops.h"
#include <stdio.h>
#include <omp.h>

/* Matrix multiplication with specified scheduling */
double benchmark_matmul_scheduling(const tensor_t *a, const tensor_t *b, tensor_t *result, 
                                  const char *schedule, int chunk_size) {
    /* Warm up the cache */
    tensor_matmul(result, a, b);
    
    /* Measure performance */
    double start_time = omp_get_wtime();
    
    /* Perform matrix multiplication with specified scheduling */
    if (strcmp(schedule, "static") == 0) {
        #pragma omp parallel for collapse(2) schedule(static, chunk_size)
        for (size_t i = 0; i < a->rows; i++) {
            for (size_t j = 0; j < b->cols; j++) {
                tensor_elem_t sum = 0.0;
                for (size_t k = 0; k < a->cols; k++) {
                    size_t idx_a = i * a->row_stride + k * a->col_stride;
                    size_t idx_b = k * b->row_stride + j * b->col_stride;
                    sum += a->data[idx_a] * b->data[idx_b];
                }
                size_t idx_result = i * result->row_stride + j * result->col_stride;
                result->data[idx_result] = sum;
            }
        }
    } else if (strcmp(schedule, "dynamic") == 0) {
        #pragma omp parallel for collapse(2) schedule(dynamic, chunk_size)
        for (size_t i = 0; i < a->rows; i++) {
            for (size_t j = 0; j < b->cols; j++) {
                /* ... (same computation) ... */
            }
        }
    } else if (strcmp(schedule, "guided") == 0) {
        #pragma omp parallel for collapse(2) schedule(guided, chunk_size)
        for (size_t i = 0; i < a->rows; i++) {
            for (size_t j = 0; j < b->cols; j++) {
                /* ... (same computation) ... */
            }
        }
    }
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main() {
    tensor_t a, b, result;
    size_t size = 1000;
    
    /* Create tensors */
    tensor_create_ts(&a, size, size);
    tensor_create_ts(&b, size, size);
    tensor_create_ts(&result, size, size);
    
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
    
    /* Try different scheduling strategies */
    printf("Schedule | Chunk Size | Time (seconds)\n");
    printf("---------|------------|---------------\n");
    
    const char *schedules[] = {"static", "dynamic", "guided"};
    int chunk_sizes[] = {1, 4, 16, 64, 256};
    
    for (int s = 0; s < 3; s++) {
        for (int c = 0; c < 5; c++) {
            double time = benchmark_matmul_scheduling(&a, &b, &result, 
                                                    schedules[s], chunk_sizes[c]);
            printf("%8s | %10d | %15.6f\n", schedules[s], chunk_sizes[c], time);
        }
    }
    
    /* Clean up */
    tensor_free(&a);
    tensor_free(&b);
    tensor_free(&result);
    
    return 0;
}
```

This benchmark compares static, dynamic, and guided scheduling with different chunk sizes to find the optimal configuration for your hardware.

## Visualizing Parallel Tensor Operations

Let's visualize how parallel tensor operations work to better understand the concepts.

### Thread Distribution in Matrix Multiplication

```
Matrix Multiplication with 4 Threads:

Matrix A (4x4)            Matrix B (4x4)            Result Matrix C (4x4)
+----+----+----+----+     +----+----+----+----+     +----+----+----+----+
| A00| A01| A02| A03|     | B00| B01| B02| B03|     | C00| C01| C02| C03|
+----+----+----+----+     +----+----+----+----+     +----+----+----+----+
| A10| A11| A12| A13|     | B10| B11| B12| B13|     | C10| C11| C12| C13|
+----+----+----+----+     +----+----+----+----+     +----+----+----+----+
| A20| A21| A22| A23|     | B20| B21| B22| B23|     | C20| C21| C22| C23|
+----+----+----+----+     +----+----+----+----+     +----+----+----+----+
| A30| A31| A32| A33|     | B30| B31| B32| B33|     | C30| C31| C32| C33|
+----+----+----+----+     +----+----+----+----+     +----+----+----+----+

Thread Assignment (static scheduling):
Thread 0: C00, C01, C10, C11
Thread 1: C02, C03, C12, C13
Thread 2: C20, C21, C30, C31
Thread 3: C22, C23, C32, C33
```

### Blocked Matrix Multiplication with Parallel Processing

```
Blocked Matrix Multiplication with 4 Threads:

Matrix A (4x4)            Matrix B (4x4)            Result Matrix C (4x4)
+--------+--------+       +--------+--------+       +--------+--------+
|        |        |       |        |        |       |        |        |
| Block 0| Block 1|       | Block 0| Block 1|       | Block 0| Block 1|
|        |        |       |        |        |       |        |        |
+--------+--------+       +--------+--------+       +--------+--------+
|        |        |       |        |        |       |        |        |
| Block 2| Block 3|       | Block 2| Block 3|       | Block 2| Block 3|
|        |        |       |        |        |       |        |        |
+--------+--------+       +--------+--------+       +--------+--------+

Thread Assignment (block-level parallelism):
Thread 0: Process Block 0 of C
Thread 1: Process Block 1 of C
Thread 2: Process Block 2 of C
Thread 3: Process Block 3 of C
```

### Reduction Operation with Thread-Local Accumulation

```
Parallel Sum Reduction with 4 Threads:

Tensor Data (4x4):
+----+----+----+----+
| 1  | 2  | 3  | 4  |
+----+----+----+----+
| 5  | 6  | 7  | 8  |
+----+----+----+----+
| 9  | 10 | 11 | 12 |
+----+----+----+----+
| 13 | 14 | 15 | 16 |
+----+----+----+----+

Thread-Local Sums:
Thread 0: 1 + 2 + 5 + 6 = 14
Thread 1: 3 + 4 + 7 + 8 = 22
Thread 2: 9 + 10 + 13 + 14 = 46
Thread 3: 11 + 12 + 15 + 16 = 54

Final Reduction:
14 + 22 + 46 + 54 = 136
```

## Common Pitfalls and Debugging

Let's discuss some common issues you might encounter when parallelizing tensor operations.

### Race Conditions

Race conditions occur when multiple threads access and modify the same data concurrently. Symptoms include:

1. Results that change between runs
2. Results that differ from single-threaded execution
3. Crashes or hangs in parallel sections

To diagnose race conditions, you can use tools like Helgrind (part of Valgrind):

```bash
valgrind --tool=helgrind ./your_program
```

Common solutions include:

1. Using OpenMP's `reduction` clause for accumulation operations
2. Using `critical` sections or `atomic` updates for shared data
3. Ensuring each thread works on independent data

### False Sharing

False sharing occurs when threads access different variables that happen to be on the same cache line, causing cache coherence overhead. Symptoms include:

1. Poor scaling as the number of threads increases
2. Performance that's worse than expected

To mitigate false sharing:

1. Align thread-local data to cache line boundaries (typically 64 bytes)
2. Pad structures to avoid sharing cache lines

Here's an example of padding to avoid false sharing:

```c
/* Thread-local data with padding to avoid false sharing */
typedef struct {
    tensor_elem_t sum;
    char padding[64 - sizeof(tensor_elem_t)];  /* Pad to cache line size */
} thread_data_t;

tensor_elem_t tensor_sum_padded(const tensor_t *t) {
    int num_threads = omp_get_max_threads();
    thread_data_t *thread_data = (thread_data_t *)malloc(num_threads * sizeof(thread_data_t));
    
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].sum = 0.0;
    }
    
    #pragma omp parallel if(t->rows * t->cols > 1000)
    {
        int thread_id = omp_get_thread_num();
        
        #pragma omp for collapse(2) nowait
        for (size_t i = 0; i < t->rows; i++) {
            for (size_t j = 0; j < t->cols; j++) {
                size_t idx = i * t->row_stride + j * t->col_stride;
                thread_data[thread_id].sum += t->data[idx];
            }
        }
    }
    
    /* Combine thread-local results */
    tensor_elem_t total_sum = 0.0;
    for (int i = 0; i < num_threads; i++) {
        total_sum += thread_data[i].sum;
    }
    
    free(thread_data);
    return total_sum;
}
```

### Load Imbalance

Load imbalance occurs when some threads finish their work much earlier than others, leading to inefficient resource utilization. Symptoms include:

1. Some CPU cores showing high utilization while others are idle
2. Performance that doesn't scale linearly with the number of threads

To diagnose load imbalance, you can use tools like `perf` on Linux to see thread utilization:

```bash
perf record -g ./your_program
perf report
```

Solutions include:

1. Using dynamic or guided scheduling
2. Manually balancing workloads based on estimated computation time
3. Using smaller chunk sizes to distribute work more evenly

### Overhead from Thread Creation and Synchronization

Thread management overhead can outweigh the benefits of parallelization for small workloads. Symptoms include:

1. Parallel version being slower than sequential for small inputs
2. Performance that improves only for large inputs

To mitigate this overhead:

1. Use the `if` clause to only parallelize large workloads
2. Reuse threads for multiple operations instead of creating new ones
3. Reduce synchronization points by using `nowait` where possible

## Exercises

### Exercise 1: Implement a Parallel Tensor Convolution Function

Implement a function `tensor_convolve` that performs 2D convolution of a tensor with a kernel, using OpenMP for parallelization.

Hint: Convolution involves sliding a kernel over the input tensor and computing the sum of element-wise products.

Partial solution:

```c
int tensor_convolve(tensor_t *result, const tensor_t *input, const tensor_t *kernel) {
    /* Check dimensions */
    if (kernel->rows % 2 == 0 || kernel->cols % 2 == 0) {
        return TENSOR_INVALID_PARAMETER;  /* Kernel dimensions should be odd */
    }
    
    size_t result_rows = input->rows - kernel->rows + 1;
    size_t result_cols = input->cols - kernel->cols + 1;
    
    /* Create or resize result tensor */
    int status = tensor_create_ts(result, result_rows, result_cols);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Compute half sizes for kernel */
    size_t half_krows = kernel->rows / 2;
    size_t half_kcols = kernel->cols / 2;
    
    /* Perform convolution in parallel */
    #pragma omp parallel for collapse(2) if(result_rows * result_cols > 1000)
    for (size_t i = 0; i < result_rows; i++) {
        for (size_t j = 0; j < result_cols; j++) {
            tensor_elem_t sum = 0.0;
            
            /* Apply kernel */
            for (size_t ki = 0; ki < kernel->rows; ki++) {
                for (size_t kj = 0; kj < kernel->cols; kj++) {
                    size_t input_i = i + ki;
                    size_t input_j = j + kj;
                    
                    size_t kernel_idx = ki * kernel->row_stride + kj * kernel->col_stride;
                    size_t input_idx = input_i * input->row_stride + input_j * input->col_stride;
                    
                    sum += kernel->data[kernel_idx] * input->data[input_idx];
                }
            }
            
            size_t result_idx = i * result->row_stride + j * result->col_stride;
            result->data[result_idx] = sum;
        }
    }
    
    return TENSOR_SUCCESS;
}
```

### Exercise 2: Implement a Thread Pool for Tensor Operations

Implement a thread pool that can be used to execute multiple tensor operations concurrently, avoiding the overhead of repeatedly creating and destroying threads.

Hint: Use OpenMP tasks or implement your own thread pool with pthreads.

### Exercise 3: Benchmark Different Parallel Strategies for Tensor Transposition

Implement and benchmark different parallel strategies for tensor transposition, including:

1. Row-parallel transposition
2. Block-parallel transposition
3. Cache-oblivious recursive transposition

Compare their performance for different tensor sizes and numbers of threads.

Partial solution:

```c
/* Row-parallel transposition */
int tensor_transpose_row_parallel(tensor_t *result, const tensor_t *t) {
    /* Create result tensor with swapped dimensions */
    int status = tensor_create_ts(result, t->cols, t->rows);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* Transpose in parallel by rows */
    #pragma omp parallel for if(t->rows > 100)
    for (size_t i = 0; i < t->rows; i++) {
        for (size_t j = 0; j < t->cols; j++) {
            size_t src_idx = i * t->row_stride + j * t->col_stride;
            size_t dst_idx = j * result->row_stride + i * result->col_stride;
            result->data[dst_idx] = t->data[src_idx];
        }
    }
    
    return TENSOR_SUCCESS;
}

/* Block-parallel transposition */
int tensor_transpose_block_parallel(tensor_t *result, const tensor_t *t, size_t block_size) {
    /* Create result tensor with swapped dimensions */
    int status = tensor_create_ts(result, t->cols, t->rows);
    if (status != TENSOR_SUCCESS) {
        return status;
    }
    
    /* If block_size is 0, use a default */
    if (block_size == 0) {
        block_size = 64;
    }
    
    /* Calculate number of blocks */
    size_t num_row_blocks = (t->rows + block_size - 1) / block_size;
    size_t num_col_blocks = (t->cols + block_size - 1) / block_size;
    
    /* Transpose in parallel by blocks */
    #pragma omp parallel for collapse(2) if(num_row_blocks * num_col_blocks > 4)
    for (size_t i0 = 0; i0 < t->rows; i0 += block_size) {
        for (size_t j0 = 0; j0 < t->cols; j0 += block_size) {
            size_t i_end = (i0 + block_size < t->rows) ? i0 + block_size : t->rows;
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

## Summary and Key Takeaways

In this chapter, we've explored parallelizing tensor workloads with OpenMP:

- We implemented thread-safe tensor allocation and initialization to prevent race conditions.
- We parallelized core tensor operations using OpenMP directives, achieving significant speedups on multi-core systems.
- We addressed race conditions in accumulation operations using reduction clauses and thread-local storage.
- We explored techniques for balancing workloads across heterogeneous cores, including dynamic scheduling and manual load balancing.

Key takeaways:

1. **Thread Safety**: Ensure your tensor infrastructure is thread-safe before parallelizing operations.
2. **Parallelization Strategy**: Choose the right parallelization strategy based on the operation and data size.
3. **Race Conditions**: Be vigilant about race conditions in accumulation operations and use appropriate synchronization mechanisms.
4. **Load Balancing**: Use dynamic scheduling or manual load balancing to ensure efficient resource utilization.
5. **Overhead Awareness**: Be mindful of parallelization overhead and only parallelize operations that benefit from it.

By effectively parallelizing tensor operations, you can achieve near-linear speedups on multi-core systems, dramatically improving the performance of your tensor programming applications.

In the next chapter, we'll explore vectorizing tensor code with SIMD intrinsics to extract even more performance from modern CPUs.

## Further Reading

1. "Using OpenMP: Portable Shared Memory Parallel Programming" by Barbara Chapman, Gabriele Jost, and Ruud van der Pas - A comprehensive guide to OpenMP programming.

2. "Patterns for Parallel Programming" by Timothy G. Mattson, Beverly A. Sanders, and Berna L. Massingill - Explores patterns and best practices for parallel programming.

3. "OpenMP Application Programming Interface" (official specification) - The definitive reference for OpenMP directives and functions.