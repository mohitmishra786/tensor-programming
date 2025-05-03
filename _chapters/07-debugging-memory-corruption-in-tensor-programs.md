---
layout: chapter
title: Debugging Memory Corruption in Tensor Programs
number: 7
description: Tackle the most challenging bugs in tensor programming - memory corruption and leaks. This chapter equips you with powerful tools and techniques to find and fix these issues.
---

## You Will Learn To...

- Identify and diagnose common memory corruption issues in tensor programs
- Use Valgrind, AddressSanitizer, and other memory debugging tools effectively
- Implement defensive programming techniques to prevent memory errors
- Create custom debugging utilities specific to tensor operations
- Develop a systematic approach to tracking down elusive memory bugs

## Introduction

Last week, I spent three days tracking down a segmentation fault that only appeared when processing tensors larger than 10,000 elements. The culprit? A single off-by-one error in an index calculation that occasionally wrote past the end of an allocated buffer. Memory corruption bugs like these are among the most frustrating issues in C programming—they can manifest far from their cause, appear intermittently, and sometimes even seem to disappear when you add debugging code.

In tensor programming, these issues are particularly common due to the complex memory access patterns, pointer arithmetic, and large data structures involved. This chapter isn't just theoretical—it's drawn from years of painful debugging sessions and the techniques that actually saved real projects.

## Common Memory Corruption Issues in Tensor Programs

Before diving into debugging tools, let's examine the typical memory issues that plague tensor implementations:

### Buffer Overflows

The classic out-of-bounds access occurs frequently in tensor code due to index calculations:

```c
// Incorrect: potential buffer overflow
void tensor_add(Tensor *result, const Tensor *a, const Tensor *b) {
    int size = a->dims[0] * a->dims[1]; // What if tensor has more dimensions?
    for (int i = 0; i <= size; i++) {   // <= is wrong! Should be <
        result->data[i] = a->data[i] + b->data[i];
    }
}
```

This seemingly innocent mistake (using `<=` instead of `<`) writes one element past the allocated buffer, potentially corrupting heap metadata or other variables.

### Use-After-Free

Tensor operations often involve temporary buffers that get passed around between functions:

```c
// Dangerous: returning pointer to freed memory
float* compute_tensor_means(Tensor *t) {
    float *means = malloc(t->dims[0] * sizeof(float));
    // Calculate means for each row...
    return means;  // Caller must free this!
}

// Later in the code...
float *row_means = compute_tensor_means(&my_tensor);
// Use row_means...
// Oops! Forgot to free row_means (memory leak)

// Even worse: double free
float *means1 = compute_tensor_means(&tensor1);
float *means2 = means1;  // Alias
free(means1);
// ... later ...
free(means2);  // BOOM! Double free
```

### Uninitialized Memory

Failing to initialize tensor data before use leads to unpredictable behavior:

```c
// Allocates memory but doesn't initialize values
Tensor* create_tensor(int rows, int cols) {
    Tensor *t = malloc(sizeof(Tensor));
    t->dims[0] = rows;
    t->dims[1] = cols;
    t->data = malloc(rows * cols * sizeof(float));
    // Oops! Forgot to initialize t->data elements
    return t;
}
```

### Dimension Mismatch

Tensor-specific issues often involve dimension mismatches:

```c
// Assumes tensors have same dimensions without checking
void tensor_multiply(Tensor *result, const Tensor *a, const Tensor *b) {
    int size = a->dims[0] * a->dims[1];
    // No validation that b or result have the same dimensions!
    for (int i = 0; i < size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
}
```

## Essential Memory Debugging Tools

### Valgrind: Your First Line of Defense

Valgrind is an invaluable tool for detecting memory issues. It instruments your program at runtime to track memory operations.

#### Basic Valgrind Usage

```bash
# Compile with debugging symbols
gcc -g tensor_ops.c -o tensor_program -lm

# Run with Valgrind
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./tensor_program
```

Let's look at a typical Valgrind output for a tensor program with memory issues:

```
==12345== Invalid write of size 4
==12345==    at 0x4008F1: tensor_add (tensor_ops.c:45)
==12345==    by 0x400A3B: main (main.c:28)
==12345==  Address 0x5204040 is 0 bytes after a block of size 400 alloc'd
==12345==    at 0x4C2AB80: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x400812: create_tensor (tensor_ops.c:15)
==12345==    by 0x4009CF: main (main.c:22)
```

This output tells us we're writing past the end of an allocated buffer in the `tensor_add` function. The buffer was allocated in `create_tensor` and is 400 bytes (100 floats on a typical system).

#### Interpreting Common Valgrind Messages

- **Invalid read/write**: Accessing memory outside allocated blocks
- **Use of uninitialized value**: Using memory before setting a value
- **Memory leak**: Failing to free allocated memory
- **Source/destination overlap in memcpy**: Overlapping memory regions in copy operations

### AddressSanitizer (ASan): Faster Alternative to Valgrind

ASan is a compiler feature that instruments code at compile time, making it faster than Valgrind but with similar detection capabilities:

```bash
# Compile with AddressSanitizer
gcc -g -fsanitize=address -fno-omit-frame-pointer tensor_ops.c -o tensor_program -lm

# Run normally - errors will be reported automatically
./tensor_program
```

ASan output for the same buffer overflow might look like:

```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x614000000044 at pc 0x55a12a pc 0x55a12b bp 0x7ffd53a bp 0x7ffd53a sp 0x7ffd53a
WRITE of size 4 at 0x614000000044 thread T0
    #0 0x55a12a in tensor_add /home/user/tensor_ops.c:45:23
    #1 0x55a2bc in main /home/user/main.c:28:5

0x614000000044 is located 0 bytes after 400-byte region [0x614000000040,0x614000000044)
allocated by thread T0 here:
    #0 0x7f2a1d in malloc /usr/local/lib/clang/10.0.0/lib/linux/libclang_rt.asan-x86_64.a
    #1 0x55a0a1 in create_tensor /home/user/tensor_ops.c:15:14
    #2 0x55a23a in main /home/user/main.c:22:15
```

### UndefinedBehaviorSanitizer (UBSan)

UBSan catches undefined behavior like signed integer overflow, which can be particularly useful for tensor index calculations:

```bash
gcc -g -fsanitize=undefined tensor_ops.c -o tensor_program -lm
```

### Custom Guard Bytes

For situations where external tools aren't available, you can implement guard bytes around your tensor allocations:

```c
Tensor* create_tensor_with_guards(int rows, int cols) {
    Tensor *t = malloc(sizeof(Tensor));
    t->dims[0] = rows;
    t->dims[1] = cols;
    
    // Allocate extra space for guard bytes
    size_t data_size = rows * cols * sizeof(float);
    t->data = malloc(data_size + 2 * sizeof(uint32_t));
    
    // Set guard bytes at beginning and end
    uint32_t *guard_begin = (uint32_t*)t->data;
    *guard_begin = 0xDEADBEEF;  // Guard pattern
    
    // Actual data starts after guard
    t->data = (float*)(guard_begin + 1);
    
    // Set guard at end
    uint32_t *guard_end = (uint32_t*)(((char*)t->data) + data_size);
    *guard_end = 0xDEADBEEF;
    
    return t;
}

// Check guards when freeing
void free_tensor_with_guards(Tensor *t) {
    if (!t) return;
    
    // Get pointer to beginning guard
    uint32_t *guard_begin = ((uint32_t*)t->data) - 1;
    
    // Calculate pointer to end guard
    size_t data_size = t->dims[0] * t->dims[1] * sizeof(float);
    uint32_t *guard_end = (uint32_t*)(((char*)t->data) + data_size);
    
    // Check guards
    if (*guard_begin != 0xDEADBEEF || *guard_end != 0xDEADBEEF) {
        fprintf(stderr, "MEMORY CORRUPTION DETECTED in tensor at %p\n", t);
        if (*guard_begin != 0xDEADBEEF)
            fprintf(stderr, "  Beginning guard corrupted: 0x%08X\n", *guard_begin);
        if (*guard_end != 0xDEADBEEF)
            fprintf(stderr, "  Ending guard corrupted: 0x%08X\n", *guard_end);
        abort();  // Crash immediately to help with debugging
    }
    
    // Free the actual allocation (guard_begin)
    free(guard_begin);
    t->data = NULL;
    free(t);
}
```

## Systematic Debugging Approach

When facing memory corruption in tensor code, follow this systematic approach:

### 1. Reproduce the Issue Reliably

Create a minimal test case that consistently reproduces the problem:

```c
// Test case to reproduce buffer overflow
int main() {
    Tensor *a = create_tensor(10, 10);
    Tensor *b = create_tensor(10, 10);
    Tensor *result = create_tensor(10, 10);
    
    // Initialize with test data
    for (int i = 0; i < 100; i++) {
        a->data[i] = i;
        b->data[i] = i * 2;
    }
    
    // This should trigger the overflow
    tensor_add(result, a, b);
    
    // Print results to verify
    for (int i = 0; i < 100; i++) {
        printf("%f ", result->data[i]);
        if ((i+1) % 10 == 0) printf("\n");
    }
    
    free_tensor(a);
    free_tensor(b);
    free_tensor(result);
    return 0;
}
```

### 2. Add Defensive Validation

Implement defensive checks in your tensor functions:

```c
void tensor_add(Tensor *result, const Tensor *a, const Tensor *b) {
    // Validate inputs
    if (!result || !a || !b) {
        fprintf(stderr, "Error: NULL tensor pointer in tensor_add\n");
        return;
    }
    
    // Validate dimensions match
    if (a->dims[0] != b->dims[0] || a->dims[1] != b->dims[1] ||
        result->dims[0] != a->dims[0] || result->dims[1] != a->dims[1]) {
        fprintf(stderr, "Error: Dimension mismatch in tensor_add\n");
        fprintf(stderr, "  A: %d x %d\n", a->dims[0], a->dims[1]);
        fprintf(stderr, "  B: %d x %d\n", b->dims[0], b->dims[1]);
        fprintf(stderr, "  Result: %d x %d\n", result->dims[0], result->dims[1]);
        return;
    }
    
    // Calculate size correctly
    int size = a->dims[0] * a->dims[1];
    
    // Perform operation with correct bounds
    for (int i = 0; i < size; i++) {  // Note: < not <=
        result->data[i] = a->data[i] + b->data[i];
    }
}
```

### 3. Use Logging for Tensor Operations

Implement a logging system to track tensor operations:

```c
// Define log levels
enum LogLevel { ERROR, WARNING, INFO, DEBUG };
enum LogLevel current_log_level = INFO;

void tensor_log(enum LogLevel level, const char *format, ...) {
    if (level > current_log_level) return;
    
    va_list args;
    va_start(args, format);
    
    // Print timestamp and log level
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char time_str[20];
    strftime(time_str, 20, "%Y-%m-%d %H:%M:%S", tm_info);
    
    const char *level_str = "UNKNOWN";
    switch (level) {
        case ERROR: level_str = "ERROR"; break;
        case WARNING: level_str = "WARNING"; break;
        case INFO: level_str = "INFO"; break;
        case DEBUG: level_str = "DEBUG"; break;
    }
    
    fprintf(stderr, "[%s] %s: ", time_str, level_str);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    
    va_end(args);
}

// Example usage
void tensor_add_with_logging(Tensor *result, const Tensor *a, const Tensor *b) {
    tensor_log(DEBUG, "tensor_add called with tensors at %p, %p, %p", result, a, b);
    tensor_log(DEBUG, "Dimensions: result=%dx%d, a=%dx%d, b=%dx%d", 
              result->dims[0], result->dims[1],
              a->dims[0], a->dims[1],
              b->dims[0], b->dims[1]);
              
    // ... rest of function ...
    
    tensor_log(DEBUG, "tensor_add completed successfully");
}
```

### 4. Implement Tensor Metadata Tracking

Keep track of tensor allocations to detect leaks and double-frees:

```c
// Global tracking structure
#define MAX_TRACKED_TENSORS 1000
struct {
    Tensor *tensors[MAX_TRACKED_TENSORS];
    char *allocation_sites[MAX_TRACKED_TENSORS];
    int count;
} tensor_tracker;

// Initialize tracker
void init_tensor_tracker() {
    memset(&tensor_tracker, 0, sizeof(tensor_tracker));
}

// Record tensor allocation
void track_tensor(Tensor *t, const char *file, int line) {
    if (tensor_tracker.count >= MAX_TRACKED_TENSORS) {
        tensor_log(ERROR, "Too many tensors to track!");
        return;
    }
    
    // Allocate and store location string
    char *location = malloc(strlen(file) + 20);
    sprintf(location, "%s:%d", file, line);
    
    tensor_tracker.tensors[tensor_tracker.count] = t;
    tensor_tracker.allocation_sites[tensor_tracker.count] = location;
    tensor_tracker.count++;
    
    tensor_log(DEBUG, "Tracked new tensor %p allocated at %s", t, location);
}

// Remove tensor from tracking
void untrack_tensor(Tensor *t) {
    for (int i = 0; i < tensor_tracker.count; i++) {
        if (tensor_tracker.tensors[i] == t) {
            tensor_log(DEBUG, "Untracking tensor %p allocated at %s", 
                      t, tensor_tracker.allocation_sites[i]);
            
            free(tensor_tracker.allocation_sites[i]);
            
            // Move the last element to this position
            tensor_tracker.tensors[i] = tensor_tracker.tensors[tensor_tracker.count-1];
            tensor_tracker.allocation_sites[i] = tensor_tracker.allocation_sites[tensor_tracker.count-1];
            tensor_tracker.count--;
            return;
        }
    }
    
    tensor_log(ERROR, "Attempted to untrack unknown tensor %p - possible double free!", t);
}

// Print leak report
void tensor_tracker_report() {
    if (tensor_tracker.count == 0) {
        tensor_log(INFO, "No tensor leaks detected");
        return;
    }
    
    tensor_log(WARNING, "Detected %d leaked tensors:", tensor_tracker.count);
    for (int i = 0; i < tensor_tracker.count; i++) {
        Tensor *t = tensor_tracker.tensors[i];
        tensor_log(WARNING, "  Tensor %p (%dx%d) allocated at %s", 
                  t, t->dims[0], t->dims[1], tensor_tracker.allocation_sites[i]);
    }
}

// Macro to automatically capture file and line
#define CREATE_TENSOR(rows, cols) \
    create_tensor_tracked((rows), (cols), __FILE__, __LINE__)

// Tracked version of tensor creation
Tensor* create_tensor_tracked(int rows, int cols, const char *file, int line) {
    Tensor *t = create_tensor(rows, cols);
    track_tensor(t, file, line);
    return t;
}

// Tracked version of tensor freeing
void free_tensor_tracked(Tensor *t) {
    if (!t) return;
    untrack_tensor(t);
    free_tensor(t);
}
```

## Real-World Debugging Session

Let's walk through a real debugging session for a memory corruption issue in a tensor program:

```
$ valgrind --leak-check=full ./tensor_program
==12345== Memcheck, a memory error detector
==12345== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==12345== Using Valgrind-3.15.0 and LibVEX; rerun with -h for copyright info
==12345== Command: ./tensor_program
==12345== 
Initializing tensors...
Performing tensor operations...
==12345== Invalid write of size 4
==12345==    at 0x109BB2: tensor_matmul (tensor_ops.c:87)
==12345==    by 0x109F8A: main (main.c:42)
==12345==  Address 0x4A49C90 is 0 bytes after a block of size 400 alloc'd
==12345==    at 0x483B7F3: malloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x109936: create_tensor (tensor_ops.c:15)
==12345==    by 0x109EE2: main (main.c:38)
==12345== 
Program completed.
==12345== 
==12345== HEAP SUMMARY:
==12345==     in use at exit: 1,248 bytes in 3 blocks
==12345==   total heap usage: 4 allocs, 1 frees, 2,048 bytes allocated
==12345== 
==12345== 416 bytes in 1 blocks are definitely lost in loss record 2 of 3
==12345==    at 0x483B7F3: malloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x109936: create_tensor (tensor_ops.c:15)
==12345==    by 0x109EE2: main (main.c:36)
==12345== 
==12345== LEAK SUMMARY:
==12345==    definitely lost: 416 bytes in 1 blocks
==12345==    indirectly lost: 0 bytes in 0 blocks
==12345==      possibly lost: 0 bytes in 0 blocks
==12345==    still reachable: 832 bytes in 2 blocks
==12345==         suppressed: 0 bytes in 0 blocks
==12345== 
==12345== For lists of detected and suppressed errors, rerun with: -s
==12345== ERROR SUMMARY: 2 errors from 2 contexts (suppressed: 0 from 0)
```

Analyzing this output:

1. We have a buffer overflow in `tensor_matmul` at line 87
2. We're writing past the end of a 400-byte block (100 floats)
3. We also have a memory leak of 416 bytes (a Tensor struct + data)

Let's look at the problematic function:

```c
// Original function with bug
void tensor_matmul(Tensor *result, const Tensor *a, const Tensor *b) {
    // Assume matrices: a is m×n, b is n×p, result is m×p
    int m = a->dims[0];
    int n = a->dims[1];
    int p = b->dims[1];
    
    // No dimension validation!
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a->data[i * n + k] * b->data[k * p + j];
            }
            // Bug: result->data[i * p + j] can go out of bounds if
            // result dimensions are incorrect
            result->data[i * p + j] = sum;
        }
    }
}
```

The fixed version would be:

```c
// Fixed version with validation
void tensor_matmul(Tensor *result, const Tensor *a, const Tensor *b) {
    // Validate inputs
    if (!result || !a || !b) {
        tensor_log(ERROR, "NULL tensor pointer in tensor_matmul");
        return;
    }
    
    // Assume matrices: a is m×n, b is n×p, result is m×p
    int m = a->dims[0];
    int n = a->dims[1];
    int p = b->dims[1];
    
    // Validate dimensions
    if (b->dims[0] != n || result->dims[0] != m || result->dims[1] != p) {
        tensor_log(ERROR, "Dimension mismatch in tensor_matmul");
        tensor_log(ERROR, "  A: %d×%d, B: %d×%d, Result: %d×%d", 
                  m, n, b->dims[0], p, result->dims[0], result->dims[1]);
        tensor_log(ERROR, "  Expected: A: %d×%d, B: %d×%d, Result: %d×%d",
                  m, n, n, p, m, p);
        return;
    }
    
    // Initialize result to zeros
    memset(result->data, 0, m * p * sizeof(float));
    
    // Perform matrix multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a->data[i * n + k] * b->data[k * p + j];
            }
            result->data[i * p + j] = sum;
        }
    }
    
    tensor_log(DEBUG, "tensor_matmul completed successfully");
}
```

## Debugging Memory Corruption with GDB

For particularly tricky issues, GDB can help set watchpoints on memory addresses:

```bash
# Compile with debugging symbols
gcc -g tensor_ops.c -o tensor_program -lm

# Run with GDB
gdb ./tensor_program
```

In GDB, you can set a watchpoint on a specific memory address:

```
(gdb) start
(gdb) break create_tensor
(gdb) continue
(gdb) finish
(gdb) print t->data
$1 = (float *) 0x55555576a2a0
(gdb) watch *((float *) 0x55555576a2a0 + 99)
(gdb) continue
```

This will stop execution whenever the last element of the tensor is modified.

## Diagram: Memory Debugging Workflow

```
+----------------+     +-----------------+     +------------------+
| Compile with   |     | Run with        |     | Analyze output   |
| debugging flags|---->| memory checker  |---->| for error        |
| -g -O0         |     | (Valgrind/ASan) |     | locations        |
+----------------+     +-----------------+     +------------------+
                                                       |
+----------------+     +-----------------+     +------------------+
| Fix issue and  |     | Add defensive   |     | Locate source   |
| verify with    |<----| checks and      |<----| of corruption in |
| memory checker |     | logging         |     | identified area  |
+----------------+     +-----------------+     +------------------+
```

## Diagram: Tensor Memory Layout with Guard Bytes

```
+----------------+----------------+------------------+----------------+
| Tensor struct  | Guard bytes    | Actual tensor    | Guard bytes    |
| (metadata)     | (0xDEADBEEF)   | data (float[])   | (0xDEADBEEF)   |
+----------------+----------------+------------------+----------------+
      ^                 ^                 ^                 ^
      |                 |                 |                 |
   t pointer      guard_begin        t->data          guard_end
```

## Diagram: Valgrind Detection Process

```
+-------------+     +----------------+     +----------------+
| Your Program|     | Valgrind       |     | Shadow Memory  |
| tensor_ops.c|---->| Instrumentation|---->| Tracking       |
+-------------+     +----------------+     +----------------+
                            |                     |
                            v                     v
                    +----------------+     +----------------+
                    | Memory Access  |     | Error Detection|
                    | Monitoring    |---->| & Reporting    |
                    +----------------+     +----------------+
```

## Common Pitfalls and Solutions

### Pitfall 1: Inconsistent Dimension Order

Mixing row-major and column-major indexing:

```c
// Inconsistent indexing
float get_element_row_major(Tensor *t, int i, int j) {
    return t->data[i * t->dims[1] + j];  // Row-major
}

float get_element_col_major(Tensor *t, int i, int j) {
    return t->data[j * t->dims[0] + i];  // Column-major
}

// Solution: Standardize on one approach and document it
typedef enum { ROW_MAJOR, COLUMN_MAJOR } TensorLayout;

float get_element(Tensor *t, int i, int j, TensorLayout layout) {
    if (layout == ROW_MAJOR) {
        return t->data[i * t->dims[1] + j];
    } else {
        return t->data[j * t->dims[0] + i];
    }
}
```

### Pitfall 2: Incorrect Tensor Reshaping

```c
// Dangerous: doesn't check total size
void reshape_tensor(Tensor *t, int new_rows, int new_cols) {
    // No validation that new_rows * new_cols == t->dims[0] * t->dims[1]
    t->dims[0] = new_rows;
    t->dims[1] = new_cols;
}

// Solution: Validate total size remains the same
bool reshape_tensor_safe(Tensor *t, int new_rows, int new_cols) {
    int old_size = t->dims[0] * t->dims[1];
    int new_size = new_rows * new_cols;
    
    if (old_size != new_size) {
        tensor_log(ERROR, "Cannot reshape %dx%d tensor to %dx%d (size mismatch)",
                  t->dims[0], t->dims[1], new_rows, new_cols);
        return false;
    }
    
    t->dims[0] = new_rows;
    t->dims[1] = new_cols;
    return true;
}
```

### Pitfall 3: Forgetting to Free Temporary Tensors

```c
// Memory leak: temporary tensor never freed
Tensor* compute_tensor_sum(Tensor *a, Tensor *b) {
    Tensor *result = create_tensor(a->dims[0], a->dims[1]);
    tensor_add(result, a, b);
    return result;
}

// Later in code
Tensor *a = create_tensor(100, 100);
Tensor *b = create_tensor(100, 100);
// Initialize a and b...

// This creates a temporary result that's never freed
Tensor *c = compute_tensor_sum(a, b);
Tensor *d = compute_tensor_sum(c, b);  // Another leak

// Solution: Document ownership transfer
// In header: "// Caller takes ownership of returned tensor and must free it"
// And in calling code:
Tensor *c = compute_tensor_sum(a, b);
Tensor *d = compute_tensor_sum(c, b);
free_tensor(c);  // Free intermediate result
free_tensor(d);  // Free final result
```

## Practical Exercises

### Exercise 1: Implement a Memory-Safe Tensor Library

Create a tensor library with built-in memory safety features:

1. Implement the `Tensor` struct with guard bytes
2. Create functions for tensor creation, operations, and destruction
3. Add validation for all tensor operations
4. Implement a tensor tracking system to detect leaks

**Partial Solution:**

```c
// tensor_safe.h
#ifndef TENSOR_SAFE_H
#define TENSOR_SAFE_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int dims[2];           // For simplicity, just 2D tensors
    float *data;           // Actual data pointer (points to data after guard)
    uint32_t *guard_begin; // Internal use only
    bool tracked;          // Whether this tensor is being tracked
} Tensor;

// Creation and destruction
Tensor* tensor_create(int rows, int cols);
void tensor_free(Tensor *t);

// Basic operations
bool tensor_add(Tensor *result, const Tensor *a, const Tensor *b);
bool tensor_multiply(Tensor *result, const Tensor *a, const Tensor *b);

// Debugging helpers
void tensor_print(const Tensor *t, const char *name);
bool tensor_validate(const Tensor *t);  // Check guard bytes

// Memory tracking
void tensor_tracking_enable(void);
void tensor_tracking_disable(void);
void tensor_tracking_report(void);

#endif // TENSOR_SAFE_H
```

### Exercise 2: Debug a Tensor Program with Memory Corruption

Given the following program with memory corruption issues, use the techniques from this chapter to find and fix the bugs:

```c
// buggy_tensor.c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int dims[2];
    float *data;
} Tensor;

Tensor* create_tensor(int rows, int cols) {
    Tensor *t = malloc(sizeof(Tensor));
    t->dims[0] = rows;
    t->dims[1] = cols;
    t->data = malloc(rows * cols * sizeof(float));
    return t;
}

void free_tensor(Tensor *t) {
    free(t->data);
    free(t);
}

void tensor_transpose(Tensor *result, const Tensor *input) {
    // Bug 1: No dimension validation
    
    int rows = input->dims[0];
    int cols = input->dims[1];
    
    for (int i = 0; i <= rows; i++) {  // Bug 2: <= should be <
        for (int j = 0; j < cols; j++) {
            result->data[j * rows + i] = input->data[i * cols + j];
        }
    }
}

int main() {
    Tensor *a = create_tensor(3, 4);
    
    // Initialize with values
    for (int i = 0; i < 12; i++) {
        a->data[i] = i + 1;
    }
    
    // Print original
    printf("Original:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%5.1f ", a->data[i * 4 + j]);
        }
        printf("\n");
    }
    
    // Create result tensor with wrong dimensions
    Tensor *result = create_tensor(4, 2);  // Bug 3: Should be 4x3
    
    // Transpose
    tensor_transpose(result, a);
    
    // Print result
    printf("\nTransposed:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {  // Bug 4: Accessing out of bounds
            printf("%5.1f ", result->data[i * 3 + j]);
        }
        printf("\n");
    }
    
    free_tensor(a);
    // Bug 5: Missing free(result)
    
    return 0;
}
```

**Hints:**
1. Use Valgrind or AddressSanitizer to identify memory issues
2. Check for off-by-one errors in loops
3. Verify tensor dimensions match the expected operation
4. Look for memory leaks

### Exercise 3: Create a Custom Memory Debugging Tool for Tensors

Implement a specialized memory debugging tool for tensor operations that:

1. Tracks all tensor allocations and deallocations
2. Validates tensor dimensions before operations
3. Checks for buffer overflows using guard bytes
4. Generates a report of memory usage and potential issues

**Partial Solution:**

```c
// Start with this skeleton and implement the missing functions

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define GUARD_PATTERN 0xDEADBEEF
#define MAX_TRACKED_TENSORS 1000

typedef struct {
    int dims[2];
    float *data;
    // Internal tracking fields
    uint32_t *guard_begin;
    uint32_t *guard_end;
    const char *alloc_file;
    int alloc_line;
} DebugTensor;

// Global tracking state
static struct {
    DebugTensor *tensors[MAX_TRACKED_TENSORS];
    int count;
    bool enabled;
} tensor_tracker = {0};

// Enable/disable tracking
void tensor_debug_enable() {
    tensor_tracker.enabled = true;
}

void tensor_debug_disable() {
    tensor_tracker.enabled = false;
}

// Create a tracked tensor
DebugTensor* tensor_debug_create(int rows, int cols, const char *file, int line) {
    // TODO: Implement tensor creation with guard bytes and tracking
}

// Free a tracked tensor
void tensor_debug_free(DebugTensor *t, const char *file, int line) {
    // TODO: Implement tensor freeing with tracking and validation
}

// Validate a tensor's guard bytes
bool tensor_debug_validate(DebugTensor *t) {
    // TODO: Implement guard byte validation
}

// Generate a memory report
void tensor_debug_report() {
    // TODO: Implement memory usage report
}

// Validate dimensions for an operation
bool tensor_debug_validate_op(const char *op_name, DebugTensor *result, 
                             DebugTensor *a, DebugTensor *b) {
    // TODO: Implement dimension validation for common operations
}
```

## Summary

In this chapter, we've explored the challenging world of memory corruption in tensor programs and learned systematic approaches to diagnose and fix these issues:

- Memory corruption in tensor code often stems from buffer overflows, use-after-free errors, uninitialized memory, and dimension mismatches
- Tools like Valgrind, AddressSanitizer, and UndefinedBehaviorSanitizer can help detect memory issues at runtime
- Custom techniques like guard bytes and tensor tracking provide additional protection when external tools aren't available
- A systematic debugging approach involves reproducing the issue, adding defensive validation, implementing logging, and tracking tensor metadata
- Real-world debugging sessions often require a combination of tools and techniques to track down elusive memory bugs

By applying these techniques, you can build more robust tensor libraries that handle memory safely and reliably, even when processing large datasets or performing complex operations.

## Further Reading

1. Valgrind Documentation: [https://valgrind.org/docs/manual/mc-manual.html](https://valgrind.org/docs/manual/mc-manual.html)
2. AddressSanitizer Documentation: [https://github.com/google/sanitizers/wiki/AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer)
3. "Debugging with GDB" Manual: [https://sourceware.org/gdb/current/onlinedocs/gdb/](https://sourceware.org/gdb/current/onlinedocs/gdb/)