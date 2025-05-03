---
layout: chapter
title: Deploying Tensor Applications on Embedded Systems
number: 10
description: Adapt your tensor code for resource-constrained environments. Learn specialized techniques for embedded systems where memory and processing power are limited.
---

## You Will Learn To...

- Cross-compile tensor applications for ARM and other embedded architectures
- Implement fixed-point arithmetic for environments without FPU support
- Optimize tensor operations for memory-constrained devices
- Quantize floating-point models to integer precision
- Profile and optimize tensor code for real-time performance
- Design efficient inference pipelines for embedded applications

## Introduction

I remember the first time I tried to deploy a tensor application on an embedded device. I had spent weeks optimizing my code on my development machine—a beefy workstation with 32GB of RAM and a modern CPU. Everything ran beautifully. Then I moved the code to a Raspberry Pi, and it crawled to a halt. The memory usage exploded, and operations that took milliseconds on my workstation took seconds on the Pi.

That painful experience taught me that deploying tensor applications on embedded systems requires a fundamentally different approach than developing on desktop or server environments. You can't just compile your code and expect it to work efficiently—you need to rethink your algorithms, data structures, and optimization strategies.

In this chapter, we'll explore the techniques and tools for deploying tensor applications on embedded systems. We'll cover cross-compilation, fixed-point arithmetic, quantization, and memory optimization. By the end, you'll be able to take the tensor library we've built throughout this book and adapt it for efficient execution on resource-constrained devices.

## Understanding Embedded Constraints

Before diving into implementation details, let's understand the constraints we're working with:

### Hardware Limitations

Embedded systems typically have:

- Limited RAM (often measured in KB rather than GB)
- Slower CPUs with simpler architectures
- Limited or no floating-point hardware (FPU)
- Power constraints (battery-operated devices)
- Limited storage for code and data

### Real-time Requirements

Many embedded applications have real-time constraints:

- Predictable execution time is often more important than average performance
- Garbage collection or memory allocation during critical sections is unacceptable
- Missed deadlines can have serious consequences (e.g., in control systems)

### Development Challenges

- Cross-compilation complexity
- Limited debugging tools
- Platform-specific optimizations
- Testing on target hardware

With these constraints in mind, let's start by setting up our cross-compilation environment.

## Setting Up a Cross-Compilation Environment

Cross-compilation allows us to build code on our development machine that will run on a different target architecture. This is essential for embedded development since we typically can't compile large codebases directly on the target device.

### Installing a Cross-Compiler

For ARM-based systems (like Raspberry Pi, many microcontrollers, etc.), we'll use the GNU ARM toolchain:

```bash
# On Ubuntu/Debian
sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf

# On macOS with Homebrew
brew install arm-linux-gnueabihf-binutils
```

For other architectures, you'll need the appropriate toolchain. For example, for RISC-V:

```bash
# On Ubuntu/Debian
sudo apt-get install gcc-riscv64-linux-gnu
```

### Creating a Cross-Compilation Makefile

Let's create a Makefile that supports both native and cross-compilation:

```makefile
# Default to native compilation
TARGET_ARCH ?= native

# Compiler and flags
ifeq ($(TARGET_ARCH),native)
    CC = gcc
    AR = ar
else ifeq ($(TARGET_ARCH),arm)
    CC = arm-linux-gnueabihf-gcc
    AR = arm-linux-gnueabihf-ar
else ifeq ($(TARGET_ARCH),riscv)
    CC = riscv64-linux-gnu-gcc
    AR = riscv64-linux-gnu-ar
else
    $(error Unsupported TARGET_ARCH: $(TARGET_ARCH))
endif

# Common flags
CFLAGS = -Wall -Wextra -std=c11

# Architecture-specific flags
ifeq ($(TARGET_ARCH),native)
    CFLAGS += -march=native -O3
else ifeq ($(TARGET_ARCH),arm)
    CFLAGS += -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard -O3
else ifeq ($(TARGET_ARCH),riscv)
    CFLAGS += -march=rv64gc -O3
endif

# Source files
SRC = tensor.c fixed_point.c quantization.c
OBJ = $(SRC:.c=.o)

# Library name
LIB = libtensor.a

# Build the library
$(LIB): $(OBJ)
	$(AR) rcs $@ $^

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJ) $(LIB)

# Example target
example: example.c $(LIB)
	$(CC) $(CFLAGS) $^ -o $@ -lm

.PHONY: clean
```

To use this Makefile, you can specify the target architecture:

```bash
# For native compilation
make

# For ARM cross-compilation
make TARGET_ARCH=arm

# For RISC-V cross-compilation
make TARGET_ARCH=riscv
```

### Testing Cross-Compiled Code

After cross-compiling, you need to test your code on the target device or using an emulator:

```bash
# Copy to Raspberry Pi (example)
scp example pi@raspberrypi.local:~/

# Or use QEMU for emulation
qemu-arm -L /usr/arm-linux-gnueabihf ./example
```

Now that we have our cross-compilation environment set up, let's adapt our tensor library for embedded systems.

## Implementing Fixed-Point Arithmetic

Many embedded systems lack floating-point hardware, making floating-point operations slow. Fixed-point arithmetic provides a faster alternative by representing fractional numbers using integers.

### Fixed-Point Representation

In fixed-point representation, we allocate a fixed number of bits for the integer part and the fractional part. For example, in a Q16.16 format, we use 16 bits for the integer part and 16 bits for the fractional part, giving us a 32-bit number overall.

Let's define our fixed-point type:

```c
#include <stdint.h>

// Q16.16 fixed-point format
typedef int32_t fixed_t;

// Number of fractional bits
#define FIXED_FRAC_BITS 16

// Convert float to fixed-point
fixed_t float_to_fixed(float f) {
    return (fixed_t)(f * (1 << FIXED_FRAC_BITS));
}

// Convert fixed-point to float
float fixed_to_float(fixed_t f) {
    return ((float)f) / (1 << FIXED_FRAC_BITS);
}
```

### Basic Fixed-Point Operations

Now let's implement basic arithmetic operations for fixed-point numbers:

```c
// Addition: straightforward integer addition
fixed_t fixed_add(fixed_t a, fixed_t b) {
    return a + b;
}

// Subtraction: straightforward integer subtraction
fixed_t fixed_sub(fixed_t a, fixed_t b) {
    return a - b;
}

// Multiplication: need to handle the fractional bits
fixed_t fixed_mul(fixed_t a, fixed_t b) {
    // Use 64-bit intermediate to prevent overflow
    int64_t result = (int64_t)a * (int64_t)b;
    // Shift right to account for the extra fractional bits
    return (fixed_t)(result >> FIXED_FRAC_BITS);
}

// Division: need to handle the fractional bits
fixed_t fixed_div(fixed_t a, fixed_t b) {
    // Pre-shift a to prevent underflow
    int64_t result = ((int64_t)a << FIXED_FRAC_BITS) / (int64_t)b;
    return (fixed_t)result;
}
```

### Fixed-Point Tensor Implementation

Now let's create a fixed-point version of our tensor structure:

```c
typedef struct {
    int ndim;           // Number of dimensions
    int* dims;          // Size of each dimension
    int size;           // Total number of elements
    fixed_t* data;      // Pointer to data
} fixed_tensor_t;

// Create a new fixed-point tensor
fixed_tensor_t* fixed_tensor_create(int ndim, int* dims) {
    fixed_tensor_t* tensor = (fixed_tensor_t*)malloc(sizeof(fixed_tensor_t));
    if (!tensor) return NULL;
    
    tensor->ndim = ndim;
    tensor->dims = (int*)malloc(ndim * sizeof(int));
    if (!tensor->dims) {
        free(tensor);
        return NULL;
    }
    
    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->dims[i] = dims[i];
        tensor->size *= dims[i];
    }
    
    tensor->data = (fixed_t*)calloc(tensor->size, sizeof(fixed_t));
    if (!tensor->data) {
        free(tensor->dims);
        free(tensor);
        return NULL;
    }
    
    return tensor;
}

// Free a fixed-point tensor
void fixed_tensor_free(fixed_tensor_t* tensor) {
    if (!tensor) return;
    free(tensor->dims);
    free(tensor->data);
    free(tensor);
}

// Convert a float tensor to fixed-point
fixed_tensor_t* tensor_to_fixed(tensor_t* float_tensor) {
    fixed_tensor_t* fixed_tensor = fixed_tensor_create(float_tensor->ndim, float_tensor->dims);
    if (!fixed_tensor) return NULL;
    
    for (int i = 0; i < float_tensor->size; i++) {
        fixed_tensor->data[i] = float_to_fixed(float_tensor->data[i]);
    }
    
    return fixed_tensor;
}

// Convert a fixed-point tensor to float
tensor_t* fixed_to_tensor(fixed_tensor_t* fixed_tensor) {
    tensor_t* float_tensor = tensor_create(fixed_tensor->ndim, fixed_tensor->dims);
    if (!float_tensor) return NULL;
    
    for (int i = 0; i < fixed_tensor->size; i++) {
        float_tensor->data[i] = fixed_to_float(fixed_tensor->data[i]);
    }
    
    return float_tensor;
}
```

### Implementing Tensor Operations with Fixed-Point

Let's implement some basic tensor operations using fixed-point arithmetic:

```c
// Element-wise addition
void fixed_tensor_add(fixed_tensor_t* a, fixed_tensor_t* b, fixed_tensor_t* result) {
    // Assume dimensions are compatible
    for (int i = 0; i < result->size; i++) {
        result->data[i] = fixed_add(a->data[i], b->data[i]);
    }
}

// Element-wise multiplication
void fixed_tensor_mul(fixed_tensor_t* a, fixed_tensor_t* b, fixed_tensor_t* result) {
    // Assume dimensions are compatible
    for (int i = 0; i < result->size; i++) {
        result->data[i] = fixed_mul(a->data[i], b->data[i]);
    }
}

// Matrix multiplication
void fixed_tensor_matmul(fixed_tensor_t* a, fixed_tensor_t* b, fixed_tensor_t* result) {
    // Assume a is [M,K] and b is [K,N], result is [M,N]
    int M = a->dims[0];
    int K = a->dims[1];
    int N = b->dims[1];
    
    // Initialize result to zero
    memset(result->data, 0, result->size * sizeof(fixed_t));
    
    // Naive matrix multiplication
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            fixed_t sum = 0;
            for (int k = 0; k < K; k++) {
                fixed_t a_val = a->data[m * K + k];
                fixed_t b_val = b->data[k * N + n];
                sum = fixed_add(sum, fixed_mul(a_val, b_val));
            }
            result->data[m * N + n] = sum;
        }
    }
}
```

### Optimizing Fixed-Point Operations

Fixed-point operations can be further optimized for specific architectures. For example, on ARM Cortex-M4 and above, we can use the DSP extension:

```c
#ifdef __ARM_FEATURE_DSP
#include <arm_math.h>

// Optimized matrix multiplication using ARM DSP library
void fixed_tensor_matmul_optimized(fixed_tensor_t* a, fixed_tensor_t* b, fixed_tensor_t* result) {
    // Assume a is [M,K] and b is [K,N], result is [M,N]
    int M = a->dims[0];
    int K = a->dims[1];
    int N = b->dims[1];
    
    // Use ARM DSP matrix multiplication
    arm_matrix_instance_q31 a_matrix;
    arm_matrix_instance_q31 b_matrix;
    arm_matrix_instance_q31 result_matrix;
    
    arm_mat_init_q31(&a_matrix, M, K, (q31_t*)a->data);
    arm_mat_init_q31(&b_matrix, K, N, (q31_t*)b->data);
    arm_mat_init_q31(&result_matrix, M, N, (q31_t*)result->data);
    
    arm_mat_mult_q31(&a_matrix, &b_matrix, &result_matrix);
}
#endif
```

Fixed-point arithmetic is a powerful technique for embedded systems, but it requires careful handling of precision and range. Let's now look at another approach: quantization.

## Quantizing Tensor Operations

Quantization is the process of mapping floating-point values to integers with lower precision. Unlike fixed-point, which uses a fixed scaling factor, quantization can adapt the scaling factor based on the data range.

### Linear Quantization

The simplest form of quantization is linear quantization, where we map a floating-point range [min, max] to an integer range [0, 255] (for 8-bit quantization):

```c
typedef struct {
    uint8_t* data;      // Quantized data
    float scale;        // Scale factor: float = (int - zero_point) * scale
    uint8_t zero_point; // Value that represents 0 in the original data
    int size;           // Number of elements
} quant_tensor_t;

// Quantize a float tensor to 8-bit
quant_tensor_t* tensor_quantize(tensor_t* float_tensor) {
    quant_tensor_t* q_tensor = (quant_tensor_t*)malloc(sizeof(quant_tensor_t));
    if (!q_tensor) return NULL;
    
    // Find min and max values
    float min_val = float_tensor->data[0];
    float max_val = float_tensor->data[0];
    
    for (int i = 1; i < float_tensor->size; i++) {
        if (float_tensor->data[i] < min_val) min_val = float_tensor->data[i];
        if (float_tensor->data[i] > max_val) max_val = float_tensor->data[i];
    }
    
    // Compute scale and zero_point
    float scale = (max_val - min_val) / 255.0f;
    uint8_t zero_point = (uint8_t)(-min_val / scale);
    
    // Allocate and fill quantized data
    q_tensor->data = (uint8_t*)malloc(float_tensor->size * sizeof(uint8_t));
    if (!q_tensor->data) {
        free(q_tensor);
        return NULL;
    }
    
    for (int i = 0; i < float_tensor->size; i++) {
        float val = float_tensor->data[i];
        int quantized = (int)roundf(val / scale) + zero_point;
        q_tensor->data[i] = (uint8_t)CLAMP(quantized, 0, 255);
    }
    
    q_tensor->scale = scale;
    q_tensor->zero_point = zero_point;
    q_tensor->size = float_tensor->size;
    
    return q_tensor;
}

// Dequantize back to float
tensor_t* tensor_dequantize(quant_tensor_t* q_tensor, int ndim, int* dims) {
    tensor_t* float_tensor = tensor_create(ndim, dims);
    if (!float_tensor) return NULL;
    
    for (int i = 0; i < q_tensor->size; i++) {
        float val = ((int)q_tensor->data[i] - q_tensor->zero_point) * q_tensor->scale;
        float_tensor->data[i] = val;
    }
    
    return float_tensor;
}
```

### Quantized Matrix Multiplication

Quantized operations require careful handling of scales and zero points. Here's an implementation of quantized matrix multiplication:

```c
// Helper macro to clamp values
#define CLAMP(x, low, high) ((x) < (low) ? (low) : ((x) > (high) ? (high) : (x)))

// Quantized matrix multiplication
void quant_tensor_matmul(quant_tensor_t* a, quant_tensor_t* b, quant_tensor_t* result,
                        int M, int K, int N) {
    // Compute the result scale
    float result_scale = a->scale * b->scale;
    
    // Perform matrix multiplication with dequantization and requantization
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                int a_val = (int)a->data[m * K + k] - a->zero_point;
                int b_val = (int)b->data[k * N + n] - b->zero_point;
                sum += a_val * b_val;
            }
            
            // Requantize the result
            float float_result = sum * result_scale;
            int quantized = (int)roundf(float_result / result->scale) + result->zero_point;
            result->data[m * N + n] = (uint8_t)CLAMP(quantized, 0, 255);
        }
    }
}
```

### Optimizing Quantized Operations

Many modern embedded processors have instructions specifically designed for quantized operations. For example, ARM NEON provides SIMD instructions for 8-bit integer arithmetic:

```c
#ifdef __ARM_NEON
#include <arm_neon.h>

// NEON-optimized quantized matrix multiplication
void quant_tensor_matmul_neon(quant_tensor_t* a, quant_tensor_t* b, quant_tensor_t* result,
                             int M, int K, int N) {
    // Compute the result scale
    float result_scale = a->scale * b->scale;
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            
            // Process 16 elements at a time using NEON
            int k = 0;
            for (; k <= K - 16; k += 16) {
                uint8x16_t a_vec = vld1q_u8(&a->data[m * K + k]);
                uint8x16_t b_vec = vld1q_u8(&b->data[k * N + n]);
                
                // Convert to int16 and subtract zero points
                int16x8_t a_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a_vec)));
                int16x8_t a_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a_vec)));
                int16x8_t b_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_vec)));
                int16x8_t b_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b_vec)));
                
                a_low = vsubq_s16(a_low, vdupq_n_s16(a->zero_point));
                a_high = vsubq_s16(a_high, vdupq_n_s16(a->zero_point));
                b_low = vsubq_s16(b_low, vdupq_n_s16(b->zero_point));
                b_high = vsubq_s16(b_high, vdupq_n_s16(b->zero_point));
                
                // Multiply and accumulate
                sum += vaddvq_s32(vpaddlq_s16(vmulq_s16(a_low, b_low)));
                sum += vaddvq_s32(vpaddlq_s16(vmulq_s16(a_high, b_high)));
            }
            
            // Handle remaining elements
            for (; k < K; k++) {
                int a_val = (int)a->data[m * K + k] - a->zero_point;
                int b_val = (int)b->data[k * N + n] - b->zero_point;
                sum += a_val * b_val;
            }
            
            // Requantize the result
            float float_result = sum * result_scale;
            int quantized = (int)roundf(float_result / result->scale) + result->zero_point;
            result->data[m * N + n] = (uint8_t)CLAMP(quantized, 0, 255);
        }
    }
}
#endif
```

Quantization is a powerful technique for reducing memory usage and improving performance on embedded systems. However, it comes with a trade-off in accuracy. Let's now look at memory optimization techniques.

## Memory Optimization Techniques

Embedded systems often have severe memory constraints. Here are some techniques to optimize memory usage in tensor applications:

### In-Place Operations

In-place operations modify tensors directly without creating temporary copies:

```c
// In-place element-wise addition
void tensor_add_inplace(tensor_t* a, tensor_t* b) {
    // Assume dimensions are compatible
    for (int i = 0; i < a->size; i++) {
        a->data[i] += b->data[i];
    }
}

// In-place ReLU activation
void tensor_relu_inplace(tensor_t* tensor) {
    for (int i = 0; i < tensor->size; i++) {
        if (tensor->data[i] < 0) tensor->data[i] = 0;
    }
}
```

### Memory Pooling

Instead of allocating and freeing memory for each operation, use a memory pool:

```c
typedef struct {
    void* memory;       // Pointer to the memory pool
    size_t size;        // Total size of the pool
    size_t used;        // Amount of memory currently used
} memory_pool_t;

// Initialize a memory pool
memory_pool_t* memory_pool_create(size_t size) {
    memory_pool_t* pool = (memory_pool_t*)malloc(sizeof(memory_pool_t));
    if (!pool) return NULL;
    
    pool->memory = malloc(size);
    if (!pool->memory) {
        free(pool);
        return NULL;
    }
    
    pool->size = size;
    pool->used = 0;
    
    return pool;
}

// Allocate memory from the pool
void* memory_pool_alloc(memory_pool_t* pool, size_t size) {
    // Align size to 8 bytes
    size = (size + 7) & ~7;
    
    if (pool->used + size > pool->size) {
        return NULL;  // Not enough memory
    }
    
    void* ptr = (char*)pool->memory + pool->used;
    pool->used += size;
    
    return ptr;
}

// Reset the pool (free all allocations)
void memory_pool_reset(memory_pool_t* pool) {
    pool->used = 0;
}

// Free the pool
void memory_pool_free(memory_pool_t* pool) {
    if (!pool) return;
    free(pool->memory);
    free(pool);
}

// Create a tensor using the memory pool
tensor_t* tensor_create_from_pool(memory_pool_t* pool, int ndim, int* dims) {
    tensor_t* tensor = (tensor_t*)memory_pool_alloc(pool, sizeof(tensor_t));
    if (!tensor) return NULL;
    
    tensor->ndim = ndim;
    tensor->dims = (int*)memory_pool_alloc(pool, ndim * sizeof(int));
    if (!tensor->dims) return NULL;
    
    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->dims[i] = dims[i];
        tensor->size *= dims[i];
    }
    
    tensor->data = (float*)memory_pool_alloc(pool, tensor->size * sizeof(float));
    if (!tensor->data) return NULL;
    
    // Initialize to zero
    memset(tensor->data, 0, tensor->size * sizeof(float));
    
    return tensor;
}
```

### Tensor Reuse

Instead of creating new tensors for each operation, reuse existing ones:

```c
// Reuse a tensor for a new operation
void tensor_reuse(tensor_t* tensor, int ndim, int* dims) {
    // Check if dimensions match
    int new_size = 1;
    for (int i = 0; i < ndim; i++) {
        new_size *= dims[i];
    }
    
    // If the new size is larger, we need to reallocate
    if (new_size > tensor->size) {
        free(tensor->data);
        tensor->data = (float*)malloc(new_size * sizeof(float));
        if (!tensor->data) {
            // Handle allocation failure
            tensor->size = 0;
            return;
        }
    }
    
    // Update dimensions
    free(tensor->dims);
    tensor->dims = (int*)malloc(ndim * sizeof(int));
    if (!tensor->dims) {
        // Handle allocation failure
        free(tensor->data);
        tensor->data = NULL;
        tensor->size = 0;
        return;
    }
    
    tensor->ndim = ndim;
    tensor->size = new_size;
    for (int i = 0; i < ndim; i++) {
        tensor->dims[i] = dims[i];
    }
    
    // Initialize to zero
    memset(tensor->data, 0, tensor->size * sizeof(float));
}
```

### Static Memory Allocation

For extremely constrained systems, use static allocation instead of dynamic:

```c
#define MAX_TENSOR_SIZE 1024

typedef struct {
    int ndim;
    int dims[4];  // Maximum 4 dimensions
    int size;
    float data[MAX_TENSOR_SIZE];
} static_tensor_t;

// Initialize a static tensor
void static_tensor_init(static_tensor_t* tensor, int ndim, int* dims) {
    tensor->ndim = ndim;
    
    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->dims[i] = dims[i];
        tensor->size *= dims[i];
    }
    
    // Check if the tensor fits in the static buffer
    if (tensor->size > MAX_TENSOR_SIZE) {
        fprintf(stderr, "Error: Tensor size exceeds maximum\n");
        tensor->size = 0;
        return;
    }
    
    // Initialize to zero
    memset(tensor->data, 0, tensor->size * sizeof(float));
}
```

These memory optimization techniques are crucial for embedded systems with limited RAM. Now let's look at optimizing for real-time performance.

## Real-Time Inference Pipeline

Many embedded applications require real-time inference, where tensor operations must complete within strict time constraints. Let's design a real-time inference pipeline for a simple neural network:

```c
typedef struct {
    // Layer weights and biases (quantized)
    quant_tensor_t* weights1;
    quant_tensor_t* bias1;
    quant_tensor_t* weights2;
    quant_tensor_t* bias2;
    
    // Layer dimensions
    int input_size;
    int hidden_size;
    int output_size;
    
    // Pre-allocated buffers for intermediate results
    quant_tensor_t* input_buffer;
    quant_tensor_t* hidden_buffer;
    quant_tensor_t* output_buffer;
    
    // Memory pool for temporary allocations
    memory_pool_t* pool;
} rt_network_t;

// Initialize the real-time network
rt_network_t* rt_network_create(int input_size, int hidden_size, int output_size,
                              float* weights1, float* bias1,
                              float* weights2, float* bias2) {
    rt_network_t* net = (rt_network_t*)malloc(sizeof(rt_network_t));
    if (!net) return NULL;
    
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;
    
    // Create a memory pool for temporary allocations
    net->pool = memory_pool_create(1024 * 1024);  // 1MB pool
    if (!net->pool) {
        free(net);
        return NULL;
    }
    
    // Create and quantize weights and biases
    // (Implementation omitted for brevity)
    
    // Pre-allocate buffers for intermediate results
    // (Implementation omitted for brevity)
    
    return net;
}

// Perform inference with the real-time network
void rt_network_inference(rt_network_t* net, float* input, float* output) {
    // Reset the memory pool for this inference
    memory_pool_reset(net->pool);
    
    // Quantize the input
    for (int i = 0; i < net->input_size; i++) {
        // Simplified quantization for illustration
        net->input_buffer->data[i] = (uint8_t)(input[i] * 255.0f);
    }
    
    // First layer: input -> hidden
    quant_tensor_matmul_neon(net->input_buffer, net->weights1, net->hidden_buffer,
                           1, net->input_size, net->hidden_size);
    
    // Add bias and apply ReLU
    for (int i = 0; i < net->hidden_size; i++) {
        int val = net->hidden_buffer->data[i] + net->bias1->data[i] - net->bias1->zero_point;
        net->hidden_buffer->data[i] = (uint8_t)CLAMP(val, 0, 255);
        
        // ReLU: max(0, x)
        if (val < net->hidden_buffer->zero_point) {
            net->hidden_buffer->data[i] = net->hidden_buffer->zero_point;
        }
    }
    
    // Second layer: hidden -> output
    quant_tensor_matmul_neon(net->hidden_buffer, net->weights2, net->output_buffer,
                           1, net->hidden_size, net->output_size);
    
    // Add bias
    for (int i = 0; i < net->output_size; i++) {
        int val = net->output_buffer->data[i] + net->bias2->data[i] - net->bias2->zero_point;
        net->output_buffer->data[i] = (uint8_t)CLAMP(val, 0, 255);
    }
    
    // Dequantize the output
    for (int i = 0; i < net->output_size; i++) {
        // Simplified dequantization for illustration
        output[i] = ((int)net->output_buffer->data[i] - net->output_buffer->zero_point) * 
                    net->output_buffer->scale;
    }
}
```

This real-time inference pipeline uses pre-allocated buffers and a memory pool to avoid dynamic memory allocation during inference, which is crucial for predictable execution time.

## Profiling and Benchmarking on Embedded Systems

To optimize tensor operations for embedded systems, you need to profile and benchmark your code on the target hardware. Here's a simple benchmarking framework:

```c
#include <time.h>

// Benchmark a function
double benchmark_function(void (*func)(void*), void* args, int iterations) {
    struct timespec start, end;
    
    // Warm-up run
    func(args);
    
    // Start timing
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Run the function multiple times
    for (int i = 0; i < iterations; i++) {
        func(args);
    }
    
    // End timing
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    // Calculate elapsed time in milliseconds
    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsed += (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    return elapsed / iterations;
}

// Example usage
typedef struct {
    tensor_t* a;
    tensor_t* b;
    tensor_t* result;
} matmul_args_t;

void matmul_func(void* args) {
    matmul_args_t* matmul_args = (matmul_args_t*)args;
    tensor_matmul(matmul_args->a, matmul_args->b, matmul_args->result);
}

int main() {
    // Create test tensors
    int dims_a[2] = {64, 64};
    int dims_b[2] = {64, 64};
    int dims_result[2] = {64, 64};
    
    tensor_t* a = tensor_create(2, dims_a);
    tensor_t* b = tensor_create(2, dims_b);
    tensor_t* result = tensor_create(2, dims_result);
    
    // Initialize with random values
    for (int i = 0; i < a->size; i++) a->data[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < b->size; i++) b->data[i] = (float)rand() / RAND_MAX;
    
    // Prepare benchmark arguments
    matmul_args_t args = {a, b, result};
    
    // Run benchmark
    double avg_time = benchmark_function(matmul_func, &args, 100);
    printf("Average execution time: %.3f ms\n", avg_time);
    
    // Clean up
    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    
    return 0;
}
```

When profiling on embedded systems, consider these factors:

1. **CPU Utilization**: Measure the percentage of CPU time your application uses.
2. **Memory Usage**: Track peak memory usage and memory fragmentation.
3. **Cache Performance**: Monitor cache hit/miss rates using hardware performance counters.
4. **Power Consumption**: For battery-powered devices, measure power usage during different operations.

## Case Study: Tensor Operations on Raspberry Pi

Let's put everything together with a case study of optimizing tensor operations on a Raspberry Pi:

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensor.h"
#include "fixed_point.h"
#include "quantization.h"

// Benchmark different implementations
void benchmark_matrix_multiplication() {
    printf("Benchmarking matrix multiplication implementations...\n");
    
    // Create test matrices
    int dims_a[2] = {128, 128};
    int dims_b[2] = {128, 128};
    int dims_c[2] = {128, 128};
    
    tensor_t* a = tensor_create(2, dims_a);
    tensor_t* b = tensor_create(2, dims_b);
    tensor_t* c = tensor_create(2, dims_c);
    
    // Initialize with random values
    for (int i = 0; i < a->size; i++) a->data[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < b->size; i++) b->data[i] = (float)rand() / RAND_MAX;
    
    // Convert to fixed-point
    fixed_tensor_t* a_fixed = tensor_to_fixed(a);
    fixed_tensor_t* b_fixed = tensor_to_fixed(b);
    fixed_tensor_t* c_fixed = fixed_tensor_create(2, dims_c);
    
    // Quantize to 8-bit
    quant_tensor_t* a_quant = tensor_quantize(a);
    quant_tensor_t* b_quant = tensor_quantize(b);
    quant_tensor_t* c_quant = (quant_tensor_t*)malloc(sizeof(quant_tensor_t));
    c_quant->data = (uint8_t*)malloc(c->size * sizeof(uint8_t));
    c_quant->scale = 0.1f;
    c_quant->zero_point = 128;
    c_quant->size = c->size;
    
    // Benchmark floating-point implementation
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < 10; i++) {
        tensor_matmul(a, b, c);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double float_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                       (end.tv_nsec - start.tv_nsec) / 1000000.0;
    float_time /= 10.0;
    
    // Benchmark fixed-point implementation
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < 10; i++) {
        fixed_tensor_matmul(a_fixed, b_fixed, c_fixed);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double fixed_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                       (end.tv_nsec - start.tv_nsec) / 1000000.0;
    fixed_time /= 10.0;
    
    // Benchmark quantized implementation
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < 10; i++) {
        quant_tensor_matmul(a_quant, b_quant, c_quant, dims_a[0], dims_a[1], dims_b[1]);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double quant_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                       (end.tv_nsec - start.tv_nsec) / 1000000.0;
    quant_time /= 10.0;
    
    // Benchmark NEON-optimized quantized implementation
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < 10; i++) {
        quant_tensor_matmul_neon(a_quant, b_quant, c_quant, dims_a[0], dims_a[1], dims_b[1]);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double neon_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                      (end.tv_nsec - start.tv_nsec) / 1000000.0;
    neon_time /= 10.0;
    
    // Print results
    printf("Floating-point: %.3f ms\n", float_time);
    printf("Fixed-point: %.3f ms (%.1fx speedup)\n", 
           fixed_time, float_time / fixed_time);
    printf("Quantized: %.3f ms (%.1fx speedup)\n", 
           quant_time, float_time / quant_time);
    printf("NEON-optimized: %.3f ms (%.1fx speedup)\n", 
           neon_time, float_time / neon_time);
    
    // Clean up
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    fixed_tensor_free(a_fixed);
    fixed_tensor_free(b_fixed);
    fixed_tensor_free(c_fixed);
    free(a_quant->data);
    free(a_quant);
    free(b_quant->data);
    free(b_quant);
    free(c_quant->data);
    free(c_quant);
}

int main() {
    // Seed random number generator
    srand(time(NULL));
    
    // Run benchmarks
    benchmark_matrix_multiplication();
    
    return 0;
}
```

Typical results on a Raspberry Pi 4 might look like:

```
Benchmarking matrix multiplication implementations...
Floating-point: 45.782 ms
Fixed-point: 12.345 ms (3.7x speedup)
Quantized: 5.678 ms (8.1x speedup)
NEON-optimized: 1.234 ms (37.1x speedup)
```

These results demonstrate the significant performance improvements possible with fixed-point arithmetic, quantization, and SIMD optimizations on embedded systems.

## Summary

In this chapter, we've explored techniques for deploying tensor applications on embedded systems:

- Setting up a cross-compilation environment for ARM and other architectures
- Implementing fixed-point arithmetic for systems without FPU support
- Quantizing floating-point models to 8-bit integers
- Optimizing memory usage with pooling, reuse, and static allocation
- Building real-time inference pipelines with predictable performance
- Profiling and benchmarking tensor operations on embedded hardware

By applying these techniques, you can adapt the tensor library we've built throughout this book for efficient execution on resource-constrained devices. The key is to understand the specific constraints of your target platform and choose the appropriate optimizations.

Remember that embedded development often involves trade-offs between performance, memory usage, and accuracy. The best approach depends on your specific application requirements and hardware constraints.

## Exercises

1. **Implement a Fixed-Point Convolution Layer**
   
   Extend the fixed-point tensor implementation to support 2D convolution operations, which are common in image processing and computer vision applications.
   
   Hint: Start with a naive implementation using nested loops, then optimize it with techniques like im2col to convert convolution to matrix multiplication.

```c
// Partial solution: im2col function for fixed-point convolution
void fixed_im2col(fixed_tensor_t* input, fixed_tensor_t* output,
                 int kernel_h, int kernel_w, int stride_h, int stride_w,
                 int pad_h, int pad_w) {
    // Assume input is [batch, height, width, channels]
    int batch = input->dims[0];
    int height = input->dims[1];
    int width = input->dims[2];
    int channels = input->dims[3];
    
    // Calculate output dimensions
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // Each column of output corresponds to a location in the output feature map
    // Each row corresponds to a element in the kernel times input channels
    int col_idx = 0;
    for (int b = 0; b < batch; b++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int h = oh * stride_h - pad_h + kh;
                        int w = ow * stride_w - pad_w + kw;
                        
                        if (h >= 0 && h < height && w >= 0 && w < width) {
                            for (int c = 0; c < channels; c++) {
                                int input_idx = ((b * height + h) * width + w) * channels + c;
                                int output_idx = (kh * kernel_w * channels + kw * channels + c) * 
                                                (batch * out_h * out_w) + col_idx;
                                output->data[output_idx] = input->data[input_idx];
                            }
                        } else {
                            // Zero padding
                            for (int c = 0; c < channels; c++) {
                                int output_idx = (kh * kernel_w * channels + kw * channels + c) * 
                                                (batch * out_h * out_w) + col_idx;
                                output->data[output_idx] = 0;
                            }
                        }
                    }
                }
                col_idx++;
            }
        }
    }
}
```

2. **Optimize Memory Usage for a Neural Network**
   
   Take the neural network implementation from Chapter 9 and optimize it for memory-constrained embedded systems. Use techniques like memory pooling, tensor reuse, and in-place operations to minimize memory usage.
   
   Hint: Analyze the memory usage pattern of the forward and backward passes to identify opportunities for reuse.

3. **Benchmark Different Quantization Schemes**
   
   Implement and compare different quantization schemes (symmetric vs. asymmetric, per-tensor vs. per-channel) for a neural network inference task. Measure both performance and accuracy on a real embedded device.
   
   Hint: Start with a pre-trained model, quantize it using different schemes, and compare the results on a validation dataset.

## Further Reading

1. **ARM NEON Programming Guide**
   - ARM Developer: https://developer.arm.com/architectures/instruction-sets/simd-isas/neon
   - NEON Programmer's Guide: https://developer.arm.com/documentation/den0018/latest/

2. **Embedded Deep Learning**
   - TensorFlow Lite for Microcontrollers: https://www.tensorflow.org/lite/microcontrollers
   - Pete Warden's "TinyML": https://tinymlbook.com/

3. **Fixed-Point Arithmetic**
   - "Fixed-Point Arithmetic: An Introduction" by Randy Yates: http://www.digitalsignallabs.com/fp.pdf