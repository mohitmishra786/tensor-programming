---
layout: chapter
title: Building a Neural Network Layer with Tensors
number: 9
description: Apply your tensor programming skills to machine learning by implementing neural network components from scratch. This chapter bridges the gap between low-level tensor operations and high-level ML applications.
---

## You Will Learn To...

- Implement a fully connected neural network layer using tensor operations
- Design efficient forward and backward propagation algorithms in C
- Optimize memory usage for batch processing of inputs
- Fuse operations to minimize memory traffic and improve performance
- Debug and validate neural network implementations with test cases

## Introduction

I've spent countless hours debugging neural network implementations that looked correct on paper but failed spectacularly in practice. The gap between textbook descriptions and efficient C implementations is substantial, especially when performance matters. In this chapter, we'll bridge that gap by building a fully connected neural network layer from scratch using our tensor library.

Neural networks might seem like a topic better suited for Python or other high-level languages, but there are compelling reasons to implement them in C. Maybe you're working on an embedded system with strict memory constraints, or perhaps you need predictable performance without garbage collection pauses. Whatever your motivation, implementing neural networks in C gives you complete control over memory and computation.

Let's roll up our sleeves and get to work.

## The Anatomy of a Fully Connected Layer

Before diving into code, let's understand what we're building. A fully connected (or dense) layer transforms an input vector into an output vector through a matrix multiplication followed by a bias addition and an activation function:

```
output = activation(weights × input + bias)
```

Where:
- `input` is an N-dimensional vector
- `weights` is an M×N matrix (M is the output dimension)
- `bias` is an M-dimensional vector
- `activation` is a non-linear function like ReLU, sigmoid, or tanh

For batch processing, we extend this to handle multiple inputs simultaneously:

```
outputs = activation(weights × inputs + bias)
```

Where `inputs` is a B×N matrix (B is the batch size) and `outputs` is a B×M matrix.

## Implementing the Layer Structure

Let's start by defining our layer structure:

```c
typedef enum {
    ACTIVATION_NONE,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH
} activation_type_t;

typedef struct {
    // Layer dimensions
    int input_size;   // N
    int output_size;  // M
    
    // Layer parameters
    tensor_t* weights;  // M×N matrix
    tensor_t* bias;     // M-dimensional vector
    
    // Activation function
    activation_type_t activation;
    
    // Cache for backpropagation
    tensor_t* last_input;     // Input to the layer
    tensor_t* last_output;    // Output before activation
    tensor_t* activated;      // Output after activation
} fc_layer_t;
```

This structure contains everything we need: dimensions, weights, biases, activation type, and caches for backpropagation. The caches store intermediate results that we'll need during the backward pass.

## Layer Initialization and Cleanup

Now let's implement functions to initialize and free our layer:

```c
fc_layer_t* fc_layer_create(int input_size, int output_size, activation_type_t activation) {
    fc_layer_t* layer = (fc_layer_t*)malloc(sizeof(fc_layer_t));
    if (!layer) {
        fprintf(stderr, "Failed to allocate memory for layer\n");
        return NULL;
    }
    
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;
    
    // Initialize weights with Xavier/Glorot initialization
    // This helps with training convergence
    float weight_scale = sqrtf(6.0f / (input_size + output_size));
    
    // Create tensors for weights and bias
    int weight_dims[2] = {output_size, input_size};
    layer->weights = tensor_create(2, weight_dims);
    
    int bias_dims[1] = {output_size};
    layer->bias = tensor_create(1, bias_dims);
    
    if (!layer->weights || !layer->bias) {
        fc_layer_free(layer);
        return NULL;
    }
    
    // Initialize weights with random values scaled by weight_scale
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            float random_val = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * weight_scale;
            tensor_set(layer->weights, (int[]){i, j}, random_val);
        }
    }
    
    // Initialize bias to zeros
    for (int i = 0; i < output_size; i++) {
        tensor_set(layer->bias, (int[]){i}, 0.0f);
    }
    
    // Initialize cache tensors to NULL, we'll create them as needed
    layer->last_input = NULL;
    layer->last_output = NULL;
    layer->activated = NULL;
    
    return layer;
}

void fc_layer_free(fc_layer_t* layer) {
    if (!layer) return;
    
    // Free all tensors
    tensor_free(layer->weights);
    tensor_free(layer->bias);
    tensor_free(layer->last_input);
    tensor_free(layer->last_output);
    tensor_free(layer->activated);
    
    // Free the layer itself
    free(layer);
}
```

Notice how we're using Xavier/Glorot initialization for the weights. This is a common technique in neural networks that helps with training convergence by setting the initial scale of the weights based on the layer dimensions.

## Activation Functions

Before implementing the forward and backward passes, let's define our activation functions:

```c
// Apply activation function element-wise
void apply_activation(tensor_t* input, tensor_t* output, activation_type_t activation) {
    int size = tensor_total_size(input);
    
    for (int i = 0; i < size; i++) {
        float val = tensor_get_flat(input, i);
        float result;
        
        switch (activation) {
            case ACTIVATION_RELU:
                result = val > 0.0f ? val : 0.0f;
                break;
            case ACTIVATION_SIGMOID:
                result = 1.0f / (1.0f + expf(-val));
                break;
            case ACTIVATION_TANH:
                result = tanhf(val);
                break;
            case ACTIVATION_NONE:
            default:
                result = val;
                break;
        }
        
        tensor_set_flat(output, i, result);
    }
}

// Compute derivative of activation function given the activated values
void activation_derivative(tensor_t* activated, tensor_t* derivative, activation_type_t activation) {
    int size = tensor_total_size(activated);
    
    for (int i = 0; i < size; i++) {
        float act_val = tensor_get_flat(activated, i);
        float deriv;
        
        switch (activation) {
            case ACTIVATION_RELU:
                deriv = act_val > 0.0f ? 1.0f : 0.0f;
                break;
            case ACTIVATION_SIGMOID:
                // For sigmoid: f'(x) = f(x) * (1 - f(x))
                deriv = act_val * (1.0f - act_val);
                break;
            case ACTIVATION_TANH:
                // For tanh: f'(x) = 1 - f(x)^2
                deriv = 1.0f - act_val * act_val;
                break;
            case ACTIVATION_NONE:
            default:
                deriv = 1.0f;
                break;
        }
        
        tensor_set_flat(derivative, i, deriv);
    }
}
```

These functions apply the activation function and compute its derivative, which we'll need for backpropagation. We're using the `tensor_get_flat` and `tensor_set_flat` functions to access tensor elements by their flat index, which is more efficient for element-wise operations.

## Forward Pass

Now let's implement the forward pass, which computes the output of the layer given an input:

```c
tensor_t* fc_layer_forward(fc_layer_t* layer, tensor_t* input, bool is_training) {
    // Check input dimensions
    if (input->dims[input->ndim - 1] != layer->input_size) {
        fprintf(stderr, "Input dimension mismatch: expected %d, got %d\n", 
                layer->input_size, input->dims[input->ndim - 1]);
        return NULL;
    }
    
    // Determine batch size and reshape input if needed
    int batch_size = 1;
    if (input->ndim > 1) {
        // For multi-dimensional input, the batch size is the product of all dimensions except the last
        for (int i = 0; i < input->ndim - 1; i++) {
            batch_size *= input->dims[i];
        }
    }
    
    // Reshape input to 2D: [batch_size, input_size]
    tensor_t* reshaped_input = NULL;
    if (input->ndim != 2 || input->dims[0] != batch_size || input->dims[1] != layer->input_size) {
        int new_dims[2] = {batch_size, layer->input_size};
        reshaped_input = tensor_reshape(input, 2, new_dims);
    } else {
        reshaped_input = tensor_clone(input);
    }
    
    if (!reshaped_input) {
        fprintf(stderr, "Failed to reshape input\n");
        return NULL;
    }
    
    // Allocate output tensor: [batch_size, output_size]
    int output_dims[2] = {batch_size, layer->output_size};
    tensor_t* output = tensor_create(2, output_dims);
    if (!output) {
        tensor_free(reshaped_input);
        fprintf(stderr, "Failed to allocate output tensor\n");
        return NULL;
    }
    
    // Perform matrix multiplication: output = input × weights^T
    // Note: weights is [output_size, input_size], so we need to transpose it
    tensor_t* weights_t = tensor_transpose(layer->weights, 0, 1);
    if (!weights_t) {
        tensor_free(reshaped_input);
        tensor_free(output);
        fprintf(stderr, "Failed to transpose weights\n");
        return NULL;
    }
    
    // output[b, o] = sum_i(input[b, i] * weights[o, i])
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < layer->output_size; o++) {
            float sum = 0.0f;
            for (int i = 0; i < layer->input_size; i++) {
                sum += tensor_get(reshaped_input, (int[]){b, i}) * 
                       tensor_get(weights_t, (int[]){i, o});
            }
            tensor_set(output, (int[]){b, o}, sum);
        }
    }
    
    tensor_free(weights_t);
    
    // Add bias: output = output + bias
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < layer->output_size; o++) {
            float val = tensor_get(output, (int[]){b, o}) + 
                        tensor_get(layer->bias, (int[]){o});
            tensor_set(output, (int[]){b, o}, val);
        }
    }
    
    // Store pre-activation output for backpropagation
    if (is_training) {
        // Cache the input and pre-activation output
        tensor_free(layer->last_input);
        layer->last_input = tensor_clone(reshaped_input);
        
        tensor_free(layer->last_output);
        layer->last_output = tensor_clone(output);
    }
    
    // Apply activation function
    tensor_t* activated = tensor_clone(output);
    if (!activated) {
        tensor_free(reshaped_input);
        tensor_free(output);
        fprintf(stderr, "Failed to allocate activated tensor\n");
        return NULL;
    }
    
    apply_activation(output, activated, layer->activation);
    
    // Store activated output for backpropagation
    if (is_training) {
        tensor_free(layer->activated);
        layer->activated = tensor_clone(activated);
    }
    
    tensor_free(output);
    tensor_free(reshaped_input);
    
    return activated;
}
```

This function performs several steps:
1. Reshapes the input to a 2D tensor with shape [batch_size, input_size]
2. Performs matrix multiplication between the input and the transposed weights
3. Adds the bias to each output
4. Applies the activation function
5. Caches intermediate results for backpropagation if in training mode

Note that we're using a naive matrix multiplication implementation here. In a production system, you'd want to use a more optimized approach, possibly using BLAS as we discussed in Chapter 6.

## Backward Pass

The backward pass computes the gradients of the loss with respect to the layer's parameters and inputs. This is the heart of neural network training:

```c
tensor_t* fc_layer_backward(fc_layer_t* layer, tensor_t* output_gradient, float learning_rate) {
    // Check if we have cached values from forward pass
    if (!layer->last_input || !layer->last_output || !layer->activated) {
        fprintf(stderr, "No cached values from forward pass. Was forward called with is_training=true?\n");
        return NULL;
    }
    
    int batch_size = layer->last_input->dims[0];
    
    // Step 1: Compute gradient through activation function
    // d_preact = d_output * activation'(preact)
    tensor_t* activation_grad = tensor_create_like(layer->activated);
    activation_derivative(layer->activated, activation_grad, layer->activation);
    
    tensor_t* preact_grad = tensor_create_like(layer->last_output);
    for (int i = 0; i < tensor_total_size(preact_grad); i++) {
        float grad = tensor_get_flat(output_gradient, i) * tensor_get_flat(activation_grad, i);
        tensor_set_flat(preact_grad, i, grad);
    }
    
    tensor_free(activation_grad);
    
    // Step 2: Compute gradient for bias
    // d_bias = sum(d_preact, axis=0)
    for (int o = 0; o < layer->output_size; o++) {
        float bias_grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            bias_grad += tensor_get(preact_grad, (int[]){b, o});
        }
        
        // Update bias using gradient descent
        float current_bias = tensor_get(layer->bias, (int[]){o});
        tensor_set(layer->bias, (int[]){o}, current_bias - learning_rate * bias_grad);
    }
    
    // Step 3: Compute gradient for weights
    // d_weights[o, i] = sum_b(d_preact[b, o] * input[b, i])
    for (int o = 0; o < layer->output_size; o++) {
        for (int i = 0; i < layer->input_size; i++) {
            float weight_grad = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                weight_grad += tensor_get(preact_grad, (int[]){b, o}) * 
                              tensor_get(layer->last_input, (int[]){b, i});
            }
            
            // Update weight using gradient descent
            float current_weight = tensor_get(layer->weights, (int[]){o, i});
            tensor_set(layer->weights, (int[]){o, i}, 
                      current_weight - learning_rate * weight_grad);
        }
    }
    
    // Step 4: Compute gradient with respect to input
    // d_input[b, i] = sum_o(d_preact[b, o] * weights[o, i])
    tensor_t* input_grad = tensor_create_like(layer->last_input);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < layer->input_size; i++) {
            float grad_sum = 0.0f;
            for (int o = 0; o < layer->output_size; o++) {
                grad_sum += tensor_get(preact_grad, (int[]){b, o}) * 
                           tensor_get(layer->weights, (int[]){o, i});
            }
            tensor_set(input_grad, (int[]){b, i}, grad_sum);
        }
    }
    
    tensor_free(preact_grad);
    
    return input_grad;
}
```

The backward pass performs these steps:
1. Computes the gradient through the activation function
2. Computes and applies the gradient for the bias
3. Computes and applies the gradient for the weights
4. Computes the gradient with respect to the input, which will be passed to the previous layer

We're using simple gradient descent here, but in practice, you might want to use more advanced optimizers like Adam or RMSProp.

## Operation Fusion for Performance

One way to optimize neural network computations is to fuse operations. Instead of computing each operation separately and storing intermediate results, we can combine operations to reduce memory traffic.

Let's implement a fused version of the forward pass that combines matrix multiplication, bias addition, and activation:

```c
tensor_t* fc_layer_forward_fused(fc_layer_t* layer, tensor_t* input, bool is_training) {
    // Similar input checking and reshaping as before...
    
    // Allocate output tensor directly
    int output_dims[2] = {batch_size, layer->output_size};
    tensor_t* activated = tensor_create(2, output_dims);
    
    // Fused operation: activated[b, o] = activation(sum_i(input[b, i] * weights[o, i]) + bias[o])
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < layer->output_size; o++) {
            float sum = 0.0f;
            
            // Matrix multiplication
            for (int i = 0; i < layer->input_size; i++) {
                sum += tensor_get(reshaped_input, (int[]){b, i}) * 
                       tensor_get(layer->weights, (int[]){o, i});
            }
            
            // Add bias
            sum += tensor_get(layer->bias, (int[]){o});
            
            // Apply activation
            float result;
            switch (layer->activation) {
                case ACTIVATION_RELU:
                    result = sum > 0.0f ? sum : 0.0f;
                    break;
                case ACTIVATION_SIGMOID:
                    result = 1.0f / (1.0f + expf(-sum));
                    break;
                case ACTIVATION_TANH:
                    result = tanhf(sum);
                    break;
                case ACTIVATION_NONE:
                default:
                    result = sum;
                    break;
            }
            
            tensor_set(activated, (int[]){b, o}, result);
        }
    }
    
    // For training, we still need to cache intermediate results
    if (is_training) {
        // Cache the input
        tensor_free(layer->last_input);
        layer->last_input = tensor_clone(reshaped_input);
        
        // We need to reconstruct the pre-activation output for backprop
        tensor_free(layer->last_output);
        layer->last_output = tensor_create(2, output_dims);
        
        for (int b = 0; b < batch_size; b++) {
            for (int o = 0; o < layer->output_size; o++) {
                float sum = 0.0f;
                for (int i = 0; i < layer->input_size; i++) {
                    sum += tensor_get(reshaped_input, (int[]){b, i}) * 
                           tensor_get(layer->weights, (int[]){o, i});
                }
                sum += tensor_get(layer->bias, (int[]){o});
                tensor_set(layer->last_output, (int[]){b, o}, sum);
            }
        }
        
        // Cache the activated output
        tensor_free(layer->activated);
        layer->activated = tensor_clone(activated);
    }
    
    tensor_free(reshaped_input);
    
    return activated;
}
```

This fused implementation reduces memory traffic by computing the final result directly, without storing intermediate tensors. However, for training, we still need to cache the intermediate results for backpropagation.

## Batch Processing Optimization

Batch processing is essential for efficient neural network training. Let's optimize our implementation for batch processing by using cache-aware blocking:

```c
#define BLOCK_SIZE 16  // Adjust based on your cache size

tensor_t* fc_layer_forward_blocked(fc_layer_t* layer, tensor_t* input, bool is_training) {
    // Similar input checking and reshaping as before...
    
    // Allocate output tensors
    int output_dims[2] = {batch_size, layer->output_size};
    tensor_t* output = tensor_create(2, output_dims);
    tensor_t* activated = tensor_create(2, output_dims);
    
    // Initialize output to zeros
    tensor_fill(output, 0.0f);
    
    // Blocked matrix multiplication
    for (int b_block = 0; b_block < batch_size; b_block += BLOCK_SIZE) {
        for (int o_block = 0; o_block < layer->output_size; o_block += BLOCK_SIZE) {
            for (int i_block = 0; i_block < layer->input_size; i_block += BLOCK_SIZE) {
                // Process a block of the matrices
                for (int b = b_block; b < MIN(b_block + BLOCK_SIZE, batch_size); b++) {
                    for (int o = o_block; o < MIN(o_block + BLOCK_SIZE, layer->output_size); o++) {
                        float sum = tensor_get(output, (int[]){b, o});
                        
                        for (int i = i_block; i < MIN(i_block + BLOCK_SIZE, layer->input_size); i++) {
                            sum += tensor_get(reshaped_input, (int[]){b, i}) * 
                                   tensor_get(layer->weights, (int[]){o, i});
                        }
                        
                        tensor_set(output, (int[]){b, o}, sum);
                    }
                }
            }
        }
    }
    
    // Add bias and apply activation
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < layer->output_size; o++) {
            float val = tensor_get(output, (int[]){b, o}) + tensor_get(layer->bias, (int[]){o});
            
            // Apply activation
            float result;
            switch (layer->activation) {
                case ACTIVATION_RELU:
                    result = val > 0.0f ? val : 0.0f;
                    break;
                case ACTIVATION_SIGMOID:
                    result = 1.0f / (1.0f + expf(-val));
                    break;
                case ACTIVATION_TANH:
                    result = tanhf(val);
                    break;
                case ACTIVATION_NONE:
                default:
                    result = val;
                    break;
            }
            
            tensor_set(activated, (int[]){b, o}, result);
        }
    }
    
    // Cache for backpropagation if in training mode
    if (is_training) {
        tensor_free(layer->last_input);
        layer->last_input = tensor_clone(reshaped_input);
        
        tensor_free(layer->last_output);
        layer->last_output = tensor_clone(output);
        
        tensor_free(layer->activated);
        layer->activated = tensor_clone(activated);
    }
    
    tensor_free(output);
    tensor_free(reshaped_input);
    
    return activated;
}
```

This implementation uses blocking to improve cache utilization. By processing the matrices in small blocks, we keep the working set in cache, reducing memory traffic and improving performance.

## Building a Simple Neural Network

Now that we have our fully connected layer implementation, let's build a simple neural network for classification:

```c
typedef struct {
    int num_layers;
    fc_layer_t** layers;
} neural_network_t;

neural_network_t* nn_create(int num_layers, int* layer_sizes, activation_type_t* activations) {
    neural_network_t* nn = (neural_network_t*)malloc(sizeof(neural_network_t));
    if (!nn) return NULL;
    
    nn->num_layers = num_layers - 1;  // Number of layers is one less than number of sizes
    nn->layers = (fc_layer_t**)malloc(nn->num_layers * sizeof(fc_layer_t*));
    if (!nn->layers) {
        free(nn);
        return NULL;
    }
    
    // Create each layer
    for (int i = 0; i < nn->num_layers; i++) {
        nn->layers[i] = fc_layer_create(layer_sizes[i], layer_sizes[i+1], activations[i]);
        if (!nn->layers[i]) {
            // Clean up previously created layers
            for (int j = 0; j < i; j++) {
                fc_layer_free(nn->layers[j]);
            }
            free(nn->layers);
            free(nn);
            return NULL;
        }
    }
    
    return nn;
}

void nn_free(neural_network_t* nn) {
    if (!nn) return;
    
    if (nn->layers) {
        for (int i = 0; i < nn->num_layers; i++) {
            fc_layer_free(nn->layers[i]);
        }
        free(nn->layers);
    }
    
    free(nn);
}

tensor_t* nn_forward(neural_network_t* nn, tensor_t* input, bool is_training) {
    tensor_t* current = tensor_clone(input);
    
    for (int i = 0; i < nn->num_layers; i++) {
        tensor_t* next = fc_layer_forward(nn->layers[i], current, is_training);
        tensor_free(current);
        
        if (!next) {
            fprintf(stderr, "Forward pass failed at layer %d\n", i);
            return NULL;
        }
        
        current = next;
    }
    
    return current;
}

void nn_backward(neural_network_t* nn, tensor_t* output_gradient, float learning_rate) {
    tensor_t* current_gradient = tensor_clone(output_gradient);
    
    for (int i = nn->num_layers - 1; i >= 0; i--) {
        tensor_t* next_gradient = fc_layer_backward(nn->layers[i], current_gradient, learning_rate);
        tensor_free(current_gradient);
        
        if (!next_gradient && i > 0) {
            fprintf(stderr, "Backward pass failed at layer %d\n", i);
            return;
        }
        
        current_gradient = next_gradient;
    }
    
    tensor_free(current_gradient);  // Free the final gradient
}
```

This neural network implementation stacks multiple fully connected layers and provides forward and backward pass functions for training.

## Example: Training a Neural Network for MNIST

Let's put everything together with an example of training a neural network for MNIST digit classification:

```c
// Assume we have functions to load MNIST data
extern tensor_t* load_mnist_images(const char* filename);
extern tensor_t* load_mnist_labels(const char* filename);

// Compute cross-entropy loss and its gradient
void compute_loss_and_gradient(tensor_t* predictions, tensor_t* targets, 
                              float* loss, tensor_t* gradient) {
    int batch_size = predictions->dims[0];
    int num_classes = predictions->dims[1];
    
    *loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        int target_class = 0;
        float max_target = tensor_get(targets, (int[]){b, 0});
        
        // Find the target class (one-hot encoding)
        for (int c = 1; c < num_classes; c++) {
            float val = tensor_get(targets, (int[]){b, c});
            if (val > max_target) {
                max_target = val;
                target_class = c;
            }
        }
        
        // Compute softmax and cross-entropy loss
        float max_val = tensor_get(predictions, (int[]){b, 0});
        for (int c = 1; c < num_classes; c++) {
            float val = tensor_get(predictions, (int[]){b, c});
            if (val > max_val) max_val = val;
        }
        
        float sum_exp = 0.0f;
        float* exp_vals = (float*)malloc(num_classes * sizeof(float));
        
        for (int c = 0; c < num_classes; c++) {
            float val = tensor_get(predictions, (int[]){b, c}) - max_val;
            exp_vals[c] = expf(val);
            sum_exp += exp_vals[c];
        }
        
        for (int c = 0; c < num_classes; c++) {
            float softmax = exp_vals[c] / sum_exp;
            
            // Cross-entropy loss
            if (c == target_class) {
                *loss -= logf(softmax);
            }
            
            // Gradient: softmax - one_hot_target
            float target_val = (c == target_class) ? 1.0f : 0.0f;
            tensor_set(gradient, (int[]){b, c}, softmax - target_val);
        }
        
        free(exp_vals);
    }
    
    *loss /= batch_size;
    
    // Scale gradient by batch size
    for (int i = 0; i < tensor_total_size(gradient); i++) {
        float val = tensor_get_flat(gradient, i) / batch_size;
        tensor_set_flat(gradient, i, val);
    }
}

int main() {
    // Load MNIST data
    tensor_t* train_images = load_mnist_images("train-images-idx3-ubyte");
    tensor_t* train_labels = load_mnist_labels("train-labels-idx1-ubyte");
    
    if (!train_images || !train_labels) {
        fprintf(stderr, "Failed to load MNIST data\n");
        return 1;
    }
    
    // Normalize images to [0, 1]
    for (int i = 0; i < tensor_total_size(train_images); i++) {
        float val = tensor_get_flat(train_images, i) / 255.0f;
        tensor_set_flat(train_images, i, val);
    }
    
    // Create a neural network: 784 -> 128 -> 64 -> 10
    int layer_sizes[] = {784, 128, 64, 10};
    activation_type_t activations[] = {ACTIVATION_RELU, ACTIVATION_RELU, ACTIVATION_NONE};
    
    neural_network_t* nn = nn_create(4, layer_sizes, activations);
    if (!nn) {
        fprintf(stderr, "Failed to create neural network\n");
        return 1;
    }
    
    // Training parameters
    int num_epochs = 10;
    int batch_size = 64;
    float learning_rate = 0.01f;
    
    // Number of batches
    int num_samples = train_images->dims[0];
    int num_batches = (num_samples + batch_size - 1) / batch_size;  // Ceiling division
    
    printf("Training neural network for MNIST classification...\n");
    printf("Samples: %d, Batch size: %d, Batches: %d\n", num_samples, batch_size, num_batches);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = MIN(start_idx + batch_size, num_samples);
            int current_batch_size = end_idx - start_idx;
            
            // Create batch tensors
            int batch_image_dims[2] = {current_batch_size, 784};
            tensor_t* batch_images = tensor_create(2, batch_image_dims);
            
            int batch_label_dims[2] = {current_batch_size, 10};
            tensor_t* batch_labels = tensor_create(2, batch_label_dims);
            tensor_fill(batch_labels, 0.0f);
            
            // Fill batch tensors
            for (int i = 0; i < current_batch_size; i++) {
                int idx = start_idx + i;
                
                // Copy image
                for (int j = 0; j < 784; j++) {
                    float val = tensor_get_flat(train_images, idx * 784 + j);
                    tensor_set(batch_images, (int[]){i, j}, val);
                }
                
                // Set one-hot label
                int label = (int)tensor_get_flat(train_labels, idx);
                tensor_set(batch_labels, (int[]){i, label}, 1.0f);
            }
            
            // Forward pass
            tensor_t* predictions = nn_forward(nn, batch_images, true);
            if (!predictions) {
                fprintf(stderr, "Forward pass failed\n");
                return 1;
            }
            
            // Compute loss and gradient
            float batch_loss;
            tensor_t* loss_gradient = tensor_create_like(predictions);
            compute_loss_and_gradient(predictions, batch_labels, &batch_loss, loss_gradient);
            
            total_loss += batch_loss;
            
            // Count correct predictions
            for (int i = 0; i < current_batch_size; i++) {
                int pred_class = 0;
                float max_pred = tensor_get(predictions, (int[]){i, 0});
                
                for (int c = 1; c < 10; c++) {
                    float val = tensor_get(predictions, (int[]){i, c});
                    if (val > max_pred) {
                        max_pred = val;
                        pred_class = c;
                    }
                }
                
                int true_class = 0;
                float max_true = tensor_get(batch_labels, (int[]){i, 0});
                
                for (int c = 1; c < 10; c++) {
                    float val = tensor_get(batch_labels, (int[]){i, c});
                    if (val > max_true) {
                        max_true = val;
                        true_class = c;
                    }
                }
                
                if (pred_class == true_class) {
                    correct++;
                }
            }
            
            // Backward pass
            nn_backward(nn, loss_gradient, learning_rate);
            
            // Clean up
            tensor_free(batch_images);
            tensor_free(batch_labels);
            tensor_free(predictions);
            tensor_free(loss_gradient);
            
            // Print progress
            if (batch % 100 == 0) {
                printf("Epoch %d, Batch %d/%d, Loss: %.4f\n", 
                       epoch + 1, batch + 1, num_batches, batch_loss);
            }
        }
        
        // Print epoch results
        float avg_loss = total_loss / num_batches;
        float accuracy = (float)correct / num_samples;
        
        printf("Epoch %d: Avg Loss = %.4f, Accuracy = %.2f%%\n", 
               epoch + 1, avg_loss, accuracy * 100.0f);
    }
    
    // Clean up
    nn_free(nn);
    tensor_free(train_images);
    tensor_free(train_labels);
    
    return 0;
}
```

This example demonstrates how to use our neural network implementation to train a classifier for MNIST digits. It includes loading data, creating a network, training with mini-batches, and evaluating accuracy.

## Common Pitfalls and Debugging

Implementing neural networks in C comes with its own set of challenges. Here are some common pitfalls and how to address them:

### Memory Leaks

Neural networks involve many temporary tensors, making memory management crucial. Use tools like Valgrind to detect leaks:

```
$ valgrind --leak-check=full ./neural_network
```

Common sources of leaks include:
- Forgetting to free tensors after use
- Not freeing tensors on error paths
- Not freeing cached tensors in the layer structure

### Numerical Stability

Floating-point operations can lead to numerical instability. Common issues include:

1. **Exploding/Vanishing Gradients**: Use proper weight initialization (like Xavier/Glorot) and activation functions (like ReLU).

2. **Softmax Overflow**: Subtract the maximum value before computing exponentials:

```c
float max_val = tensor_get(logits, (int[]){0});
for (int i = 1; i < size; i++) {
    float val = tensor_get(logits, (int[]){i});
    if (val > max_val) max_val = val;
}

float sum_exp = 0.0f;
for (int i = 0; i < size; i++) {
    float val = expf(tensor_get(logits, (int[]){i}) - max_val);
    tensor_set(softmax, (int[]){i}, val);
    sum_exp += val;
}

for (int i = 0; i < size; i++) {
    float val = tensor_get(softmax, (int[]){i}) / sum_exp;
    tensor_set(softmax, (int[]){i}, val);
}
```

3. **Log of Zero**: Add a small epsilon when computing logarithms:

```c
float epsilon = 1e-7f;
float loss = -logf(MAX(prediction, epsilon));
```

### Debugging with Gradient Checking

Gradient checking is a powerful technique to verify your backpropagation implementation:

```c
void gradient_check(fc_layer_t* layer, tensor_t* input, tensor_t* output_gradient) {
    // Perform forward and backward passes
    tensor_t* output = fc_layer_forward(layer, input, true);
    tensor_t* analytical_grad = fc_layer_backward(layer, output_gradient, 0.0f);
    
    // Compute numerical gradient for a few elements
    float epsilon = 1e-4f;
    
    printf("Gradient check for weights:\n");
    for (int o = 0; o < MIN(3, layer->output_size); o++) {
        for (int i = 0; i < MIN(3, layer->input_size); i++) {
            // Save original weight
            float original = tensor_get(layer->weights, (int[]){o, i});
            
            // Compute loss with weight + epsilon
            tensor_set(layer->weights, (int[]){o, i}, original + epsilon);
            tensor_t* output_plus = fc_layer_forward(layer, input, false);
            float loss_plus = compute_loss(output_plus, output_gradient);
            
            // Compute loss with weight - epsilon
            tensor_set(layer->weights, (int[]){o, i}, original - epsilon);
            tensor_t* output_minus = fc_layer_forward(layer, input, false);
            float loss_minus = compute_loss(output_minus, output_gradient);
            
            // Restore original weight
            tensor_set(layer->weights, (int[]){o, i}, original);
            
            // Compute numerical gradient
            float numerical_grad = (loss_plus - loss_minus) / (2 * epsilon);
            
            // Get analytical gradient
            float analytical = 0.0f;
            for (int b = 0; b < input->dims[0]; b++) {
                analytical += tensor_get(output_gradient, (int[]){b, o}) * 
                             tensor_get(input, (int[]){b, i});
            }
            
            // Compare
            printf("Weight[%d,%d]: Numerical = %.6f, Analytical = %.6f, Diff = %.6f\n",
                   o, i, numerical_grad, analytical, fabsf(numerical_grad - analytical));
            
            tensor_free(output_plus);
            tensor_free(output_minus);
        }
    }
    
    tensor_free(output);
    tensor_free(analytical_grad);
}
```

This function compares the gradients computed by backpropagation with numerical gradients computed using finite differences. If they match closely, your implementation is likely correct.

## Optimizing Neural Network Performance

Here are some techniques to optimize neural network performance in C:

1. **Use BLAS for Matrix Operations**: Replace naive matrix multiplication with BLAS calls (as discussed in Chapter 6).

2. **Parallelize with OpenMP**: Add OpenMP pragmas to parallelize computations (as discussed in Chapter 4).

3. **Vectorize with SIMD**: Use SIMD intrinsics for element-wise operations (as discussed in Chapter 5).

4. **Optimize Memory Layout**: Use cache-aware blocking and ensure proper alignment.

5. **Fuse Operations**: Combine multiple operations to reduce memory traffic.

6. **Quantize Weights**: Use lower precision (e.g., int8) for inference to improve performance.

## Summary

In this chapter, we've built a fully connected neural network layer from scratch using our tensor library. We've covered:

- Implementing forward and backward passes for fully connected layers
- Applying various activation functions
- Optimizing performance with operation fusion and cache-aware blocking
- Building a complete neural network for classification
- Debugging common issues in neural network implementations

This implementation provides a solid foundation for building more complex neural network architectures in C. By understanding the low-level details of neural network operations, you can create highly optimized implementations tailored to your specific needs.

## Exercises

1. **Extend the Layer Implementation**
   
   Implement a convolutional layer that operates on 4D tensors (batch, height, width, channels). Start with a naive implementation, then optimize it using the techniques discussed in this chapter.
   
   Hint: Begin with the forward pass, implementing the convolution operation as nested loops. Then add the backward pass to compute gradients for the filters and input.

2. **Implement Dropout Regularization**
   
   Add dropout regularization to the fully connected layer to prevent overfitting. Dropout randomly sets a fraction of the inputs to zero during training.
   
   Hint: Use a random number generator to create a mask tensor during the forward pass. Store this mask for the backward pass to ensure consistent gradient flow.

```c
// Partial solution
void apply_dropout(tensor_t* input, tensor_t* output, float dropout_rate, tensor_t* mask) {
    float scale = 1.0f / (1.0f - dropout_rate);
    
    for (int i = 0; i < tensor_total_size(input); i++) {
        float random = (float)rand() / RAND_MAX;
        float mask_val = random > dropout_rate ? 1.0f : 0.0f;
        tensor_set_flat(mask, i, mask_val);
        
        float val = tensor_get_flat(input, i) * mask_val * scale;
        tensor_set_flat(output, i, val);
    }
}
```

3. **Optimize for Cache Efficiency**
   
   Profile the neural network implementation using tools like `perf` and identify cache-related bottlenecks. Implement cache-aware optimizations to improve performance.
   
   Hint: Experiment with different block sizes and loop orders. Measure the impact on performance using a benchmark that tracks execution time and cache miss rates.

## Further Reading

1. **BLAS and LAPACK Documentation**
   - Netlib BLAS: http://www.netlib.org/blas/
   - OpenBLAS: https://www.openblas.net/
   - Intel MKL: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html

2. **Neural Network Optimization Techniques**
   - Efficient Backprop by Yann LeCun: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
   - Deep Learning by Goodfellow, Bengio, and Courville: https://www.deeplearningbook.org/

3. **Memory and Cache Optimization**
   - What Every Programmer Should Know About Memory by Ulrich Drepper: https://people.freebsd.org/~lstewart/articles/cpumemory.pdf