/*
 * model_utils.c
 *
 *  Created on: Jul 16, 2024
 *      Author: karim
 */


#include "main.h"

#define DEBUG_LEVEL 0

/**
 * @brief Apply a linear activation function (identity function).
 *
 * @param input  Pointer to input data.
 * @param output Pointer to output data.
 * @param size   Number of elements in the input and output arrays.
 */
void linear_activation(float *input, float *output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i];
    }
}

/**
 * @brief Apply the ReLU activation function.
 *
 * @param input  Pointer to input data.
 * @param output Pointer to output data.
 * @param size   Number of elements in the input and output arrays.
 */
void relu_activation(float *input, float *output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = fmaxf(0, input[i]);
    }
}

/**
 * @brief Perform a forward pass through a dense layer.
 *
 * @param p_forward_pass_args Pointer to the structure containing the arguments for the forward pass.
 */
void forward_pass(forward_pass_args_t *p_forward_pass_args) {
    dense_layer_t *p_dense_layer = p_forward_pass_args->p_dense_layer;

#if DEBUG_LEVEL >= 1
    printf("Forward Pass:\n");
    printf("  Input Size: %ld\n", p_dense_layer->input_size);
    printf("  Output Size: %ld\n", p_dense_layer->output_size);
#endif

    for (size_t i = 0; i < p_dense_layer->output_size; i++) {
        p_dense_layer->p_outputs[i] = p_dense_layer->p_biases[i];

#if DEBUG_LEVEL >= 2
        printf("  Neuron %ld: %f +", i, p_dense_layer->p_outputs[i]);
#endif

        for (size_t j = 0; j < p_dense_layer->input_size; j++) {
            p_dense_layer->p_outputs[i] += p_forward_pass_args->p_inputs[j] * p_dense_layer->p_weights[i * p_dense_layer->input_size + j];

#if DEBUG_LEVEL >= 3
            printf("(%f * %f) + ", p_forward_pass_args->p_inputs[j], p_dense_layer->p_weights[i * p_dense_layer->input_size + j]);
#endif
        }

#if DEBUG_LEVEL >= 2
        printf("= %f\n", p_dense_layer->p_outputs[i]);
#endif
    }

#if DEBUG_LEVEL >= 1
    printf("Before Activation Output: [");
    for (size_t i = 0; i < p_dense_layer->output_size; i++) {
        printf("%f, ", p_dense_layer->p_outputs[i]);
    }
    printf("]\n");
#endif

    p_dense_layer->activation(p_dense_layer->p_outputs, p_dense_layer->p_outputs, p_dense_layer->output_size);

#if DEBUG_LEVEL >= 1
    printf("Activation Output: [");
    for (size_t i = 0; i < p_dense_layer->output_size; i++) {
        printf("%f, ", p_dense_layer->p_outputs[i]);
    }
    printf("]\n");
#endif
}

/**
 * @brief Initialize the model with given layers.
 *
 * @param model      Pointer to the model structure.
 * @param p_layers   Pointer to the array of dense layers.
 * @param num_layers Number of layers in the model.
 */
void model_init(model_t *model, dense_layer_t *p_layers, size_t num_layers) {
    model->p_layers = p_layers;
    model->num_layers = num_layers;
    model->predict = model_predict;
    model->init = model_init;
}

/**
 * @brief Predict the output for given input using the model.
 *
 * @param model   Pointer to the model structure.
 * @param inputs  Pointer to the input data.
 * @param outputs Pointer to the output data.
 */
void model_predict(model_t *model, const float *inputs, float *outputs) {
#if DEBUG_LEVEL >= 1
    printf("\n---Entering model_predict function---\n");
#endif

    const float *current_input = inputs;

#if DEBUG_LEVEL >= 1
    printf("Looping for each layer: number of layers: %ld\n", model->num_layers);
#endif

    for (size_t l = 0; l < model->num_layers; l++) {
#if DEBUG_LEVEL >= 1
        printf("Inputs: [");
        for (size_t i = 0; i < model->p_layers[l].input_size; i++) {
            printf("%f, ", current_input[i]);
        }
        printf("]\n");
#endif

#if DEBUG_LEVEL >= 2
        printf("Creating forward pass arguments for layer %ld...\n", l);
#endif

        forward_pass_args_t args = {
            .p_inputs = current_input,
            .p_dense_layer = &model->p_layers[l]
        };

        forward_pass(&args);

#if DEBUG_LEVEL >= 2
        printf("Intermediate output for layer %ld:\n", l);
        printf("Outputs: [");
        for (size_t i = 0; i < args.p_dense_layer->output_size; i++) {
            printf("%f, ", args.p_dense_layer->p_outputs[i]);
        }
        printf("]\n");
#endif

        if (l == model->num_layers - 1) {
            // For the last layer, store the output in the `outputs` array
#if DEBUG_LEVEL >= 2
            printf("Storing output for the last layer...\n");
#endif
            for (size_t i = 0; i < args.p_dense_layer->output_size; i++) {
                outputs[i] = args.p_dense_layer->p_outputs[i];
            }
        } else {
            // For intermediate layers, update the `current_input` for the next layer
#if DEBUG_LEVEL >= 2
            printf("Updating current_input for the next layer...\n");
#endif
            current_input = args.p_dense_layer->p_outputs;
        }
    }

#if DEBUG_LEVEL >= 1
    printf("---Exiting model_predict function---\n");
#endif
}
