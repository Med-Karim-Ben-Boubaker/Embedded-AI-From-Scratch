#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief Function pointer type for activation functions.
 *
 * @param input  Pointer to input data.
 * @param output Pointer to output data.
 * @param size   Number of elements in the input and output arrays.
 */
typedef void (*activation_function_t)(float*, float*, size_t);

/**
 * @brief Structure representing a dense (fully connected) layer in a neural network.
 */
typedef struct {
    float *p_weights;            /**< Pointer to the weights array */
    float *p_biases;             /**< Pointer to the biases array */
    size_t input_size;           /**< Number of inputs to the layer */
    size_t output_size;          /**< Number of outputs from the layer */
    activation_function_t activation; /**< Activation function for the layer */
    float *p_outputs;            /**< Pointer to the output data array */
} dense_layer_t;

/**
 * @brief Structure for holding the arguments needed for a forward pass.
 */
typedef struct {
    const float *p_inputs;       /**< Pointer to the input data */
    dense_layer_t *p_dense_layer; /**< Pointer to the dense layer structure */
} forward_pass_args_t;

/**
 * @brief Structure representing a neural network model.
 */
typedef struct model_t {
    dense_layer_t *p_layers;     /**< Pointer to the array of dense layers */
    size_t num_layers;           /**< Number of layers in the model */

    /**
     * @brief Function pointer for the model's predict function.
     *
     * @param model   Pointer to the model structure.
     * @param inputs  Pointer to the input data.
     * @param outputs Pointer to the output data.
     */
    void (*predict)(struct model_t* model, const float* inputs, float* outputs);

    /**
     * @brief Function pointer for the model's initialization function.
     *
     * @param model      Pointer to the model structure.
     * @param layers     Pointer to the array of dense layers.
     * @param num_layers Number of layers in the model.
     */
    void (*init)(struct model_t* model, dense_layer_t* layers, size_t num_layers);

    /**
     * @brief Function pointer for adding a layer to the model.
     *
     * @param model Pointer to the model structure.
     * @param layer Pointer to the dense layer to be added.
     */
    void (*add)(struct model_t* model, dense_layer_t* layer);

} model_t;

/**
 * @brief Perform a forward pass through a dense layer.
 *
 * @param p_forward_pass_args Pointer to the structure containing the arguments for the forward pass.
 */
void forward_pass(forward_pass_args_t* p_forward_pass_args);

/**
 * @brief Apply the ReLU activation function.
 *
 * @param input  Pointer to input data.
 * @param output Pointer to output data.
 * @param size   Number of elements in the input and output arrays.
 */
void relu_activation(float *input, float *output, size_t size);

/**
 * @brief Apply a linear activation function (identity function).
 *
 * @param input  Pointer to input data.
 * @param output Pointer to output data.
 * @param size   Number of elements in the input and output arrays.
 */
void linear_activation(float *input, float *output, size_t size);

/**
 * @brief Predict the output for given input using the model.
 *
 * @param model   Pointer to the model structure.
 * @param inputs  Pointer to the input data.
 * @param outputs Pointer to the output data.
 */
void model_predict(model_t *model, const float *inputs, float *outputs);

/**
 * @brief Initialize the model with given layers.
 *
 * @param model      Pointer to the model structure.
 * @param layers     Pointer to the array of dense layers.
 * @param num_layers Number of layers in the model.
 */
void model_init(model_t *model, dense_layer_t *layers, size_t num_layers);

/**
 * @brief Add a layer to the model.
 *
 * @param model Pointer to the model structure.
 * @param layer Pointer to the dense layer to be added.
 */
void model_add(model_t *model, dense_layer_t* layer);

#endif /* MODEL_UTILS_H */
