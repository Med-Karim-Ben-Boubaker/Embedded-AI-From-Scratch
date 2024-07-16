#include "model_utils.h"
// This is an example program for a neural networks that generates sin wave.
// The model is trained in python from the neural network from scratch project

#define DEBUG_STATE 1

#define INPUT_SIZE  1
#define OUTPUT_SIZE 1

// importing model's parameters
extern float layer0_weights[16];
extern float layer1_weights[256];
extern float layer2_weights[16];

extern float layer0_biases[16];
extern float layer1_biases[16];
extern float layer2_biases[1];

int main() {

    // Define the first layer
    int layer0_neurons_num = 16;
    float layer0_outputs[layer0_neurons_num];
    dense_layer_t layer0 = {
        layer0_weights,
        layer0_biases,
        INPUT_SIZE,
        layer0_neurons_num,
        relu_activation,        // Define the activation function ( keep the same architecture as the trained model )
        layer0_outputs
    };

    // Define the second layer
    int layer1_neurons_num = 16;
    int layer1_input_size = 16;
    float layer1_outputs[layer1_neurons_num];
    dense_layer_t layer1 = {
        layer1_weights,
        layer1_biases,
        layer1_input_size,
        layer1_neurons_num,
        relu_activation,        // Define the activation function ( keep the same architecture as the trained model )
        layer1_outputs
    };

    // Define the third layer
    int layer2_input_number = 16;
    int layer2_neurons_num = 1;
    float layer2_outputs[layer2_neurons_num];
    dense_layer_t layer2 = {
        layer2_weights,
        layer2_biases,
        layer2_input_number,
        layer2_neurons_num,
        linear_activation,      // Define the activation function ( keep the same architecture as the trained model )
        layer2_outputs
    };

    // define number of layers and the layers array
    int layers_num = 3;
    dense_layer_t layers[3] = { layer0, layer1, layer2 };

    // Define an input array
    float input[INPUT_SIZE];

    // Define the output array
    float output[OUTPUT_SIZE];

    // Define the model
    model_t model;
    model.init = model_init;
    model.init(&model, layers, layers_num);

  // Testing the sin wave model
  #ifdef DEBUG_STATE
  for (int i = 0; i < 6; i++) {
      input[0] = i;
      model.predict(&model, input, output);
      printf("Predicted Output of sin(%f): %f -- Actual Value Sin(%f) = %f\n", input[0], output[0], input[0], sin(input[0]));
  }
  #endif
    
    return 0;
}
