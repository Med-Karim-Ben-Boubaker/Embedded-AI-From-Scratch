# neural_network.py
# importing dependencies
import numpy as np
import math

class DenseLayer:
    # Initialize a Dense Layer
    def __init__(self, number_inputs: int, number_neurons: int):
        self.weights = 0.1 * np.random.randn(number_inputs, number_neurons)
        self.biases = np.zeros((1, number_neurons))
        
    def forward_pass(self, inputs):
        # Save Inputs for backprobagation
        self.inputs = inputs
        # Calculating output
        self.output = np.dot(inputs, self.weights) + self.biases
        
        return self.output
    
    def backward_pass(self, prev_upstream_gradient):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, prev_upstream_gradient)
        self.dbiases = np.sum(prev_upstream_gradient, axis=0, keepdims=True)
        # Gradients on inputs
        self.upstream_gradient = np.dot(prev_upstream_gradient, self.weights.T)
        
        return self.upstream_gradient
    
class Activation():
    # An Interface for the different Activation Functions
    def __init__(self):
        pass
    
    def forward(self, inputs):
        pass
    
    def backward(self, prev_upstream_gradient):
        pass
    
class ActivationReLU(Activation):
    # ReLU Activation Function.
    def forward_pass(self, inputs):
        # Save Inputs for backprobagation
        self.inputs = inputs
        # Implimentation of the ReLU Activation Function
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward_pass(self, prev_upstream_gradient):
        # This is an implimentation of the derivative of ReLU Activation Function.
        self.upstream_gradient = prev_upstream_gradient.copy()
        
        # Zero gradient where input values were negative
        self.upstream_gradient[self.inputs <= 0] = 0
        
        return self.upstream_gradient
        
class ActivationLinear(Activation):
    # Linear Activation Function
    def forward_pass(self, inputs):
        # Save Inputs for backprobagation
        self.inputs = inputs
        # Implimentation of the Linear Activation function.
        self.output = inputs
        return self.output
        
    def backward_pass(self, prev_upstream_gradient):
        # derivative of the Linear Activation Function which is 1.
        self.upstream_gradient = prev_upstream_gradient.copy()
        
        return self.upstream_gradient
    
class Cost:
    # Calculate the cost depending on the loss function defined by the forward_pass method.
    def calculate(self, y_pred, y_true):
        # Get the sample losses array
        sample_losses = self.forward_pass(y_pred, y_true)
        # Average the losses values to get the cost value
        data_losses = np.mean(sample_losses)
        return data_losses
    
class LossMeanSquareError(Cost):
    # Forward pass
    def forward_pass(self, y_pred, y_true):
        # Calculate squared errors for each sample
        squared_errors = np.square(y_true - y_pred)
        return squared_errors

    def backward_pass(self, predictions, y_true):
        # Number of samples
        samples = len(predictions)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        output_number = len(predictions[0])
        
        # Calculate the derivative of the Mean Square Error wrt predictions
        self.upstream_gradient = -2 * (y_true - predictions) / output_number
        self.upstream_gradient = self.upstream_gradient / samples
        
        return self.upstream_gradient
    
class Optimizer_SGD():
    def __init__(self, learning_rate: float) ->  None:
        # Initialize optimizer - set settings
        self.learning_rate = learning_rate
    
    # Update parameters
    def update_params(self, layer: DenseLayer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases 
    


    
