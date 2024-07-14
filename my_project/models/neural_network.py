# neural_network.py

# Import necessary modules
import numpy as np
import math

class DenseLayer:
    """
    A fully connected layer in a neural network.
    """
    def __init__(self, number_inputs: int, number_neurons: int):
        """
        Initialize a Dense Layer.

        Args:
            number_inputs: Number of input features.
            number_neurons: Number of neurons in the layer.
        """
        self.weights = 0.1 * np.random.randn(number_inputs, number_neurons).astype(np.float32)
        self.biases = np.zeros((1, number_neurons)).astype(np.float32)
        
    def forward_pass(self, inputs):
        """
        Perform the forward pass.

        Args:
            inputs: Input data.

        Returns:
            output: The result of the forward pass.
        """
        # Save Inputs for backpropagation
        self.inputs = inputs
        # Calculating output
        self.output = np.dot(inputs, self.weights) + self.biases
        
        return self.output
    
    def backward_pass(self, prev_upstream_gradient):
        """
        Perform the backward pass.

        Args:
            prev_upstream_gradient: The gradient flowing from the subsequent layer.

        Returns:
            upstream_gradient: The gradient to be passed to the preceding layer.
        """
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, prev_upstream_gradient)
        self.dbiases = np.sum(prev_upstream_gradient, axis=0, keepdims=True)
        # Gradients on inputs
        self.upstream_gradient = np.dot(prev_upstream_gradient, self.weights.T)
        
        return self.upstream_gradient

class Activation:
    """
    An interface for activation functions.
    """
    def __init__(self):
        pass
    
    def forward(self, inputs):
        pass
    
    def backward(self, prev_upstream_gradient):
        pass

class ActivationReLU(Activation):
    """
    ReLU Activation Function.
    """
    def forward_pass(self, inputs):
        """
        Perform the forward pass.

        Args:
            inputs: Input data.

        Returns:
            output: The result of the ReLU activation.
        """
        # Save Inputs for backpropagation
        self.inputs = inputs
        # Implementation of the ReLU Activation Function
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward_pass(self, prev_upstream_gradient):
        """
        Perform the backward pass.

        Args:
            prev_upstream_gradient: The gradient flowing from the subsequent layer.

        Returns:
            upstream_gradient: The gradient to be passed to the preceding layer.
        """
        # Implementation of the derivative of ReLU Activation Function
        self.upstream_gradient = prev_upstream_gradient.copy()
        
        # Zero gradient where input values were negative
        self.upstream_gradient[self.inputs <= 0] = 0
        
        return self.upstream_gradient

class ActivationLinear(Activation):
    """
    Linear Activation Function.
    """
    def forward_pass(self, inputs):
        """
        Perform the forward pass.

        Args:
            inputs: Input data.

        Returns:
            output: The result of the Linear activation.
        """
        # Save Inputs for backpropagation
        self.inputs = inputs
        # Implementation of the Linear Activation function
        self.output = inputs
        return self.output
        
    def backward_pass(self, prev_upstream_gradient):
        """
        Perform the backward pass.

        Args:
            prev_upstream_gradient: The gradient flowing from the subsequent layer.

        Returns:
            upstream_gradient: The gradient to be passed to the preceding layer.
        """
        # Derivative of the Linear Activation Function which is 1
        self.upstream_gradient = prev_upstream_gradient.copy()
        
        return self.upstream_gradient

class Cost:
    """
    Base class for calculating the cost (loss).
    """
    def calculate(self, y_pred, y_true):
        """
        Calculate the cost.

        Args:
            y_pred: Predicted values.
            y_true: True values.

        Returns:
            data_losses: The mean loss value.
        """
        # Get the sample losses array
        sample_losses = self.forward_pass(y_pred, y_true)
        # Average the losses values to get the cost value
        data_losses = np.mean(sample_losses)
        return data_losses

class LossMeanSquareError(Cost):
    """
    Mean Squared Error Loss.
    """
    def forward_pass(self, y_pred, y_true):
        """
        Calculate the forward pass for MSE.

        Args:
            y_pred: Predicted values.
            y_true: True values.

        Returns:
            squared_errors: The squared errors for each sample.
        """
        # Calculate squared errors for each sample
        squared_errors = np.square(y_true - y_pred)
        return squared_errors

    def backward_pass(self, predictions, y_true):
        """
        Calculate the backward pass for MSE.

        Args:
            predictions: Predicted values.
            y_true: True values.

        Returns:
            upstream_gradient: The gradient to be passed to the preceding layer.
        """
        # Number of samples
        samples = len(predictions)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        output_number = len(predictions[0])
        
        # Calculate the derivative of the Mean Square Error with respect to predictions
        self.upstream_gradient = -2 * (y_true - predictions) / output_number
        self.upstream_gradient = self.upstream_gradient / samples
        
        return self.upstream_gradient

class Optimizer_SGD:
    """
    Stochastic Gradient Descent (SGD) Optimizer.
    """
    def __init__(self, learning_rate: float):
        """
        Initialize the optimizer.

        Args:
            learning_rate: Learning rate for the optimizer.
        """
        self.learning_rate = learning_rate
    
    def update_params(self, layer: DenseLayer):
        """
        Update parameters of the layer.

        Args:
            layer: The layer to update.
        """
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
