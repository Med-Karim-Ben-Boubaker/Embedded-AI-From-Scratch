# Import necessary modules
from .neural_network import DenseLayer, Activation
import numpy as np

# model.py
class Model:
    def __init__(self):
        """
        Initialize the Model with no layers, loss function, or optimizer.
        """
        self.layers = []
        self.loss = None
        self.optimizer = None
        
    def add(self, layer):
        """
        Add a layer to the model.
        
        Args:
            layer: An instance of a neural network layer (e.g., DenseLayer, Activation).
        """
        self.layers.append(layer)
        
    def compile(self, loss, optimizer):
        """
        Compile the model with a loss function and an optimizer.
        
        Args:
            loss: An instance of a loss function.
            optimizer: An instance of an optimizer.
        """
        self.loss = loss
        self.optimizer = optimizer
    
    def fit(self, X, Y, epochs, print_every=10000):
        """
        Train the model on the given data.
        
        Args:
            X: Input data.
            Y: Target data.
            epochs: Number of epochs to train.
            print_every: Frequency of printing loss information.
        """
        self.loss_history = []
        
        for epoch in range(epochs):
            # Forward pass
            output = X
            for layer in self.layers:
                output = layer.forward_pass(output)
            
            # Calculate loss
            loss_value = self.loss.calculate(output, Y)
            self.loss_history.append(loss_value)
            
            if epoch % print_every == 0:
                print(f'Epoch: {epoch}, Loss: {loss_value}')
                
            # Backward pass
            upstream_gradient = self.loss.backward_pass(output, Y)
            for layer in reversed(self.layers):
                upstream_gradient = layer.backward_pass(upstream_gradient)
                
            # Update weights and biases
            for layer in self.layers:
                if isinstance(layer, DenseLayer):
                    self.optimizer.update_params(layer)
        
        print("Neural Network Training has been completed")
        
    def predict(self, X):
        """
        Make predictions with the model on the given input data.
        
        Args:
            X: Input data.
        
        Returns:
            output: Predictions made by the model.
        """
        output = X
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output
    
    def info(self):
        """
        Print model information including the number and details of each layer.
        """
        print("Model Information:")
        print("===================")
        print(f"Number of layers: {len(self.layers)}")
        for i, layer in enumerate(self.layers):
            print(f"\nLayer {i + 1}:")
            if isinstance(layer, DenseLayer):
                print(f"  Type: DenseLayer")
                print(f"  Number of neurons: {layer.weights.shape[1]}")
                print(f"  Weights shape: {layer.weights.shape}")
                print(f"  Biases shape: {layer.biases.shape}")
                print(f"  Weights: {layer.weights}")
                print(f"  Biases: {layer.biases}")
            elif isinstance(layer, Activation):
                print(f"  Type: {layer.__class__.__name__}")
                
    def save_weights(self, file_path):
        """
        Save the weights and biases of the model to a file.
        
        Args:
            file_path: Path to save the weights and biases.
        """
        weights_biases = {}
        for index, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                weights_biases[f'layer_{index}_weights'] = layer.weights
                weights_biases[f'layer_{index}_biases'] = layer.biases
                
        np.savez_compressed(file_path, **weights_biases)
        print(f"Weights and biases saved to {file_path}")
        
    def load_weights(self, file_path):
        """
        Load the weights and biases of the model from a file.
        
        Args:
            file_path: Path to load the weights and biases from.
        """
        with np.load(file_path) as data:
            for index, layer in enumerate(self.layers):
                if isinstance(layer, DenseLayer):
                    layer.weights = data[f'layer_{index}_weights']
                    layer.biases = data[f'layer_{index}_biases']
        print(f'Weights and biases loaded from {file_path}')
        
    def extract_weights_biases(self):
        """
        Extract the weights and biases of the model as a dictionary.
        
        Returns:
            weights_biases: Dictionary containing the weights and biases.
        """
        weights_biases = {}
        for index, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                weights_biases[f'layer_{index}_weights'] = layer.weights.astype(np.float32)
                weights_biases[f'layer_{index}_biases'] = layer.biases.astype(np.float32)
        return weights_biases
