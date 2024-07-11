from .neural_network import DenseLayer, Activation
import numpy as np

# model.py
class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        
    def add(self, layer):
        self.layers.append(layer)
        
    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
    
    def fit(self, X, Y, epochs, print_every=100):
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
        print("Neural Network Training has been complete")
        
    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output
    
    def info(self):
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
                print(f"{layer.__class__.__name__}")
                
    def save_weights(self, file_path):
        weights_biases = {}
        for index, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                weights_biases[f'layer_{index}_weights'] = layer.weights
                weights_biases[f'layer_{index}_biases'] = layer.biases
                
        np.savez_compressed(file_path, **weights_biases)
        print(f"Weights and biases saved to {file_path}")
        
    def load_weights(self, file_path):
        with np.load(file_path) as data:
            for index, layer in enumerate(self.layers):
                if isinstance(layer, DenseLayer):
                    layer.weights = data[f'layer_{index}_weights']
                    layer.biases = data[f'layer_{i}_biases']
        print(f'Weights and biases loaded from {file_path}')
                
            
                
            
        
        
                