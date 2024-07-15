# Importing the necessary modules needed for the demo.
from models import neural_network as nn
from models import model as md
import utils as ut
import numpy as np
import matplotlib.pyplot as plt

# Defining some global variables
samples = 1000
seed = 42
batch_size = 32
epochs = 30000
learning_rate = 0.01

if __name__ == '__main__':
    # Create DataSet
    sin_wave = ut.SinData(samples, seed, batch_size)
    x_values, y_values = sin_wave.x_values.reshape(-1, 1), sin_wave.y_values.reshape(-1, 1)
    
    # Create the model
    model = md.Model()

    # Add layers to the model
    model.add(nn.DenseLayer(1, 16))
    model.add(nn.ActivationReLU())
    model.add(nn.DenseLayer(16, 16))
    model.add(nn.ActivationReLU())
    model.add(nn.DenseLayer(16, 1))
    model.add(nn.ActivationLinear())

    # Compile the model
    # Defining the loss function and the optimizer that will be used in the learning process.
    model.compile(loss=nn.LossMeanSquareError(), optimizer=nn.Optimizer_SGD(learning_rate=learning_rate))
    
    # Train the model using the fit function.
    model.fit(x_values, y_values, epochs=epochs)
    
    ut.ModelConverter.generate_c_code(model, './c_model_implementation/model_weights.c')
    
    