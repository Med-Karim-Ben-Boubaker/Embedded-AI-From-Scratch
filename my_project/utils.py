import numpy as np
import math

class SinData():
    def __init__(self, samples: int, seed: int, batch_size : int) -> None:
        # Generating Data Sin wave
        self.samples = samples
        self.seed = seed
        self.batch_size = batch_size
        np.random.seed(self.seed)
        
        # Generate a uniformly distributed set of random numbers in the range from
        # 0 to 2n, which covers a complete sine wave oscillation
        self.x_values = np.random.uniform(low=0, high=2*math.pi, size=self.samples)

        # Shuffle the values to guarantee they're not in order
        np.random.shuffle(self.x_values)

        # Calculate the corresponding sine values
        self.y_values = np.sin(self.x_values)
        
        # Add a small random number to each y value
        self.y_values += 0.2 * np.random.rand(*self.y_values.shape)
        
        # Split data into batches
        self.x_batches, self.y_batches = self.create_batches()
    
    def create_batches(self):
        # Calculate the number of full batches
        num_batches = self.samples // self.batch_size
        # Handle any remaining samples that don't fit into a full batch
        remainder = self.samples % self.batch_size
        
        # Split the x_values and y_values into batches
        x_batches = np.array_split(self.x_values[:num_batches * self.batch_size], num_batches)
        y_batches = np.array_split(self.y_values[:num_batches * self.batch_size], num_batches)
        
        x_batches = np.array(x_batches)
        y_batches = np.array(y_batches)
         
        return x_batches, y_batches

class ModelConverter():
    def generate_c_code(model, output_c_file_path):
        weights_biases = model.extract_weights_biases()
        
        layer_index = 0
        with open(output_c_file_path, 'w') as f:
            f.write('#include "model_utils.h"\n\n')
            for key in weights_biases:
                layer_type, _, param_type = key.split('_')
                param_name = f'{layer_type}{layer_index}_{param_type}'
                param_values = weights_biases[key].flatten()

                f.write(f'// Define {param_name.replace("_", " ")}\n')
                f.write(f'float {param_name}[{len(param_values)}] = {{\n')

                for i, value in enumerate(param_values):
                    f.write(f'    {value}')
                    if i != len(param_values) - 1:
                        f.write(',')
                    f.write('\n')

                f.write('};\n\n')
                
                if param_type == 'biases': layer_index += 1