# Embedded AI Using Neural Networks from Scratch

This project documents the process of building your own neural network for embedded applications from scratch. Starting with building neural network utilities using Python with the help of NumPy, we will develop the training part in Python. For the embedded part, we will use the C programming language to build a neural network that can use pre-trained models generated in Python and perform forward propagation successfully.

We will use the example of generating a sine wave for demonstration purposes. The model will consist of 1 input neuron representing `x`, 1 output neuron representing `y = sin(x)`, and two hidden layers with 16 neurons each.

## Documentation

The [docs](docs/documentation.md) folder contains all the documentation about the theory and the thinking process behind creating the neural network. It includes simple explanations of fundamental topics such as forward propagation, backpropagation, layers, neurons, activation functions, etc.

## Setup

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/Med-Karim-Ben-Boubaker/Neural-Network-From-Scratch.git
    cd Neural-Network-From-Scratch
    ```

2. **Create a virtual environment:**

    ```sh
    python3 -m venv myenv
    ```

3. **Activate the virtual environment:**

    On Windows:
    ```sh
    myenv\Scripts\activate
    ```

    On macOS and Linux:
    ```sh
    source myenv/bin/activate
    ```

4. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

5. **Run the project:**

    Open the `main.ipynb` Jupyter notebook and run all the cells to get plots of predictions and the training process, or you can simply run the `run.py` file:

    ```sh
    cd my_project/
    python3 run.py
    ```

    This script will create, train, and generate the model ready to be used by the C program simulating its implementation in an embedded system.

    You can find the generated file named `model_weights.c`. Below is an example of generated weights for a layer:

    ```c
    // Define layer0 weights
    float layer0_weights[16] = {
        -0.08466117829084396,
        0.5169710516929626,
        0.18730637431144714,
        0.5202388167381287,
        0.31740641593933105,
        -0.09392301738262177,
        0.3741433918476105,
        -0.15243594348430634,
        0.17482277750968933,
        0.3164679706096649,
        0.12593820691108704,
        0.5449357628822327,
        -0.09343516081571579,
        -0.06662387400865555,
        0.11048824340105057,
        0.3304210305213928
    };
    ```

6. **Build the C model implementation:**

    Go to the `c_model_implementation/` folder and run `make` to build the project:

    ```sh
    cd c_model_implementation
    make
    ```

7. **Run the executable:**

    ```sh
    ./main_program
    ```

