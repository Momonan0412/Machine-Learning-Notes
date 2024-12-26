import numpy as np
class NetworkInitializer:
    @staticmethod
    def initialize_weights(layers):
        """
        Initialize random weights for the connections between each pair of consecutive layers.
        
        Example:
            num_inputs = 3, num_hidden = [3, 5], num_outputs = 2
            Layer structure: [3 (input), 3 (hidden), 5 (hidden), 2 (output)]
            
            This creates weight matrices for:
                - Connections from the input layer (3 neurons) to the first hidden layer (3 neurons): 3 x 3
                - Connections from the first hidden layer (3 neurons) to the second hidden layer (5 neurons): 3 x 5
                - Connections from the second hidden layer (5 neurons) to the output layer (2 neurons): 5 x 2
        """

        weights = []  # Store all weight matrices

        for i in range(len(layers) - 1):
            """
            Generate a random weight matrix for layers[i] to layers[i+1].
            Shape: [number of neurons in current layer, number of neurons in next layer].
            """
            w = np.random.rand(layers[i], layers[i + 1])  # Random values in the range [0, 1)
            weights.append(w)  # Append the weight matrix to the list

            # Debug/Visualization
            # print("Layer connection:", i, "->", i + 1)  # Display the connection between layers
            # print("Weight matrix:\n", w)  # Display the generated weight matrix
        return weights
    
    @staticmethod
    def initialize_derivatives(layers):
        """
        Initialize a container for storing derivatives (gradients).

        Explanation:
        - The number of weight matrices in the network is equal to the number of 
        derivative matrices required during backpropagation.
        - Each derivative matrix corresponds to the weights between two consecutive layers
        and has the same shape as the weight matrix.

        Process:
        - For each pair of consecutive layers, a derivative matrix is created with
        dimensions equal to (number of neurons in the current layer) x (number of neurons in the next layer).
        - The container `derivatives` stores all these matrices.

        Debugging:
        - Prints each derivative matrix during initialization for visualization.
        """
        derivatives = []
        for i in range(len(layers) - 1):
            # Create a zero matrix for derivatives matching the weight dimensions
            derivative = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(derivative)
            # Debugging output
            # print("Derivative #", i)
            # print(derivative)
        return derivatives

    @staticmethod
    def initialize_activations(layers):
        """
        Initialize a container for storing activations.

        Explanation:
        - The number of activation arrays in the network is equal to the number of layers.
        - Each activation array corresponds to the outputs (activations) of the neurons in a layer.

        Process:
        - For each layer in the network, an activation array is created with
        dimensions equal to the number of neurons in that layer.
        - The container `_activations` stores all these arrays.

        Debugging:
        - Prints each activation array during initialization for visualization.
        """
        activations = []
        for i in range(len(layers)):
            # Create a zero array for activations matching the number of neurons in the layer
            activation = np.zeros(layers[i])
            activations.append(activation)
            # Debugging output
            # print("Activation #", i)
            # print(activation)
        return activations