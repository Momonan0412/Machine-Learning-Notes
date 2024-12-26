import numpy as np
from activate_neuron import ActivationFunctions
from initialize_neuron import NetworkInitializer
from backpropagation import Backpropagate
class MLP:
    
    def __init__(self, num_inputs=2, num_hidden=[5], num_outputs=1):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        # Combine input, hidden, and output layers into a single list
        self._layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        # Call Set Weights
        self._initialize_network()
        
    def __str__(self):
        """
        Returns a string representation of the MLP class, including the number of inputs,
        hidden layers, outputs, and the current state of weights, activations, and derivatives.
        """
        # Create a readable representation of the weights
        weights_str = "\n".join(
            [f"Layer {i} -> {i + 1}:\n{w}" for i, w in enumerate(self._get_weights())]
        )
        
        # Create a readable representation of the activations
        activations_str = "\n".join(
            [f"Layer {i} activations: {a}" for i, a in enumerate(self._get_activations())]
        )
        
        # Create a readable representation of the derivatives
        derivatives_str = "\n".join(
            [f"Layer {i} -> {i + 1} derivatives:\n{d}" for i, d in enumerate(self._get_derivatives())]
        )

        # Combine everything into a single formatted string
        return (
            f"MLP Configuration:\n"
            f"Number of Inputs: {self._get_num_inputs()}\n"
            f"Hidden Layers: {self._get_num_hidden()}\n"
            f"Number of Outputs: {self._get_num_outputs()}\n\n"
            f"Weights:\n{weights_str}\n\n"
            f"Activations:\n{activations_str}\n\n"
            f"Derivatives:\n{derivatives_str}"
        )

    
    def _initialize_network(self):
        self._weights = NetworkInitializer.initialize_weights(self._layers)
        self._activations = NetworkInitializer.initialize_activations(self._layers)
        self._derivatives = NetworkInitializer.initialize_derivatives(self._layers)
        self._biases = NetworkInitializer.initialize_bias(self._layers)
        self._biases_derivatives = NetworkInitializer.initialize_derivatives(self._layers)
        
    """
    Example:
    Step 1: Input to Hidden Layer (with Bias)

    1. Input Vector (x):
        x = [0.3, 0.1]
        
    2. Weight Matrix (W):
        W = [[0.73136551, 0.94651301, 0.83236155, 0.91204298, 0.2280451 ],
            [0.8482137,  0.07064327, 0.56306696, 0.18116871, 0.53751426]]
            
    3. Bias Vector (b):
        b = [0.04941694, 0.89881235, 0.43248694, 0.23011943, 0.3978821]
        
    4. Perform Dot Product (Input Vector x Weight Matrix):
        net_input = x . W
        net_input = [
            (0.3 * 0.73136551 + 0.1 * 0.8482137),
            (0.3 * 0.94651301 + 0.1 * 0.07064327),
            (0.3 * 0.83236155 + 0.1 * 0.56306696),
            (0.3 * 0.91204298 + 0.1 * 0.18116871),
            (0.3 * 0.2280451 + 0.1 * 0.53751426)
        ]

        net_input = [0.30423102, 0.29101823, 0.30601516, 0.29172977, 0.12216496]

    5. Add Bias to the Net Input:
        activation = net_input + b
        activation = [
            0.30423102 + 0.04941694,
            0.29101823 + 0.89881235,
            0.30601516 + 0.43248694,
            0.29172977 + 0.23011943,
            0.12216496 + 0.3978821
        ]

        activation = [0.35364796, 1.18983058, 0.7385021, 0.5218492, 0.52004706]

    ---

    Step 2: Hidden to Output Layer (with Bias)

    1. Input Vector from Hidden Layer (x):
        x = [0.58750192, 0.76671076, 0.67666822, 0.62758007, 0.62715877]

    2. Weight Matrix (W):
        W = [[0.81583146],
            [0.49811756],
            [0.10897875],
            [0.27196133],
            [0.77692976]]
            
    3. Bias Vector (b):
        b = [0.91637537]

    4. Perform Dot Product (Hidden Layer Input Vector x Weight Matrix):
        net_input_output = x . W
        net_input_output = [
            (0.58750192 * 0.81583146),
            (0.76671076 * 0.49811756),
            (0.67666822 * 0.10897875),
            (0.62758007 * 0.27196133),
            (0.62715877 * 0.77692976)
        ]
        
        net_input_output = [0.47912966, 0.38186682, 0.07367063, 0.17085558, 0.48777492]

    5. Add Bias to the Net Input:
        activation_output = net_input_output + b
        activation_output = [
            0.47912966 + 0.91637537
        ]

        activation_output = [2.50967298]
    """
    def forward_propagate(self, inputs):
        """
        Perform forward propagation through the network to calculate output activations.

        Explanation:
        - Forward propagation is the process of passing inputs through the network
        layer by layer to compute the activations for each layer.
        - These activations are used to make predictions or as the basis for backpropagation.

        Steps:
        1. The input layer activations are set directly from the provided `inputs` parameter.
        - The input layer is constant and does not undergo any transformation.
        - It serves as the starting point for calculations in subsequent layers.
        2. For each layer, the net inputs are calculated using a weighted sum of the 
        activations from the previous layer (matrix multiplication with weights).
        3. The net inputs are passed through the activation function (sigmoid) to produce
        the current layer's activations.
        4. The activations are stored in `_activations` for each layer.

        Parameters:
        - inputs: A list or numpy array representing the input values to the network.

        Returns:
        - activations: The final output activations of the network (predictions).

        Example Process:
        - For a network with 3 layers:
        1. Input Layer: Activations are directly the input values.
        2. Hidden Layer 1:
            - Net Input: h1 = a0 * w0 (weighted sum of input activations and weights).
            - Activation: a1 = sigmoid(h1).
        3. Output Layer:
            - Net Input: h2 = a1 * w1.
            - Activation: a2 = sigmoid(h2).
        """
        # Initialize activations with the input values
        activations = inputs  
        self._activations[0] = inputs  # The input layer is stored as is since it remains constant

        # Iterate through each layer's weights
        for i, weight in enumerate(self._weights):  
            # Calculate the net inputs to the next layer using matrix multiplication
            net_inputs = np.dot(activations, weight)  + self._biases[i] # 12/26/2024 Added Bias
            # print(f"( Input:{activations} * Weight:{weight} ) + {self._biases[i]}")
            # Compute the activations using the sigmoid function
            activations = ActivationFunctions.sigmoid(net_inputs)
            
            # Store the activations for the current layer in the activations list
            self._activations[i + 1] = activations  

            """
            Example:
            - For the second layer:
            h2 = a1 * w1  # Weighted sum of previous activations and weights
            a2 = sigmoid(h2)  # Apply sigmoid to get current activations
            """
            
        # Return the final layer's activations (network's output)
        return activations


    def _get_weights(self):
        return self._weights
    
    def _get_biases(self):
        return self._biases
    
    def _get_activations(self):
        return self._activations
    
    def _get_derivatives(self):
        return self._derivatives
    
    def _get_biases_derivatives(self):
        return self._biases_derivatives
    
    def _get_num_inputs(self):
        return self.num_inputs
    
    def _get_num_hidden(self):
        return self.num_hidden 
    
    def _get_num_outputs(self):
        return self.num_outputs
