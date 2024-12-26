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
            net_inputs = np.dot(activations, weight)  
            
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
    
    def _get_activations(self):
        return self._activations
    
    def _get_derivatives(self):
        return self._derivatives
    
    def _get_num_inputs(self):
        return self.num_inputs
    
    def _get_num_hidden(self):
        return self.num_hidden 
    
    def _get_num_outputs(self):
        return self.num_outputs
