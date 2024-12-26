import numpy as np
from activate_neuron import ActivationFunctions
class Backpropagate:
    def __init__(self, activations, derivatives, weights, biases, biases_derivatives, learning_rate = 0.1):
        self._activations = activations
        self._derivatives = derivatives
        self._biases_derivatives = biases_derivatives
        self._weights = weights
        self._biases = biases
        self._learning_rate = learning_rate
    
    
    """
    dE/dW_i = (y - a_[i + 1]) s'(h_[i + 1]) a_i 
    Where:
    (y - a_[i + 1]) => Error
    s'(h_[i + 1]) => Derivative of Sigmoid
    a_i => Activation Function
    
    
    dE/dW_[i - 1] = (y - a_[i + 1]) s'(h_[i + 1]) w_i s'(h_[i]) a_[i-1] 
    Where:
    (y - a_[i + 1]) s'(h_[i + 1]) w_i => the `error = np.dot(delta, self._weights[i].T)`
    """
    def backward_propagate(self, error):
        # Calculate the error between the predicted and actual output values
        # Loop over the list of derivatives in reverse order to propagate the error back from the output to the input layers
        for i in reversed(range(len(self._derivatives))):
            # Get the activation of the next layer (current layer's activation after forward propagation)
            activation = self._activations[i + 1]

            # Calculate the delta (error term) for the current layer
            # It is the error multiplied by the derivative of the activation function (sigmoid here)
            delta = error * ActivationFunctions.sigmoid_derivative(activation)
            self._biases_derivatives[i] = delta
            # Reshape delta to a column vector for matrix multiplication with the current activation
            # This is done to align the dimensions for proper dot product with the activations
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            # Get the activation values from the current layer
            current_activation = self._activations[i]

            # Reshape the current activation into a column vector (matching the shape needed for matrix multiplication)
            current_activation_reshaped = current_activation.reshape(current_activation.shape[0], -1)
            
            # Update the derivative (gradient) for the weight matrix at layer `i` using the dot product
            # This calculates the gradient of the error with respect to the weights for backpropagation
            self._derivatives[i] = np.dot(current_activation_reshaped, delta_reshaped)

            # Propagate the error backwards by calculating the error for the previous layer
            # It is done by multiplying the delta with the transpose of the weight matrix of the current layer
            error = np.dot(delta, self._weights[i].T)

            # Print the derivatives (gradients) for the current layer, useful for debugging
            # print(f"Derivatives #{i}", self._derivatives[i])
        # Return the error after backpropagation to be used in the next iteration or training step
        return error

    def gradient_descent(self):
        self._update_weights()
        # self._update_biases()

    def _update_weights(self):
        for i in range(len(self._weights)):
            weight = self._weights[i]
            # print("Before: ", weight)
            derivative = self._derivatives[i]
            weight += derivative * self._learning_rate
            # print("After: ", weight)
            
    def _update_biases(self):
        # Update biases
        # print("Biases: ")
        # print(self._biases[1])
        # print("Biases Derivatives: ")
        # print(self._bias_derivatives[-2])
        
        
        for i in range(len(self._biases)):
            bias = self._biases[i]
            bias_derivative = self._biases_derivatives[i]  # Gradient for the biases
            # print(f"Bias {i}: ", bias)
            # print(f"Bias Derivative Reshaped {i}: ", bias_derivative)
            # Update the biases: b_j = b_j - learning_rate * bias_derivative
            self._biases[i] -= self._learning_rate * bias_derivative
            print("Updated Bias: ", self._biases[i])
