import numpy as np

class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid activation function."""
        return x * (1.0 - x)
    
    @staticmethod
    def calculate_error(predicted, actual):
        """Calculate the error (predicted - actual)."""
        return predicted - actual

    @staticmethod
    def mean_squared_error(target, output):
        return np.average((target - output)**2)