from backpropagation import Backpropagate
from multilayer_perceptron import MLP
from activate_neuron import ActivationFunctions
class Trainer:
    def __init__(self, mlp : MLP):
        self._mlp = mlp
        # print(self._mlp)
        
    def train(self, inputs, targets, epochs):
        """Orchestrates the training process."""
        for epoch in range(epochs):
            total_error = 0
            for input, target in zip(inputs, targets):
                # Step 1: Forward pass
                predicted = self._mlp.forward_propagate(input)

                # Step 2: Backward pass (calculates gradients)
                backpropagate = Backpropagate(self._mlp._get_activations(), 
                                              self._mlp._get_derivatives(), 
                                              self._mlp._get_weights(),
                                              self._mlp._get_biases(),
                                              self._mlp._get_biases_derivatives())
                
                backpropagate.backward_propagate(target-predicted)

                # Step 3: Weight updates (gradient descent)
                backpropagate.gradient_descent()

                # Step 4: Compute error
                total_error += ActivationFunctions.mean_squared_error(target, predicted)

            # Log progress
            # print(f"Epoch {epoch + 1}, Error: {total_error / len(inputs)}")