import numpy as np
from random import random
from backpropagation import Backpropagate
from multilayer_perceptron import MLP
from neuron_trainer import Trainer

if __name__ == "__main__":
    # Scale inputs and targets
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])  # Scale targets to 0-1 range

    # Initialize MLP with proper weight initialization
    mlp = MLP(2, [5], 1)

    # Train the model
    trainer = Trainer(mlp)
    trainer.train(inputs, targets, epochs=50)

    # Test the model
    test_input = np.array([0.3, 0.1])
    test_target = np.array([0.4]) / 1.0  # Scale target
    output = mlp.forward_propagate(test_input)

    print()
    print(f"Expected: {test_target[0]}, Predicted: {output[0]}")