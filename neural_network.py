from random import random
from typing import List

import numpy as np

from neuron import Neuron

# np.random.seed(1)


class NeuralNetwork:
    def __init__(self, layers: List[int]):
        self.layers: List[List[Neuron]] = []
        self.add_layers(layers)
        self.setup()

    def __repr__(self):
        string = ""
        for layer in self.layers:
            for neuron in layer:
                string += neuron.__repr__()
            string += "\n"
        return string

    def add_layers(self, layers):  # Agrega neuronas sin parametros
        for i, layer in enumerate(layers, start=1):
            self.layers.append([Neuron(i, j + 1) for j in range(layer)])

    def setup(self):  # Agrega listas de pesos, sesgos, neuronas previas y neuronas siguientes
        for i, layer in enumerate(self.layers):
            for neuron in layer:
                if i == 0:  # Capa de entrada
                    neuron.next_neurons = self.layers[i + 1]
                elif i == len(self.layers) - 1:  # Capas salida
                    neuron.weights = [0.10 * np.random.randn() for _ in range(len(self.layers[i - 1]))]
                    neuron.bias = 1
                    neuron.prev_neurons = self.layers[i - 1]
                else:  # Capa ocultas
                    neuron.weights = [0.10 * np.random.randn() for _ in range(len(self.layers[i - 1]))]
                    neuron.bias = 1
                    neuron.prev_neurons = self.layers[i - 1]
                    neuron.next_neurons = self.layers[i + 1]

    def predict(self, inputs):
        # outputs = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                for neuron, x in zip(layer, inputs):
                    neuron.value = x
                    neuron.need_activation = False
            else:
                for neuron in layer:
                    neuron.calculate_value()
                    if i == len(self.layers) - 1:
                        output = neuron.activate()
        return output

    def update_weights(self, label, lr):
        for layer in reversed(self.layers[1:]):
            for neuron in layer:
                neuron.calculate_new_weight(label, lr)

        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.weights = neuron.new_weights
                neuron.bias = neuron.new_bias

    def reset_values(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.value = 0

    def train(self, X, y, n_epochs, lr=0.1):
        for epoch in range(n_epochs):
            correct_prediction = 0
            error = 0
            for x, label in zip(X, y):
                output = self.predict(x)
                predicted = 0 if output < 0.5 else 1
                correct_prediction += int(predicted == label)
                error += (output - label) ** 2.0
                self.update_weights(label, lr)
                self.reset_values()

            accuracy = correct_prediction / len(X) * 100
            loss = error / len(X)
            print("Epoch {}/{}, Loss: {:.8f}, Accuracy: {:.3f}".format(epoch + 1, n_epochs, loss, accuracy))


def mean_squared_error(actual, predicted):
    sum_square_error = sum(
        (actual[i] - predicted[i]) ** 2.0 for i in range(len(actual))
    )
    return 1.0 / len(actual) * sum_square_error


if __name__ == "__main__":

    nn = NeuralNetwork([13, 2, 1])

    nn.predict([1, 3, 4])

    nn.update_weights(1)

    for i, layer in enumerate(nn.layers, start=1):
        for neuron in layer:
            print(neuron)
            print(f"x -> {neuron.value}")
            if i == 1:
                print(f"Sgtes -> {neuron.next_neurons}")
            elif i != len(nn.layers):
                print(f"Pesos actuales -> {neuron.weights}")
                print(f"Pesos nuevos -> {neuron.new_weights}")
                print(f"Previas -> {neuron.prev_neurons}")
                print(f"Sgtes -> {neuron.next_neurons}")
            else:
                print(f"Pesos actuales -> {neuron.weights}")
                print(f"Pesos nuevos -> {neuron.new_weights}")
                print(f"Previas -> {neuron.prev_neurons}\n")
        print()
