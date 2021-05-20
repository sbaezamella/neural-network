from random import random, uniform
from typing import List, Optional

import numpy as np


class Neuron:
    def __init__(self, weights=None, prev_neurons=None):
        self.value = 0
        self.weights: List[float] = weights
        self.new_weights = None
        self.prev_neurons: List[Neuron] = prev_neurons
        self.next_neurons: List[Neuron] = None

    def activate(self):
        return 1 / (1 + np.exp(-self.value))

    def derivative(self):
        activate = self.activate()
        return activate * (1 - activate)

    def __str__(self):
        if self.weights and self.prev_neurons:
            format = f"Valor -> {self.value}, Pesos -> {self.weights}\n"
            for neuron in self.prev_neurons:
                format += f"Previas -> {neuron.value}\n"
            return format
        return f"Valor -> {self.value}\n"

    def calculate_value(self):
        for neuron, weight in zip(self.prev_neurons, self.weights):
            self.value += neuron.activate() * weight

    def calculate_new_weight(self, learning_rate, label):
        if not self.next_neurons:
            self.delta = (self.activate() - label) * self.derivative()
        else:
            next_sum = []
            for next_neuron, weight in zip(self.next_neurons, self.weights):
                next_sum += next_neuron.delta * weight
            self.delta = next_sum * self.derivative()
        
        gradiente = self.delta * self.activate()
        
        new_weights = []
        for weight in self.weights:
            new_weights = weight - (learning_rate * gradiente)


class NeuralNetwork:
    def __init__(self, layers: List[int]):
        self.layers: List[Neuron] = []
        for i, layer in enumerate(layers):
            neurons = []
            for j in range(layer):
                if i > 0:
                    weights = [random() for _ in range(len(self.layers[i - 1]))]
                    prev_neurons = [neuron for neuron in self.layers[i - 1]]
                    neuron = Neuron(weights=weights, prev_neurons=prev_neurons)
                    neurons.append(neuron)
                else:
                    neurons.append(Neuron())
            self.layers.append(neurons)

    def predict(self, inputs):
        output = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                for neuron, x in zip(layer, inputs):
                    neuron.value = x
            else:
                for neuron in layer:
                    neuron.calculate_value()
                    if i == len(self.layers) - 1:
                        output.append(neuron.value)
        return output


nn = NeuralNetwork([2, 5, 3])
print(nn.predict([1, 2]))