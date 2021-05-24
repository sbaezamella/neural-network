from random import random
from typing import List

import numpy as np


class Neuron:
    def __init__(self, layer, pos):
        self.layer = layer
        self.pos = pos
        self.value = 0
        self.weights: List[float] = []
        self.new_weights: List[float] = []
        self.prev_neurons: List[Neuron] = []
        self.next_neurons: List[Neuron] = []
    
    def __repr__(self) -> str:
        return f"Neuron ({self.layer},{self.pos})"

    def activate(self):
        return 1 / (1 + np.exp(-self.value))

    def derivative(self):
        activate = self.activate()  # Calcular una sola vez la funcion sigmoid
        return activate * (1 - activate)

    def calculate_value(self):
        for neuron, weight in zip(self.prev_neurons, self.weights):
            self.value += neuron.activate() * weight

    def calculate_new_weight(self, learning_rate, label):
        if not self.next_neurons:
            self.delta = (self.activate() - label) * self.derivative()
        else:
            next_sum = []
            for next_neuron in self.next_neurons: # Fix
                for prev_neuron, weight in zip(next_neuron.prev_neurons, next_neuron.weights):
                    if self == prev_neuron:
                        # print(prev_neuron, weight)
                        next_sum += next_neuron.delta * weight
            self.delta = next_sum * self.derivative()

        gradiente = self.delta * self.activate()

        self.new_weights = [weight - (learning_rate * gradiente) for weight in self.weights]


class NeuralNetwork:
    def __init__(self, layers): # nn = NeuralNetwork([5, 3, 2])
        self.layers: List[List[Neuron]] = []
        self.add_layers(layers)
        self.randomize_weights()

    def add_layers(self, layers):  # Agrega neuronas sin parametros
        for i, layer in enumerate(layers, start=1):
            self.layers.append([Neuron(i, j + 1) for j in range(layer)]) # list comprehension

    def randomize_weights(self):  # Agrega listas de pesos, neuronas previas y neuronas siguientes
        for i, layer in enumerate(self.layers):
            for neuron in layer:
                if i == 0:  # Capa de entrada
                    neuron.next_neurons = self.layers[i + 1]
                elif i == len(self.layers) - 1:  # Capas salida
                    neuron.weights = [random() for _ in range(len(self.layers[i - 1]))]
                    neuron.prev_neurons = self.layers[i - 1]
                else:  # Capa ocultas
                    neuron.weights = [random() for _ in range(len(self.layers[i - 1]))]
                    neuron.prev_neurons = self.layers[i - 1]
                    neuron.next_neurons = self.layers[i + 1]

    def predict(self, inputs):
        outputs = []
        for i, layer in enumerate(self.layers):
            if i == 0:  # Capa de entrada
                for neuron, x in zip(layer, inputs):
                    neuron.value = x
            else:  # Forward propagate
                for neuron in layer:
                    neuron.calculate_value()
                    if i == len(self.layers) - 1:
                        outputs.append(neuron.value)
        self.outputs = outputs

    def update_weights(self):
        pass


nn = NeuralNetwork([3, 3, 1])
for i, layer in enumerate(nn.layers, start=1):
    print(f"Capa {i}")
    for j, neuron in enumerate(layer, start=1):
        if i == 1:
            print(neuron)
            print(f"Sgtes -> {neuron.next_neurons}")
            print()
        elif i != len(nn.layers):
            print(neuron)
            print(f"Peso -> {neuron.weights}")
            print(f"Previas -> {neuron.prev_neurons}")
            print(f"Sgtes -> {neuron.next_neurons}")
            print()
        else:
            print(neuron)
            print(f"Peso -> {neuron.weights}")
            print(f"Previas -> {neuron.prev_neurons}")
            print()
            
for layer in reversed(nn.layers):
    for neuron in layer:
        # print(neuron)
        neuron.calculate_new_weight(0.1, 1)


# nn.predict([4, 6, 8, 7, 1, 3])
# print(nn.outputs)
