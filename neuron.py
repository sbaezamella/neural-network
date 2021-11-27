import math
from typing import List


class Neuron:
    def __init__(self, layer, pos):
        self.layer = layer
        self.pos = pos
        self.value = 0
        self.weights: List[float] = []
        self.new_weights: List[float] = []
        self.prev_neurons: List[Neuron] = []
        self.next_neurons: List[Neuron] = []
        self.need_activation: bool = True

    def __repr__(self) -> str:
        if not self.prev_neurons:
            return f"""Neuron ({self.layer},{self.pos})
x -> {self.value}
"""
        else:
            return f"""Neuron ({self.layer},{self.pos})
x -> {self.activate()}
Pesos -> {self.weights}
Bias -> {self.bias}
"""

    def activate(self):
        return 1 / (1 + math.exp(-self.value))

    def derivative(self):
        activate = self.activate()  # Calcular una sola vez
        return activate * (1 - activate)

    def calculate_value(self):
        for prev_neuron, weight in zip(self.prev_neurons, self.weights):
            if not prev_neuron.need_activation:
                self.value += prev_neuron.value * weight
            else:
                self.value += prev_neuron.activate() * weight
        self.value += self.bias

    def calculate_new_weight(self, label, learning_rate):
        if not self.next_neurons:  # Capa de salida
            self.delta = (self.activate() - label) * self.derivative()
        else:
            next_sum = 0
            for next_neuron in self.next_neurons:
                for prev_neuron, next_weight in zip(
                    next_neuron.prev_neurons, next_neuron.weights
                ):
                    if self == prev_neuron:
                        next_sum += next_neuron.delta * next_weight
                        break
            self.delta = next_sum * self.derivative()

        new_weights = [
            weight - (learning_rate * self.delta * prev_neuron.activate())
            for weight, prev_neuron in zip(self.weights, self.prev_neurons)
        ]
        self.new_weights = new_weights
        self.new_bias = self.bias - (learning_rate * self.delta)
