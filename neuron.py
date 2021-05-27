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
        elif not self.next_neurons:
            return f"""Neuron ({self.layer},{self.pos})
x -> {self.activate()}
Pesos -> {self.weights}
Bias -> {self.bias}
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
        # print(f"Value for neuron {self.layer},{self.pos}...")
        for prev_neuron, weight in zip(self.prev_neurons, self.weights):
            if not prev_neuron.need_activation:
                self.value += prev_neuron.value * weight
                # print(f"prev value {prev_neuron.value} weight {weight} value {self.value}")
            else:
                self.value += prev_neuron.activate() * weight
                # print(f"prev value {prev_neuron.activate()} weight {weight} value {self.value}")
        self.value += self.bias
        # print(f"after bias {self.value}")
        # print()

    def calculate_new_weight(self, label, learning_rate):
        # print(f"Weights and bias for neuron {self.layer},{self.pos}...")
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
        # print(f"activation {self.activate()} derivative {self.derivative()}")
        # print(f"delta {self.delta}")

        # for weight, prev_neuron in zip(self.weights, self.prev_neurons):
        #     print(f"weight {weight} prev neuron activate {prev_neuron.activate()}")
        #     print(f"new weight {weight - (learning_rate * self.delta * prev_neuron.activate())}")
        new_weights = [
            weight - (learning_rate * self.delta * prev_neuron.activate())
            for weight, prev_neuron in zip(self.weights, self.prev_neurons)
        ]
        self.new_weights = new_weights
        self.new_bias = self.bias - (learning_rate * self.delta)
        # print(f"bias {self.bias}")
        # print(f"new bias {self.new_bias}")
        # print()
