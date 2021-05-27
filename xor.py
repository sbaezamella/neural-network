from neural_network import NeuralNetwork

nn = NeuralNetwork([2, 2, 1])

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

y = [0, 1, 1, 0]

n_epochs = 100_000

nn.train(X, y, n_epochs)
        
for x, label in zip(X, y):
    output = nn.predict(x)
    print()
    print(f"input {x} output {output} esperado {label}")
    nn.reset_values()
