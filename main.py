import pandas as pd

from neural_network import NeuralNetwork

nn = NeuralNetwork([13, 5, 1])

# Lectura de datos desde dataset.csv
dataset_df = pd.read_csv("dataset.csv", usecols=[i for i in range(1, 15)])

# Reordenamiento
dataset_df_copy = dataset_df.sample(frac=1).reset_index(drop=True)

# Datos de entrada y etiquetas
n_columns = len(dataset_df_copy.columns)
data_input = dataset_df_copy.iloc[:, 0 : n_columns - 1]
data_label = dataset_df_copy.iloc[:, n_columns - 1]

# Normalizacion
norm_cols = ["duration_ms", "key", "loudness", "tempo", "time_signature"]
for feature_name in data_input.columns:
    if feature_name in norm_cols:
        min_val = data_input[feature_name].max()
        max_val = data_input[feature_name].min()
        data_input[feature_name] = (data_input[feature_name] - min_val) / (max_val - min_val)

# Division de datos
training_percent = 0.7
n_rows = len(dataset_df_copy.index)
n_train = round(training_percent * n_rows)
n_test = n_rows - n_train

X_train = data_input.iloc[0:n_train]
y_train = data_label.iloc[0:n_train]
X_test = data_input.iloc[n_test:]
y_test = data_label.iloc[n_test:]

# Entrenamiento
n_epochs = 1
nn.train(X_train.values, y_train.values, n_epochs, lr=0.1)
