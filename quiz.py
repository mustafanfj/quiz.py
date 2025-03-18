import numpy as np
import pandas as pd
from IPython.display import display

inputs = [
    [70, 2.5, 0.35, 80],
    [180, 6.5, 4.5, 160],
    [75, 2.8, 0.40, 85],
    [72, 2.6, 0.38, 82],
    [175, 6.3, 4.3, 155],
    [185, 6.7, 4.7, 165]
]

# Pigeon = 1, Eagle = 0
target = [1, 0, 1, 1, 0, 0]

weights = [0.05, -0.15, 0.25, 0.10]
bias = 0.07
learning_rate = 0.08

# Training stage
for epoch in range(1000):
    total_error = 0
    history = []
    for i in range(len(inputs)):
        sum_value = np.dot(inputs[i], weights) + bias
        sum_value = np.clip(sum_value, -500, 500)  # Prevent overflow
        sigmoid = 1 / (1 + np.exp(-sum_value))

        error = target[i] - sigmoid
        total_error += error**2

        output = "Pigeon" if sigmoid >= 0.5 else "Eagle"

        history.append([*inputs[i], *weights, bias, output, sigmoid, error])

        # Weight and bias updates
        for j in range(len(weights)):
            weights[j] += learning_rate * error * inputs[i][j]
        bias += learning_rate * error

    RMS = np.sqrt(total_error / len(inputs))
    history.append(["Epoch:", epoch + 1, "", "", "", "", "", "", "RMS:", RMS])

    columns = ["Wingspan(cm)", "Beak Length(cm)", "Weight(kg)", "Flying Speed(km/h)",
               "W1", "W2", "W3", "W4", "Bias", "Output", "Sigmoid", "Error"]
    df = pd.DataFrame(history, columns=columns)
    display(df)

print("FINISHED TRAINING")

# Testing stage
testing = [
    [178, 6.4, 4.4, 157],
    [74, 2.7, 0.39, 83]
]

testing_history = []
for i in range(len(testing)):
    sum_value = np.dot(testing[i], weights) + bias
    sum_value = np.clip(sum_value, -500, 500)  # Prevent overflow
    sigmoid = 1 / (1 + np.exp(-sum_value))

    output = "Pigeon" if sigmoid >= 0.5 else "Eagle"
    testing_history.append([*testing[i], *weights, bias, output, sigmoid])

print("\nTesting stage:")
columns = ["Wingspan(cm)", "Beak Length(cm)", "Weight(kg)", "Flying Speed(km/h)",
           "W1", "W2", "W3", "W4", "Bias", "Output", "Sigmoid"]
df = pd.DataFrame(testing_history, columns=columns)
display(df)
