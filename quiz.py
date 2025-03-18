import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
inputs = [
    [0.70, 2.5, 0.35, 0.80],
    [1.80, 6.5, 4.5, 1.60],
    [0.75, 2.8, 0.40, 0.85],
    [0.72, 2.6, 0.38, 0.82],
    [1.75, 6.3, 4.3, 1.55],
    [1.85, 6.7, 4.7, 1.65]
]

#Piegeon = 1, Eagle = 0
target = [1, 0, 1, 1, 0, 0]


weights = [0.05, -0.15, 0.25, 0.10]
bias = 0.07
learning_rate = 0.08

#training stage
history = []
rms_history = []

for epoch in range(0, 1000):
    i = 0
    history = []
    total_error = 0
    while (i < len(inputs)):
        sum = inputs[i][0] * weights[0] + inputs[i][1] * weights[1] + inputs[i][2] * weights[2] +inputs[i][3] * weights[3]+ bias
        sigmoid =  1/(1 + np.exp(-sum))
        Error = target[i] - sigmoid
        total_error += Error**2
        if (sigmoid >= 0.5):
            output = "Piegon"
        else:
            output = "Eagle"
        history.append([inputs[i][0], inputs[i][1], inputs[i][2], inputs[i][3], weights[0], weights[1], weights[2], weights[3], bias, output, sigmoid, Error])
        weights[0] += learning_rate * (Error) * inputs[i][0]
        weights[1] += learning_rate * (Error) * inputs[i][1]
        weights[2] += learning_rate * (Error) * inputs[i][2]
        weights[3] += learning_rate * (Error) * inputs[i][3]
        bias += learning_rate * (Error)
        i += 1
    RMS = np.sqrt(total_error/len(inputs))
    rms_history.append(RMS)  # Store RMS values

    history.append(["Epoch: ", epoch + 1, "", "", "", "", "", "", "RMS:", RMS])
    columns = [	"Wingspan (cm)", "Beak Length (cm)", "Weight (kg)", "Flying Speed (km/h)", "W1", "W2", "W3", "W4", "bias", "output", "sigmoid", "Error"]
    df = pd.DataFrame(history, columns=columns)
    display(df)
    print()
print("FINISHED TRAINING")

#testing stage
testing = [
    [1.78, 6.4, 4.4,1.57],
    [0.74, 2.7, 0.39, 0.83]
]
testing_history = []
for i in range(0,2):
    sum = testing[i][0] * weights[0] + testing[i][1] * weights[1] + testing[i][2] * weights[2] + testing[i][3] * weights[3] + bias
    sigmoid = 1/(1 + np.exp(-sum))
    if (sigmoid >= 0.5):
        output = "Piegeon"
    else:
        output = "Eagle"
    testing_history.append([testing[i][0], testing[i][1], testing[i][2], testing[i][3], weights[0], weights[1], weights[2], weights[3], bias, output, sigmoid])

print("\nTesting stage:")
columns = [	"Wingspan (cm)", "Beak Length (cm)", "Weight (kg)", "Flying Speed (km/h)", "W1", "W2", "W3", "W4", "bias", "output", "sigmoid"]
df = pd.DataFrame(testing_history, columns=columns)
display(df)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(rms_history) + 1), rms_history)  
plt.xlabel("Epochs")
plt.ylabel("RMS Error")
plt.title("RMS Error Over Epochs")
plt.grid()
plt.show()
