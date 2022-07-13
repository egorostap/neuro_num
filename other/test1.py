# нейронка самописная без скрытого слоя

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0, 0, 1],
                        [1, 1, 0],
                        [1, 0, 0]]) # 3 примера 3 входа на 3 нейрона
# training_inputs = np.array([0, 1, 1]) # 1 пример 3 входа на 3 нейрона
training_outputs = np.array([0, 1, 1]).T # 3 выхода на 3 нейрона
weights = np.array([0, 0, 0], dtype=float)# 3 веса
test_inputs = np.array([1, 0, 0]) #тестовая выборка

for i in range(100):
    outputs = sigmoid(training_inputs @ weights) # работа выходного нейрона
    err = training_outputs - outputs
    weights += training_inputs @ (err * outputs * (1-outputs))
    # weights += training_inputs @ (0.1 * err)


# print(err, weights, outputs)
print('__________________')
outputs_test = sigmoid(test_inputs @ weights) # работа выходного нейрона
print(outputs_test)