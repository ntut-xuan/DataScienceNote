import numpy as np
import random
import time
import matplotlib.pyplot as plt

def standard_random():
    std = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    return std


def active_function(x):
    return np.log(1+np.exp(x))


def calc(x_1s, x_2s, x_1, x_2, y_1, y_2, w, b):
    x_1s = np.transpose(data)[0]
    x_2s = np.transpose(data)[1]
    output = w[5] * y_1 + w[6] * y_2 + b[3]
    return (x_1s * w[1] + x_2s * w[3], x_1s * w[2] + x_2s * w[4], active_function(x_1), active_function(x_2), output)

data = [[35, 67], [12, 75], [16, 89], [45, 56], [10, 90]]
label = [1, 0, 1, 1, 0]

data_array = np.array(data)
label_array = np.array(label)

w = standard_random()
b = [0 for i in range(4)]

print(w)

x_1s = np.transpose(data)[0]
x_2s = np.transpose(data)[1]
x_1 = x_1s * w[1] + x_2s * w[3]
x_2 = x_1s * w[2] + x_2s * w[4]
y_1 = np.log(1+np.exp(x_1))
y_2 = np.log(1+np.exp(x_2))
output = w[5] * active_function(x_1*w[1]+x_2*w[3] + b[1]) + w[6] * active_function(x_1*w[2]+x_2*w[4] + b[2]) + b[3]
learn_rate = 0.01

for i in range(1000):
    x_1s = np.transpose(data)[0]
    x_2s = np.transpose(data)[1]
    x_1, x_2, y_1, y_2, output = calc(x_1s, x_2s, x_1, x_2, y_1, y_2, w, b)
    db3 = np.sum(-2 * (label_array - output))
    dw5 = np.sum((label_array - output) * y_1 * -2)
    dw6 = np.sum((label_array - output) * y_2 * -2)
    db1 = np.sum((label_array - output) * w[5] * (np.exp(x_1) / (1 + np.exp(x_1))) * -2)
    db2 = np.sum((label_array - output) * w[6] * (np.exp(x_2) / (1 + np.exp(x_2))) * -2)
    dw1 = np.sum((label_array - output) * w[5] * (np.exp(x_1) / (1 + np.exp(x_1))) * x_1s * -2)
    dw2 = np.sum((label_array - output) * w[6] * (np.exp(x_2) / (1 + np.exp(x_2))) * x_1s * -2)
    dw3 = np.sum((label_array - output) * w[5] * (np.exp(x_1) / (1 + np.exp(x_1))) * x_2s * -2)
    dw4 = np.sum((label_array - output) * w[6] * (np.exp(x_2) / (1 + np.exp(x_2))) * x_2s * -2)
    print("SSR =", np.sum(label_array - output) * np.sum(label_array - output))
    b[3] = b[3] - db3 * learn_rate
    b[2] = b[2] - db2 * learn_rate
    b[1] = b[1] - db1 * learn_rate
    w[6] = w[6] - dw6 * learn_rate
    w[5] = w[5] - dw5 * learn_rate
    w[4] = w[4] - dw4 * learn_rate
    w[3] = w[3] - dw3 * learn_rate
    w[2] = w[2] - dw2 * learn_rate
    w[1] = w[1] - dw1 * learn_rate
    print("b=", b)
    print("w=", w)

fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')

plot_x = np.linspace(0, 100, num=100)

table = np.array([[0 for _ in range(100)] for _ in range(100)])

for i in range(100):
    for j in range(100):
        x1 = i
        x2 = j
        #print(x1, x2)
        table[i][j] =   w[5] * (active_function(x1 * w[1] + x2 * w[3]) + b[1]) \
                      + w[6] * (active_function(x1 * w[2] + x2 * w[4]) + b[2]) \
                      + b[3]

ax.scatter(np.transpose(data_array)[0], np.transpose(data_array)[1], label_array)
ax.plot_surface(plot_x, plot_x, table, cmap='seismic')
plt.show()

print(table)

print(table[45][56])
print(table[25][70])