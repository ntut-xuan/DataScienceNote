from sklearn.neural_network import MLPClassifier
import numpy as np

data = [[35, 67], [12, 75], [16, 89], [45, 56], [10, 90]]
label = [1, 0, 1, 1, 0]

data_array = np.array(data)
label_array = np.array(label)

clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=2, random_state=1, verbose=1, max_iter=400)
clf.fit(data_array, label_array)

predicted = clf.predict(np.array([25, 70]).reshape((1, -1)))

print(clf.coefs_)
print(predicted[0])