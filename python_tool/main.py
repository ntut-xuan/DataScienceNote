import numpy as np
import tool
import time
import warnings
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

n_samples = 300
random_state = 170
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.01)

points = noisy_circles[0]

array = np.zeros((n_samples, n_samples))

for i in range(n_samples):
    for j in range(n_samples):
        array[i][j] = np.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)

algo = tool.WPGMAClustering(n_clusters=2)
algo.fit(array)
algo.clustering()

print(algo.labels_)

colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(algo.labels_) + 1),
                )
            ))


plt.scatter(points[:, 0], points[:, 1], color=colors[algo.labels_])
plt.show()