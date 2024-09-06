import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import d3rlpy

dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-v0')

tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', verbose=2)
data = dataset.observations
reduced_data = tsne.fit_transform(data)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', marker='o')
plt.title('t-SNE Projection of 11-Dimensional Data to 2 Dimensions')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)
plt.savefig('tsne')