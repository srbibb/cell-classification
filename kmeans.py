import numpy as np
import pandas as pd
import os

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import math

# Clusters the cells based on the features output by the VGG16 neural network. Clustering is done
# using a K-Means model. The cells are collected into 5 clusters.

# Reads in features extracted from the neural network and removes any empty values
cells = pd.read_csv("D:/icm/data/results/pca_features.txt", header=None)
cells.dropna(axis=1, how="all", inplace=True)

# Reads in the filenames read by the neural network and adds their path name to allow them to be viewed later
filenames = []
filenames = np.loadtxt(fname="D:/icm/data/results/filenames.txt", dtype=str, delimiter=",")
filenames = ["D:/icm/data/resized/composite/" + x for x in filenames if x]

# Initialises the K-Means model with 5 clusters and clusters the data
kmeans = KMeans(n_clusters=5, random_state=25, n_init=5)
kmeans.fit(cells)
predictions = kmeans.predict(cells)

# Assigns each file to the cluster it was predicted as belonging to
groups = {}
for file, cluster in zip(filenames,predictions):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

# Finds the largest cluster to set the size of the output
largest = 0
for i in range(0,4):
    if len(groups[i]) > largest:
        largest = len(groups[i])

group_size = math.ceil(math.sqrt(largest))

# Displays the final clusters as a grid containing each image assigned to the chosen cluster
# Also saves the images of the clusters
def view_cluster(cluster):
    plt.figure(figsize = (group_size, group_size))
    files = groups[cluster]
    for index, file in enumerate(files):
        plt.subplot(group_size, group_size,index+1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    filename = "kmeans_cluster_" + str(cluster) + ".png"
    cluster_path = os.path.join("D:/icm/data/cluster output/", filename)
    plt.savefig(cluster_path)

# Displays the cluster for each class
for i in range(len(groups)):
    print("Plotting cluster " + str(i))
    print(len(groups[i]))
    view_cluster(i)
plt.show()