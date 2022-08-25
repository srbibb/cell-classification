from multiprocessing.util import ForkAwareLocal
import numpy as np
import pandas as pd
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from keras.preprocessing.image import load_img

import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

cells = pd.read_csv("D:/icm/data/results/pca_features.txt", header=None)
cells.dropna(axis=1, how="all", inplace=True)
filenames = []
with os.scandir("D:/icm/data/resized/") as files:
    for file in files:
        if file.name.endswith('.png'):
            filenames.append(file.name)
filenames = ["D:/icm/data/resized/" + x for x in filenames if x]

le = LabelEncoder()
kmeans = KMeans(n_clusters=5, random_state=25, n_init=5)
data = cells.apply(le.fit_transform)
#testing = testing.apply(le.fit_transform)

kmeans.fit(data)
predictions = kmeans.predict(data)

groups = {}
for file, cluster in zip(filenames,predictions):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

def view_cluster(cluster):
    plt.figure(figsize = (25,25))
    files = groups[cluster]
    for index, file in enumerate(files):
        plt.subplot(25,25,index+1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    filename = "kmeans_cluster_" + str(cluster) + ".png"
    cluster_path = os.path.join("D:/icm/data/cluster output/", filename)
    plt.savefig(cluster_path)

for i in range(len(groups)):
    print("Plotting cluster " + str(i))
    print(len(groups[i]))
    view_cluster(i)

plt.close("all")

def show_clusters():
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init="k-means++", n_clusters=5, n_init=4)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.8  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on the digits dataset (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

show_clusters()
