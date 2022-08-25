from fileinput import filename
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import VGG16
from keras.models import Model

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA

import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd

path = "D:/icm/data/resized"
os.chdir(path)

cells = []
scaler = StandardScaler()
norm = Normalizer()

with os.scandir(path) as files:
    for file in files:
        if file.name.endswith('.png'):
            cells.append(file.name)

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def extract_features(file,model):
    img = load_img(file)
    img = np.array(img)
    reshaped_img = img.reshape(1,224,224,3)
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx)
    return features

data = {}
for cell in cells:
    features = extract_features(cell, model)
    data[cell] = features

filenames = np.array(list(data.keys()))
features = np.array(list(data.values()))
features = features.reshape(-1,4096)
labels = ["0", "1", "2", "3", "4", "5"]

file = open("D:/icm/data/results/features.txt", "w")
for i in range(len(features)):
    for j in range(len(features[i])):
        file.write(str(features[i][j]) + ",")
    file.write("\n")

file.close()
#features = scaler.fit_transform(features)
features = norm.fit_transform(features)

pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(features)

file = open("D:/icm/data/results/pca_features.txt", "w")
for i in range(len(x_pca)):
    for j in range(len(x_pca[i])):
        file.write(str(x_pca[i][j]) + ",")
    file.write("\n")

file.close()

file = open("D:/icm/data/results/filenames.txt", "w")
for i in range(len(filenames)):
    file.write(str(filenames[i]) + ",")

file.close()

'''
kmeans = KMeans(n_clusters=len(labels), random_state=None)
kmeans.fit(x_pca)

groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

def view_cluster(cluster):
    plt.figure(figsize = (30,30))
    files = groups[cluster]
    for index, file in enumerate(files):
        plt.subplot(30,30,index+1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    filename = "cluster_standard" + str(cluster) + ".png"
    cluster_path = os.path.join("D:/icm/data/cropped/zprojected output/resized/results/", filename)
    plt.savefig(cluster_path)

for i in range(len(labels)):
    print("Plotting cluster " + str(i))
    view_cluster(i)

'''