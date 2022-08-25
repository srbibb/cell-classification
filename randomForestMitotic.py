from random import Random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from pathlib import Path
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

from keras.preprocessing.image import load_img

cells = pd.read_csv("D:/icm/data/results/features.txt", header=None)
cells.dropna(axis=1, how="all", inplace=True)
labels = pd.read_csv("D:/icm/data/MitoticData.csv")
filenames = []
with os.scandir("D:/icm/data/resized/") as files:
    for file in files:
        if file.name.endswith('.png'):
            filenames.append(file.name)
filenames = ["D:/icm/data/resized/" + x for x in filenames if x]

labels = labels[['id', 'choice']]
data = pd.merge(cells, labels, left_on=cells.index, right_on='id')
dataset_y = data['choice']
dataset_x = data.drop(['id', 'choice'], axis=1)
classes = ['Interphase', 'Mitotic']
le = LabelEncoder()
lab_filenames = []

for i in range(len(cells)):
    if (i+1 in labels.id.values):
        lab_filenames.append(filenames[i])
        print(filenames[i] + " " + str(i))

dataset_x = np.array(dataset_x)
dataset_y = np.array(dataset_y)

training_x, testing_x, training_y, testing_y = train_test_split(dataset_x, dataset_y, test_size=0.2, stratify=dataset_y)

accuracies = []

rf = RandomForestClassifier(random_state=25, class_weight='balanced')
skf = StratifiedKFold(n_splits=5, random_state=27, shuffle=True)
for train_index, test_index in skf.split(training_x, training_y):
    X_train = training_x[train_index]
    y_train = training_y[train_index]
    X_test = training_x[test_index]
    y_test = training_y[test_index]
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    print("Accuracy:",accuracy_score(y_test, y_pred))

avg = 0
for i in range (len(accuracies)):
    avg = avg + accuracies[i]

avg = avg/len(accuracies)
print("Average accuracy: " + str(avg))

groups = {}
for file, cluster in zip(lab_filenames, y_pred):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

largest = 0
for elem in classes:
    if elem not in groups.keys():
        groups[elem] = []
    else:
        if len(groups[elem]) > largest:
            largest = len(groups[elem])

group_size = math.ceil(math.sqrt(largest))

def view_cluster(cluster):
    plt.figure(figsize = (group_size, group_size))
    files = groups[cluster]
    for index, file in enumerate(files):
        plt.subplot(group_size, group_size,index+1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    filename = "rf_cluster_" + str(cluster) + ".png"
    cluster_path = os.path.join("D:/icm/data/cluster output/", filename)
    plt.savefig(cluster_path)

for i in classes:
    print("Plotting cluster " + str(i))
    print(len(groups[i]))
    view_cluster(i)
plt.show()

