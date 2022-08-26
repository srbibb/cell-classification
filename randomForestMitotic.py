from random import Random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import load_img

# Takes the features and files from the neural network and classes them between being mitotic or interphase.
# Cells are classed using a Random Forest Classification model.

# Reads in features extracted from the neural network and removes any empty values
cells = pd.read_csv("D:/icm/data/results/features.txt", header=None)
cells.dropna(axis=1, how="all", inplace=True)

# Reads in the labels assigned to each data point
labels = pd.read_csv("D:/icm/data/Data.csv")
labels = labels[['id', 'choice']]

# Reads in the filenames read by the neural network and adds their path name to allow them to be viewed later
filenames = []
filenames = np.loadtxt(fname="D:/icm/data/results/filenames.txt", dtype=str, delimiter=",")
filenames = ["D:/icm/data/resized/" + x for x in filenames if x]

# Matches the cells which are labelled with their features to create the dataset
data = pd.merge(cells, labels, left_on=cells.index, right_on='id')
dataset_y = data['choice']
dataset_x = data.drop(['id', 'choice'], axis=1)
classes = ['Interphase', 'Mitotic']

# Collects the filenames of the labelled data
lab_filenames = []
for i in range(len(cells)):
    if (i in labels.id.values):
        lab_filenames.append(filenames[i])

dataset_x = np.array(dataset_x)
dataset_y = np.array(dataset_y)

# Splits the dataset into training and validation data
training_x, testing_x, training_y, testing_y = train_test_split(dataset_x, dataset_y, test_size=0.2, stratify=dataset_y)

accuracies = []

# Carries out cross-validation with a random forest classifier using the training data
# The training data is split into 5 folds and each fold is used as the testing data in turn
# The average accuracy can then be used to assess the model
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

# Assigns each file to the cluster it was predicted as belonging to
groups = {}
for file, cluster in zip(lab_filenames, y_pred):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

# Finds the largest cluster to set the size of the output
largest = 0
for elem in classes:
    if elem not in groups.keys():
        groups[elem] = []
    else:
        if len(groups[elem]) > largest:
            largest = len(groups[elem])
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
    filename = "rf_cluster_" + str(cluster) + ".png"
    cluster_path = os.path.join("D:/icm/data/cluster output/", filename)
    plt.savefig(cluster_path)

# Displays the cluster for each class
for i in classes:
    print("Plotting cluster " + str(i))
    print(len(groups[i]))
    view_cluster(i)
plt.show()

