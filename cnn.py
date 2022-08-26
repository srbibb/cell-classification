from fileinput import filename
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import VGG16
from keras.models import Model

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

import os
import numpy as np
from random import randint

# Reads in the dataset of cropped cell images and processes them using the VGG16 neural network, taking
# the features calculated by the network as the output.
# The output features are normalised and have PCA performed on them before classification. 
# The filenames which have been processed, the output features and the features after PCA 
# are written as separate text files to be used to classify.

cells = []
norm = Normalizer()

# Reads in the data from the chosen folder and stores the filename
path = "D:/icm/data/resized"
os.chdir(path)
with os.scandir(path) as files:
    for file in files:
        if file.name.endswith('.png'):
            cells.append(file.name)

# The VGG16 model is initialised. The final layer of the model is removed, to get the final calculated
# features the output instead of the final prediction from the model
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Loads each image as an array in the shape expected by the network and gets the features predicted
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

# Collects the filenames and features to write to file
filenames = np.array(list(data.keys()))
features = np.array(list(data.values()))
features = features.reshape(-1,4096)

file = open("D:/icm/data/results/features.txt", "w")
for i in range(len(features)):
    for j in range(len(features[i])):
        file.write(str(features[i][j]) + ",")
    file.write("\n")

file.close()

# Normalises the features
features = norm.fit_transform(features)

# Performs PCA to achieve 95% variance
pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(features)

# Writes the features after PCA to file
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