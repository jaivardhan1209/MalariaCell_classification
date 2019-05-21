#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:42:14 2019

@author: Surya
"""

import numpy as np
import os
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
x = []
y = []
SIZE = 64   
DATA_DIR = '/Users/Surya/Downloads/cell_images/'
parasitized_images = os.listdir(DATA_DIR + 'Parasitized/')
for i, image_name in enumerate(parasitized_images):
    try:
        if (image_name.split('.')[1] == 'png'):
            image = cv2.imread(DATA_DIR + 'Parasitized/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE, SIZE))
            x.append(np.array(image))
            y.append(0)
    except Exception:
        print("Could not read image {} with name {}".format(i, image_name))
uninfected_images = os.listdir(DATA_DIR + 'Uninfected/')
for i, image_name in enumerate(uninfected_images):
    try:
        if (image_name.split('.')[1] == 'png'):
            image = cv2.imread(DATA_DIR + 'Uninfected/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE, SIZE))
            x.append(np.array(image))
            y.append(1)
    except Exception:
        print("Could not read image {} with name {}".format(i, image_name))
x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.1, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
x_train = [i.flatten() for i in x_train]
print(np.array(x_train).shape)
x_test = [i.flatten() for i in x_test]
print(np.array(x_test).shape)
#model = DecisionTreeClassifier()
model=RandomForestClassifier(n_estimators=100)
model.fit(np.array(x_train),np.array(y_train))
y_pred = model.predict(np.array(x_test))
print("Accuracy: {:.2f}% " .format(metrics.accuracy_score(y_test, y_pred)*100))