import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

#flattening y values
y_train = y_train.reshape(-1,)

classes = ["airplane", "automobile","bird","cat","deer","	dog","frog","horse","ship","truck"]

def plot_images(X,y,index):
    plt.figure(figsize=(5,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    
#Normalizing the values
X_train = X_train/255
X_test = X_test/255

ann_model = keras.models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='sigmoid')
    ])

ann_model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

ann_model.fit(X_train, y_train, epochs=5)

ann_model.evaluate(X_test, y_test)