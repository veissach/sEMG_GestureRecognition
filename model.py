#%%
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#%%
Input_path = 'EMG-data.csv'
df = pd.read_csv(Input_path)

features = df.drop(columns=["label","class","time"])

#%%
Class = df["class"]
Class = Class.values
features = features.values
x_train, x_test, y_train, y_test = train_test_split(features, Class, test_size=0.2, random_state=1)

#%%
# Normalizing data
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train -= mean
x_train /= std

x_test -= mean
x_test /= std

#%%
# one hot encoding Labels
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#%%