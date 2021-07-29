import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

concrete = pd.read_csv('./concrete.csv')
concrete.head()

# The target for this task is the column `'CompressiveStrength'`. The remaining columns are the features we'll use as inputs.
input_shape = [8]

# defining the model
model = keras.Sequential([
    layers.Dense(units=512, activation='relu', input_shape=[8]),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=1),
])

# Activation layers
model = keras.Sequential([
    #     layers.Dense(32, activation='relu', input_shape=[8]),
    layers.Dense(units=32, input_shape=[8]),
    layers.Activation('relu'),
    #     layers.Dense(32, activation='relu'),
    layers.Dense(units=32),
    layers.Activation('relu'),
    layers.Dense(1),
])
