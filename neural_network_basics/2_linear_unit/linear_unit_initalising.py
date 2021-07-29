# The bias enables the neuron to modify the output independently of its inputs.
# Bias allows you to shift the activation function by adding a constant (i.e. the given bias) to the input.

# Setup plotting
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

red_wine = pd.read_csv('./winequality-red.csv')
print(red_wine.head())

# step 1 check input shape
print(red_wine.shape)  # (rows, columns)
input_shape = [11]

# step 2 define a linear model
model = keras.Sequential([layers.Dense(units=1, input_shape=input_shape)])

# step 3 Weights
w, b = model.weights
print("Weights\n{}\n\nBias\n{}".format(w, b))
