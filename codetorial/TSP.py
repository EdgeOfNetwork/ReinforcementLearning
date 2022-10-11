import numpy as np
import random as rd
import tensorflow as tf
from tensorflow.keras.utils import plot_model


def rargmin(vector):
  m = np.amin(vector)
  indices = np.nonzero(vector == m)[0]
  return rd.choice(indices)


# Create a neural network model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, input_shape=(8,), activation=tf.nn.relu),
  tf.keras.layers.Dense(4, activation='linear')
])

# Create and Set weight values
weight0 = np.random.rand(8, 128) * 1.0
weight1 = np.zeros(128)
weight2 = np.random.rand(128, 4) * 0.5
weight3 = np.zeros(4)

weights = np.array([weight0, weight1, weight2, weight3])
model.set_weights(weights)

# Visualize
plot_model(model, to_file='model_shapes.png', show_shapes=True)