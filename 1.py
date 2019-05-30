#
# REGRESSION
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# set logs to error only
tf.logging.set_verbosity(tf.logging.ERROR)

# features
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float) 

# labels
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float) 

# print examples
for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

#create model
model = tf.keras.Sequential()

#add 1 dense layer with 1 neuron
layer0 = tf.keras.layers.Dense(units=1, input_shape=[1], activation=tf.nn.relu)
model.add(layer0)

#compile model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.5))

# train model
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=True)
print("Finished training the model")

# plot the training history
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

#use model to predict data
print(model.predict([100.0])) 

print("These are the layer variables: {}".format(layer0.get_weights()))