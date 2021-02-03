from matplotlib import pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import random
import pandas as pd
import datetime
import requests_cache
from tensorflow.python.training import saver

def get_variables(test_size):
    with open('./rawText/variables.pkl', 'rb') as f:
        arr = pickle.load(f)  # 단 한줄씩 읽어옴
    arr = arr[['KospiDeT','Rvol','SNET_pos','SNEM_PCA_PS','SNEM_PCA_N','SNEM_PCA_BT']]
    arr = arr[test_size:]
    return arr.values.tolist()

def step(a, arr, arr_idx, fwd_idx):

    gain = arr[arr_idx+fwd_idx][0] - arr[arr_idx][0]
    if a == 1:
        r = gain
    else:
        r = -gain
    return r

hist = 21
inputdim = hist * 6
iterations = 101
update_period = 10
gamma = 0.9

variables = get_variables(test_size = 0)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_dim=inputdim, activation='relu'))
model.add(tf.keras.layers.Dense(52, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

learning_period= 21



memory = []
s = []
score = 0
done = False
startpoint = 500
model_date = startpoint
fwd_idx = 5
model.load_weights('./weights/test_historic_'+ str(model_date) +'.h5')

for idx in range(startpoint, len(variables) - fwd_idx, fwd_idx):
    if idx >= model_date + 21:
        model_date += 21
        model.load_weights('./weights/test_historic_' + str(model_date) + '.h5')
    s = np.asmatrix(np.concatenate(variables[idx - hist:idx]))
    if idx + fwd_idx >= len(variables):
        break
    logits = model(s)
    a_dist = logits.numpy()
    a = np.random.choice(a_dist[0], p=a_dist[0])
    a = np.argmax(a_dist == a)

    r = step(a, variables, idx, fwd_idx)
    score += r
    memory.append([a, score])

pd.DataFrame(memory)[1].plot()
pd.DataFrame(memory)[0].plot(secondary_y=True)
plt.show()
a = 1
