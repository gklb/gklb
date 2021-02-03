from matplotlib import pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import random
import pandas as pd
import datetime
import requests_cache
from tensorflow.python.training import saver

def get_variables(train_size):
    with open('./rawText/variables.pkl', 'rb') as f:
        arr = pickle.load(f)  # 단 한줄씩 읽어옴
    arr = arr[['KospiDeT','Rvol','SNET_pos','SNEM_PCA_PS','SNEM_PCA_N','SNEM_PCA_BT']]
    arr = arr[:train_size]
    return arr.values.tolist()

def discount_rewards(r, gamma):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in range(0, r.size):
        running_add = running_add*gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def step(a, arr, arr_idx, fwd_idx):

    gain = arr[arr_idx+fwd_idx][0] - arr[arr_idx][0]
    if a == 1:
        r = gain
    else:
        r = -gain
    return r

def reinforceLearning(train_size, model_load, learning_period):

    variables = get_variables(train_size=train_size)
    hist = 63
    inputdim = hist * 6
    iterations = 801
    update_period = 10
    gamma = 0.8254 #set weight of 6 month past reward become 0.01 
    #model_load = False

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=inputdim, activation='relu'))
    model.add(tf.keras.layers.Dense(52, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if model_load == False:
        pass
    else:
        model.load_weights('./weights/test_historic_'+str(train_size - learning_period)+'.h5')

    gradBuffer = model.trainable_variables
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    scores = []
    for iter in range(iterations):

        memory=[]
        s=[]
        score=0
        done=False
        startpoint = hist + random.randrange(0,hist)
        fwd_idx = 5

        for idx in range(startpoint,len(variables)-fwd_idx,fwd_idx):
            s = np.asmatrix(np.concatenate(variables[idx-hist:idx]))
            if idx + fwd_idx >= len(variables):
                break
            with tf.GradientTape() as tape:
                logits = model(s)
                a_dist = logits.numpy()
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist==a)
                loss = compute_loss([a],logits)
            grads = tape.gradient(loss, model.trainable_variables)
            r= step(a, variables, idx, fwd_idx)
            score += r
            memory.append([grads,r])

        scores.append(score)
        memory=np.array(memory)
        memory[:,1]=discount_rewards(memory[:,1],gamma)

        for grads, r in memory:
            for ix, grad in enumerate(grads):
                gradBuffer[ix] += grad * r

        if iter % update_period == 0:
            optimizer.apply_gradients(zip(gradBuffer,model.trainable_variables))
            for ix, grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0

        if iter % 10 == 0:
            print("Learning  {}  Score  {}".format(iter, np.mean(scores[-10:])))

        if iter % 100 == 0:
            model.save_weights('./weights/test_historic_'+str(train_size)+'.h5')

    model.save_weights('test.h5')
    a = 1

if __name__ == '__main__':

    learning_period = 21
    firsttime = False
    for idx in range(0, 2260, learning_period):
        if firsttime == True:
            model_load = False
            firsttime = False
        else:
            model_load = True

        reinforceLearning(train_size = idx, model_load=model_load, learning_period=learning_period)
