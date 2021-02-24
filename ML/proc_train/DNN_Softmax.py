'''참고
https://tykimos.github.io/2017/01/27/Keras_Talk/ about keras , myth
https://rfriend.tistory.com/553 construction DNN model
출처: https://3months.tistory.com/424 [Deep Play]
'''

import numpy as np
import pickle
import tensorflow as tf
import random
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler
import torch
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm.keras import TqdmCallback

scaler = StandardScaler()
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def labeling(arr):
    gain = arr[-1] - arr[0]
    if gain>0:
        return 1
    else:
        return 0

def flattenSeries(arr, hist):
    flattedArray = []
    for idx in range(hist, len(arr)):
        falttedPiece = []
        arrPiece = arr[idx-hist:idx]
        flattedPiece = np.ravel(arrPiece)
        flattedArray.append(flattedPiece)

    return flattedArray

def get_variables(train_size,direct):
    with open(direct, 'rb') as f:
        arr = pickle.load(f)
    arr = arr[:train_size]
    return arr.values.tolist()

def preprocFeatures(arr):
    gainArr = arr.iloc[:,0].copy()
    scaled_arr = scaler.transform(arr)
    return gainArr, scaled_arr

def step(a, arr, arr_idx, fwd_idx):

    gain = arr[arr_idx+fwd_idx][0] - arr[arr_idx][0]
    if a == 2: # Bull
        r = gain
    elif a == 1: # Neutral
        r = 0
    else: # Bear
        r = -gain
    return r

def DNN_Classify(train_data,
                      train_size,
                      #model_load,
                      #learning_period,
                      hist, # time length of input data
                      iterations
                      #update_period, # gradient will be updated for every 10 iterations
                      #save_direct # location of saving weights
                      ):

    variables = train_data.iloc[:train_size].copy()
    scaler.fit(variables)
    _, features = preprocFeatures(variables.drop('Label', axis=1).copy())
    #gainArr = gainArr.values.tolist()[hist:]

    features = flattenSeries(features,hist)
    labels = torch.as_tensor(variables.Label.iloc[hist:].copy().values)
    labels = torch.nn.functional.one_hot(labels.long())

    inputdim = len(features[-1])
    #model_load = False

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=inputdim, activation='relu'))
    model.add(tf.keras.layers.Dense(52, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=50)
    mc = ModelCheckpoint(save_direct + '/test_historic_' + str(train_size) + '.h5', monitor='loss', mode='min',
                         save_best_only=True)

    if model_load == False:
        pass
    else:
        model.load_weights(save_direct+'/test_historic_'+str(train_size - learning_period)+'.h5')

    model.fit(np.array(features), np.array(labels), epochs=iterations, verbose=0, callbacks=[TqdmCallback(verbose=0), early_stopping, mc])

    model.save_weights(save_direct+'/test_historic_'+str(train_size)+'.h5')

if __name__ == '__main__':

    basedir = 'C:/Users/admin/PycharmProjects/pythonProject_tf_2'
    inputdata_direct = basedir + '/pickle_var/variables1.pkl'
    hist = 63
    learning_period = 21
    fwd_idx = 5
    iterations = 1024
    update_period = 32
    save_direct = basedir + '/weights/dnn/weights1'

    with open(inputdata_direct, 'rb') as f:
        arr = pickle.load(f)

    arr['Label'] = arr['KospiDeT'].rolling(fwd_idx).apply(labeling).shift(-4)
    arr = arr.dropna()
    firsttime = True
    for idx in range(756, len(arr), learning_period):
        if firsttime == True:
            model_load = False
            firsttime = False
        else:
            model_load = True

        DNN_Classify(train_data=arr, train_size=idx, hist=hist, iterations=iterations)