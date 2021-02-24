import numpy as np
import pickle
import random
from tqdm import tqdm
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def get_variables(train_size,direct):
    with open(direct, 'rb') as f:
        arr = pickle.load(f)
    arr = arr[:train_size]
    return arr.values.tolist()

def logistic_reg():
    a = 1

def labeling(arr):
    gain = arr[-1] - arr[0]
    if gain>0:
        return 1
    else:
        return 0

def preprocFeatures(arr):
    gainArr = arr.iloc[:,0].copy()
    scaled_arr = scaler.transform(arr)
    return gainArr, scaled_arr

def LogisticTrain(arr, train_size):
    variables = arr.iloc[0:train_size-5].copy()
    model = LogisticRegression(max_iter=800)
    features = variables.drop('Label',axis=1).copy()
    scaler.fit(features)
    gainArr, features = preprocFeatures(features)

    labels = variables.Label.copy()

    model.fit(features,labels)
    pickle.dump(model, open(save_direct + '/test_historic_' + str(train_size) + '.sav','wb'))

if __name__ == '__main__':

    basedir = 'C:/pythonProject_tf/textAnalysis'
    inputdata_direct = basedir+ '/pickle_var/variables1_3.pkl'
    learning_period = 21
    #hist = 63
    #iterations = 801
    #update_period = 50
    fwd_idx = 5
    save_direct = basedir + '/weights/logistics/weights1_3'

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

        LogisticTrain(arr, idx)