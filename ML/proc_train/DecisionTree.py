from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import numpy as np
import pickle
import random
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def get_variables(train_size,direct):
    with open(direct, 'rb') as f:
        arr = pickle.load(f)
    return arr.values.tolist()

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

def DecisionTreeTrain(arr, train_size):
    #startpoint=max(train_size-756-5,0)
    startpoint=0
    variables = arr.iloc[startpoint:train_size-5].copy()
    labels = variables.Label.copy()
    features = variables.drop('Label',axis=1).copy()
    scaler.fit(features)
    gainArr, features = preprocFeatures(features)

    for grid_max_depth in range(5,15):
        model = DecisionTreeClassifier(criterion='gini', max_depth=grid_max_depth)
        model.fit(features, labels)
        score_model = model.score(features, labels)
        if score_model > 0.9:
            break

    export_graphviz(model, out_file='tree.dot',
                    feature_names=arr.columns[:-1],
                    filled=True,
                    rounded=True)
    pickle.dump(model, open(save_direct + '/test_historic_' + str(train_size) + '.sav','wb'))
    if (train_size - 756) % 300==0:
        print(train_size,'done')

if __name__ == '__main__':

    basedir = 'C:/pythonProject_tf/textAnalysis'
    inputdata_direct = basedir+ '/pickle_var/variables1_3.pkl'
    learning_period = 21
    fwd_idx = 5
    save_direct = basedir + '/weights/decisiontree/weights1_3'

    with open(inputdata_direct, 'rb') as f:
        arr = pickle.load(f)
    scaler.fit(arr)

    arr['Label'] = arr['KospiDeT'].rolling(fwd_idx).apply(labeling).shift(-4)
    arr = arr.dropna()
    firsttime = True
    for idx in range(756, len(arr), learning_period):
        if firsttime == True:
            model_load = False
            firsttime = False
        else:
            model_load = True

        DecisionTreeTrain(arr, idx)


