import xgboost as xgb
from sklearn.tree import export_graphviz
import numpy as np
import pickle
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

def RandomForestTrain(arr, train_size):
    #startpoint=max(train_size-756-5,0)
    startpoint=0
    variables = arr.iloc[startpoint:train_size-5].copy()
    labels = variables.Label.copy()
    features = variables.drop('Label',axis=1).copy()
    scaler.fit(features)
    gainArr, features = preprocFeatures(features)

    '''params = {
        'objective':'binary:logistic',
        'max_depth':5,
        'n_estimators':150,
        'subsample':0.4
    }'''

    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(features, labels, test_size=0.3)
    model = xgb.XGBClassifier(max_depth=5,
                                colsample_bytree=0.8,
                                subsample=0.3,
                                n_estimators=1024)
    modelGrid = GridSearchCV(model,
                                {'max_depth':[3,5,10]
                                }, verbose=0)
    modelGrid.fit(train_data_x,train_data_y,early_stopping_rounds=100, eval_metric='auc',eval_set=[(test_data_x,test_data_y)], verbose=0)
    bestmodel = modelGrid.best_estimator_
    #model.fit(train_data_x,train_data_y,early_stopping_rounds=10, eval_metric='auc',eval_set=[(test_data_x,test_data_y)])

    pickle.dump(bestmodel, open(save_direct + '/test_historic_' + str(train_size) + '.pkl','wb'))

if __name__ == '__main__':

    basedir = 'C:/pythonProject_tf/textAnalysis'
    inputdata_direct = basedir+ '/pickle_var/variables1_2.pkl'
    learning_period = 21
    fwd_idx = 5
    save_direct = basedir + '/weights/randomforest/weights1_2'

    with open(inputdata_direct, 'rb') as f:
        arr = pickle.load(f)
    scaler.fit(arr)

    arr['Label'] = arr['KospiDeT'].rolling(fwd_idx).apply(labeling).shift(-4)
    arr = arr.dropna()
    firsttime = True
    with tqdm(range(756, len(arr), learning_period)) as tqd:
        for idx in tqd:
            if firsttime == True:
                model_load = False
                firsttime = False
            else:
                model_load = True

            RandomForestTrain(arr, idx)
