import DatastreamDSWS as dsws
import numpy as np
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import os
import graphviz
import pydotplus
import joblib
from sklearn.metrics import classification_report
from sklearn import tree


arr = pd.read_csv('../decisiontree_VIX/sourcedata.csv')
arr.index = pd.to_datetime(arr.iloc[:,0])
arr = arr.drop(arr.iloc[:,0].name, axis=1)
arr = arr.apply(pd.to_numeric)
arr = arr.dropna()

predictM = 5
windowLen = 252*5
testarr = arr.drop(['CBOEVIX','S&PCOMP','SMB','HML','CBOEVIX_d','CBOEVIX_roll'],axis=1)
predResult = np.array(arr.classifier.iloc[0:windowLen+predictM])
for index in range(windowLen+2*predictM,len(testarr),predictM):

    temparr = testarr.iloc[:index]
    train_x = temparr.iloc[:len(temparr)-2*predictM, 0:len(testarr.columns)-1]
    train_y = temparr.iloc[:len(temparr)-2*predictM, len(testarr.columns)-1]
    test_x = temparr.iloc[len(temparr)-predictM:, 0:len(testarr.columns)-1]
    test_y = temparr.iloc[len(temparr)-predictM:, len(testarr.columns)-1]

    model = RandomForestClassifier(random_state=5, criterion='gini').fit(train_x, train_y)
    #print(model.score(train_x, train_y))
    pred_test = model.predict(test_x)
    #print(classification_report(test_y, pred_test))
    predResult = np.hstack([predResult, pred_test])
    if index % 100 == 0:
        print(index/len(testarr)*100, 'model score:',model.score(test_x, test_y),
            'Accuracy:',classification_report(arr['classifier'].iloc[windowLen:len(predResult)],predResult[windowLen:],output_dict=True)['weighted avg']['f1-score'])

print(classification_report(arr['classifier'].iloc[windowLen:len(predResult)],predResult[windowLen:]))
print(np.array(testarr.columns))
print(model.feature_importances_)
# http://www.webgraphviz.com/

arr = arr.iloc[:len(predResult)]
arr['Prediction'] = predResult
arr = arr.iloc[windowLen:]
for index_ in range(1,2):
    arr['PortSize'] = 100
    arr['PortVal'] = 100
    caseKey = arr.Prediction.iloc[0]
    for index in range(5,len(arr)):
        if index % 5 == 0 and index >= 5:
            caseKey = arr['Prediction'].iloc[index-index_]
        if caseKey == 0:
            arr.PortSize.iloc[index] = arr.PortSize.iloc[index-1] * (1+arr.SMB.iloc[index])
            arr.PortVal.iloc[index] = arr.PortVal.iloc[index - 1] * (1+arr.HML.iloc[index])
        elif caseKey == 1:
            arr.PortSize.iloc[index] = arr.PortSize.iloc[index-1] * (1+arr.SMB.iloc[index])
            arr.PortVal.iloc[index] = arr.PortVal.iloc[index - 1] * (1-arr.HML.iloc[index])
        elif caseKey == 2:
            arr.PortSize.iloc[index] = arr.PortSize.iloc[index-1] * (1-arr.SMB.iloc[index])
            arr.PortVal.iloc[index] = arr.PortVal.iloc[index - 1] * (1+arr.HML.iloc[index])
        elif caseKey == 3:
            arr.PortSize.iloc[index] = arr.PortSize.iloc[index-1] * (1-arr.SMB.iloc[index])
            arr.PortVal.iloc[index] = arr.PortVal.iloc[index - 1] * (1-arr.HML.iloc[index])
        else:
            arr.PortSize.iloc[index] = arr.PortSize.iloc[index-1]
            arr.PortVal.iloc[index] = arr.PortVal.iloc[index-1]
    print(index_,'Val',arr.PortVal.iloc[-1]/arr.PortVal.iloc[0],'Size:',arr.PortSize.iloc[-1]/arr.PortSize.iloc[0])


a = 1