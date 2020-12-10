import DatastreamDSWS as dsws
import numpy as np
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
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
arr['SMB_mom'] = (1+arr.SMB).rolling(22).apply(np.prod)
arr['HML_mom'] = (1+arr.HML).rolling(22).apply(np.prod)
arr['classifier_'] = arr.classifier
arr = arr.drop(['classifier'],axis=1)
arr = arr.dropna()

predictM = 66
windowLen = 252*5
testarr = arr.drop(['CBOEVIX','S&PCOMP','SMB','HML','CBOEVIX_d','CBOEVIX_roll'],axis=1)
predResult = np.array(arr.classifier_.iloc[0:windowLen])
for index in range(windowLen+predictM,len(testarr),predictM):

    temparr = testarr.iloc[:index]
    train_x = temparr.iloc[:index-predictM, 0:len(testarr.columns)-1]
    train_y = temparr.iloc[:index-predictM, len(testarr.columns)-1]
    test_x = temparr.iloc[index-predictM:, 0:len(testarr.columns)-1]
    test_y = temparr.iloc[index-predictM:, len(testarr.columns)-1]

    model = DecisionTreeClassifier(random_state=5, criterion='gini').fit(train_x, train_y)
    #print(model.score(train_x, train_y))
    if index % 100 == 0:
        print(index/len(testarr)*100)
    #    print(model.score(test_x, test_y))
    pred_test = model.predict(test_x)
    #print(classification_report(test_y, pred_test))
    predResult = np.hstack([predResult, pred_test])

print(classification_report(arr['classifier_'].iloc[windowLen:len(predResult)],predResult[windowLen:]))
print(np.array(testarr.columns))
print(model.feature_importances_)
export_graphviz(model, out_file='vix_model.dot',
                class_names=np.array(['SH', 'SL', 'BH', 'BL', 'NA']),
                feature_names=testarr.columns.tolist()[0:len(testarr.columns)-1],
                filled=True,
                rounded=True)
# http://www.webgraphviz.com/

arr = arr.iloc[:len(predResult)]
arr['Prediction'] = predResult
arr = arr.iloc[windowLen:]
arr['PortSize'] = 100
arr['PortVal'] = 100
caseKey = arr.Prediction.iloc[0]
for index in range(5,len(arr)):
    if index % 5 == 0 and index >= 5:
        caseKey = arr['Prediction'].iloc[index-5]
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
print(5,'Val',arr.PortVal.iloc[-1]/arr.PortVal.iloc[0],'Size:',arr.PortSize.iloc[-1]/arr.PortSize.iloc[0])


a = 1