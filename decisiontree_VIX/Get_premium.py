import DatastreamDSWS as dsws
import numpy as np
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import os
import graphviz
import pydotplus
from IPython.display import display
import joblib
from sklearn.metrics import classification_report
from sklearn import tree

def get_data_csv(Items):

    startdate = '1990-01-01'
    enddate = '-0d'

    ds = dsws.Datastream(username='ZSEJ010', password='POINT410')

    arr = ds.get_data(tickers=Items[0], fields=['PI'], start=startdate, end=enddate, freq='D')
    for Item in Items[1:]:
        arr = pd.concat([arr, ds.get_data(tickers=Item, fields=['PI'], start=startdate, end=enddate, freq='D')], axis=1)

    arr.to_csv('../decisiontree_VIX/backdata.csv')

def get_Arr(address, backdataarr):

    arr = pd.read_csv(address)
    arr.index = pd.to_datetime(arr['DATE'])
    arr = arr.drop(['DATE'],axis=1)
    arr = arr.apply(pd.to_numeric)
    arr = arr.loc[str(backdataarr.iloc[0].name):]
    return arr

def iterativePremium(startIndex, testarr,):
    premiumArr = np.hstack([np.array(testarr.columns[1:]),'rsq'])
    for index in range(startIndex, len(testarr)):
        y = np.array(testarr[index - startIndex:index])[:, 0]
        x = np.array(testarr[index - startIndex:index])[:, 1:]
        mlr = LinearRegression()
        premiumArr = np.vstack((premiumArr, np.hstack([mlr.fit(x, y).coef_, mlr.fit(x, y).score(x,y)])))
    return premiumArr

if __name__ == '__main__':

    '''
    #download up to date time series
    Items = ['CBOEVIX', 'S&PCOMP']
    get_data_csv(Items)
    '''


    const_Decile = True
    
    arr = pd.read_csv('../decisiontree_VIX/backdata.csv')
    arr = arr.drop([0, 1])
    arr.index = pd.to_datetime(arr['Instrument'])
    arr = arr.drop(['Instrument'],axis=1)
    arr = arr.apply(pd.to_numeric)

    PBarr = get_Arr('../VolSpreadPort/PBRTNdata.csv', arr)
    MEarr = get_Arr('../VolSpreadPort/MERTNdata.csv', arr)

    if const_Decile == True:
        MEarr['SMB'] = MEarr['Lo 10'] - MEarr['Hi 10']
        MEarr['SBcov'] = MEarr['Lo 10'].rolling(252).cov(MEarr['Hi 10'].rolling(252))
        PBarr['HML'] = PBarr['Hi 10'] - PBarr['Lo 10']
        PBarr['HLcov'] = PBarr['Hi 10'].rolling(252).cov(PBarr['Lo 10'].rolling(252))

    else:
        MEarr['SMB'] = MEarr['Lo 20'] - MEarr['Hi 20']
        MEarr['SBcov'] = MEarr['Lo 20'].rolling(252).cov(MEarr['Hi 20'].rolling(252))
        PBarr['HML'] = PBarr['Hi 20'] - PBarr['Lo 20']
        PBarr['HLcov'] = PBarr['Hi 20'].rolling(252).cov(PBarr['Lo 20'].rolling(252))

    arr = pd.concat([arr,MEarr['SMB'],PBarr['HML'],MEarr['SBcov'],PBarr['HLcov']],axis=1).dropna()

    arr['SMB'] = arr['SMB'].div(100)
    arr['HML'] = arr['HML'].div(100)

    arr['CBOEVIX_d'] = arr['CBOEVIX'] / 100 / np.sqrt(252)
    arr['CBOEVIX_roll'] = arr['CBOEVIX_d'].rolling(252).mean().rolling(2).mean()
    arr['CBOEVIX_spr'] = arr['CBOEVIX_d'] - arr['CBOEVIX_roll']

    arr['SMBVol'] = arr['SMB'].apply(lambda x: np.square(x)).rolling(252).mean()
    arr['HMLVol'] = arr['HML'].apply(lambda x: np.square(x)).rolling(252).mean()
    arr['SHCov'] = arr['SMB'].rolling(252).cov(arr['HML'].rolling(252)) * 2

    arr = arr.dropna()

    testarr = arr[['CBOEVIX_spr', 'SMBVol', 'HMLVol', 'SHCov']]
    premiumArr = iterativePremium(252,testarr)
    premiumArr = pd.DataFrame(premiumArr[1:], columns=['SMBp', 'HMLp', 'SHCovp','Rsqr'], index=testarr[252:].index)
    arr = pd.concat([arr, premiumArr], axis=1).dropna()

    #make label by month forward returns
    arr['classifier'] = 1
    for index in range(22,len(arr)):
        if (arr.SMB.iloc[index-22:index]+1).prod() > 1.01 and (arr.HML.iloc[index-22:index]+1).prod() > 1.01:
            arr.classifier.iloc[index-1] = 0
        elif (arr.SMB.iloc[index-22:index]+1).prod() > 1.01 and (arr.HML.iloc[index-22:index]+1).prod() <= 0.99:
            arr.classifier.iloc[index-1] = 1
        elif (arr.SMB.iloc[index-22:index]+1).prod() <= 0.99 and (arr.HML.iloc[index-22:index]+1).prod() > 1.01:
            arr.classifier.iloc[index-1] = 2
        elif (arr.SMB.iloc[index-22:index]+1).prod() <= 0.99 and (arr.HML.iloc[index-22:index]+1).prod() <= 0.99:
            arr.classifier.iloc[index-1] = 3
        else:
            arr.classifier.iloc[index-1] = 4

    arr.to_csv('../decisiontree_VIX/sourcedata.csv')
    #we've just made backdata set completely. Now task only left is testing iterative DecisionTree Model

    '''
    arr = arr.drop(['CBOEVIX','S&PCOMP','SMB','HML','CBOEVIX_d','CBOEVIX_roll','SHCov','SHCovp'],axis=1)

    train_x = arr.iloc[0:round(len(arr)*0.8),0:5]
    train_y = arr.iloc[0:round(len(arr)*0.8),5]
    test_x = arr.iloc[round(len(arr)*0.2)+1:len(arr),0:5]
    test_y = arr.iloc[round(len(arr)*0.2)+1:len(arr),5]

    model = DecisionTreeClassifier(max_depth=15, random_state=1).fit(train_x,train_y)
    print(model.score(train_x,train_y))
    print(model.score(test_x,test_y))
    pred_test = model.predict(test_x)
    print(classification_report(test_y,pred_test))

    export_graphviz(model, out_file='vix_model.dot',
                    class_names=np.array(['SH','SL','BH','BL']),
                    feature_names=arr.columns.tolist()[0:5],
                    filled= True,
                    rounded=True)
    #http://www.webgraphviz.com/
    '''
    a=1


