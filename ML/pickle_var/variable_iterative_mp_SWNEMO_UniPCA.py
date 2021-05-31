import pickle
import pandas as pd
import DatastreamDSWS as dsws
import matplotlib.pyplot as plt
import statsmodels
import numpy as np
import datetime
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as ss
from sklearn.decomposition import PCA
import multiprocessing
import math
import os
import csv

def calc_slope(x):
    slope = np.polyfit(range(len(x)), x, 1)[0]
    return slope

def sentimentForm(pos,neg):
    return (pos - neg) / (pos + neg)

def get_PCA(arr, index):
    pca = PCA(n_components=index)
    printcipalComponents = pca.fit_transform(arr)
    principalDf = pd.DataFrame(data=printcipalComponents)
    return principalDf, pca

def plot_cumsum(pca):
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"})
    fig.show()

def normalization_PCA(regex_str, sourceArray, exp_variance, rebalancingDate, pcaDate, model_code):
    vararr = sourceArray.filter(regex=regex_str).dropna()
    varsc = pd.DataFrame(ss.zscore(vararr), index=vararr.index, columns=vararr.columns)
    varsc = varsc.fillna(0)

    if rebalancingDate == True:
        principalDf, pca = get_PCA(varsc, len(vararr.columns))
        exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
        num_pca = np.where([exp_var_cumul >= exp_variance])[1][0] + 1
        varpca, varpc = get_PCA(varsc, num_pca)
        varpc = pd.DataFrame(varpc.components_, columns=vararr.columns)
        varpca = pd.DataFrame(varpca.values, index=varsc.index, columns=['PCA_' + str(i) for i in range(num_pca)])

        varpc.to_pickle(dir +"/pickle_var/variables_pca/" + model_code + pcaDate.strftime('%Y-%m-%d')+".pkl")
    else:
        varpc = pd.read_pickle(dir +"/pickle_var/variables_pca/" + model_code + pcaDate.strftime('%Y-%m-%d')+".pkl")
        varpca = np.matmul(varsc.values,varpc.values.T)
        varpca = pd.DataFrame(varpca, index=varsc.index, columns=['PCA_' + str(i) for i in range(len(varpc))])

    variables = pd.concat([sourceArray.KospiDeT, varpca], ignore_index=True, axis=1)
    variables.columns = ['KospiDeT'] + list(varpca.columns)
    variables = variables.dropna()

    return variables

def process_data(idx):

    rebalancingDate = False
    matchDate = BaseDate + datetime.timedelta(days=7*idx)
    next_matchDate = BaseDate + datetime.timedelta(days=7*(idx+1))
    #rebalancingDate와 최초 학습일을 제외하면 data는 이전 pca를 기준으로 제작되어야한다.
    #pca는 data의 미미한 변화에도 변동하므로 이를 고려하지 않으면 잘못된 결과를 도출할 수도 있다.
    if matchDate.month != next_matchDate.month:
        rebalancingDate = True
        pcaDate = matchDate
    elif matchDate == BaseDate:
        rebalancingDate = True
        pcaDate = BaseDate
    elif matchDate.month == BaseDate.month:
        pcaDate = BaseDate
    else:
        formeridx = 0
        while matchDate.month == (BaseDate + datetime.timedelta(days=7*(idx+formeridx))).month:
            formeridx += -1
        pcaDate = BaseDate + datetime.timedelta(days=7*(idx+formeridx))
    arrSliceIdx = sourcearr.index.get_loc(matchDate, method = 'ffill')
    arr = sourcearr.copy().iloc[:arrSliceIdx]
    print(matchDate)
    kospiLT_STL = statsmodels.tsa.seasonal.STL(arr.Kospi, trend=253*10-1, period=253).fit()
    trend_adj = kospiLT_STL.trend[0] / kospiLT_STL.trend.copy()
    resid_adj = kospiLT_STL.resid.copy() * trend_adj
    #지수 level이 상승할수록 점차 level을 사용한 변동값은 실제를 반영하기 어려워짐. 이에 따라 조정을 줌

    arr['KospiLTrd'] = kospiLT_STL.trend
    arr['KospiDeT'] = resid_adj

    #arr.SNET_pos = arr.SNET_pos.rolling(252).mean()
    #이번에는 SNET_pos에 lagging을 가하지 말자
    arr['F_Rvol'] = arr.KospiDeT.rolling(21).std()
    arr = arr.dropna()

    variables_all= normalization_PCA('F_|S_|E_', arr, 0.9, rebalancingDate, pcaDate, 'All')
    variables_excSent = normalization_PCA('F_|E_', arr, 0.9, rebalancingDate, pcaDate, 'excSent')
    variables_excFdmt = normalization_PCA('S_|E_', arr, 0.9, rebalancingDate, pcaDate, 'excFdmt')
    variables_excEcon = normalization_PCA('F_|S_', arr, 0.9, rebalancingDate, pcaDate, 'excEcon')
    variables_onlySent = normalization_PCA('S_', arr, 0.9, rebalancingDate, pcaDate, 'onlySent')

    variables_all.to_pickle(dir +"/pickle_var/variables/All"+matchDate.strftime('%Y-%m-%d')+".pkl")
    variables_excSent.to_pickle(dir +"/pickle_var/variables/excSent"+matchDate.strftime('%Y-%m-%d')+".pkl")
    variables_excFdmt.to_pickle(dir +"/pickle_var/variables/excFdmt"+matchDate.strftime('%Y-%m-%d')+".pkl")
    variables_excEcon.to_pickle(dir +"/pickle_var/variables/excEcon"+matchDate.strftime('%Y-%m-%d')+".pkl")
    variables_onlySent.to_pickle(dir +"/pickle_var/variables/onlySent"+matchDate.strftime('%Y-%m-%d')+".pkl")

dir = 'C:/pythonProject_tf/textAnalysis'
with open(dir + './rawText/SWNEMO_Score.pkl', 'rb') as f:
    SWNEMO_pd = pickle.load(f) # 단 한줄씩 읽어옴
SWNEMO_pd.columns = SWNEMO_pd.columns.values

ECON_pd = pd.read_csv(dir + './rawText/economic_raw.csv')
ECON_pd.index = pd.to_datetime(ECON_pd.Date, format='%Y-%m-%d')
ECON_pd = ECON_pd.drop(labels='Date',axis=1)

arr = pd.concat([SWNEMO_pd, ECON_pd], ignore_index=True, axis=1)
arr.columns = list('S_'+SWNEMO_pd.columns) + list(ECON_pd.columns)
arr.rename(columns={'KORCOMP(PI)':'Kospi'}, inplace=True)
arr = arr.fillna(method='ffill')
arr = arr.apply(pd.to_numeric)
sourcearr = arr.copy()

#initial date of data is 2011-03-18, but as we need at least 252 days to train
BaseDate = datetime.datetime.strptime('2012-03-16', '%Y-%m-%d')
LoopEndCount = int(((sourcearr.index[-1] - BaseDate)/7).days)

if __name__ == '__main__':
    a_pool = multiprocessing.Pool(os.cpu_count())
    a_pool.map(process_data, range(LoopEndCount + 1))
