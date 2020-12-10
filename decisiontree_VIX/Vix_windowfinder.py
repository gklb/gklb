import DatastreamDSWS as dsws
import numpy as np
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns

def get_data_csv(Items):

    startdate = '1990-01-01'
    enddate = '-0d'

    ds = dsws.Datastream(username='ZSEJ010', password='POINT410')

    arr = ds.get_data(tickers=Items[0], fields=['PI'], start=startdate, end=enddate, freq='D')
    for Item in Items[1:]:
        arr = pd.concat([arr, ds.get_data(tickers=Item, fields=['PI'], start=startdate, end=enddate, freq='D')], axis=1)

    arr.to_csv('../decisiontree_VIX/backdata.csv')

def get_slope(array):
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return slope

def monthlyRTN(array):
    array = np.array(array)
    return array[-1]/array[0] - 1

if __name__ == '__main__':

    '''
    #download up to date time series
    Items = ['CBOEVIX', 'S&PCOMP']
    get_data_csv(Items)
    '''

    arr = pd.read_csv('../decisiontree_VIX/backdata.csv')
    arr = arr.drop([0, 1])
    arr.index = pd.to_datetime(arr['Instrument'])
    arr = arr.drop(['Instrument'],axis=1)
    arr = arr.apply(pd.to_numeric)

    '''1st
    arr['YearAndWeek'] = arr['CBOEVIX'].index.year * 100 + arr['CBOEVIX'].index.week
    arr_group = arr.groupby(['YearAndWeek']).mean()

    arr['CBOEVIX'] = arr['CBOEVIX'].rolling(52).mean().rolling(2).mean()
    arr_group = arr_group.rolling(52).mean().rolling(2).mean()
    arrMerged = arr.merge(arr_group, left_on='YearAndWeek', right_on=arr_group.index).set_index(arr.index)
    '''

    '''#2nd
    arr['CBOEVIX'] = arr['CBOEVIX'] / 100 * np.sqrt(30/365)
    arr['CBOEVIX_roll'] = arr['CBOEVIX'].rolling(252).mean().rolling(2).mean()
    arr['S&PCOMP_rtn'] = arr['S&PCOMP']/ arr['S&PCOMP'].shift(1) - 1
    arr['S&PCOMP_roll'] = arr['S&PCOMP_rtn'].rolling(252).mean().rolling(2).mean()
    arr['CBOEVIX_spr'] = arr['CBOEVIX'] - arr['CBOEVIX_roll']
    arr['S&PCOMP_spr'] = arr['S&PCOMP_rtn'] - arr['S&PCOMP_roll']
    '''

    #ranges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #df.groupby(pd.cut(df.a, ranges)).count()

    '''
    for index in range(1,253):
        arr['CBOEVIX_d'] = arr['CBOEVIX'] / 100 * np.sqrt(index / 252)
        arr['CBOEVIX_roll'] = arr['CBOEVIX_d'].rolling(252).mean().rolling(2).mean()
        arr['S&PCOMP_rtn'] = arr['S&PCOMP'].shift(-index) / arr['S&PCOMP'] - 1
        arr['S&PCOMP_roll'] = arr['S&PCOMP_rtn'].rolling(252).mean().rolling(2).mean()
        arr['CBOEVIX_spr'] = arr['CBOEVIX_d'] - arr['CBOEVIX_roll']
        arr['S&PCOMP_spr'] = arr['S&PCOMP_rtn'] - arr['S&PCOMP_roll']

        print(index,': ',arr['CBOEVIX_spr'].corr(arr['S&PCOMP_rtn'])
              ,arr['CBOEVIX_spr'].corr(arr['S&PCOMP_roll'])
              ,arr['CBOEVIX_spr'].corr(arr['S&PCOMP_spr']))
    '''

    index = 1
    arr['CBOEVIX_d'] = arr['CBOEVIX'] / 100 * np.sqrt(index / 252)
    arr['CBOEVIX_roll'] = arr['CBOEVIX_d'].rolling(252).mean().rolling(2).mean()
    arr['S&PCOMP_rtn'] = arr['S&PCOMP'].shift(-index) / arr['S&PCOMP'] - 1
    arr['S&PCOMP_roll'] = arr['S&PCOMP_rtn'].rolling(252).mean().rolling(2).mean()
    arr['CBOEVIX_spr'] = arr['CBOEVIX_d'] - arr['CBOEVIX_roll']
    arr['S&PCOMP_spr'] = arr['S&PCOMP_rtn'] - arr['S&PCOMP_roll']

    sns.distplot(arr['CBOEVIX_spr'])
    plt.show()
    #CBOEVIX_spr 분포도 정규분포로 잘 나온다
    #56일 선행 S&PCOMP와 상관관계가 -0.3 이상으로 나온다는 점은 긍정적

    a = 1