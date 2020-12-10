import DatastreamDSWS as dsws
import numpy as np
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress

def get_data_csv(Items):

    startdate = '1990-01-01'
    enddate = '-0d'

    ds = dsws.Datastream(username='ZSEJ010', password='POINT410')

    arr = ds.get_data(tickers=Items[0], fields=['PI'], start=startdate, end=enddate, freq='D')
    for Item in Items[1:]:
        arr = pd.concat([arr, ds.get_data(tickers=Item, fields=['PI'], start=startdate, end=enddate, freq='D')], axis=1)

    arr.to_csv('../vixtest/backdata.csv')

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

    arr = pd.read_csv('../vixtest/backdata.csv')
    arr = arr.drop([0, 1])
    arr.index = pd.to_datetime(arr['Instrument'])
    arr = arr.drop(['Instrument'],axis=1)
    arr = arr.apply(pd.to_numeric)

    #1st#arr['YearAndWeek'] = arr['CBOEVIX'].index.year * 100 + arr['CBOEVIX'].index.week
    #1st#arr_group = arr.groupby(['YearAndWeek']).mean()

    #1st#arr['CBOEVIX'] = arr['CBOEVIX'].rolling(52).mean().rolling(2).mean()
    #1st#arr_group = arr_group.rolling(52).mean().rolling(2).mean()
    #1st#arrMerged = arr.merge(arr_group, left_on='YearAndWeek', right_on=arr_group.index).set_index(arr.index)

    '''d = np.array([0,0,0])
    for index in range(1,253):
        arr['CBOEVIX_d'] = arr['CBOEVIX'] / 100 * np.sqrt((1)/252)
        arr['CBOEVIX_roll'] = arr['CBOEVIX_d'].rolling(252).mean().rolling(2).mean()
        arr['S&PCOMP_rtn'] = arr['S&PCOMP'].shift(-index)/ arr['S&PCOMP'] - 1
        arr['S&PCOMP_roll'] = arr['S&PCOMP_rtn'].rolling(252).mean().rolling(2).mean()
        arr['CBOEVIX_spr'] = arr['CBOEVIX_d'] - arr['CBOEVIX_roll']
        arr['S&PCOMP_spr'] = arr['S&PCOMP_rtn'] - arr['S&PCOMP_roll']
        #arr = arr.dropna()
        #53일일 때 과거 연관성이 가장 높다

        d = np.vstack([d,[arr['S&PCOMP_rtn'].corr(arr['CBOEVIX_spr']),
              arr['S&PCOMP_roll'].corr(arr['CBOEVIX_spr']),
              arr['S&PCOMP_spr'].corr(arr['CBOEVIX_spr'])]])
        print(index,': ',arr['S&PCOMP_rtn'].corr(arr['CBOEVIX_spr']),
              arr['S&PCOMP_roll'].corr(arr['CBOEVIX_spr']),
              arr['S&PCOMP_spr'].corr(arr['CBOEVIX_spr']))
    '''

    index = 53
    arr['CBOEVIX_d'] = arr['CBOEVIX'] / 100 * np.sqrt((1)/252)
    arr['CBOEVIX_roll'] = arr['CBOEVIX_d'].rolling(252).mean().rolling(2).mean()
    arr['S&PCOMP_rtn'] = arr['S&PCOMP'].shift(-index)/ arr['S&PCOMP'] - 1
    arr['S&PCOMP_roll'] = arr['S&PCOMP_rtn'].rolling(252).mean().rolling(2).mean()
    arr['CBOEVIX_spr'] = arr['CBOEVIX_d'] - arr['CBOEVIX_roll']
    arr['S&PCOMP_spr'] = arr['S&PCOMP_rtn'] - arr['S&PCOMP_roll']
    arr['S&PCOMP_rtn2'] = arr['S&PCOMP']/ arr['S&PCOMP'].shift(1) - 1
    #arr = arr.dropna()
    #53일일 때 과거 연관성이 가장 높다

    d = np.vstack([[arr['S&PCOMP_rtn'].corr(arr['CBOEVIX_spr']),
          arr['S&PCOMP_roll'].corr(arr['CBOEVIX_spr']),
          arr['S&PCOMP_spr'].corr(arr['CBOEVIX_spr'])]])

    arr = arr.dropna()
    arr['PortVal'] = 100
    arr['S&PCOMP_index'] = 100
    for index in range(1,len(arr)):
        if arr['S&PCOMP_roll'].iloc[index-1] >= 0:
            arr['PortVal'].iloc[index] = arr['PortVal'].iloc[index-1] * (1+arr['S&PCOMP_rtn2'].iloc[index])
            arr['S&PCOMP_index'].iloc[index] = arr['S&PCOMP_index'].iloc[index - 1] * (1 + arr['S&PCOMP_rtn2'].iloc[index])
        else:
            arr['PortVal'].iloc[index] = arr['PortVal'].iloc[index - 1] * (1-arr['S&PCOMP_rtn2'].iloc[index])
            arr['S&PCOMP_index'].iloc[index] = arr['S&PCOMP_index'].iloc[index - 1] * (1 + arr['S&PCOMP_rtn2'].iloc[index])

    a = 1

    '''
    S&PCOMP_spr은 252일 rolling rtn(1년 기준)과 -0.3의 상관관계가 있다
    유의미해보인다다 
   '''