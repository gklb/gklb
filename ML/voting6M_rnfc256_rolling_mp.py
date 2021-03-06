from matplotlib import pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import random
import pandas as pd
import os
import multiprocessing
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import DatastreamDSWS as dsws
import datetime
import csv
scaler = StandardScaler()
local_scaler = StandardScaler()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def preprocFeatures(arr):
    gainArr = arr.iloc[:,0].copy()
    scaled_arr = scaler.transform(arr)
    return gainArr, scaled_arr

def get_variables(test_size, var_dir):
    with open(var_dir, 'rb') as f:
        arr = pickle.load(f)  # 단 한줄씩 읽어옴
    arr = arr[test_size:]
    return arr

def get_mainvariables(test_size, var_dir):
    with open(var_dir, 'rb') as f:
        arr = pickle.load(f)  # 단 한줄씩 읽어옴
    arr = arr[test_size:]
    return arr

def makeModel_RNFC(inputdim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(252, input_dim=inputdim, activation='relu'))
    model.add(tf.keras.layers.Dense(126, activation='relu'))
    model.add(tf.keras.layers.Dense(63, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    return model

def loadModel_RNFC(model, weight_dir, extension, matchDate):

    model_loaded = False
    model.load_weights(weight_dir + BaseDate + extension)
    loc_num = 0
    while model_loaded == False:
        try:
            matchDate_dt = datetime.datetime.strptime(matchDate, '%Y-%m-%d') + datetime.timedelta(days=loc_num)
            model.load_weights(weight_dir + matchDate_dt.strftime('%Y-%m-%d') + extension)
        except:
            loc_num += -1
            matchDate_dt = datetime.datetime.strptime(matchDate, '%Y-%m-%d') + datetime.timedelta(days=loc_num)
            if matchDate_dt.strftime('%Y-%m-%d') == BaseDate:
                model_loaded = True
        else:
            model_loaded = True

def scalize_variable(variables, hist, idx):
    local_s = variables.iloc[idx - hist:idx].copy()

    local_scaler.fit(local_s)

    local_s = local_scaler.transform(local_s)

    local_s = tf.expand_dims(np.concatenate(local_s), 0)

    return local_s

def stepTernary(a, matchDate):

    matchDate_dt = datetime.datetime.strptime(matchDate, '%Y-%m-%d')
    fwdDate_dt = datetime.datetime.strptime(matchDate, '%Y-%m-%d') + datetime.timedelta(days=gain_term)
    idx = kospi_var.index.get_loc(matchDate_dt, method='ffill')
    fwd_idx = kospi_var.index.get_loc(fwdDate_dt, method='ffill')
    gain = kospi_var.iloc[fwd_idx][0]/kospi_var.iloc[idx][0] - 1

    if a == 2:
        r = gain
    elif a == 1:
        r = 0
    else:
        r = -gain

    return r

def prob_to_action(logits):
    a_dist = logits.numpy()
    a = np.random.choice(a_dist[0], p=a_dist[0])
    a = np.argmax(a_dist == a)

    return a

def VoteToStep(actions, prob, history_rewards, matchDate):

    action_policy = "weight_average"
    sum_range = 8
    model_weight = [-0.5, -0.5, -0.5, 0., 0. , 1.5]

    if action_policy == "reward":
        gamma = 1 #gamma for 13 weeks 7017
        gamma_arr = np.ones(sum_range)
        for idx in range(sum_range):
            gamma_arr[idx] = gamma_arr[idx] * (gamma**idx)
        gamma_arr = gamma_arr[::-1]

        np_rewards = np.array(history_rewards)
        weighted_actions = np_rewards[-sum_range:,]
        weighted_actions = weighted_actions*gamma_arr[-len(weighted_actions):].reshape(len(weighted_actions),1)
        weighted_actions = weighted_actions.mean(axis=0)
        offset = min(weighted_actions)
        for idx in range(len(weighted_actions)):
            weighted_actions[idx] = weighted_actions[idx] - offset + 1
        #a = random.choices(actions, weights=weighted_actions)[0]
        max_model = np.argmax(weighted_actions)
        a = random.choices(actions, weights=weighted_actions)[0]

    elif action_policy == "average":
        weighted_prob = (model_weight * np.array(prob).T).T.sum(axis=0)
        a = random.choices([0,1,2], weights=weighted_prob)[0]

    elif action_policy == "weight_average":
        np_rewards = np.array(history_rewards)[-sum_range:, ]
        if len(np_rewards) == sum_range:
            sharp_rewards = np_rewards.mean(axis=0)/np_rewards.std()
        else:
            sharp_rewards = np_rewards.mean(axis=0)
        offset = min(sharp_rewards)
        for idx in range(len(sharp_rewards)):
            sharp_rewards[idx] = sharp_rewards[idx] - offset + 1
        weighted_prob = (sharp_rewards * model_weight * np.array(prob).T).T.sum(axis=0)
        a = random.choices([0,1,2], weights=weighted_prob)[0]

    elif action_policy == 'model_switch':
        np_rewards = np.array(history_rewards)[-sum_range:, ]
        if len(np_rewards) == sum_range:
            sharp_rewards = np_rewards.mean(axis=0)/np_rewards.std()
        else:
            sharp_rewards = np_rewards.mean(axis=0)
        offset = min(sharp_rewards)
        for idx in range(len(sharp_rewards)):
            sharp_rewards[idx] = sharp_rewards[idx] - offset + 1

    r = stepTernary(a, matchDate)
    group_r = []
    for action in actions:
        temp_r = stepTernary(action, matchDate)
        group_r.append(temp_r)
    #weighted_a = np.mean(actions)
    return a, r, group_r


def VoteCounsel(s_E, s_S, matchDate, inputdim):

    RNFC_E_1Y = makeModel_RNFC(inputdim)
    RNFC_E_3Y = makeModel_RNFC(inputdim)
    RNFC_E_All = makeModel_RNFC(inputdim)
    RNFC_S_1Y = makeModel_RNFC(inputdim)
    RNFC_S_3Y = makeModel_RNFC(inputdim)
    RNFC_S_All = makeModel_RNFC(inputdim)

    loadModel_RNFC(RNFC_E_1Y, weightdir_RNFC1Y, '.h5', matchDate)
    loadModel_RNFC(RNFC_E_3Y, weightdir_RNFC3Y, '.h5', matchDate)
    loadModel_RNFC(RNFC_E_All, weightdir_RNFCAll, '.h5', matchDate)
    loadModel_RNFC(RNFC_S_1Y, weightdir_RNFC1Y, '.h5', matchDate)
    loadModel_RNFC(RNFC_S_3Y, weightdir_RNFC3Y, '.h5', matchDate)
    loadModel_RNFC(RNFC_S_All, weightdir_RNFCAll, '.h5', matchDate)

    p_RNFC_E_1Y = RNFC_E_1Y(s_E)
    p_RNFC_E_3Y = RNFC_E_3Y(s_E)
    p_RNFC_E_All = RNFC_E_All(s_E)
    p_RNFC_S_1Y = RNFC_S_1Y(s_S)
    p_RNFC_S_3Y = RNFC_S_3Y(s_S)
    p_RNFC_S_All = RNFC_S_All(s_S)

    a_RNFC_E_1Y = prob_to_action(p_RNFC_E_1Y)
    a_RNFC_E_3Y = prob_to_action(p_RNFC_E_3Y)
    a_RNFC_E_All = prob_to_action(p_RNFC_E_All)
    a_RNFC_S_1Y = prob_to_action(p_RNFC_S_1Y)
    a_RNFC_S_3Y = prob_to_action(p_RNFC_S_3Y)
    a_RNFC_S_All = prob_to_action(p_RNFC_S_All)

    return [a_RNFC_E_1Y,a_RNFC_E_3Y, a_RNFC_E_All,a_RNFC_S_1Y,a_RNFC_S_3Y, a_RNFC_S_All], [p_RNFC_E_1Y[0], p_RNFC_E_3Y[0], p_RNFC_E_All[0],p_RNFC_S_1Y[0], p_RNFC_S_3Y[0], p_RNFC_S_All[0]]

def variables_input_control(variables, x):

    variables.ECON_PCA_1 = x[0] * variables.ECON_PCA_1
    variables.ECON_PCA_2 = x[1] * variables.ECON_PCA_2
    variables.ECON_PCA_3 = x[2] * variables.ECON_PCA_3
    variables.ECON_PCA_4 = x[3] * variables.ECON_PCA_4
    variables.KospiDeT = x[4] * variables.KospiDeT
    # variables.KospiDeT = -variables.KospiDeT
    variables.SNET_pos = x[5] * variables.SNET_pos
    variables.EMOS_PCA_1 = x[6] * variables.EMOS_PCA_1
    variables.EMOS_PCA_2 = x[7] * variables.EMOS_PCA_2
    variables.EMOS_PCA_3 = x[8] * variables.EMOS_PCA_3
    variables.EMOS_PCA_4 = x[9] * variables.EMOS_PCA_4
    scaler.fit(variables)
    variables = scaler.transform(variables)[-hist:]

    return tf.expand_dims(np.concatenate(variables), 0)


def PredModel(x):

    x_E = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    x_S = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    memory = []
    history_rewards = [[1,1,1,1,1,1]]
    score = 1

    with tqdm(list) as tqd:

        for element in tqd:

            matchDate = element.replace('variablesEnS','').replace('.pkl','')
            variables = get_variables(0, var_dir + '/' + element)
            s_E = variables_input_control(variables, x_E)
            s_S = variables_input_control(variables, x_S)

            actions, prob = VoteCounsel(s_E, s_S, matchDate, inputdim)
            a, r, rewards = VoteToStep(actions, prob, history_rewards, matchDate)
            score = score * (1 + r)
            history_rewards.append(rewards)
            memory.append(score)

    return memory

hist = 126
gain_term = 7

dir = 'C:/Users/admin/PycharmProjects/pythonProject_tf_2'
var_dir = dir + '/pickle_var/variables'

weightdir_RNFC1Y = dir + '/weights/reinforce4L/all/test_historic_'
weightdir_RNFC3Y = dir + '/weights/reinforce4L/all/test_historic_'
weightdir_RNFCAll = dir + '/weights/reinforce4L/all/test_historic_'

#get historical kospi price data
'''startdate = '2012-03-16'
enddate = '-0d'
ds = dsws.Datastream(username='ZSEJ010', password='POINT410')
kospi_var = ds.get_data(tickers='KORCOMP', fields=['PI'], start=startdate, end=enddate, freq='D')
kospi_var.index = pd.to_datetime(kospi_var.index, format='%Y-%m-%d')'''
kospi_var = pd.read_csv(dir + '/pickle_var/kospi_index.csv')
kospi_var.index = kospi_var.Date
kospi_var = kospi_var.drop('Date',axis=1)
kospi_var.index = pd.to_datetime(kospi_var.index, format='%Y-%m-%d')

#get list of input datas
list = os.listdir(var_dir)
#BaseDate = list[0].replace('variablesEnS','').replace('.pkl','')
BaseDate = '2014-03-28'
BaseDate_dt = datetime.datetime.strptime(BaseDate, '%Y-%m-%d')
list = list[list.index('variablesEnS' + BaseDate + '.pkl'):]

#get input variables for find out dimension of input
variables = get_mainvariables(test_size = 0, var_dir=var_dir + '/' + list[0])

inputdim = tf.constant(hist * len(variables.iloc[-1]))

'''after initialization, start calculation portfolio return'''
total_memory = []

if __name__ == '__main__':

    a_pool = multiprocessing.Pool(int(os.cpu_count()/2))
    total_memory = a_pool.map(PredModel, range(8))
    #total_memory = a_pool.map(PredModel, range(0, 8))
    total_memory_df = pd.DataFrame(np.array(total_memory).T)

    total_memory_df.plot()
    #total_memory_df.to_csv('C:/Users/admin/Desktop/EKSM.csv')
    kospi = kospi_var.iloc[kospi_var.index.get_loc(BaseDate_dt):]
    plt.show()

    '''for idx in range(len(main_variables)):
        if idx % fwd_idx != 0:
            main_variables.iloc[idx,0] = np.nan
    main_variables = main_variables.dropna(axis = 0)'''

    '''final_port = pd.DataFrame(np.array(total_memory).T,index=kospi.index)
    final_port = final_port + main_variables.Kospi[0]
    final_port['Kospi'] = main_variables.Kospi.copy()

    final_port.to_csv(dir+ '/total_memory_temp2.csv')
    #pickle.dump(final_port, open(dir+ '/total_memory_temp2.pkl','wb'))
    final_port.plot()
    plt.show()'''
    a = 1