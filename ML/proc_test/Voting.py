from matplotlib import pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import random
import pandas as pd
import os
import xgboost as xgb
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
local_scaler = StandardScaler()

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

def stepBinary(a, arr, arr_idx, fwd_idx):

    gain = arr[arr_idx+fwd_idx] - arr[arr_idx]
    if a == 1:
        r = gain
    else:
        r = -gain
    return r

def stepTernary(a, arr, arr_idx, fwd_idx):

    gain = arr[arr_idx+fwd_idx] - arr[arr_idx]
    if a == 1: # Bull
        r = gain
    elif a == 0.5: # Neutral
        r = 0
    else:
        r = -gain

    return r

def VoteToStep(actions, history_rewards, arr, arr_idx, fwd_idx):

    np_rewards = np.array(history_rewards)
    if len(np_rewards[-52:])==52:
        weighted_actions = np_rewards[-52:,].sum(axis=0)/np_rewards[-12:,].std(axis=0)
    else:
        weighted_actions = np_rewards[-52:, ].sum(axis=0)
    #act_rew_raw = pd.DataFrame(np.transpose(np.vstack([np.array(actions),weighted_actions])))
    #grouped_act_rew = act_rew_raw.groupby(act_rew_raw[0]).sum()
    #actions_list = grouped_act_rew.index.values
    #actions_weight = grouped_act_rew[1].values
    #a = random.choices(actions_list, weights=actions_weight)[0]
    a = random.choices(actions, weights=weighted_actions)[0]
    r = stepTernary(a, arr, arr_idx, fwd_idx)
    group_r = []
    for action in actions:
        temp_r = stepTernary(action, arr, arr_idx, fwd_idx)
        group_r.append(temp_r)
    #weighted_a = np.mean(actions)
    return a, r, group_r

def loadModel_RndFst(weight_dir, extension, idx):
    model_loaded = False
    loc_num = 0
    while model_loaded == False:
        try:
            model = pickle.load(open(weight_dir + str(idx + loc_num) + extension, 'rb'))
        except:
            loc_num += -1
            if loc_num < -21:
                model_loaded = True
        else:
            model_loaded = True

    return model

def makeModel_DNN(inputdim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=inputdim, activation='relu'))
    model.add(tf.keras.layers.Dense(52, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model

def loadModel_DNN(model, weight_dir, extension, idx):

    model_loaded = False
    loc_num = 0
    model.load_weights(weight_dir + str(756) + extension)
    while model_loaded == False:
        try:
            model.load_weights(weight_dir + str(idx+loc_num) + extension)
        except:
            loc_num += -1
            if loc_num < -21:
                model_loaded = True
        else:
            model_loaded = True

def makeModel_RNFC(inputdim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=inputdim, activation='relu'))
    model.add(tf.keras.layers.Dense(52, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    return model

def loadModel_RNFC(model, weight_dir, extension, idx):

    model_loaded = False
    loc_num = 0
    model.load_weights(weight_dir + str(756) + extension)
    while model_loaded == False:
        try:
            model.load_weights(weight_dir + str(idx+loc_num) + extension)
        except:
            loc_num += -1
            if loc_num < -21:
                model_loaded = True
        else:
            model_loaded = True

def scalize_variable(variables, hist, idx):
    local_s = variables.iloc[idx - hist:idx].copy()
    s = variables.iloc[:idx + 1].copy()

    local_scaler.fit(local_s)
    scaler.fit(s)

    local_s = local_scaler.transform(local_s)
    s = scaler.transform(s)

    local_s = tf.expand_dims(np.concatenate(local_s), 0)
    s = np.transpose(s[idx].reshape((-1, 1)))

    return s, local_s


def VoteCounsel(s2, local_s2, s3, local_s3, idx, inputdim2, inputdim3):

    loadModel_DNN(DNN_2, weightdir_DNN1_2, '.h5', idx)
    loadModel_DNN(DNN_3, weightdir_DNN1_3, '.h5', idx)
    loadModel_RNFC(RNFC_2, weightdir_RNFC1_2, '.h5', idx)
    loadModel_RNFC(RNFC_3, weightdir_RNFC1_3, '.h5', idx)
    RndFst_2 = loadModel_RndFst(weightdir_RndFst1_2, '.pkl', idx)
    RndFst_3 = loadModel_RndFst(weightdir_RndFst1_3, '.pkl', idx)

    a_DNN_2 = np.random.choice([0,1],p=DNN_2.predict(local_s2)[0])
    a_DNN_3 = np.random.choice([0,1],p=DNN_3.predict(local_s3)[0])
    a_RNFC_2 = np.random.choice([0,0.5,1],p=RNFC_2.predict(local_s2)[0])
    a_RNFC_3 = np.random.choice([0,0.5,1],p=RNFC_3.predict(local_s3)[0])
    a_RndFst_2 = RndFst_2.predict(s2)[0]
    a_RndFst_3 = RndFst_3.predict(s3)[0]

    return [a_DNN_2,a_DNN_3,a_RNFC_2,a_RNFC_3,a_RndFst_2,a_RndFst_3]

def PredModel(variables2, variables3, kospi_var):

    memory = []
    s = []
    actions = []
    history_rewards = [[1,1,1,1,1,1]]
    action = 0
    score = 0
    counter=0
    startpoint = 756
    model_date = startpoint
    hist = 63

    with tqdm(range(startpoint, len(variables2) - fwd_idx, fwd_idx)) as tqd:

        for idx in tqd:

            s2, local_s2 = scalize_variable(variables2, hist, idx)
            s3, local_s3 = scalize_variable(variables3, hist, idx)

            actions = VoteCounsel(s2,local_s2, s3, local_s3, idx, inputdim2, inputdim3)
            a, r, rewards = VoteToStep(actions, history_rewards, kospi_var, idx, fwd_idx)
            score += r
            history_rewards.append(rewards)
            memory.append([a, score])

    return memory

learning_period = 21
fwd_idx = 5
hist = 63

dir = 'C:/pythonProject_tf/textAnalysis'
mainvar_dir = dir + '/pickle_var/variables.pkl'
var_dir1_2 = dir + '/pickle_var/variables1_2.pkl'
var_dir1_3 = dir + '/pickle_var/variables1_3.pkl'

weightdir_DNN1_2 = dir + '/weights/DNN_Softmax/weights1_2/test_historic_'
weightdir_DNN1_3 = dir + '/weights/DNN_Softmax/weights1_3/test_historic_'
weightdir_RndFst1_2 = dir + '/weights/randomforest/weights1_2/test_historic_'
weightdir_RndFst1_3 = dir + '/weights/randomforest/weights1_3/test_historic_'
weightdir_RNFC1_2 = dir + '/weights/reinforce/weights1_2/test_historic_'
weightdir_RNFC1_3 = dir + '/weights/reinforce/weights1_3/test_historic_'

main_variables = get_mainvariables(test_size = 0, var_dir=mainvar_dir)
variables2 = get_mainvariables(test_size = 0, var_dir=var_dir1_2)
variables3 = get_mainvariables(test_size = 0, var_dir=var_dir1_3)
kospi_var = main_variables.Kospi.copy()

inputdim2 = tf.constant(hist * len(variables2.iloc[-1]))
inputdim3 = tf.constant(hist * len(variables3.iloc[-1]))
DNN_2 = makeModel_DNN(inputdim2)
DNN_3 = makeModel_DNN(inputdim3)
RNFC_2 = makeModel_RNFC(inputdim2)
RNFC_3 = makeModel_RNFC(inputdim3)

'''after initialization, start calculation portfolio return'''
total_memory = []
memory = PredModel(variables2, variables3, kospi_var)
total_memory.append(memory)

main_variables = main_variables[756:]

for idx in range(len(main_variables)):
    if idx % fwd_idx != 0:
        main_variables.iloc[idx,0] = np.nan
main_variables = main_variables.dropna(axis = 0)

final_port = pd.DataFrame(total_memory[0],index=main_variables[:-1].index)
final_port['Kospi'] = main_variables.Kospi.copy()
final_port[1] = final_port[1] + final_port.Kospi[0]

final_port[1].plot()
final_port.Kospi.plot()
final_port[0].plot(secondary_y=True)
plt.show()
a = 1