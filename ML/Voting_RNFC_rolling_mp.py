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

def makeModel_RNFC(inputdim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=inputdim, activation='relu'))
    model.add(tf.keras.layers.Dense(52, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    return model

def loadModel_RNFC(model, weight_dir, extension, idx):

    model_loaded = False
    loc_num = -5
    model.load_weights(weight_dir + str(252) + extension)
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

    local_scaler.fit(local_s)

    local_s = local_scaler.transform(local_s)

    local_s = tf.expand_dims(np.concatenate(local_s), 0)

    return local_s

def stepTernary(a, arr, arr_idx, fwd_idx):

    gain = arr[arr_idx+fwd_idx] - arr[arr_idx]
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

def VoteToStep(actions, prob, history_rewards, arr, arr_idx, fwd_idx):

    action_policy = "reward"

    if action_policy == "reward":
        sum_range = 13
        gamma = 1 #gamma for 13 weeks 7017
        gamma_arr = np.ones(sum_range)
        for idx in range(sum_range):
            gamma_arr[idx] = gamma_arr[idx] * (gamma**idx)
        gamma_arr = gamma_arr[::-1]

        np_rewards = np.array(history_rewards)
        weighted_actions = np_rewards[-sum_range:,]
        weighted_actions = weighted_actions*gamma_arr[-len(weighted_actions):].reshape(len(weighted_actions),1)
        weighted_actions = weighted_actions.sum(axis=0)
        offset = min(weighted_actions)
        for idx in range(len(weighted_actions)):
            weighted_actions[idx] = weighted_actions[idx] - offset + 1
        #a = random.choices(actions, weights=weighted_actions)[0]
        max_model = np.argmax(weighted_actions)
        a = random.choices(actions, weights=weighted_actions)[0]

    elif action_policy == "average":
        avg_prob = np.array(prob).mean(axis=0)
        a = np.random.choice([0,1,2], p=avg_prob)

    elif action_policy == "weight_average":
        sum_range = 13
        np_rewards = np.array(history_rewards)[-sum_range:, ]
        if len(np_rewards) == sum_range:
            sharp_rewards = np_rewards.sum(axis=0)/np_rewards.std()
        else:
            sharp_rewards = np_rewards.sum(axis=0)
        offset = min(sharp_rewards)
        for idx in range(len(sharp_rewards)):
            sharp_rewards[idx] = sharp_rewards[idx] - offset + 1
        weighted_prob = (sharp_rewards * np.array(prob).T).T.mean(axis=0)
        a = random.choices([0,1,2], weights=weighted_prob)[0]

    r = stepTernary(a, arr, arr_idx, fwd_idx)
    group_r = []
    for action in actions:
        temp_r = stepTernary(action, arr, arr_idx, fwd_idx)
        group_r.append(temp_r)
    #weighted_a = np.mean(actions)
    return a, r, group_r


def VoteCounsel(s, idx, inputdim):

    RNFC_1Y = makeModel_RNFC(inputdim)
    RNFC_3Y = makeModel_RNFC(inputdim)
    RNFC_All = makeModel_RNFC(inputdim)

    loadModel_RNFC(RNFC_1Y, weightdir_RNFC1Y, '.h5', idx)
    loadModel_RNFC(RNFC_3Y, weightdir_RNFC3Y, '.h5', idx)
    loadModel_RNFC(RNFC_All, weightdir_RNFCAll, '.h5', idx)

    p_RNFC_1Y = RNFC_1Y(s)
    p_RNFC_3Y = RNFC_3Y(s)
    p_RNFC_All = RNFC_All(s)
    a_RNFC_1Y = prob_to_action(p_RNFC_1Y)
    a_RNFC_3Y = prob_to_action(p_RNFC_3Y)
    a_RNFC_All = prob_to_action(p_RNFC_All)

    return [a_RNFC_1Y,a_RNFC_3Y, a_RNFC_All], [p_RNFC_1Y[0],p_RNFC_3Y[0], p_RNFC_All[0]]

def PredModel(x):

    memory = []
    s = []
    actions = []
    history_rewards = [[1,1,1]]
    action = 0
    score = 0
    counter=0
    startpoint = 252
    model_date = startpoint
    hist = 63

    with tqdm(range(startpoint, 2677, fwd_idx)) as tqd:

        for idx in tqd:

            variables = get_variables(0, var_dir + str(idx + 5) + '.pkl')
            variables.ECON_PCA_1 = 0
            variables.ECON_PCA_2 = 0
            variables.ECON_PCA_3 = 0
            variables.ECON_PCA_4 = 0
            variables.KospiDeT = 0
            #variables.KospiDeT = -variables.KospiDeT
            #variables.SNET_pos = 0
            #variables.EMOS_PCA_1 = 0
            #variables.EMOS_PCA_2 = 0
            #variables.EMOS_PCA_3 = 0
            #variables.EMOS_PCA_4 = 0
            scaler.fit(variables)
            variables = scaler.transform(variables)[-63:]
            s = tf.expand_dims(np.concatenate(variables), 0)
            #s = scalize_variable(variables, hist, idx)
            actions, prob = VoteCounsel(s, idx, inputdim)
            a, r, rewards = VoteToStep(actions, prob, history_rewards, kospi_var, idx, fwd_idx)
            score += r
            history_rewards.append(rewards)
            memory.append(score)

    return memory

learning_period = 21
fwd_idx = 5
hist = 63

dir = 'C:/pythonProject_tf/textAnalysis'
mainvar_dir = dir + '/pickle_var/variables.pkl'
var_dir = dir + '/pickle_var/test_variables_rolling/variables1_2_'

weightdir_RNFC1Y = dir + '/weights/reinforcestd_rolling/weights252/test_historic_'
weightdir_RNFC3Y = dir + '/weights/reinforcestd_rolling/weights756/test_historic_'
weightdir_RNFCAll = dir + '/weights/reinforcestd_rolling/weightsAll/test_historic_'

main_variables = get_mainvariables(test_size = 0, var_dir=mainvar_dir)
variables = get_mainvariables(test_size = 0, var_dir=var_dir + str(257) + '.pkl')

kospi_var = main_variables.Kospi.copy()

inputdim = tf.constant(hist * len(variables.iloc[-1]))

'''after initialization, start calculation portfolio return'''
total_memory = []

if __name__ == '__main__':

    a_pool = multiprocessing.Pool(os.cpu_count())
    total_memory = a_pool.map(PredModel, range(0, 16))

    main_variables = main_variables[252:]

    for idx in range(len(main_variables)):
        if idx % fwd_idx != 0:
            main_variables.iloc[idx,0] = np.nan
    main_variables = main_variables.dropna(axis = 0)

    final_port = pd.DataFrame(np.array(total_memory).T,index=main_variables[1:].index)
    final_port = final_port + main_variables.Kospi[0]
    final_port['Kospi'] = main_variables.Kospi.copy()

    final_port.to_csv(dir+ '/total_memory_temp2.csv')
    #pickle.dump(final_port, open(dir+ '/total_memory_temp2.pkl','wb'))
    final_port.plot()
    plt.show()
    a = 1