from matplotlib import pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import random
import pandas as pd
import os

def get_variables(test_size, var_dir):
    with open(var_dir, 'rb') as f:
        arr = pickle.load(f)  # 단 한줄씩 읽어옴
    arr = arr[test_size:]
    return arr.values.tolist()

def get_mainvariables(test_size, var_dir):
    with open(var_dir, 'rb') as f:
        arr = pickle.load(f)  # 단 한줄씩 읽어옴
    arr = arr[test_size:]
    return arr

def step(a, arr, arr_idx, fwd_idx):

    gain = arr[arr_idx+fwd_idx] - arr[arr_idx]
    if a == 1: # Bull
        r = gain
    else: # Bear
        r = -gain
    return r

def step2(a, arr, arr_idx, fwd_idx):

    gain = arr[arr_idx+fwd_idx] - arr[arr_idx]
    r = a * gain
    return r


def predModel(weight_dir, variables, kospi_var):

    hist = 63
    inputdim = hist * len(variables[-1])
    iterations = 101
    update_period = 10
    gamma = 0.9
    memory = []
    s = []
    score = 0
    counter=0
    done = False
    startpoint = 756
    model_date = startpoint
    fwd_idx = 5

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=inputdim, activation='relu'))
    model.add(tf.keras.layers.Dense(52, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    model_loaded = False
    model.load_weights(weight_dir + str(model_date) + '.h5')

    for idx in range(startpoint, len(variables) - fwd_idx, fwd_idx):
        model_loaded = False
        loc_num = 0
        while model_loaded == False:
            try:
                model.load_weights(weight_dir + str(idx+loc_num) + '.h5')
            except:
                loc_num += -1
            else:
                model_loaded = True

        if idx + fwd_idx >= len(variables):
            break
        s = tf.expand_dims(np.concatenate(variables[idx - hist:idx]), 0)
        logits = model(s)
        a_dist = logits.numpy()
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)

        #r = step(a, kospi_var, idx, fwd_idx)
        r = step(a, kospi_var, idx, fwd_idx)
        score += r
        memory.append([a, score])

        counter+=1
        if counter%100==0:
            print(weight_dir + ' / ' + str(idx))

    return memory

dir = 'C:/pythonProject_tf/textAnalysis/'
var_dir = dir + "pickle_var/"
weight_dir = dir + 'weights/DNN_softmax/'
var_list = os.listdir(var_dir)
weight_list = os.listdir(weight_dir)
main_variables = get_mainvariables(test_size = 0, var_dir=var_dir+var_list[0])
kospi_var = main_variables.Kospi.copy()
var_list = var_list[1:]

total_memory = []
for weight_el in weight_list:
    memory = []
    var_num = weight_el.replace('weights','')
    var_el_dir = var_dir + 'variables' + var_num + '.pkl'
    weight_el_dir = weight_dir + weight_el + '/test_historic_'
    try:
        variables = get_variables(test_size=0, var_dir = var_el_dir)
        memory = predModel(weight_el_dir, variables, kospi_var)
        total_memory.append(memory)
    except:
        pass

fwd_idx = 5
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