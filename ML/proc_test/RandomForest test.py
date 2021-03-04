from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.tree import export_graphviz
import numpy as np
import pickle
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

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

def step2(a, arr, arr_idx, fwd_idx):

    gain = arr[arr_idx+fwd_idx] - arr[arr_idx]
    if a == 1:
        r = gain
    else:
        r = -gain
    return r

def predModel(weight_dir, variables, kospi_var):

    model = pickle.load(open(weight_dir + str(756) + '.pkl', 'rb'))
    memory = []
    score = 0
    counter=0
    startpoint = 756
    for idx in range(startpoint, len(variables) - fwd_idx, fwd_idx):

        model_loaded = False
        loc_num = 0
        while model_loaded == False:
            try:
                model = pickle.load(open(weight_dir + str(idx+loc_num) + '.pkl', 'rb'))
            except:
                loc_num += -1
                if loc_num < -21:
                    model_loaded=True
            else:
                model_loaded = True

        local_variables = variables.iloc[:idx+1].copy()
        scaler.fit(local_variables)
        local_variables = scaler.transform(local_variables)
        s = np.transpose(local_variables[idx].reshape((-1, 1)))
        a = model.predict(s)[0]
        #a = np.random.choice([0,1],p=[a_p,1-a_p])

        #r = step(a, kospi_var, idx, fwd_idx)
        r = step2(a, kospi_var, idx, fwd_idx)
        score += r
        memory.append([a, score])

        counter+=1
        if counter%100==0:
            print(weight_dir + ' / ' + str(idx))

    return memory

learning_period = 21
fwd_idx = 5

dir = 'C:/pythonProject_tf/textAnalysis/'
var_dir = dir + "pickle_var/"
weight_dir = dir + 'weights/randomforest/'
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

    variables = get_variables(test_size=0, var_dir = var_el_dir)

    memory = predModel(weight_el_dir, variables, kospi_var)
    total_memory.append(memory)

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