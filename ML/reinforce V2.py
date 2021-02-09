import numpy as np
import pickle
import tensorflow as tf
import random

def get_variables(train_size,direct):
    with open(direct, 'rb') as f:
        arr = pickle.load(f)
    arr = arr[:train_size]
    return arr.values.tolist()

def discount_rewards(r, gamma, random_rate):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in range(0, r.size): # give more weight one recent value
        running_add = running_add*gamma + r[t]
        discounted_r[t] = running_add
    for t in range(0, r.size):
        discounted_r[t] = random.choices([discounted_r[t],-discounted_r[t]],weights=[1000-random_rate,random_rate], k=1)
        #give false information to escape from false local optimal
    return discounted_r

def gammaDrvF(gamma,n):
    return 1/(gamma-1) * np.log(gamma)*np.exp(n*np.log(gamma))

def findgamma(varLen, maxRewardPeriod, stopSlope):
    #to give premium to recent data, should make bandwidth formula of time
    #As length of series increase, gamma should be changed for longer series
    temp_idx = 0
    for gamma_r in range(800,1000, 1):
        gamma = gamma_r/1000
        for idx in range(1,varLen):
            if gammaDrvF(gamma, idx) <= stopSlope:
                temp_idx = idx
                break
        if (varLen - temp_idx) <= (varLen * maxRewardPeriod):
            return gamma

def step(a, arr, arr_idx, fwd_idx):

    gain = arr[arr_idx+fwd_idx][0] - arr[arr_idx][0]
    if a == 2: # Bull
        r = gain
    elif a == 1: # Neutral
        r = 0
    else: # Bear
        r = -gain
    return r

def reinforceLearning(train_data,
                      train_size,
                      model_load,
                      learning_period,
                      hist, # time length of input data
                      iterations,
                      update_period, # gradient will be updated for every 10 iterations
                      stopSlope, # reward decaying High Pass Filter
                      maxRewardPeriod, # length of top of High Pass Filter
                      inputdata_direct, # location of input data pickle
                      save_direct # location of saving weights
                      ):

    variables = train_data[:train_size].values.tolist()
    gamma = 0.9153 #-> rewards after 52weeks(9153), 26weeks(0.8376) 0.01% #findgamma(train_size,maxRewardPeriod,stopSlope)
    former_max = 0
    random_rate = 1
    inputdim = hist * len(variables[-1])
    #model_load = False

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=inputdim, activation='relu'))
    model.add(tf.keras.layers.Dense(52, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if model_load == False:
        pass
    else:
        model.load_weights(save_direct+'/test_historic_'+str(train_size - learning_period)+'.h5')

    gradBuffer = model.trainable_variables
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    scores = []
    for iter in range(iterations):

        memory=[]
        score=0
        startpoint = hist + random.randrange(0,hist)
        fwd_idx = 5

        for idx in range(startpoint,len(variables)-fwd_idx,fwd_idx):
            s = tf.expand_dims(np.concatenate(variables[idx-hist:idx]),0)
            if idx + fwd_idx >= len(variables):
                break
            with tf.GradientTape() as tape:
                logits = model(s)
                a_dist = logits.numpy()
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist==a)
                loss = compute_loss([a],logits)
            grads = tape.gradient(loss, model.trainable_variables)
            r= step(a, variables, idx, fwd_idx)
            score += r
            memory.append([grads,r])

        scores.append(score)
        memory=np.array(memory)
        memory[:,1]=discount_rewards(memory[:,1],gamma, random_rate)

        for grads, r in memory:
            for ix, grad in enumerate(grads):
                gradBuffer[ix] += grad*r

        if iter % update_period == 0:
            optimizer.apply_gradients(zip(gradBuffer,model.trainable_variables))
            for ix, grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0

        if iter % update_period == 0:
            print("{} Learning  {}  Score  {}   Var  {}   Max  {}   Min  {}"
                  .format(train_size, iter, np.mean(scores[-50:]), np.std(scores[-50:]), np.max(scores[-50:]), np.min(scores[-50:])))
            if former_max == np.max(scores[-50:]):
                random_rate = 100
            else:
                random_rate = 1
            former_max = np.max(scores[-50:])

        if iter % 400 == 0:
            model.save_weights(save_direct+'/test_historic_'+str(train_size)+'.h5')

    model.save_weights(save_direct+'/test_historic_'+str(train_size)+'.h5')

if __name__ == '__main__':

    inputdata_direct = './pickle_var/variables1.pkl'
    learning_period = 21
    hist = 63
    iterations = 801
    update_period = 50
    stopSlope = 0.01
    maxRewardPeriod = 0.3
    save_direct = './weights'

    with open(inputdata_direct, 'rb') as f:
        arr = pickle.load(f)

    firsttime = True
    for idx in range(500, len(arr), learning_period):
        if firsttime == True:
            model_load = False
            firsttime = False
        else:
            model_load = True

        reinforceLearning(train_data=arr, train_size = idx, model_load=model_load, learning_period=learning_period, hist=hist,
                          iterations=iterations, update_period=update_period,
                          stopSlope=stopSlope, maxRewardPeriod=maxRewardPeriod,
                          inputdata_direct=inputdata_direct,save_direct=save_direct)
