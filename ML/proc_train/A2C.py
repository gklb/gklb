'''참고
grad tape adopted version
https://github.com/abhisheksuran/Reinforcement_Learning/blob/master/ActorCritic.ipynb
description for upper code
https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-1-of-2-d1e26a54ce97
tensforflow official version
https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
tf1 version
https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/ce39055e492798625a42afb24352b63a2a1f5954/4.%20A2C/A2C_CartPole-v0.py#L84
https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
https://www.youtube.com/watch?v=2I1pE0Cx7Qk
'''

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pickle
from tqdm import tqdm
import random

def choose_action(a_prob):
    a_prob = a_prob.numpy()
    a_dist = tfp.distributions.Categorical(probs=a_prob, dtype=tf.float32)
    a = int(a_dist.sample().numpy()[0])
    return a

'''def get_loss(a_prob):
    a = choose_action(a_prob)
    loss = compute_loss([a], a_prob)
    return loss'''

def preprocess_step(states, actions, rewards, gamma):
    #compute Q Vals
    discount_rewards = []
    sum_reward = 0
    for r in rewards:
        sum_reward = r + gamma*sum_reward
        discount_rewards.append(sum_reward)

    return discount_rewards

def comp_loss(probs, actions, td):
    probability = []
    log_probability = []
    for pb, a in zip(probs,actions):
        log_prob = tf.math.log(pb+1e-8)
        probability.append(pb)
        log_probability.append(log_prob)

    p_loss = []
    e_loss = []
    td = td.numpy()
    for pb, t, lpb in zip(probability, td, log_probability):
        t = tf.constant(t)
        policy_loss = tf.math.multiply(lpb,t)
        entropy_loss = tf.math.multiply(pb,lpb)
        p_loss.append(policy_loss)
        e_loss.append(entropy_loss)
    p_loss = tf.stack(p_loss)
    e_loss = tf.stack(e_loss)
    p_loss = tf.reduce_mean(p_loss)
    e_loss = tf.reduce_mean(e_loss)
    loss = p_loss + 0.001 * e_loss

    return loss

'''every time before learning gradient, get action from current state,
and then get next_state, reward from environment'''
'''learn'''
def learn(states, actions, discount_rewards):
    discount_rewards = tf.reshape(discount_rewards, (len(discount_rewards),))

    action_prob = []
    value = []
    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

        for state in states:
            action_prob.append(actor(state))
            value.append(critic(state))
        value = tf.reshape(value, (len(value),))
        # compute advantage
        td = tf.math.subtract(discount_rewards, value)
        actor_loss = comp_loss(action_prob, actions, td)
        critic_loss = 0.5*tf.keras.losses.mean_squared_error(discount_rewards, value)

    grad1 = tape1.gradient(actor_loss, actor.trainable_variables)
    grad2 = tape2.gradient(critic_loss, critic.trainable_variables)
    actor_opt.apply_gradients(zip(grad1, actor.trainable_variables))
    critic_opt.apply_gradients(zip(grad2, critic.trainable_variables))

    return actor_loss, critic_loss

def step(a, arr, arr_idx, fwd_idx):

    gain = arr[arr_idx+fwd_idx][0] - arr[arr_idx][0]
    if a == 2: # Bull
        r = gain
    elif a == 1: # Neutral
        r = -gain/10
    else: # Bear
        r = -gain
    return r

def Agent(train_size):
    variables = arr[:train_size].values.tolist()
    ep_reward = []
    total_avgr = []
    with tqdm(range(iterations)) as tqd:
        for iter in tqd:

            startpoint = max(len(variables) - 756 + random.randrange(0, hist), hist + random.randrange(0, round(hist/2)))
            total_reward = 0
            rewards = []
            states = []
            actions = []
            for step_idx in range(startpoint, len(variables) - fwd_idx, fwd_idx):
                state = tf.expand_dims(np.concatenate(variables[step_idx - hist:step_idx], axis=None), 0)
                if step_idx + fwd_idx >= len(variables):
                    break
                action_prob = actor(state)
                action = choose_action(action_prob)
                reward = step(action, variables, step_idx, fwd_idx)
                rewards.append(reward)
                actions.append(action)
                states.append(state)
                total_reward += reward

            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            tqd.set_postfix(Time=train_size, Score=total_reward, AVG=avg_reward)

            discount_rewards = preprocess_step(states,actions,rewards,gamma)
            aloss, closs = learn(states, actions, discount_rewards)
            tqd.set_postfix(Time=idx, Score=total_reward)

basedir = 'C:/pythonProject_tf/textAnalysis'
inputdata_direct = basedir + '/pickle_var/variables1_2.pkl'
learning_period = 5
hist = 63
iterations = 1601
fwd_idx = 5
save_direct = basedir + '/weights/52w'

with open(inputdata_direct, 'rb') as f:
    arr = pickle.load(f)

lr_actor = 0.001
lr_critic = 0.01

gamma = 0.7017  # -> rewards after 52weeks(9153), 26weeks(0.8376), 13weeks(0.7017), 7weeks(0.5179) 0.01% #findgamma(train_size,maxRewardPeriod,stopSlope)
inputdim = hist * arr.shape[1]


'''define models'''
#compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

actor = tf.keras.Sequential()
actor.add(tf.keras.layers.Dense(128, input_dim=inputdim, activation='relu'))
actor.add(tf.keras.layers.Dense(52, activation='relu'))
actor.add(tf.keras.layers.Dense(3, activation='softmax'))
actor_opt = tf.keras.optimizers.Adam(learning_rate=lr_actor)

critic = tf.keras.Sequential()
critic.add(tf.keras.layers.Dense(128, input_dim=inputdim, activation='relu'))
critic.add(tf.keras.layers.Dense(52, activation='relu'))
critic.add(tf.keras.layers.Dense(1, activation=None))
critic_opt = tf.keras.optimizers.Adam(learning_rate=lr_critic)

firsttime = True
for idx in range(2400, arr.shape[0], learning_period):
    if firsttime == True:
        model_load = False
        firsttime = False
    else:
        model_load = True

    Agent(idx)



