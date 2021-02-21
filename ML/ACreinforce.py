'''참고
grad tape adopted version
https://github.com/abhisheksuran/Reinforcement_Learning/blob/master/ActorCritic.ipynb
description for upper code
https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-1-of-2-d1e26a54ce97
tensforflow official version
https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
tf1 version
https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/ce39055e492798625a42afb24352b63a2a1f5954/4.%20A2C/A2C_CartPole-v0.py#L84
'''


import tensorflow as tf
import numpy as np
import pickle

def choose_action(a_prob):
    a_dist = a_prob.numpy()
    a = np.random.choice(a_dist[0], p=a_dist[0])
    a = np.argmax(a_dist == a)
    return a

def get_loss(a_prob):
    choose_action(a_prob)
    loss = compute_loss([a], a_prob)
    return loss

inputdata_direct = './pickle_var/variables1_2.pkl'
learning_period = 5
hist = 63
iterations = 1601
update_period = 50
stopSlope = 0.01
maxRewardPeriod = 0.3
save_direct = './weights/52w'

with open(inputdata_direct, 'rb') as f:
    arr = pickle.load(f)

max_episode = 1000
max_timestep = 100
gamma = 0.9
lr_actor = 0.001
lr_critic = 0.01

action_space = 1
observation_space = 1
lr_actor = 0.01
lr_critic = 0.02
gamma = 0.99
train_size = 1000

variables = arr[:train_size].values.tolist()
gamma = 0.7017  # -> rewards after 52weeks(9153), 26weeks(0.8376), 13weeks(0.7017), 7weeks(0.5179) 0.01% #findgamma(train_size,maxRewardPeriod,stopSlope)
former_max = 0
random_rate = 1
inputdim = hist * len(variables[-1])

obs = tf.variable(tf.ones(shape=(None, hist)))
obs_next = tf.variables(tf.ones(shape=(None,hist)))
action_label = tf.variable(tf.ones(shape=(None)))
Q = tf.variable(tf.ones(shape=(None,1)))

'''define models'''
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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

'''every time before learning gradient, get action from current state,
and then get next_state, reward from environment'''
'''learn'''
with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
    action_prob = actor(obs)
    value = critic(obs)

    action_prob_next = actor(obs_next)
    value_next = critic(obs_next)

    cross_entropy = get_loss(action_prob)
    advantage = Q - value

    actor_loss = tf.reduce_mean(cross_entropy * advantage)
    critic_loss = tf.reduce_mean(tf.square(advantage))

grad1 = tape1.gradient(actor_loss, actor.trainable_variables)
grad2 = tape2.gradient(critic_loss, critic.trainable_variables)
actor_opt.apply_gradients(grad1, actor.trainable_variables)
critic_opt.apply_gradients(grad2, critic.trainable_variables)





