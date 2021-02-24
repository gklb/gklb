from matplotlib import pyplot as plt
import numpy as np
import pickle
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import random
import pandas as pd
import datetime
import requests_cache
from tensorflow.python.training import saver

tf.disable_v2_behavior()

class DecisionPolicy:
    def select_action(self, current_state, step):
        pass

    def update_q(self, state, action, reward, next_state):
        pass


class RandomDecisionPolicy(DecisionPolicy):
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, current_state, step):
        action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action


class QLearningDecisionPolicy(DecisionPolicy):
    def __init__(self, actions, input_dim):
        self.epsilon = 0.9
        self.gamma = 0.9
        self.actions = actions
        output_dim = len(actions)
        h1_dim = 200

        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y = tf.placeholder(tf.float32, [output_dim])
        W1 = tf.Variable(tf.random_normal([input_dim, h1_dim]), name='W1')
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]), name='b1')
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        W2 = tf.Variable(tf.random_normal([h1_dim, output_dim]), name='W2')
        b2 = tf.Variable(tf.constant(0.1, shape=[output_dim]), name='b2')
        self.q = tf.nn.relu(tf.matmul(h1, W2) + b2, name='op_to_restore')
        loss = tf.square(self.y - self.q)
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def load_model(self):
        loader = tf.train.import_meta_graph('test_model.ckpt.meta')
        loader.restore(self.sess, tf.train.latest_checkpoint('test_model.ckpt'))

        train_op = tf.get_collection('W1')
        a=1
        '''W1 = graph.get_tensor_by_name("W1:0")
        W2 = graph.get_tensor_by_name("W2:0")
        b1 = graph.get_tensor_by_name("b1:0")
        b2 = graph.get_tensor_by_name("b2:0")

        op_to_restore = graph.get_tensor_by_name("op_to_restore:0")'''

    def save_model(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'test_model.ckpt')

    def select_action(self, current_state, step):
        threshold = min(self.epsilon, step / 100.)
        if random.random() < threshold:
            # Exploit best option with probability epsilon
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            action_idx = np.argmax(action_q_vals)  # TODO: replace w/ tensorflow's argmax
            action = self.actions[action_idx]
        else:
            # Explore random option with probability 1 - epsilon
            action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action

    def update_q(self, state, action, reward, next_state):
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
        next_action_idx = np.argmax(next_action_q_vals)
        action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})

def run_simulation(policy, variables, hist, num_tries, debug=False):
    transitions = list()
    current_gain = 0
    chg = 0
    gain = 0
    startpoint = 0#random.randrange(0,50)

    for i in range(startpoint,len(variables)-5,5):
        if i % 100 < 5:
            print(num_tries, ' progress {:.2f}%'.format(float(100*i) / (len(variables))))
        current_state = np.asmatrix(np.concatenate(variables[i:i+hist]))
        if len(variables[i+1:i+hist+1]) < hist:
            break
        current_gain = current_gain + gain
        action = policy.select_action(current_state, i)
        chg = float(variables[i+5][0]-variables[i][0])
        if action == 'Bull':
            gain = chg
        elif action == 'Bear':
            gain = -chg
        else:
            gain = 0
        reward = gain
        next_state = np.asmatrix(np.concatenate(variables[i+1:i+hist+1]))
        transitions.append((current_state, action, reward, next_state))
        policy.update_q(current_state, action, reward, next_state)

    portfolio = current_gain
    if debug:
        print('${}\t{} gain'.format(current_gain, gain))
    return portfolio


def run_simulations(policy, variables, hist, iter):
    num_tries = iter
    final_portfolios = list()
    for i in range(num_tries):
        final_portfolio = run_simulation(policy, variables, hist, i)
        final_portfolios.append(final_portfolio)
        policy.save_model()
    avg, std = np.mean(final_portfolios), np.std(final_portfolios)
    final_portfolios = pd.DataFrame(final_portfolios)
    final_portfolios.to_pickle("./rawText/final_port.pkl")
    final_portfolios.plot()
    plt.show()
    return avg, std


def get_variables(test_size):
    with open('./rawText/variables.pkl', 'rb') as f:
        arr = pickle.load(f)  # 단 한줄씩 읽어옴
    arr = arr[['KospiDeT','Rvol','SNET_pos','SNEM_PCA_PS','SNEM_PCA_N','SNEM_PCA_BT']]
    arr = arr[:test_size]
    return arr.values.tolist()

if __name__ == '__main__':

    variables = get_variables(test_size = 1800)
    actions = ['Bull','Bear']
    hist = 21
    iterations = 10
    # policy = RandomDecisionPolicy(actions)
    policy = QLearningDecisionPolicy(actions, 6*hist)
    policy.load_model()
    avg, std = run_simulations(policy, variables, hist, iterations)
    print(avg, std)
    a=1