##Inspired by andy-psai, https://github.com/andy-psai/
##Great resource which I used to recreate this here for learning about it.

import tensorflow as tf
import gym
import numpy as np
import sklearn
import sklearn.preprocessing
env = gym.envs.make("MountainCarContinuous-v0")

tf.reset_default_graph()

input_dims = 2
state_placeholder = tf.placeholder(tf.float32, [None, input_dims])

def Critic_Value(state):
    n_hidden1 = 400
    n_hidden2 = 400
    n_outputs = 1

    with tf.variable_scope("value_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()

        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.elu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu, init_xavier)
        V = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
    return V

def Actor_Policy(state):
    n_hidden1 = 40
    n_hidden2 = 40
    n_outputs = 1

    with tf.variable_scope("policy_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()

        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.elu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu, init_xavier)
        mu = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
        sigma = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
        sigma = tf.nn.softplus(sigma) + 1e-5
        norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1),axis=0)
        action_tf_var = tf.clip_by_value(action_tf_var, env.action_space.low[0], env.action_space.high[0])

    return action_tf_var, norm_dist

state_space_samples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(state_space_samples)

def scale_state(state):
    scaled = scaler.transform([state])
    return scaled

lr_actor = 0.00002
lr_critic = 0.001

action_placeholder = tf.placeholder(tf.float32)
delta_placeholder = tf.placeholder(tf.float32)
target_placeholder = tf.placeholder(tf.float32)

action_tf_var, norm_dist = Actor_Policy(state_placeholder)
V = Critic_Value(state_placeholder)

loss_actor = -tf.log(norm_dist.prob(action_placeholder) + 1e-5) * delta_placeholder
training_op_actor = tf.train.AdamOptimizer(lr_actor, name='actor_optimizer').minimize(loss_actor)

loss_critic = tf.reduce_mean(tf.squared_difference(tf.squeeze(V), target_placeholder))
training_op_critic = tf.train.AdamOptimizer(lr_critic, name='critic_optimizer').minimize(loss_critic)

gamma = 0.99
num_episodes = 300

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    episode_history = []
    for episode in range(num_episodes):

        state = env.reset()
        reward_total = 0
        steps = 0
        done = False
        while not done:

            action = sess.run(action_tf_var, feed_dict={state_placeholder:scale_state(state)})

            next_state, reward, done, _ = env.step(np.squeeze(action, axis=0))
            steps += 1
            reward_total +=reward

            V_of_next_state = sess.run(V, feed_dict = {state_placeholder: scale_state(next_state)})

            target = reward + gamma*np.squeeze(V_of_next_state)

            td_error = target - np.squeeze(sess.run(V, feed_dict={state_placeholder: scale_state(state)}))

            _, loss_actor_val = sess.run([training_op_actor,loss_actor],
                                         feed_dict={action_placeholder: np.squeeze(action),
                                                    state_placeholder: scale_state(state),
                                                    delta_placeholder: td_error})
            _, loss_critic_val = sess.run([training_op_critic, loss_critic],
                                         feed_dict={state_placeholder: scale_state(state),
                                                    target_placeholder: target})

            state = next_state
        episode_history.append(reward_total)
        print("Episode: {}, Number of Steps: {}, Cummulative reward: {:0.2f}".format(episode, steps, reward_total))

        if np.mean(episode_history[-100:]) > 90 and len(episode_history) >=101:
            print("*"*20+"High-performance"+"*"*20)
            print("Mean cumulative reward over 100 episodes:{:0.2f}".format(np.mean(episode_history[-100:])))
