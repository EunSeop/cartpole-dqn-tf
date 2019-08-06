import collections
import os
import random

import gym
import imageio
import numpy as np
import tensorflow as tf

"""
DQN STEPS:
 1. Initialize replay memory D to capacity N
 2. Initialize action-value function Q with random weights theta
 3. For episode=1, M  do
 4. Initialize squence s1={x1} and preprocess sequence pi1=pi1(s1)
 6.   For t1, T  do
 7.   select random action or predicted aciton 
 8.   execute action and observe reward r(t) and status(t+1)
 9.   S(t+1) = S(t),a(t),x(t+1) and preprocess pi(t+1) = pi(S(t+1))
 10.   store transition (pi(t), a(t), r(t), pi(t+1)) in D
 11.   sample random minibatch of transitions (pi(j), a(j), r(j), pi(j+1)) from D
 12.   set y(j) = r(j) if episode terminatese at step j+1 else r(j)+gamma
 13.   lossfunc = y(j) - Q(pi(j), a(j); theta)  
 14.   Every C steps reset Q` = Q
"""

REPLAY_MEMORY_SIZE = 5_000
TRAIN_BATCH_SIZE = 30

class DQNAgent:
    def __init__(self, action_space, state_space):
        self.replay_memory = self.initialize_memory(REPLAY_MEMORY_SIZE)
        # Gets train evry step
        self.action_space = action_space
        self.state_space = state_space
        self.model = self.create_model(action_space, state_space)
        # Target model this is what we .predict against every step
        self.target_model = self.create_model(action_space, state_space)
        self.discount = .99
        self.copy_model_to_target_model()

    def initialize_memory(self, max_capacity=10000):
        return collections.deque(maxlen=max_capacity)

    def create_model(self, action_space, state_space):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=state_space),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_space, activation='linear')
            ])
        model.compile(optimizer='sgd', loss='mse')
        return model

    def update_memory(self, transition):
        self.replay_memory.append(transition)

    def get_train_batch(self, size=30):
        return random.sample(self.replay_memory, min(size, len(self.replay_memory)))

    def predict(self, state):
        X = np.reshape(state, [1, self.state_space[0]])
        return self.model.predict(X)

    def copy_model_to_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_from_memory(self):
        x_stack = np.empty(0).reshape(0, self.state_space[0])
        y_stack = np.empty(0).reshape(0, self.action_space)
        for _ in range(10):
            for state, action, reward, next_state, done in self.get_train_batch(TRAIN_BATCH_SIZE):
                Q = self.predict(state)
                if done:
                    Q[0, action] = reward
                else:
                    Q[0, action] = reward + self.discount*np.max(self.predict(next_state))
                x_stack = np.vstack([x_stack, state])
                y_stack = np.vstack([y_stack, Q])
            self.model.fit(x_stack, y_stack, verbose=0)


class Env:
    def __init__(self, ):
        self.env_name = 'CartPole-v0'
        self.env = gym.make(self.env_name)
        self.env.reset()
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape
        self.model = DQNAgent(self.action_space, self.state_space)
        self.imgs = []
        self.total_episode_count = 0

    def decide_action(self, state):
        e = 1. / ((self.total_episode_count / 10) + 1)
        if np.random.rand(1) < e:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def run(self, n_times = 100):
        state = self.env.reset()
        # episode_image = []
        episode_count = 0
        action_count = 0
        action_counts = []
        while True:
            # episode_image.append(self.env.render(mode='rgb_array'))
            action = self.decide_action(state)
            next_state, reward, done, _ = self.env.step(action)
            action_count += 1
            reward = self.get_reward(action_count, done)
            self.model.update_memory((state, action, reward, next_state, done))
            self.model.train_from_memory()
            state = next_state
            if done:
                state = self.env.reset()
                action_counts.append(action_count)
                action_count = 0
                # self.imgs.append(episode_image)
                self.model.copy_model_to_target_model()
                # episode_image = []
                episode_count += 1
                if episode_count > n_times:
                    self.total_episode_count += episode_count
                    print("Average action count : {}".format(sum(action_counts)/episode_count))
                    break

    def get_reward(self, action_count, done):
        if done and action_count < 199:
            return -1
        elif done:
            return 1
        else:
            return 0

    def test(self, save_image_path = 'test.gif'):
        state = self.env.reset()
        episode_image = []
        action_count = 0
        while True:
            episode_image.append(self.env.render(mode='rgb_array'))
            action = np.argmax(self.model.predict(state))
            next_state, _, done, _ = self.env.step(action)
            action_count += 1
            state = next_state
            if done:
                imageio.mimsave(save_image_path, episode_image, duration=0.05)
                print("Test action count :", str(action_count))
                break

if __name__ == '__main__':
    image_folder_path = 'test_gifs'
    if not os.path.exists(image_folder_path):
        os.mkdir(image_folder_path)
    env = Env()
    for i in range(10):
        env.run(100)
        env.test(os.path.join(image_folder_path, 'test_'+str(i)+'.gif'))
