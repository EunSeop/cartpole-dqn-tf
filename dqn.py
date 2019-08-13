import collections
import os
import random
from tqdm import tqdm

import gym
import imageio
import numpy as np
import tensorflow as tf

"""
DQN STEPS:
 1. Initialize replay memory D to capacity N
 2. Initialize action-value function Q with random weights theta
 3. Initialize target action-value function Q` with weight theta- = theta
 4. For episode=1, M  do
 5. Initialize squence s1={x1} and preprocess sequence pi1=pi1(s1)
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
        self.action_space = action_space
        self.state_space = state_space
        self.model = self.create_model(action_space, state_space)
        self.target_model = self.create_model(action_space, state_space)
        self.discount = .99
        self.copy_model_to_target_model()

    def initialize_memory(self, max_capacity):
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

    def predict(self, state, target_model=False):
        X = np.reshape(state, [1, self.state_space[0]])
        if target_model:
            return self.target_model.predict(X)
        else:
            return self.model.predict(X)

    def copy_model_to_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_from_memory(self):
        STATE, ACTION, REWARD, NEXT_STATE, DONE = 0, 1, 2, 3, 4
        transitions = np.array(self.get_train_batch(TRAIN_BATCH_SIZE))
        states = np.array(transitions[:, STATE].tolist())
        Q = self.model.predict(states)
        unique_actions = np.unique(transitions[:, ACTION])
        for action in unique_actions:
            mask = transitions[:, ACTION] == action
            Q[mask, action] = transitions[mask, REWARD]
            ndone_mask = (~transitions[:, DONE] & mask).astype(bool)
            if len(ndone_mask) > 0:
                Q[ndone_mask, action] += self.discount*np.max(self.target_model.predict(np.array(transitions[ndone_mask, NEXT_STATE].tolist())), axis=1)
        self.model.fit(states, Q, verbose=0)

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
        e = 1. / ((self.total_episode_count / 10_000) + 1)
        if np.random.rand(1) < e:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def run(self, n_times = 100):
        total_action_counts = []
        for _ in tqdm(range(n_times)):
            done = False 
            episode_action_count = 0
            state = self.env.reset()
            while not done:
                action = self.decide_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_action_count += 1
                reward = self.get_reward(episode_action_count, done)
                self.model.update_memory((state.tolist(), action, reward, next_state.tolist(), done))
                self.model.train_from_memory()
                state = next_state
            total_action_counts.append(episode_action_count)
            self.model.copy_model_to_target_model()
        print("Average action count : {}".format(sum(total_action_counts)/n_times))
        self.total_episode_count += sum(total_action_counts)

    def get_reward(self, action_count, done):
        if done and action_count < 199:
            return -1
        elif done:
            return 1
        else:
            return 0

    def test(self, save_image_path = 'test.gif', n=100):
        # Test 100 episodes
        total_episode_images = []
        episode_image = []
        total_action_counts = []
        # Save 5 trial
        for _ in range(min(5, n)):
            single_action_count = 0
            state = self.env.reset()
            while True:
                episode_image.append(self.env.render(mode='rgb_array'))
                action = np.argmax(self.model.predict(state))
                next_state, _, done, _ = self.env.step(action)
                single_action_count += 1
                state = next_state
                if done:
                    total_episode_images.extend(episode_image)
                    total_action_counts.append(single_action_count)
                    episode_image = []
                    break
        self.env.close()
        imageio.mimsave(save_image_path, total_episode_images, duration=0.05)

        for _ in range(n-5):
            single_action_count = 0
            state = self.env.reset()
            while True:
                action = np.argmax(self.model.predict(state))
                next_state, _, done, _ = self.env.step(action)
                single_action_count += 1
                state = next_state
                if done:
                    total_action_counts.append(single_action_count)
                    break
        print("Test average action count {} times :".format(n), str(sum(total_action_counts)/n))


if __name__ == '__main__':
    image_folder_path = 'test_gifs'
    if not os.path.exists(image_folder_path):
        os.mkdir(image_folder_path)
    env = Env()
    for i in range(10):
        test_image_name = 'test_{}.gif'.format(i)
        env.run(100)
        env.test(os.path.join(image_folder_path, test_image_name), n=10)