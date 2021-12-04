import gym
import numpy as np
import random
import time
from IPython.display import clear_output
import pickle

env = gym.make('FrozenLake-v1')

action_space_size = env.action_space.n          # number of actions
state_space_size = env.observation_space.n       # number of states

q_table = np.zeros((state_space_size, action_space_size))  # initialize Q table

# hyperparameters
num_episodes = 10001
max_steps_per_episode = 100
learning_rate = 0.70
discount_rate = 0.95

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    # print("episode: ", episode)
    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:         # exploit
            action = np.argmax(q_table[state,:])
        else:                                                    # explore
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        # print("current state ",state)
        # print("new state ",new_state)
        # print(reward)
        #print(info)

        q_table[state,action] = q_table[state,action] * (1 - learning_rate) + \
                                learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))

        state = new_state
        rewards_current_episode += reward

        if done:
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
                          (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    
    rewards_all_episodes.append(rewards_current_episode)

    # print("rewards_current_episode: ", rewards_current_episode)

    if episode % 1000 == 0:
        print("episode: ", episode)
        running_reward = np.mean(rewards_all_episodes[episode-1000:episode])
        print(running_reward)

print(q_table)

with open('qtable.pkl', 'wb') as f:
    pickle.dump(q_table, f)


      