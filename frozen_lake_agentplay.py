import gym
import numpy as np
import random
import time
from IPython.display import clear_output
import pickle

env = gym.make('FrozenLake-v1')

action_space_size = env.action_space.n          # number of actions
state_space_size = env.observation_space.n       # number of states

max_num_of_steps = 20

##load the pkl file
q_table = pickle.load(open("qtable.pkl", "rb"))


for episode in range(1):
    state = env.reset()
    done = False
    time.sleep(1)
    for i in range(max_num_of_steps):
        env.render()
        time.sleep(3)
        action = np.argmax(q_table[state,:])
        state, reward, done, info = env.step(action)
        if done:
            # env.render()
            print("Finished after {} timesteps".format(i+1))
            time.sleep(3)
            break

    time.sleep(1)
