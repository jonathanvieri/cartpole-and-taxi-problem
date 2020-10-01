# -*- coding: utf-8 -*-
"""
Created and Modified by: Jonathan Vieri (219559949)

Code Reference

Title: Reinforcement Q-Learning from Scratch in Python with OpenAI Gym
Author: Satwik Kansal and Brendan Martin
Date: 2018
Version: 1.0
Type: Source Code
Availability: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

"""

import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Initializes the taxi problem environment
env = gym.make("Taxi-v3").env

# Making and initializing the Q table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

# This function will plot how the agent learns into 2 subplots
# The first one plots the total reward the gain over time
# The second one plots a histogram which shows the agent's total reward
def plot_res(values, title=''):   
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    clear_output(wait=True)
    
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(500, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')
    
    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(500, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()


# Trains the agent using Q-learning algorithm for 100000 episodes
for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")
print("Training finished.\n")



# Evaluating the agent's performance after training
# It will plot out the result in real time as well
total_epochs, total_penalties = 0, 0
episodes = 250     # Will be run for 250 episodes
total = 0
final = []


for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1
        total += reward

    total_penalties += penalties
    total_epochs += epochs
    final.append(total)
    plot_res(final, 'Q-Learning Strategy')

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")


