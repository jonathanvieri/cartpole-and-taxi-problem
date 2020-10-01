# -*- coding: utf-8 -*-
"""
Created and Modified by: Jonathan Vieri (219559949)

Code Reference

Title: Practical-Data-Science
Author: Kurban, R
Date: 2019
Version: 1.0
Type: Source Code
Availability: https://github.com/ritakurban/Practical-Data-Science

"""

# Importing LIbraries
import gym
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

# Environment we are going to use which is Cart Pole 
env = gym.envs.make("CartPole-v1")


# Extract one step of the simulation
def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
    return torch.from_numpy(screen)

# Speify the number of simulation steps
num_steps = 2


# Show several steps
# This function is used to show the steps specifically per frame 
# of the cart pole environment
'''
for i in range(num_steps):
    clear_output(wait=True)
    env.reset()
    plt.figure()
    plt.imshow(get_screen().cpu().permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('CartPole-v0 Environment')
    plt.xticks([])
    plt.yticks([])
    plt.show()
'''

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
    ax[0].axhline(195, c='red',ls='--', label='goal')
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
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()
    

# The random search function
# Its where the action is randomly done
def random_search(env, episodes, 
                  title='Random Strategy'):
    """ Random search strategy implementation."""
    final = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total = 0
        while not done:
            # Sample random actions
            action = env.action_space.sample()
            # Take action and extract results
            next_state, reward, done, _ = env.step(action)
            # Update reward
            total += reward
            if done:
                break
        # Add to the final reward
        final.append(total)
        plot_res(final,title)
    return final


# Get random search results
# The episodes can be changed to anything, and here I set it as 250
    
random_s = random_search(env, 250)

''' 
    Uncomment the function to run the random search result 
    WARNING it is not recommended to run both at the same time
    because it will cause lag on to your computer
'''





