#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import discreteaction_pendulum
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.layers import Dense
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os 


# In[ ]:


env= discreteaction_pendulum.Pendulum()
state_size = env.num_states
action_size = env.num_actions
batch_size = 7
n_episodes = 100
output_dir = 'hw2/dqn'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[ ]:


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) 
        self.gamma = 0.95 
        self.epsilon = 1.0 
        self.epsilon_decay = 0.995 
        self.epsilon_min = 0.01 
        self.learning_rate = 0.001
        self.model = self._build_model() 
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(64, activation='tanh')) 
        model.add(Dense(32, activation='linear')) 
        model.compile(loss= tf.keras.losses.BinaryCrossentropy(),
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        return self.memory


# In[ ]:


agent = DQNAgent(state_size, action_size) 


# In[ ]:


def actions(state):
    state = np.reshape(state, [1, state_size])
    if np.random.rand() <= agent.epsilon: 
        return random.randrange(agent.action_size)
    act_values = agent.model.predict(state) 
    return np.argmax(act_values[0]) 

for e in range(n_episodes): 
    state = env.reset()
    done = False
    while not done:  
        action = actions(state) 
        next_state, reward, done= env.step(action)        
        reward = reward if not done 
        agent.remember(state, action, reward, next_state, done)       
        state = next_state       
        if done: 
            print("episode: {}/{},e: {:.2}" 
                  .format(e, n_episodes, agent.epsilon))
            break 
    if len(agent.memory) > batch_size:   
        minibatch = random.sample(agent.remember(state, action, reward, next_state, done), batch_size) 
        for state, action, reward, next_state, done in minibatch: 
            target = reward
            state = np.reshape(state, [1, state_size])
            next_state = np.reshape(next_state, [1, state_size])
            if not done: 
                target = (reward + agent.gamma * np.amax(agent.model.predict(next_state))) 
            target_f = agent.model.predict(state) 
            target_f[0][action] = target
            agent.model.fit(state, target_f, epochs=1, verbose=0) 
        if agent.epsilon > agent.epsilon_min:

            agent.epsilon *= agent.epsilon_decay
        if e % 50 == 0:
            agent.model.save_weights(output_dir)

