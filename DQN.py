#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import discreteaction_pendulum


# In[3]:


import numpy as np


# In[4]:


import tensorflow as tf


# In[5]:


from tensorflow import keras
from keras.layers import Dense
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os 


# In[6]:


env= discreteaction_pendulum.Pendulum()
state_size = env.num_states
action_size = env.num_actions
batch_size = 32
n_episodes = 200


# In[7]:


class DQNAgent:
    def __init__(self, state_size, action_size,memory_size,):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size) 
        self.gamma = 0.99
        self.epsilon = 1.0 
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1 
        self.learning_rate = 0.001
        self.model_main = self._build_model_main() 
        self.model_target = self._build_model_target() 
    
    def _build_model_main(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(64, activation='tanh')) 
        model.add(Dense(action_size, activation='linear')) 
        model.compile(loss= 'mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def _build_model_target(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(64, activation='tanh')) 
        model.add(Dense(action_size, activation='linear')) 
        model.compile(loss= 'mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        return self.memory
    
    def replaymemory(self,batchsize,a):
        minibatch = random.sample(self.memory, batch_size) 
        
        for (state, action, reward, next_state, done) in minibatch: 
            target = reward
            
            if not done: 
                target = (reward + self.gamma * np.amax(self.model_target.predict(next_state,verbose=None)[0])) 
            
            target_f = self.model_main.predict(state,verbose=None) 
            
            target_f[0][action] = target
            
            self.model_main.fit(state, target_f, epochs=1, verbose=None)
            if  a % 30 == 0:
                self.model_target.set_weights(self.model_main.get_weights())

def trajectory(policy):
    env = discreteaction_pendulum.Pendulum()
    s = env.reset()
    s = np.reshape(s, [1, state_size])
    # Simulate an episode and save the result as an animated gif
    env.video(policy, filename='figures/test_discreteaction_pendulum.gif')

    
    # Create dict to store data from simulation
    data = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        
        a = policy(s)
        (s, r, done) = env.step(a)
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)

    # Parse data from simulation
    data['s'] = np.array(data['s'])
    theta = data['s'][:, 0]
    thetadot = data['s'][:, 1]
    tau = [env._a_to_u(a) for a in data['a']]

    # Plot data and save to png file
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(data['t'], theta, label='theta')
    ax[0].plot(data['t'], thetadot, label='thetadot')
    ax[0].legend()
    ax[1].plot(data['t'][:-1], tau, label='tau')
    ax[1].legend()
    ax[2].plot(data['t'][:-1], data['r'], label='r')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    plt.tight_layout()
    plt.savefig('figures/test_discreteaction_pendulum.png')          


# Standard Algorithm

# In[7]:


agent_11 = DQNAgent(state_size, action_size,10000) 


# In[8]:


def actions(state):
    
    if np.random.rand() <= agent_11.epsilon: 
        return random.randrange(0,action_size-1)
    else:
        act_values = agent_11.model_main.predict(state,verbose=None) 
        return np.argmax(act_values[0]) 
Q_11_Learning = np.zeros(n_episodes)
for e in range(n_episodes):     
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    t = 0
    a = 0
    while not done:  
        action = actions(state) 
        (next_state, reward, done)= env.step(action)        
        # reward = reward if not done else -10 
        next_state = np.reshape(next_state, [1, state_size])
        agent_11.remember(state, action, reward, next_state, done)
        previous_state = state       
        state = next_state       
        a = a+ 0.99**t * reward
        t = t + 1
        if done:
            break 
    if len(agent_11.memory) > batch_size:   
        Q_11 = agent_11.replaymemory(batch_size,e)
        
    Q_11_Learning[e] = a   
            
    if agent_11.epsilon > agent_11.epsilon_min:
        agent_11.epsilon *= agent_11.epsilon_decay

        
            
    
        


# With Replay , without target Q

# In[23]:


agent_10 = DQNAgent(state_size, action_size,10000) 


# In[25]:


def actions(state):
    
    if np.random.rand() <= agent_10.epsilon: 
        return random.randrange(0,action_size-1)
    else:
        act_values = agent_10.model_main.predict(state,verbose=None) 
        return np.argmax(act_values[0]) 
Q_10_Learning = np.zeros(n_episodes)  
for e in range(n_episodes):     
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    a= 0
    t = 0
    while not done:  
        action = actions(state) 
        (next_state, reward, done)= env.step(action)        
        # reward = reward if not done else -10 
        next_state = np.reshape(next_state, [1, state_size])
        agent_10.remember(state, action, reward, next_state, done)
        previous_state = state       
        state = next_state       
        a = a+ 0.99**t * reward
        t = t+1
        if done: 
            #print("episode: {}/{},e: {:.2}" 
                # .format(e, n_episodes, agent.epsilon))
            break 
    if len(agent_10.memory) > batch_size:   
        Q_10 = agent_10.replaymemory(batch_size,0)
            
    Q_10_Learning[e] = a         
    if agent_10.epsilon > agent_10.epsilon_min:
        agent_10.epsilon *= agent_10.epsilon_decay

    
        


# Without Replay, with Target Q

# In[11]:


agent_01 = DQNAgent(state_size, action_size,batch_size)


# In[12]:


Q_01_Learning = np.zeros(n_episodes) 
for e in range(n_episodes):     
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    a = 0
    t = 0
    while not done:  
        action = actions(state) 
        (next_state, reward, done)= env.step(action)        
        # reward = reward if not done else -10 
        next_state = np.reshape(next_state, [1, state_size])
        agent_01.remember(state, action, reward, next_state, done)
        previous_state = state       
        state = next_state       
        a = a+ 0.99**t * reward
        t = t+1
        if done: 
            #print("episode: {}/{},e: {:.2}" 
             #    .format(e, n_episodes, agent.epsilon))
            break 
    if len(agent_01.memory) > batch_size:   
        Q_01 = agent_01.replaymemory(batch_size,e)
            
    Q_01_Learning[e] = a             
    if agent_01.epsilon > agent_01.epsilon_min:
        agent_01.epsilon *= agent_01.epsilon_decay

    


# Without both

# In[15]:


agent_00 = DQNAgent(state_size, action_size,batch_size)
def actions(state):
    
    if np.random.rand() <= agent_00.epsilon: 
        return random.randrange(0,action_size-1)
    else:
        act_values = agent_00.model_main.predict(state,verbose=None) 
        return np.argmax(act_values[0]) 


# In[16]:


Q_00_Learning = np.zeros(n_episodes) 
for e in range(n_episodes):     
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    a = 0
    t = 0
    while not done:  
        action = actions(state) 
        (next_state, reward, done)= env.step(action)        
        # reward = reward if not done else -10 
        next_state = np.reshape(next_state, [1, state_size])
        agent_00.remember(state, action, reward, next_state, done)
        previous_state = state       
        state = next_state       
        a = a+ 0.99**t * reward
        t = t+1
        if done: 
            #print("episode: {}/{},e: {:.2}" 
                # .format(e, n_episodes, agent.epsilon))
            break 
    if len(agent_00.memory) > batch_size:   
        Q_00 = agent_00.replaymemory(batch_size,0)
            
    Q_00_Learning[e] = a         
    if agent_00.epsilon > agent_00.epsilon_min:
        agent_00.epsilon *= agent_00.epsilon_decay


# Plots_11

# In[15]:


import matplotlib.pyplot as plt
plt.plot(Q_11_Learning)
plt.ylabel('Return')
plt.xlabel('Episodes')
plt.ylim([0,20])


# In[ ]:





# In[16]:


s = env.reset()
data = {
    't': [0],
    's': [s],
    'a': [],
    'r': [],
}
policy = lambda s: a
# Simulate until episode is done
done = False
while not done:
    s = np.reshape(s, [1, state_size])
    a = np.argmax(agent_11.model_main.predict(s,verbose=None)[0])
    q = np.amax(agent_11.model_main.predict(s,verbose=None)[0])
    policy(s) == a
    (s, r, done) = env.step(a)
    data['t'].append(data['t'][-1] + 1)
    data['s'].append(s)
    data['a'].append(a)
    data['r'].append(r)

# Parse data from simulation
data['s'] = np.array(data['s'])
theta = data['s'][:, 0]
thetadot = data['s'][:, 1]
tau = [env._a_to_u(a) for a in data['a']]

# Plot data and save to png file
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(data['t'], theta, label='theta')
ax[0].plot(data['t'], thetadot, label='thetadot')
ax[0].legend()
ax[1].plot(data['t'][:-1], tau, label='tau')
ax[1].legend()
ax[2].plot(data['t'][:-1], data['r'], label='r')
ax[2].legend()
ax[2].set_xlabel('time step')
plt.tight_layout()
plt.savefig('figures/test_discreteaction_pendulum.png') 
env.video(policy, filename='figures/test_discreteaction_pendulum.gif')


# In[20]:


N = 100
X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, N), np.linspace(-15, 15, N))
Value_f = lambda s: np.amax(agent_11.model_main.predict(s,verbose=None)[0])
Q_mx = np.zeros((100,100))
s = [[0,0]]
# A low hump with a spike coming out.
for i in range(100):
    for j in range(100):
        s[0][0] = X[0][i]
        s[0][1] = Y[j][0]
        s = np.reshape(s, [1, state_size])
        Q_mx[i][j] = Value_f(s)



# c = p
    


# In[21]:


#My attempt
fig,ax = plt.subplots()
contourf_ = ax.contourf(X, Y, Q_mx)
cbar = fig.colorbar(contourf_)
# cbar.set_clim( vmin, vmax )


# In[26]:


N = 100
X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, N), np.linspace(-15, 15, N))
policy = lambda s: np.argmax(agent_11.model_main.predict(s,verbose=None)[0])
A = np.zeros((100,100))
s = [[0,0]]
# A low hump with a spike coming out.
for i in range(100):
    for j in range(100):
        s[0][0] = X[0][i]
        s[0][1] = Y[j][0]
        s = np.reshape(s, [1, state_size])
        A[i][j] = policy(s)


# In[27]:


fig,ax = plt.subplots()
contourf_ = ax.contourf(X, Y,A)
cbar = fig.colorbar(contourf_)


# Plots_10

# In[26]:


plt.plot(Q_10_Learning)
plt.ylabel('Return')
plt.xlabel('Episodes')
plt.ylim([0,20])


# In[27]:


s = env.reset()
data = {
    't': [0],
    's': [s],
    'a': [],
    'r': [],
}
policy = lambda s: a
# Simulate until episode is done
done = False
while not done:
    s = np.reshape(s, [1, state_size])
    a = np.argmax(agent_10.model_main.predict(s,verbose=None)[0])
    q = np.amax(agent_10.model_main.predict(s,verbose=None)[0])
    policy(s) == a
    (s, r, done) = env.step(a)
    data['t'].append(data['t'][-1] + 1)
    data['s'].append(s)
    data['a'].append(a)
    data['r'].append(r)

# Parse data from simulation
data['s'] = np.array(data['s'])
theta = data['s'][:, 0]
thetadot = data['s'][:, 1]
tau = [env._a_to_u(a) for a in data['a']]

# Plot data and save to png file
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(data['t'], theta, label='theta')
ax[0].plot(data['t'], thetadot, label='thetadot')
ax[0].legend()
ax[1].plot(data['t'][:-1], tau, label='tau')
ax[1].legend()
ax[2].plot(data['t'][:-1], data['r'], label='r')
ax[2].legend()
ax[2].set_xlabel('time step')
plt.tight_layout()
plt.savefig('figures/test_discreteaction_pendulum.png') 
env.video(policy, filename='figures/test_discreteaction_pendulum.gif')


# In[28]:


N = 100
X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, N), np.linspace(-15, 15, N))
Value_f = lambda s: np.amax(agent_10.model_main.predict(s,verbose=None)[0])
Q_mx = np.zeros((100,100))
s = [[0,0]]
# A low hump with a spike coming out.
for i in range(100):
    for j in range(100):
        s[0][0] = X[0][i]
        s[0][1] = Y[j][0]
        s = np.reshape(s, [1, state_size])
        Q_mx[i][j] = Value_f(s)


# In[29]:


fig,ax = plt.subplots()
contourf_ = ax.contourf(X, Y, Q_mx)
cbar = fig.colorbar(contourf_)
# cbar.set_clim( vmin, vmax )


# In[30]:


N = 100
X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, N), np.linspace(-15, 15, N))
policy = lambda s: np.argmax(agent_10.model_main.predict(s,verbose=None)[0])
A = np.zeros((100,100))
s = [[0,0]]
# A low hump with a spike coming out.
for i in range(100):
    for j in range(100):
        s[0][0] = X[0][i]
        s[0][1] = Y[j][0]
        s = np.reshape(s, [1, state_size])
        A[i][j] = policy(s)


# In[31]:


fig,ax = plt.subplots()
contourf_ = ax.contourf(X, Y,A)
cbar = fig.colorbar(contourf_)


# Plots_01

# In[18]:


plt.plot(Q_01_Learning)
plt.ylabel('Return')
plt.xlabel('Episodes')
plt.ylim([0,20])


# In[30]:


s = env.reset()
data = {
    't': [0],
    's': [s],
    'a': [],
    'r': [],
}
policy = lambda s: a
# Simulate until episode is done
done = False
while not done:
    s = np.reshape(s, [1, state_size])
    a = np.argmax(agent_01.model_main.predict(s,verbose=None)[0])
    q = np.amax(agent_01.model_main.predict(s,verbose=None)[0])
    policy(s) == a
    (s, r, done) = env.step(a)
    data['t'].append(data['t'][-1] + 1)
    data['s'].append(s)
    data['a'].append(a)
    data['r'].append(r)

# Parse data from simulation
data['s'] = np.array(data['s'])
theta = data['s'][:, 0]
thetadot = data['s'][:, 1]
tau = [env._a_to_u(a) for a in data['a']]

# Plot data and save to png file
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(data['t'], theta, label='theta')
ax[0].plot(data['t'], thetadot, label='thetadot')
ax[0].legend()
ax[1].plot(data['t'][:-1], tau, label='tau')
ax[1].legend()
ax[2].plot(data['t'][:-1], data['r'], label='r')
ax[2].legend()
ax[2].set_xlabel('time step')
plt.tight_layout()
plt.savefig('figures/test_discreteaction_pendulum.png') 
env.video(policy, filename='figures/test_discreteaction_pendulum.gif')


# In[31]:


N = 100
X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, N), np.linspace(-15, 15, N))
Value_f = lambda s: np.amax(agent_01.model_main.predict(s,verbose=None)[0])
Q_mx = np.zeros((100,100))
s = [[0,0]]
# A low hump with a spike coming out.
for i in range(100):
    for j in range(100):
        s[0][0] = X[0][i]
        s[0][1] = Y[j][0]
        s = np.reshape(s, [1, state_size])
        Q_mx[i][j] = Value_f(s)


# In[32]:


fig,ax = plt.subplots()
contourf_ = ax.contourf(X, Y, Q_mx)
cbar = fig.colorbar(contourf_)


# In[34]:


N = 100
X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, N), np.linspace(-15, 15, N))
policy = lambda s: np.argmax(agent_01.model_main.predict(s,verbose=None)[0])
A = np.zeros((100,100))
s = [[0,0]]
# A low hump with a spike coming out.
for i in range(100):
    for j in range(100):
        s[0][0] = X[0][i]
        s[0][1] = Y[j][0]
        s = np.reshape(s, [1, state_size])
        A[i][j] = policy(s)


# In[35]:


fig,ax = plt.subplots()
contourf_ = ax.contourf(X, Y,A)
cbar = fig.colorbar(contourf_)


# Plots_00

# In[17]:


plt.plot(Q_00_Learning)
plt.ylabel('Return')
plt.xlabel('Episodes')
plt.ylim([0,20])


# In[18]:


s = env.reset()
data = {
    't': [0],
    's': [s],
    'a': [],
    'r': [],
}
policy = lambda s: a
# Simulate until episode is done
done = False
while not done:
    s = np.reshape(s, [1, state_size])
    a = np.argmax(agent_00.model_main.predict(s,verbose=None)[0])
    q = np.amax(agent_00.model_main.predict(s,verbose=None)[0])
    policy(s) == a
    (s, r, done) = env.step(a)
    data['t'].append(data['t'][-1] + 1)
    data['s'].append(s)
    data['a'].append(a)
    data['r'].append(r)

# Parse data from simulation
data['s'] = np.array(data['s'])
theta = data['s'][:, 0]
thetadot = data['s'][:, 1]
tau = [env._a_to_u(a) for a in data['a']]

# Plot data and save to png file
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(data['t'], theta, label='theta')
ax[0].plot(data['t'], thetadot, label='thetadot')
ax[0].legend()
ax[1].plot(data['t'][:-1], tau, label='tau')
ax[1].legend()
ax[2].plot(data['t'][:-1], data['r'], label='r')
ax[2].legend()
ax[2].set_xlabel('time step')
plt.tight_layout()
plt.savefig('figures/test_discreteaction_pendulum.png') 
env.video(policy, filename='figures/test_discreteaction_pendulum.gif')


# In[19]:


N = 100
X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, N), np.linspace(-15, 15, N))
Value_f = lambda s: np.amax(agent_00.model_main.predict(s,verbose=None)[0])
Q_mx = np.zeros((100,100))
s = [[0,0]]
# A low hump with a spike coming out.
for i in range(100):
    for j in range(100):
        s[0][0] = X[0][i]
        s[0][1] = Y[j][0]
        s = np.reshape(s, [1, state_size])
        Q_mx[i][j] = Value_f(s)


# In[20]:


fig,ax = plt.subplots()
contourf_ = ax.contourf(X, Y, Q_mx)
cbar = fig.colorbar(contourf_)


# In[21]:


N = 100
X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, N), np.linspace(-15, 15, N))
policy = lambda s: np.argmax(agent_00.model_main.predict(s,verbose=None)[0])
A = np.zeros((100,100))
s = [[0,0]]
# A low hump with a spike coming out.
for i in range(100):
    for j in range(100):
        s[0][0] = X[0][i]
        s[0][1] = Y[j][0]
        s = np.reshape(s, [1, state_size])
        A[i][j] = policy(s)


# In[22]:


fig,ax = plt.subplots()
contourf_ = ax.contourf(X, Y,A)
cbar = fig.colorbar(contourf_)


# In[ ]:




