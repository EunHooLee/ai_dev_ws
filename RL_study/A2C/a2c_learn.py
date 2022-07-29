import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


class Actor(nn.Module):

    def __init__(self, state_dim ,action_dim, action_bound, learning_rate):
        super(Actor, self).__init__()
        self.action_bound = th.FloatTensor([action_bound])

        self.fc1 = nn.Sequential(nn.Linear(state_dim,64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64,32), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(32,16), nn.ReLU())
        self.mu  = nn.Sequential(nn.Linear(16, action_dim), nn.Tanh())
        self.std = nn.Sequential(nn.Linear(16, action_dim), nn.Softplus())

        self.actor_opt = th.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        mu = self.mu(x)
        std = self.std(x)

        return mu * self.action_bound, std

class Critic(nn.Module):
    
    def __init__(self, state_dim, learning_rate):
        self.super(Critic, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(state_dim,64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64,32), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(32,16), nn.ReLU())
        self.v   = nn.Linear(16,1)  # No activation

        self.critic_opt = th.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.v(x)

        return x

class A2Cagent(object):

    def __init__(self, env):

        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEAERNING_RATE = 0.001

        self.env = env

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        
        self.std_bound = [1e-2, 1.0]

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim, self.CRITIC_LEAERNING_RATE)

    def log_pdf(self, mu, std, action):
        std = std.clamp(self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = - 0.5 * (((action-mu) ** 2 / var) + (th.log(var * 2 * np.pi)))
        
        return th.sum(log_policy_pdf, dim=1, keepdim=True)

    def get_action(self, state):
        mu_a, std_a = self.actor(state)
        mu_a = mu_a.numpy()[0]
        std_a = std_a.numpy()[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=(1, self.action_dim))
        
        return action

    # Advantage 는 책에서는 따로 계산한다.
    # https://github.com/jk96491/RL_Algorithms/blob/master/Algorithms/Pytorch/A2C/a2c_agent.py
    def actor_learn(self, states, actions, advantages):
        actions = th.FloatTensor(actions).view(states.shape[0], self.action_dim)
        advantages = th.FloatTensor(advantages).view(states.shape[0], self.action_dim)

        mu, std = self.actor(states, training=True)
        log_policy_pdf = self.log_pdf(mu, std, actions)

        loss = th.sum(-log_policy_pdf * advantages)

        self.optimizer.zero_grad()

    
        

