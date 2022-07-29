import torch as th
import torch.nn as nn

import numpy as np


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, learning_rate):
        super(Critic,self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.fc1 = nn.Sequential(nn.Linear(self.state_dim,128),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128,128),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128,self.action_dim),nn.ReLU())

        self.fc4 = nn.Sequential(nn.Linear(128,self.action_dim),nn.ReLU())

        self.optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def forward(self,state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        # return value function
        return x

    def predict(self, state):
        v = self.forward(state)
        return v
    
    def Learn(self,states, td_target):
        td_target = th.FloatTensor(td_target)
        predict = self.forward(states)
        # MSE
        loss = th.mean((predict - td_target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def save_weights(self, path):
        th.save(self.state_dict(), path)
    
    def load_weights(self, path):
        self.load_state_dict(th.load(path))