"""
######## Function of Critic Class ########
1. Calculate value function from state
2. Predict value function from state.
3. Save weight
4. Load weight
"""
import torch as th
import torch.nn as nn

import numpy as np


class Critic(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate,
    ):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.fc1 = nn.Sequential(nn.Linear(self.state_dim, 64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())

        self.fc4 = nn.Sequential(nn.Linear(16, self.action_dim), nn.ReLU()) # output이 action_dim 만큼인지 1인지 확인하가

        self.optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        
        v = self.fc4(x)

        return v

    def predict(self, state):
        v = self.forward(state)

        return v

    def Learn(self, states, td_target):
        td_target = th.FloatTensor(td_target).detach()
        predict = self.forward(states)

        loss = th.mean((predict - td_target) ** 2)      # nn.MSELoss() 이 함수 쓰면 안되나?

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def save_weight(self, path):
        th.save(self.state_dict(), path)

    def load_weight(self, path):
        self.load_state_dict(th.load(path))



