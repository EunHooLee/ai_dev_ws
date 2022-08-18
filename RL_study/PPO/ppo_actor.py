"""
####### Functions of Actor class ######## 
1. Calculate mean and standard deviation of the policy
2. Output a action when the state inputs 
3. Calculate log policy pdf
4. Predict mean and standard deviation of policy from cuurent state
5. Save weight
6. Load weight
"""
import torch as th
import torch.nn as nn

import numpy as np


class Actor(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        action_bound,
        learning_rate,
        ratio_clipping
    ):
        super(Actor,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.ratio_clipping = ratio_clipping

        self.std_bound = [1e-2, 1.0]

        self.fc1 = nn.Sequential(nn.Linear(self.state_dim, 64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())

        self.fc4 = nn.Sequential(nn.Linear(16, self.action_dim), nn.Tanh())
        self.fc5 = nn.Sequential(nn.Linear(16, self.action_dim), nn.Softplus())

        self.optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        mu_out = self.fc4(x)
        std_out = self.fc5(x)

        return mu_out * self.action_bound, std_out

    def log_policy(self, mu, std, action): # 현재 action 에 대한 log pdf의 likelihood 값을 계산
        std = std.clamp(self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = - 0.5 *(((action-mu ** 2) / var) + (th.log(var * 2 * np.pi)))

        return th.sum(log_policy_pdf, dim=1, keepdim=True) # 이 부분 확인 !!!!!
    
    def get_action(self, state):
        mu_a, std_a = self.forward(state)
        mu_a = mu_a.item()
        std_a = std_a.item()
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, self.action_dim)

        return mu_a, std_a, action

    def predict(self, state):
        mu_a , std_a = self.forward(state)

        return mu_a 

    def Learn(
        self,
        log_old_policy_pdf,
        states,
        actions,
        advantages,
    ):
        log_old_policy_pdf = th.FloatTensor(log_old_policy_pdf)
        actions = th.FloatTensor(actions).view(states.shape[0], 1) # view(-1,1) 해도 될 것 같다.
        advantages = th.FloatTensor(advantages).view(states.shape[0], 1).detach()

        mu, std = self.forward(states)
        log_policy_pdf = self.log_policy(mu, std, actions)

        ratio = th.exp(log_policy_pdf - log_old_policy_pdf)
        clipped_ratio = ratio.clamp(1.0 - self.ratio_clipping, 1.0 + self.ratio_clipping)

        surrogate = - th.min(ratio * advantages , clipped_ratio * advantages)
        loss = surrogate.mean()     # 이 부분 왜 mean 을 구하지? sum()을 구해야 하는게 아닌가?

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save_weight(self, path):
        th.save(self.state_dict(), path)
    
    def load_weight(self, path):
        self.load_state_dict(th.load(path))
    
