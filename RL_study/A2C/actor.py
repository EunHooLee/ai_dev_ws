import torch as th
import torch.nn as nn

import numpy as np


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_bound, learning_rate):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = th.FloatTensor([action_bound])
        self.learning_rate = learning_rate
        # std bound 는 왜 제한?
        self.std_bound = [1e-2, 1.0]

        self.fc1 = nn.Sequential(nn.Linear(self.state_dim, 64),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64,32),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(32,16),nn.ReLU())

        self.fc4 = nn.Sequential(nn.Linear(16,self.action_dim),nn.Tanh())
        self.fc5 = nn.Sequential(nn.Linear(16,self.action_dim),nn.Softplus())

        self.optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def forward(self,state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        out_mu = self.fc4(x)
        out_std = self.fc5(x)
        # 평균에 action_bound 를 곱하는 이유는 뭐지??
        return out_mu * self.action_bound, out_std
    # log_pdf : object function gradient 식에서 expecttation 내부 log(pi(a|s)) 부분을 sample을 state,action sample을 이용해 estimation 하는 과정
    def log_pdf(self, mu, std, action):
        std = std.clamp(self.std_bound[0],self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5 * (((action-mu)**2)/var + (th.log(var*2*np.pi)))

        return th.sum(log_policy_pdf,dim=1, keepdim=True)

    def get_action(self, state):
        mu_a , std_a = self.forward(state)
        mu_a = mu_a.detach()
        std_a = std_a.detach()
        std_a = np.clip(std_a, self.std_bound[0],self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=(1, self.action_dim))
        
        return action
    
    def predict(self,state):
        mu_a , std_a = self.forward(state)
        return mu_a
    # Learn 함수에 대해서 왜 loss 를 구하는지 이해가 안된다.
    # gradient of OF 을 구했으면 바로 theta를 update해야되는게 아닌가? 그게 RL의 목적아닌가?
    # 왜 policy network를 update하지? 
    # policy network를 update 하면, 왜 object F가 maximize 되는거지? 
    # 또한 왜 loss 식이 왜 gradient of OF 와 같지? 
    def Learn(self, states, actions, advantages):
        actions = th.FloatTensor(actions).view(states.shape[0], self.action_dim)
        advantages = th.FloatTensor(advantages).view(states.shape[0], self.action_dim)

        mu, std = self.forward(states)
        log_policy_pdf = self.log_pdf(mu,std,actions)
        # 이 부분은 sample mean을 구해서 gradient of object function 을 계산하는 과정이다.
        loss = th.sum(-log_policy_pdf * advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # loss를 왜 return하지? 출력하려고 그런가?
        return loss
    
    def save_weights(self, path):
        th.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(th.load(path))