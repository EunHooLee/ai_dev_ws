import torch as th
import torch.nn as nn   
from optimizer import SharedAdam

class Critic(nn.Module):

    def __init__(self,state_dim, action_dim, learning_rate):
        super(Critic,self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.fc1 = nn.Sequential(nn.Linear(self.state_dim,64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64,32), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(32,16), nn.ReLU())
        
        self.fc4 = nn.Sequential(nn.Linear(16,self.action_dim),nn.ReLU())

        #self.optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer = SharedAdam(self.parameters(), lr=self.learning_rate)

    def forward(self,state):
        x = self.fc1(state)
        x = self.fc2(x.clone())
        x = self.fc3(x.clone())
        v = self.fc4(x.clone())

        return v

    def predict(self, state):
        v = self.forward(state)

        return v
    
    def Learn(self, states, n_step_td_targets):
        n_step_td_targets = th.FloatTensor(n_step_td_targets)
        predict = self.forward(states)

        loss = th.mean((predict - n_step_td_targets)**2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save_weights(self, path):
        th.save(self.state_dict(), path)
    
    def load_weights(self, path):
        self.load_state_dict(th.load(path))
        
