import torch as th
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
#https://github.com/jk96491/RL_Algorithms/tree/master/Algorithms/Pytorch/A2C


class Agent(object):

    def __init__(self, env):
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        # high 함수뭐냐 ? list, ndarray, Tensor에는 적용 안됨₩