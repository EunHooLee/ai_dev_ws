from actor import Actor
from critic import Critic
from worker import Worker

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from worker import global_episode_reward


class Agent(object):

    def __init__(self, env):
        self.WORKERS_NUM = multiprocessing.cpu_count()
        self.env = env

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.global_actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.ACTOR_LEARNING_RATE)
        self.global_critic = Critic(self.state_dim, self.action_dim, self.CRITIC_LEARNING_RATE)

        #self.global_episode_reward = []
    
    def train(self, max_episode_num):
        workers = []
        for i in range(self.WORKERS_NUM):
            worker_name = 'worker%i' % i
            workers.append(Worker(worker_name, self.env, self.global_actor, self.global_critic, max_episode_num))

        for worker in workers:
            worker.start()
        
        for worker in workers:
            worker.join()
        
        np.savetxt('./pendulum_epi_reward.txt',global_episode_reward)
        #print(self.global_episode_reward)

    def plot_result(self):
        plt.plot(global_episode_reward)
        plt.show()
    
    def save_result_graph(self):
        plt.plot(global_episode_reward)
        plt.savefig('./pendulum_epi_reward_result.png')