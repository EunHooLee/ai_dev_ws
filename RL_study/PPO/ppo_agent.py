"""
######### The Functions of Agent #########
1. 
"""
import torch as th

import numpy as np
import matplotlib.pyplot as plt

from ppo_actor import Actor
from ppo_critic import Critic
from Utils import convertToTensorInput 


class PPOAgent(object):

    def __init__(self, env):
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.RATIO_CLIPPING = 0.05
        self.EPOCHS = 5
        self.GAE_LAMBDA = 0.9
        self.load_model = False

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]/2
        
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.ACTOR_LEARNING_RATE, self.RATIO_CLIPPING)
        self.critic = Critic(self.state_dim, self.action_dim, self.CRITIC_LEARNING_RATE)

        self.save_epi_reward = []

    def gae_target(self, rewards, v_values, next_v_values, done):
        rewards = th.FloatTensor(rewards)
        n_step_targets = th.zeros_like(rewards)
        gae = th.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_values
        
        # Cumulative sum 계산을 위해 가장 마지막 부터 계산하였다.
        # 즉 n 번 반복된다면 제일 마지막 값은 n 번의 gamma 와 lambda 가 앞에 곱해질 것이다. 
        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.GAMMA * forward_val - v_values[k]
            gae_cumulative = self.GAMMA * self.GAE_LAMBDA * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]

        return gae, n_step_targets

    # 배열(8,1),(1,8) 을 벡터(8,) 로 바꿔주는 함수 / axis=0은 행 방향
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)
        
        return unpack

    def train(self, max_episode_num):
        batch_state, batch_action, batch_reward = [], [], []
        batch_log_old_policy_pdf = []

        if self.load_model is True:
            self.actor.load_weight("pendulum_actor.th")
            self.critic.load_weight("pendulum_critic.th")
            print("Actor and Critic Model is loaded Successfully")

        for ep in range(int(max_episode_num)):
            time, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                # self.env.render()
                mu_old, std_old, action = self.actor.get_action(convertToTensorInput(state, self.state_dim))
                action = np.clip(action, -self.action_bound, self.action_bound)

                var_old = std_old ** 2
                log_old_policy_pdf = -0.5 * (action - mu_old) ** 2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
                log_old_policy_pdf = np.sum(log_old_policy_pdf)

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1,1])
                log_old_policy_pdf = np.reshape(log_old_policy_pdf, [1,1])

                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(reward)
                batch_log_old_policy_pdf.append(log_old_policy_pdf)

                # Batch 에 원하는 만큼 데이터가 모이지 않았다면 다시 위로 가서 위쪽 코드 반복하기
                if len(batch_state) < self.BATCH_SIZE:
                    state = next_state
                    episode_reward += reward[0]
                    time += 1
                    continue
                
                # batch_state (1,x) 배열을 (x,) 벡터로 변환
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                rewards = self.unpack_batch(batch_reward)
                log_old_policy_pdfs = self.unpack_batch(log_old_policy_pdf)

                batch_state, batch_action, batch_reward = [], [], []
                batch_log_old_policy_pdf = []

                next_state = np.reshape(next_state, [1, self.state_dim])
                next_v_value = self.critic.predict(convertToTensorInput(next_state, self.state_dim))
                v_values = self.critic.predict(convertToTensorInput(states, self.state_dim, states.shape[0]))
                gaes, y_i = self.gae_target(rewards, v_values, next_v_value, done)

                # 같은 데이터로 여러번 반복한다? 이게 무슨 의미지? 
                for _ in range(self.EPOCHS):
                    self.critic.Learn(convertToTensorInput(states, self.state_dim, states.shape[0]), y_i)
                    self.actor.Learn(log_old_policy_pdfs,convertToTensorInput(states, self.state_dim, states.shape[0]), actions, gaes)

                state = next_state
                episode_reward +=reward[0]

                time += 1

            print('Episode: ', ep + 1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)

            if ep % 10 == 0:
                self.actor.save_weight('pendulum_actor.th')
                self.critic.save_weight('pendulum_critic.th')
        
        np.savetxt('pendulum_epi_reward.txt', self.save_epi_reward)

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
    
    def save_result_graph(self):
        plt.plot(self.save_epi_reward)
        plt.savefig('pendulum_epi_reward_result.png')


