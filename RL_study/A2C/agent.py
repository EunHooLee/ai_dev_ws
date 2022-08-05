from actor import Actor
from crititc import Critic
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import Utils

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
        # high 함수뭐냐 ? list, ndarray, Tensor에는 적용 안됨 / env에 high 라는 정보가 있나?

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim, self.action_dim, self.CRITIC_LEARNING_RATE)

        self.save_epi_reward = []
    
    def advantage_td_target(self, reward, v_value, next_v_value, done):
        if done:
            y_k = v_value
            advantage = y_k - v_value
        else:
            y_k = reward + self.GAMMA * next_v_value
            advantage = y_k - v_value
        
        return advantage, y_k
    # unpack_batch 는 기능이 뭐지? 뭐하는 함수지? 
    def unpack_batch(self, batch):
        unpack = batch[0]   
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)
        return unpack
    
    def train(self, max_episode_num):
        # 한 episode마다 초기화
        for ep in range(max_episode_num):
            batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []

            time, episode_reward, done = 0, 0, False

            state = self.env.reset()
            actor_loss, critic_loss = 0, 0
            # episode 끝이 날때까지 반복
            # 한 episode 가 끝나는 시점이 언제인지 안나왔음 : done==True 가 되는 시점이 언제인가? (아마 env에 max_ep_length가 지정되 있을 것이라고 추측됨)
            while not done:
                action = self.actor.get_action(Utils.convertToTensorInput(state,self.state_dim))
                action = np.clip(action, -self.action_bound, self.action_bound)[0]

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1,self.state_dim])
                next_state = np.reshape(next_state,[1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1,1])
            
                v_value = self.critic.predict(Utils.convertToTensorInput(state,self.state_dim)).detach().numpy()
                next_v_value = self.critic.predict(Utils.convertToTensorInput(next_state, self.state_dim)).detach().numpy()
                # 왜 8 을 더하고 나누지? 뭔가 정규화 하려는거 같은데 ? 
                train_rewrad = (reward + 8) / 8
                advantage, y_i = self.advantage_td_target(train_rewrad, v_value,next_v_value,done)
                # batch에 저장? 이거 batch 맞나? replay buffer 가 아니라?
                # RB 는 생각해보면, off-policy일 때 사용하는 거니까 A2C 에서 사용할 수 있지 않나?

                # Answer : batch 가 맞고, A2C 는 on-policy 알고리즘 이기 때문에 Replay Buffer를 이용할 수 없다. 
                batch_state.append(state)
                batch_action.append(action)
                batch_td_target.append(y_i)
                batch_advantage.append(advantage)
                
                # 여기서 batch_size 만큼 데이터를 모은다.
                # batch 가 가득차지 않으면 아래 코드가 실행되지 않고 다시 while문 초기로 돌아간다. 즉, batch 만큼 데이터 모아서 학습시킨다.
                if len(batch_state) < self.BATCH_SIZE:
                    state = next_state[0]
                    episode_reward +=reward[0]
                    time +=1
                    continue

                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                td_targets = self.unpack_batch(batch_td_target)
                advantages = self.unpack_batch(batch_advantage)
                
                batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []

                critic_loss = self.critic.Learn(Utils.convertToTensorInput(states, self.state_dim,states.shape[0]),td_targets)
                actor_loss = self.actor.Learn(Utils.convertToTensorInput(states,self.state_dim, states.shape[0]),actions, advantages)

                state = next_state[0]
                episode_reward += reward[0]
                time += 1
            # episode 가 끝나면 아래 코드 실행됨 : 즉, 에피소드 동안 총 받은 누적보상을 출력한다.
            print('Episode: ',ep+1,' Time: ', time,' Reward: ', episode_reward,' actor loss: ', actor_loss.item(),' critic loss: ',critic_loss.item())
            
            self.save_epi_reward.append(episode_reward)

            if ep % 10 == 0:
                self.actor.save_weights('pendulum_actor.th')
                self.critic.save_weights('pendulum_critic.th')
        # 다 끝나면 ep reward 저장
        np.savetxt('pendulum_epi_reward.txt', self.save_epi_reward)
    
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()