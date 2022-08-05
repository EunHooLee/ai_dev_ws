import threading
import numpy as np 

from actor import Actor
from critic import Critic


import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import Utils

global_episode_count = 0
global_step = 0
global_episode_reward = []


class Worker(threading.Thread):

    def __init__(self, worker_name, env, global_actor, global_critic, max_episode_num):
        threading.Thread.__init__(self)

        self.GAMMA = 0.95
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.t_MAX = 4 # n-step TD target

        self.max_episode_num = max_episode_num
        
        self.env = env
        self.worker_name = worker_name

        self.global_actor = global_actor
        self.global_critic = global_critic

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        
        self.worker_actor = Actor(self.state_dim,self.action_dim, self.action_bound, self.ACTOR_LEARNING_RATE)
        self.worker_critic = Critic(self.state_dim,self.action_dim,self.CRITIC_LEARNING_RATE)
        

        # 이 부분 확인!
        # self.worker_actor.apply(self.global_actor.parameter)
        # self.worker_critic.apply(self.global_critic.parameter)
        self.worker_actor.load_state_dict(self.global_actor.state_dict())
        self.worker_critic.load_state_dict(self.global_critic.state_dict())


    def n_step_td_target(self, rewards, next_v_value, done):
        y_i = np.zeros(rewards.shape)
        cumulative = 0
        if not done:
            cumulative = next_v_value
        
        for k in reversed(range(0, len(rewards))):
            cumulative = self.GAMMA * cumulative + rewards[k]
            y_i[k] = cumulative
        
        return y_i

    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)
        
        return unpack

    def run(self):
        global global_episode_count, global_step
        # global global_episode_reward

        print(self.worker_name, " starts ---")

        while global_episode_count <= int(self.max_episode_num):

            batch_state, batch_action, batch_reward = [], [], []

            step, episode_reward, done = 0, 0, False

            state = self.env.reset()
            actor_loss, critic_loss = 0, 0

            while not done:
                # self.env.render()
                action = self.worker_actor.get_action(Utils.convertToTensorInput(state, self.state_dim))
                # [0] 이 왜붙지? 
                action = np.clip(action, -self.action_bound, self.action_bound)[0]

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                # next_state = np.reshape(next_state, [1,self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1,1])

                train_reward = (reward + 8) / 8

                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(train_reward)

                state = next_state
                episode_reward += reward[0]
                step += 1

                if len(batch_state) == self.t_MAX or done:

                    states = self.unpack_batch(batch_state)
                    actions = self.unpack_batch(batch_action)
                    rewards = self.unpack_batch(batch_reward)

                    batch_state, batch_action, batch_reward = [], [], []

                    next_state = np.reshape(next_state, [1,self.state_dim])

                    next_v_value = self.worker_critic.predict(Utils.convertToTensorInput(next_state,self.state_dim)).detach().numpy()

                    n_step_td_targets = self.n_step_td_target(rewards,next_v_value,done)
                    
                    v_values = self.worker_critic.predict(Utils.convertToTensorInput(states,self.state_dim,states.shape[0])).detach().numpy()
                    
                    advantages = n_step_td_targets - v_values

                    critic_loss = self.global_critic.Learn(Utils.convertToTensorInput(states,self.state_dim,states.shape[0]), n_step_td_targets)
                    actor_loss = self.global_actor.Learn(Utils.convertToTensorInput(states, self.state_dim,states.shape[0]),actions,advantages)

                    self.worker_actor.load_state_dict(self.global_actor.state_dict())
                    self.worker_critic.load_state_dict(self.global_critic.state_dict())

                    global_step += 1
                
                if done:
                    global_episode_count += 1
                    print("Worker name: ", self.worker_name,', Episode: ',global_episode_count,', Step: ',step,', Reward: ',episode_reward)

                    global_episode_reward.append(episode_reward)

                    if global_episode_count % 10 == 0:
                        self.global_actor.save_weights('./pendulum_actor.th')
                        self.global_critic.save_weights('./pendulum_critic.th')

                        
