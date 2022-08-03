import gym
from agent import Agent
from crititc import Critic
from actor import Actor

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import Utils

def main():
    env_name = 'Pendulum-v1'
    env = gym.make('Pendulum-v1')

    agent = Agent(env)

    agent.actor.load_weights('pendulum_actor.th')
    agent.critic.load_weights('pendulum_critic.th')

    time = 0
    state = env.reset()

    while True:

        action = agent.actor.get_action(Utils.convertToTensorInput(state,agent.state_dim))

        state, reward, done, _ = env.step(action)
        env.render()
        time += 1

        print('Time: ',time,' Reward: ',reward)
        
        if time >=1000:
            break
    env.close()

if __name__ == '__main__':
    main()