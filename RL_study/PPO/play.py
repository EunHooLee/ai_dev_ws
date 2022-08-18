import gym
from ppo_agent import PPOAgent
from ppo_critic import Critic
from ppo_actor import Actor

from Utils import convertToTensorInput

def main():
    env_name = 'Pendulum-v1'
    env = gym.make('Pendulum-v1')

    agent = PPOAgent(env)

    agent.actor.load_weight('pendulum_actor.th')
    agent.critic.load_weight('pendulum_critic.th')

    time = 0
    state = env.reset()

    while True:

        action = agent.actor.get_action(convertToTensorInput(state,agent.state_dim))

        state, reward, done, _ = env.step(action)
        env.render()
        time += 1

        print('Time: ',time,' Reward: ',reward)
        
        if time >=1000:
            break
    env.close()

if __name__ == '__main__':
    main()