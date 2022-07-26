import gym
from agent import Agent


def main():
    max_episode_num = 1000
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = Agent(env)
    
    agent.train(max_episode_num) 
    
    agent.save_result_graph()

if __name__ == '__main__':
    main()