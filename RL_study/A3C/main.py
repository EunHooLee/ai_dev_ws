import gym
import agent


def main():
    max_episode_num = 1000
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    _agent = agent.Agent(env)
    
    _agent.train(max_episode_num)

    _agent.save_result_graph()

if __name__== '__main__':
    main()