from xml.sax.handler import feature_external_ges
import gym

from ppo_agent import PPOAgent


def main():

    max_episode_num = 3000
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = PPOAgent(env)

    agent.train(max_episode_num)

    agent.save_result_graph()


if __name__ == '__main__':
    main()