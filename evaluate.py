import os
import sys
import torch
import gym
from utilis.config import Config
from utilis.default_config import default_config
from model.algo import OBAC
from copy import copy


def display_model(result_path, result_epoch, config_path=None):
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Error, model file {result_path} not exists")
    if config_path is None:
        config_path = result_path + '/' + 'config.log'
    
    config = Config().load_saved(config_path)
    config.device = "cpu"

    env = gym.make(config.env_name)

    # Agent
    agent = OBAC(env.observation_space.shape[0], env.action_space, config)
    agent.load_checkpoint(result_path, result_epoch)

    # test agent
    avg_reward = 0.
    for _  in range(config.eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)

            next_state, reward, done, _ = env.step(action)
            env.render()
            episode_reward += reward

            state = next_state
        avg_reward += episode_reward
    avg_reward /= config.eval_episodes

    print("----------------------------------------")
    print("Env: {}, Test Episodes: {}, Avg. Reward: {}".format(config.env_name, config.eval_episodes, round(avg_reward, 2)))
    print("----------------------------------------")
    env.close()


if __name__ == "__main__":
    result_path = sys.argv[1]
    result_epoch = sys.argv[2] if len(sys.argv) > 2 else None
    display_model(result_path, result_epoch)