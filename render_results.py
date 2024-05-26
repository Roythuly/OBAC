import os
import sys
import torch
import gym
import numpy as np
from utilis.config import ARGConfig
from model.algo import OBAC
from utilis.video import recorder
import shutil
from envs.humanoid import make_env

# from numpngw import write_apng

# xvfb-run -s "-screen 0 640x480x24" python render_robohive.py xxx(address) xxx(epoch)


def display_model(result_path, result_epoch, config_path=None):
    if result_epoch is None:
        result_epoch = 'best'
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Error, model file {result_path} not exists")
    if config_path is None:
        config_path = result_path + '/' + 'config.log'
    
    config = ARGConfig().load_saved(config_path)
    env = make_env(config)
    config.seed = np.random.randint(0,100)
    # env.seed(config.seed)
    env.action_space.seed(config.seed)

    # Agent
    agent = OBAC(env.observation_space.shape[0], env.action_space, config)
    checkpoint_path = os.path.join(result_path, "checkpoint")
    agent.load_checkpoint(checkpoint_path, result_epoch)

    # test agent
    avg_reward = 0.
    avg_success = 0.

    #* video save path
    video_path = os.path.join('Video', config.task, config.algo, os.path.split(result_path)[-1])
    os.system('mkdir -p %s'%video_path)
    
    config.eval_times = 3
    avg_reward = 0.
    avg_success = 0.
    if video_path is not None:
        eval_recoder = recorder(video_path)
        eval_recoder.init('best.mp4', enabled=True)
    else:
        eval_recoder = None
    for _  in range(config.eval_times):
        state, _ = env.reset()
        if eval_recoder is not None:
            eval_recoder.record(env.render())
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)

            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            if eval_recoder is not None:
                eval_recoder.record(env.render())
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward

    avg_reward /= config.eval_times
    print("Env: {}, Test Episodes: {}, Avg. Reward: {}, Avg. Success: {}".format(config.task, config.eval_times, round(avg_reward, 2), round(avg_success,2)))
    eval_recoder.release('%d.mp4'%(int(avg_reward)))



if __name__ == "__main__":
    result_path = sys.argv[1]
    result_epoch = sys.argv[2] if len(sys.argv) > 2 else None
    display_model(result_path, result_epoch)