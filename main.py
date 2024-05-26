import os
import torch
import gym
import numpy as np
from utilis.config import ARGConfig
from utilis.default_config import default_config
from model.algo import OBAC
from utilis.Replaybuffer import ReplayMemory
from utilis.video import recorder
import datetime
import itertools
from torch.utils.tensorboard import SummaryWriter
import shutil
# from dm_control import suite
# from envs.meta_world_env import make_env
# import robohive


def evaluation(agent, env, total_numsteps, writer, best_reward, video_path=None):
    avg_reward = 0.
    avg_success = 0.
    if video_path is not None:
        eval_recoder = recorder(video_path)
        eval_recoder.init(f'{total_numsteps}.mp4', enabled=True)
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
        if 'solved' in info.keys():
            avg_success += float(info['solved'])
        elif 'success' in info.keys():
            avg_success += float(info['success'])
    avg_reward /= config.eval_times
    avg_success /= config.eval_times

    if eval_recoder is not None and avg_reward >= best_reward:
        eval_recoder.release('%d_%d.mp4'%(total_numsteps, int(avg_reward)))

    writer.add_scalar('test/avg_reward', avg_reward, total_numsteps)
    writer.add_scalar('test/avg_success', avg_success, total_numsteps)

    print("----------------------------------------")
    print("Env: {}, Test Episodes: {}, Avg. Reward: {}, Avg. Success: {}".format(config.task, config.eval_times, round(avg_reward, 2), round(avg_success, 2)))
    print("----------------------------------------")
    
    return avg_reward

def train_loop(config, msg = "default"):
    # set seed
    env = gym.make(config.env_name)
    # env = make_env(config)
    # env.seed(config.seed)
    env.action_space.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Agent
    agent = OBAC(env.observation_space.shape[0], env.action_space, config)

    result_path = './results/{}/{}/{}/{}_{}_{}_{}'.format(config.task, config.algo, msg, 
                                                      datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                                                      config.policy, config.seed, 
                                                      "autotune" if config.automatic_entropy_tuning else "")

    checkpoint_path = result_path + '/' + 'checkpoint'
    video_path = result_path + '/eval_video'
    
    # training logs
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    with open(os.path.join(result_path, "config.log"), 'w') as f:
        f.write(str(config))

    writer = SummaryWriter(result_path)
    current_path = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(current_path)
    files_to_save = ['main.py', 'envs', 'model', 'utilis', 'render_results.py']
    ignore_files = [x for x in files if x not in files_to_save]
    shutil.copytree('.', result_path + '/code', ignore=shutil.ignore_patterns(*ignore_files))

    # memory
    memory = ReplayMemory(config.replay_size, config.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    best_reward = -1e6
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state, _ = env.reset()

        while not done:
            if config.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if config.start_steps <= total_numsteps:
                # Number of updates per step in environment
                for i in range(config.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, value_loss, policy_loss, ent_loss, alpha, q_pi, q_behavior_pi = agent.update_parameters(memory, config.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/offline_value', value_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('parameter/alpha', alpha, updates)
                    writer.add_scalar('parameter/q_current_pi', q_pi, updates)
                    writer.add_scalar('parameter/q_behavior_pi', q_behavior_pi, updates)
                    writer.add_scalar('parameter/q_diff', q_pi - q_behavior_pi, updates)
                    updates += 1

            next_state, reward, done, truncated, info = env.step(action) # Step
            done = done or truncated
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            if "_max_episode_steps" in dir(env):
                mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            else:
                mask = 1 if episode_steps == 1000 else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            
            # test agent
            if total_numsteps % config.eval_numsteps == 0 and config.eval is True:
                video_path = None
                avg_reward = evaluation(agent, env, total_numsteps, writer, best_reward, video_path)
                if avg_reward >= best_reward and config.save is True:
                    best_reward = avg_reward
                    agent.save_checkpoint(checkpoint_path, 'best')

        if total_numsteps > config.num_steps:
            break
        
        if i_episode % 10 == 0:
            writer.add_scalar('train/reward', episode_reward, total_numsteps)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
            print()
    env.close() 



if __name__ == "__main__":
    arg = ARGConfig()
    arg.add_arg("task", "humanoid_h1hand-walk-v0", "Environment name")
    arg.add_arg("device", 0, "Computing device")
    arg.add_arg("algo", "OBAC", "choose algo")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("seed", 0, "experiment seed")
    arg.parser()

    config = default_config
    
    config.update(arg)

    print(f">>>> Training {config.algo} on {config.task} environment, on {config.device}")
    train_loop(config, msg=config.tag)