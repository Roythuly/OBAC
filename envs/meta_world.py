from collections import deque, defaultdict
from typing import Any, NamedTuple
import dm_env
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
import gym
from gym.wrappers import TimeLimit
import warnings
import metaworld
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)
	def reset(self):
		state, info = self.env.reset()
		return state
	def step(self, action):
		next_state, reward, done, flag, info = self.env.step(action)
		return next_state, reward, done, info

	def render(self, mode=None, height=384, width=384, camera_id=None):
		return self.env.render(offscreen=True, resolution=(width, height))


def make_env(env_name, seed, episode_length):
	env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=seed)
	env = MetaWorldWrapper(env)
	env = TimeLimit(env, max_episode_steps=episode_length)
	return env