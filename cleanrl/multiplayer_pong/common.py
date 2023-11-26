import importlib

import numpy as np
import torch
from torch import nn
import supersuit as ss
import gym


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def atari_network(orth_init=False):
    init = layer_init if orth_init else lambda m: m
    return nn.Sequential(
        init(nn.Conv2d(4, 32, 8, stride=4)),
        nn.ReLU(),
        init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        init(nn.Linear(64 * 7 * 7, 512)),
        nn.ReLU(),
    )


def pong_obs_modification(obs, _space, player_id):
    obs[:9, :, :] = 0
    if "second" in player_id:
        # Mirror the image
        obs = obs[:, ::-1, :]
    return obs


def get_env(args, run_name):
    env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env()
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    # Remove the score from the observation
    if "pong" in args.env_id:
        env = ss.lambda_wrappers.observation_lambda_v0(
            env,
            pong_obs_modification,
        )
    # env = ss.agent_indicator_v0(env, type_only=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(env, args.num_envs // 2, num_cpus=0, base_class="gym")
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"
    return envs
