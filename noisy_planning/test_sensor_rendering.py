import sys
sys.path.append("/home/xlju/Project/carla_099/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg")


import os
import argparse
import numpy as np
from functools import partial
from easydict import EasyDict
import copy
from tensorboardX import SummaryWriter

from core.envs import SimpleCarlaEnv, BenchmarkEnvWrapper
from core.utils.others.tcp_helper import parse_carla_tcp
from core.eval import SerialEvaluator
from ding.envs import SyncSubprocessEnvManager, BaseEnvManager
from ding.policy import DQNPolicy, PPOPolicy, TD3Policy, SACPolicy, DDPGPolicy
from ding.worker import BaseLearner, SampleSerialCollector, AdvancedReplayBuffer, NaiveReplayBuffer
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn

from demo.simple_rl.model import DQNRLModel, PPORLModel, TD3RLModel, SACRLModel, DDPGRLModel
from demo.simple_rl.env_wrapper import DiscreteEnvWrapper, ContinuousEnvWrapper
from core.utils.data_utils.bev_utils import unpack_birdview
from core.utils.others.ding_utils import compile_config
from core.utils.others.ding_utils import read_ding_config
from extended_envs.carla_env_with_detection import CarlaEnvWithDetection

def get_cls(spec):
    policy_cls, model_cls = {
        'dqn': (DQNPolicy, DQNRLModel),
        'ddpg': (DDPGPolicy, DDPGRLModel),
        'td3': (TD3Policy, TD3RLModel),
        'ppo': (PPOPolicy, PPORLModel),
        'sac': (SACPolicy, SACRLModel),
    }[spec]
    return policy_cls, model_cls


def get_cfg(args):
    if args.ding_cfg is not None:
        ding_cfg = args.ding_cfg
    else:
        ding_cfg = {
            'dqn': 'noisy_planning.config.dqn_config.py',
            # 'ppo': 'demo.simple_rl.config.ppo_config.py',
            # 'td3': 'demo.simple_rl.config.td3_config.py',
            # 'sac': 'demo.simple_rl.config.sac_config.py',
            # 'ddpg': 'demo.simple_rl.config.ddpg_config.py',
        }[args.policy]
    default_train_config = read_ding_config(ding_cfg)
    default_train_config.exp_name = args.name
    use_policy, _ = get_cls(args.policy)
    use_buffer = AdvancedReplayBuffer if args.policy != 'ppo' else None
    cfg = compile_config(
        cfg = default_train_config,
        env_manager = SyncSubprocessEnvManager,
        policy = use_policy,
        learner = BaseLearner,
        collector = SampleSerialCollector,
        buffer = use_buffer,
    )
    return cfg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simple-rl train')
    parser.add_argument('-n', '--name', type=str, default='simple-rl', help='experiment name')
    parser.add_argument('-p', '--policy', default='dqn', choices=['dqn', 'ppo', 'td3', 'sac', 'ddpg'], help='RL policy')
    parser.add_argument('-d', '--ding-cfg', default=None, help='DI-engine config path')
    args = parser.parse_args()
    cfg = get_cfg(args)

    env_cfg = cfg.env
    host = "localhost"
    port = 9000
    env = CarlaEnvWithDetection(cfg=env_cfg, host=host, port=port)
    env.reset()
    obs = env.get_observations()
    pass