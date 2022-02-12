# system
import argparse
from functools import partial
import traceback
import os
import yaml
import numpy # import numpy before torch to avoid some inexpected err
import torch
from easydict import EasyDict

# ding
from ding.envs import BaseEnvManager
from noisy_planning.env_related.carla_env_manager import CarlaSyncSubprocessEnvManager
from ding.policy import DQNPolicy, PPOPolicy, TD3Policy, SACPolicy, DDPGPolicy
from ding.worker import BaseLearner, SampleSerialCollector, AdvancedReplayBuffer, NaiveReplayBuffer
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn

# rl model
# dqn will use model from initial repo
# from demo.simple_rl.model import DQNRLModel
from rl_model import TD3RLModel, DQNRLModel, PPORLModel, SACRLModel

from demo.simple_rl.env_wrapper import DiscreteEnvWrapper, ContinuousEnvWrapper

# utils
from core.utils.others.ding_utils import compile_config
from core.utils.others.ding_utils import read_ding_config
from core.envs import SimpleCarlaEnv, BenchmarkEnvWrapper
from core.utils.others.tcp_helper import parse_carla_tcp
from core.eval import SerialEvaluator

# other module
from noisy_planning.detector.detection_model_wrapper import DetectionModelWrapper
from noisy_planning.utils.debug_utils import generate_general_logger
from tensorboardX import SummaryWriter
from noisy_planning.learner.carla_learner import CarlaLearner


def wrapped_discrete_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    env = SimpleCarlaEnv(env_cfg, host, port, tm_port)
    return BenchmarkEnvWrapper(DiscreteEnvWrapper(env), wrapper_cfg)


def wrapped_continuous_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    env = SimpleCarlaEnv(env_cfg, host, port, tm_port)
    return BenchmarkEnvWrapper(ContinuousEnvWrapper(env), wrapper_cfg)


def get_cfg(args):
    if args.ding_cfg is not None:
        ding_cfg = args.ding_cfg
    else:
        ding_cfg = {
            'dqn': 'noisy_planning.config.dqn_config.py',
            'ppo': 'noisy_planning.config.ppo_config.py',
            'td3': 'noisy_planning.config.td3_config.py',
            'sac': 'noisy_planning.config.sac_config.py',
        }[args.policy]
    default_train_config = read_ding_config(ding_cfg)
    default_train_config.exp_name = args.name
    use_policy, _ = get_cls(args.policy)
    use_buffer = AdvancedReplayBuffer if args.policy != 'ppo' else None
    cfg = compile_config(
        cfg=default_train_config,
        env_manager=CarlaSyncSubprocessEnvManager,
        policy=use_policy,
        learner=BaseLearner,
        collector=SampleSerialCollector,
        buffer=use_buffer,
    )
    return cfg

def edict2dict(edict_obj):
    if isinstance(edict_obj, list) or isinstance(edict_obj, tuple):
        return [edict2dict(i) for i in edict_obj]
    if isinstance(edict_obj, dict) or isinstance(edict_obj, EasyDict):
        dict_obj = {}
        for key, vals in edict_obj.items():
            dict_obj[key] = edict2dict(vals)
        return dict_obj
    return edict_obj


def get_cls(spec):
    policy_cls, model_cls = {
        'dqn': (DQNPolicy, DQNRLModel),
        'td3': (TD3Policy, TD3RLModel),
        'ppo': (PPOPolicy, PPORLModel),
        'sac': (SACPolicy, SACRLModel),
    }[spec]

    return policy_cls, model_cls


def main(args, seed=0):
    set_pkg_seed(seed)
    logger = generate_general_logger("MAIN")

    '''
    Config
    '''
    enable_eval = True
    cfg = get_cfg(args)
    tcp_list = parse_carla_tcp(cfg.server)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    assert len(tcp_list) >= collector_env_num + evaluator_env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(
            collector_env_num + evaluator_env_num, len(tcp_list)
        )
    # move config cfg to folder
    if not os.path.exists(cfg.exp_name):
        os.makedirs(cfg.exp_name)
    config_file = os.path.join(cfg.exp_name, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(edict2dict(cfg), f)

    '''
    Policy
    '''
    policy_cls, model_cls = get_cls(args.policy)
    model = model_cls(**cfg.policy.model)
    policy = policy_cls(cfg.policy, model=model)

    '''
    Learner and tensorboard
    '''
    learner = CarlaLearner(cfg.policy.learn, policy.learn_mode, exp_name=cfg.exp_name)
    learner.set_policy_name(args.policy)
    if args.policy == 'dqn':
        eps_cfg = cfg.policy.other.eps
        epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
        learner.set_epsilon_greedy(epsilon_greedy)
    tb_logger = learner.tb_logger


    '''
    Detector
    '''
    detection_model = None
    obs_bev_config = None
    detection_max_batch_size = None
    if cfg.env.enable_detector:
        logger.error("Detector enabled.")
        detection_model = DetectionModelWrapper(cfg=cfg.env.detector)
        obs_bev_config = [i for i in cfg.env.simulator.obs if i['name'] == 'birdview'][0]
        detection_max_batch_size = cfg.env.detector.max_batch_size
    else:
        logger.error("Detector not enabled.")


    '''
    Env and Collector
    '''
    if args.policy == 'dqn':
        wrapped_env = wrapped_discrete_env
    else:
        wrapped_env = wrapped_continuous_env

    # if detection not enabled, forbid lidar collection
    if not cfg.env.enable_detector:
        cfg.env.simulator.obs = [i for i in cfg.env.simulator.obs if i['type'] != 'lidar']

    collector_env = CarlaSyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper.collect, *tcp_list[i]) for i in range(collector_env_num)],
        cfg=cfg.env.manager.collect,
        detector=detection_model,
        detection_max_batch_size=detection_max_batch_size,
        bev_obs_config=obs_bev_config,
    )
    collector_env.seed(seed)
    collector = SampleSerialCollector(cfg.policy.collect.collector,
                                      collector_env,
                                      policy.collect_mode,
                                      tb_logger,
                                      exp_name=cfg.exp_name)
    learner.set_collector(collector, cfg.policy.collect)

    '''
    Validation
    '''
    evaluator = None
    if enable_eval:
        try:
            evaluate_env = CarlaSyncSubprocessEnvManager(
                env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper.eval, *tcp_list[collector_env_num + i]) for i in
                        range(evaluator_env_num)],
                cfg=cfg.env.manager.eval,
                detector=detection_model,
                detection_max_batch_size=detection_max_batch_size,
                bev_obs_config=obs_bev_config,
            )
            # Uncomment this to add save replay when evaluation
            # evaluate_env.enable_save_replay(cfg.env.replay_path)
            evaluate_env.seed(seed)
            evaluator = SerialEvaluator(cfg.policy.eval.evaluator,
                                        evaluate_env,
                                        policy.eval_mode,
                                        tb_logger,
                                        exp_name=cfg.exp_name)
            learner.set_evaluator(evaluator)
        except Exception as e:
            logger.error("Fail to initialize evaluator...")
            logger.error(str(e))
            logger.error(traceback.format_tb(e.__traceback__))


    '''
    Replay buffer
    '''
    if cfg.policy.get('priority', False):
        replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    else:
        replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    learner.set_replay_buffer(replay_buffer)

    '''
    Training loop
    '''
    learner.start()

    '''
    Closing
    '''
    collector.close()
    if evaluator:
        evaluator.close()
    learner.close()
    if args.policy != 'ppo':
        replay_buffer.close()
    logger.error('finish')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple-rl train')
    parser.add_argument('-n', '--name', type=str, default='simple-rl', help='experiment name')
    parser.add_argument('-p', '--policy', default='td3', choices=['dqn', 'ppo', 'td3', 'sac', 'ddpg'], help='RL policy')
    parser.add_argument('-d', '--ding-cfg', default=None, help='DI-engine config path')
    parser.add_argument('--withoutcudnn', action='store_true')
    args = parser.parse_args()
    if args.withoutcudnn:
        print("[MAIN]disable cudnn backend.")
        torch.backends.cudnn.enabled = False
    main(args)
