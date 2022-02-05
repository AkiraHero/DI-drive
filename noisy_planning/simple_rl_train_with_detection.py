# system
import argparse
from functools import partial


# ding
from ding.envs import BaseEnvManager
from noisy_planning.env_related.carla_env_manager import CarlaSyncSubprocessEnvManager
from ding.policy import DQNPolicy, PPOPolicy, TD3Policy, SACPolicy, DDPGPolicy
from ding.worker import BaseLearner, SampleSerialCollector, AdvancedReplayBuffer, NaiveReplayBuffer
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn

# rl model
from demo.simple_rl.model import DQNRLModel, PPORLModel, TD3RLModel, SACRLModel, DDPGRLModel
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
            # 'ppo': 'noisy_planning.config.ppo_config.py',
            # 'td3': 'noisy_planning.config.td3_config.py',
            # 'sac': 'noisy_planning.config.sac_config.py',
            # 'ddpg': 'noisy_planning.config.ddpg_config.py',
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


def get_cls(spec):
    policy_cls, model_cls = {
        'dqn': (DQNPolicy, DQNRLModel),
        'ddpg': (DDPGPolicy, DDPGRLModel),
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
    enable_eval = False
    cfg = get_cfg(args)
    tcp_list = parse_carla_tcp(cfg.server)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    assert len(tcp_list) >= collector_env_num + evaluator_env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(
            collector_env_num + evaluator_env_num, len(tcp_list)
        )

    '''
    Policy
    '''
    policy_cls, model_cls = get_cls(args.policy)
    model = model_cls(**cfg.policy.model)
    policy = policy_cls(cfg.policy, model=model)

    '''
    Learner and tensorboard
    '''
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = CarlaLearner(cfg.policy.learn, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    learner.set_policy_name(args.policy)
    if args.policy == 'dqn':
        eps_cfg = cfg.policy.other.eps
        epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
        learner.set_epsilon_greedy(epsilon_greedy)

    '''
    Env and Collector
    '''
    if args.policy == 'dqn':
        wrapped_env = wrapped_discrete_env
    else:
        wrapped_env = wrapped_continuous_env

    collector_env = CarlaSyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper.collect, *tcp_list[i]) for i in range(collector_env_num)],
        cfg=cfg.env.manager.collect,
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
        evaluate_env = BaseEnvManager(
            env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper.eval, *tcp_list[collector_env_num + i]) for i in
                    range(evaluator_env_num)],
            cfg=cfg.env.manager.eval,
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

    '''
    Detector
    '''
    if cfg.env.enable_detector:
        logger.error("Detector enabled.")
        detection_model = DetectionModelWrapper(cfg=cfg.env.detector)
        obs_bev_config = [i for i in cfg.env.simulator.obs if i['name'] == 'birdview'][0]
        learner.set_detection_model(detection_model, cfg.env.detector.max_batch_size, obs_bev_config)
    else:
        logger.error("Detector not enabled.")

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
    parser.add_argument('-p', '--policy', default='dqn', choices=['dqn', 'ppo', 'td3', 'sac', 'ddpg'], help='RL policy')
    parser.add_argument('-d', '--ding-cfg', default=None, help='DI-engine config path')

    args = parser.parse_args()
    main(args)
