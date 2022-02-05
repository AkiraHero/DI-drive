# system
import argparse
import time

# ding
from noisy_planning.env_related.carla_env_manager import CarlaSyncSubprocessEnvManager
from ding.policy import DQNPolicy, PPOPolicy, TD3Policy, SACPolicy, DDPGPolicy
from ding.worker import BaseLearner, SampleSerialCollector, AdvancedReplayBuffer

# rl model
from demo.simple_rl.model import DQNRLModel, PPORLModel, TD3RLModel, SACRLModel, DDPGRLModel

# utils
from core.utils.others.ding_utils import compile_config
from core.utils.others.ding_utils import read_ding_config


# other module
from noisy_planning.utils.debug_utils import generate_general_logger
from tensorboardX import SummaryWriter
from ding.torch_utils import auto_checkpoint


def get_cfg(args):
    if args.ding_cfg is not None:
        ding_cfg = args.ding_cfg
    else:
        ding_cfg = {
            'dqn': 'noisy_planning.config.dqn_config.py',
            'dqn-ini': 'noisy_planning.config.dqn_config_ini.py',
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


class TestLearner(BaseLearner):

    @auto_checkpoint
    def start(self) -> None:
        self._end_flag = False
        self._learner_done = False
        # before run hook
        self.call_hook('before_run')

        while True:
            time.sleep(0.1)
        self._learner_done = True
        # after run hook
        self.call_hook('after_run')



def main(args, seed=0):
    logger = generate_general_logger("MAIN")

    cfg = get_cfg(args)

    policy_cls, model_cls = get_cls(args.policy)
    model = model_cls(**cfg.policy.model)
    policy = policy_cls(cfg.policy, model=model)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = TestLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger,
                          exp_name=cfg.exp_name + '_' + time.strftime("%Y-%m-%d-%H-%M-%S"))
    while True:
        time.sleep(1)
    learner.start()
    learner.close()
    logger.error('finish')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple-rl train')
    parser.add_argument('-n', '--name', type=str, default='simple-rl', help='experiment name')
    parser.add_argument('-p', '--policy', default='dqn', choices=['dqn', 'ppo', 'td3', 'sac', 'ddpg'], help='RL policy')
    parser.add_argument('-d', '--ding-cfg', default=None, help='DI-engine config path')

    args = parser.parse_args()
    main(args)
