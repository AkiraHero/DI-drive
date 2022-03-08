import os
# sys.path.append("/home/xlju/carla-0.9.11-py3.7-linux-x86_64.egg")

import argparse
import torch

from core.envs import BenchmarkEnvWrapper
from core.utils.others.tcp_helper import parse_carla_tcp

from ding.utils.default_helper import deep_merge_dicts
from ding.policy import DQNPolicy, PPOPolicy, TD3Policy, SACPolicy
from ding.utils import set_pkg_seed

# rl model
# from demo.simple_rl.model import DQNRLModel
from noisy_planning.rl_model.rl_model import TD3RLModel, DQNRLModel, SACRLModel, PPORLModel
from demo.simple_rl.env_wrapper import DiscreteEnvWrapper, ContinuousEnvWrapper

# other module
from noisy_planning.detector.detection_model_wrapper import DetectionModelWrapper
from noisy_planning.eval.single_carla_evaluator_with_det import SingleCarlaEvaluatorWithDet
from noisy_planning.eval.simple_carla_env_new_render import SimpleCarlaEnvNewRender
from noisy_planning.simple_rl_train_with_detection import get_cfg


def get_cls(spec):
    policy_cls, model_cls = {
        'dqn': (DQNPolicy, DQNRLModel),
        # 'ddpg': (DDPGPolicy, DDPGRLModel),
        'td3': (TD3Policy, TD3RLModel),
        'ppo': (PPOPolicy, PPORLModel),
        'sac': (SACPolicy, SACRLModel),
    }[spec]

    return policy_cls, model_cls


def main(args, seed=0):
    # args.ckpt_path = "/home/xlju/Project/Model_behavior/training_log/simple-rl_2022-02-05-09-44-39/ckpt/iteration_14000.pth.tar"
    # args.ckpt_path = "/home/xlju/Project/Model_behavior/DI-drive/noisy_planning/tst.pkl"
    args.ckpt_path = "/cpfs2/user/juxiaoliang/project/DI-drive/noisy_planning/output_log/ppo-shorturn_nocar_add_lane-2022-02-22-08-03-32/ckpt/iteration_63000.pth.tar"
    # args.ckpt_path = ''
    cfg = get_cfg(args)

    cfg.env.visualize = dict(
        type='birdview',
        save_dir="eval_ppo_remote_with_lane",
        outputs=['video'],
    )
    cfg.policy.eval.evaluator.render = True

    eval_epchs = 50
    enable_detection = cfg.env.enable_detector

    policy_cls, model_cls = get_cls(args.policy)
    cfg.policy = deep_merge_dicts(policy_cls.default_config(), cfg.policy)

    tcp_list = parse_carla_tcp(cfg.server)
    assert len(tcp_list) > 0, "No Carla server found!"
    host, port = tcp_list[0]

    if args.policy == 'dqn':
        carla_env = BenchmarkEnvWrapper(DiscreteEnvWrapper(SimpleCarlaEnvNewRender(cfg.env, host, port)), cfg.env.wrapper.eval)
    else:
        carla_env = BenchmarkEnvWrapper(ContinuousEnvWrapper(SimpleCarlaEnvNewRender(cfg.env, host, port)), cfg.env.wrapper.eval)
    carla_env.seed(seed)
    set_pkg_seed(seed)

    model = model_cls(**cfg.policy.model)
    policy = policy_cls(cfg.policy, model=model, enable_field=['eval'])

    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
    elif cfg.policy.ckpt_path != '':
        ckpt_path = cfg.policy.ckpt_path
    else:
        ckpt_path = ''
    if ckpt_path != '':
        state_dict = torch.load(ckpt_path, map_location='cpu')
        policy.eval_mode.load_state_dict(state_dict)

    detection_model = None
    obs_bev_config = None
    if enable_detection:
        detection_model = DetectionModelWrapper(cfg=cfg.env.detector)
        obs_bev_config = [i for i in cfg.env.simulator.obs if i['name'] == 'birdview'][0]
    evaluator = SingleCarlaEvaluatorWithDet(cfg.policy.eval.evaluator, carla_env, policy.eval_mode,
                                            detector=detection_model,
                                            bev_obs_config=obs_bev_config)

    use_det_policy = True
    for i in range(eval_epchs):
        res = evaluator.eval(reset_param=dict(name="eval_episode-{}".format(i), use_det_policy=use_det_policy))
        file_name = os.path.join(cfg.env.visualize.save_dir, "eval-{}-use_det_policy-{}.txt".format(i, use_det_policy))
        with open(file_name, 'w') as f:
            f.write(str(res))
    evaluator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple-rl test')
    parser.add_argument('-n', '--name', type=str, default='simple-rl-eval', help='experiment name')
    parser.add_argument('-p', '--policy', default='dqn', choices=['dqn', 'ppo', 'td3', 'sac', 'ddpg'], help='RL policy')
    parser.add_argument('-c', '--ckpt-path', default=None, help='model ckpt path')
    parser.add_argument('-d', '--ding-cfg', default=None, help='DI-engine config path')
    args = parser.parse_args()
    main(args)
