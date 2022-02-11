import os
import sys
sys.path.append("/home/xlju/carla-0.9.11-py3.7-linux-x86_64.egg")

import argparse
import torch
from easydict import EasyDict

from core.envs import DriveEnvWrapper
from core.utils.others.tcp_helper import parse_carla_tcp

from ding.utils.default_helper import deep_merge_dicts
from ding.policy import DQNPolicy, PPOPolicy, TD3Policy, SACPolicy, DDPGPolicy
from ding.utils import set_pkg_seed

# rl model
from demo.simple_rl.model import DQNRLModel
from noisy_planning.rl_model import TD3RLModel
from demo.simple_rl.env_wrapper import DiscreteEnvWrapper, ContinuousEnvWrapper

# other module
from noisy_planning.detector.detection_model_wrapper import DetectionModelWrapper
from noisy_planning.eval.single_carla_evaluator_with_det import SingleCarlaEvaluatorWithDet
from noisy_planning.eval.simple_carla_env_new_render import SimpleCarlaEnvNewRender

from easydict import EasyDict

# eval_config = EasyDict(dict(
#     env=dict(
#         simulator=dict(
#             town='Town01',
#             delta_seconds=0.05,
#             disable_two_wheels=True,
#             verbose=False,
#             waypoint_num=32,
#             planner=dict(
#                 type='behavior',
#                 resolution=1,
#             ),
#             obs=(
#                 dict(
#                     name='birdview',
#                     type='bev',
#                     size=[160, 160],
#                     pixels_per_meter=5,
#                     pixels_ahead_vehicle=100,
#                 ),
#                 dict(
#                     name='toplidar',
#                     type='lidar',
#                     channels=64,
#                     range=32,
#                     points_per_second=1280000,
#                     rotation_frequency=20,
#                     upper_fov=10.0,
#                     lower_fov=-45.0,
#                     position=[0, 0.0, 1.6],
#                     rotation=[0, 0, 0],
#                     fixed_pt_num=40000,
#                 ),
#             ),
#         ),
#         enable_detector=False,
#         detector=dict(
#             model_repo="openpcdet",
#             model_name="pointpillar",
#             ckpt="/home/xlju/Downloads/pointpillar/pointpillar/ckpt/checkpoint_epoch_160.pth",
#             data_config=dict(
#                 class_names=['Car', 'Pedestrian'],
#                 point_feature_encoder=dict(
#                     num_point_features=4,
#                 ),
#                 depth_downsample_factor=None
#             ),
#             score_thres={
#                 "vehicle": 0.6,
#                 "walker": 0.5
#             }
#         ),
#         col_is_failure=True,
#         stuck_is_failure=True,
#         ignore_light=True,
#         visualize=dict(
#             type='birdview',
#             save_dir="eval_out_td3_without_det",
#             outputs=['show', 'gif'],
#         ),
#         wrapper=dict(
#             suite='FullTown02-v1',
#         ),
#     ),
#     policy=dict(
#         cuda=True,
#         # Pre-train model path
#         ckpt_path='',
#         model=dict(
#             obs_shape=[5, 160, 160],
#         ),
#         eval=dict(
#             evaluator=dict(
#                 render=True,
#                 transform_obs=True,
#             ),
#         ),
#     ),
#     server=[dict(
#         carla_host='localhost',
#         carla_ports=[9000, 9002, 2]
#     )],
# )
# )

from noisy_planning.config.td3_config import default_train_config
default_train_config.env.visualize = dict(
            type='birdview',
            save_dir="eval_out_td3_without_det",
            outputs=['show', 'gif'],
        )
default_train_config.policy.eval.evaluator.render = True
main_config = EasyDict(default_train_config)


def get_cls(spec):
    policy_cls, model_cls = {
        'dqn': (DQNPolicy, DQNRLModel),
        # 'ddpg': (DDPGPolicy, DDPGRLModel),
        'td3': (TD3Policy, TD3RLModel),
        # 'ppo': (PPOPolicy, PPORLModel),
        # 'sac': (SACPolicy, SACRLModel),
    }[spec]

    return policy_cls, model_cls


def main(args, seed=0):
    # args.ckpt_path = "/home/xlju/Project/Model_behavior/training_log/simple-rl_2022-02-05-09-44-39/ckpt/iteration_14000.pth.tar"
    # args.ckpt_path = "/home/xlju/Project/Model_behavior/DI-drive/noisy_planning/tst.pkl"
    args.ckpt_path = "/home/xlju/Downloads/ckpt_interrupt.pth.tar"
    cfg = main_config
    eval_epchs = 50
    enable_detection = cfg.env.enable_detector

    policy_cls, model_cls = get_cls(args.policy)
    cfg.policy = deep_merge_dicts(policy_cls.default_config(), cfg.policy)

    tcp_list = parse_carla_tcp(cfg.server)
    assert len(tcp_list) > 0, "No Carla server found!"
    host, port = tcp_list[0]

    if args.policy == 'dqn':
        carla_env = DriveEnvWrapper(DiscreteEnvWrapper(SimpleCarlaEnvNewRender(cfg.env, host, port)), cfg.env.wrapper)
    else:
        carla_env = DriveEnvWrapper(ContinuousEnvWrapper(SimpleCarlaEnvNewRender(cfg.env, host, port)), cfg.env.wrapper)
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
    parser.add_argument('-p', '--policy', default='dqn', choices=['dqn', 'ppo', 'td3', 'sac', 'ddpg'], help='RL policy')
    parser.add_argument('-c', '--ckpt-path', default=None, help='model ckpt path')
    
    args = parser.parse_args()
    main(args)
