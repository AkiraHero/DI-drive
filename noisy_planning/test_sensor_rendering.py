import sys
import time

sys.path.append("/home/xlju/carla-0.9.11-py3.7-linux-x86_64.egg")


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
from core.envs import SimpleCarlaEnv
import carla
from simple_rl_train_with_detection import post_processing_data_collection
from noisy_planning.detection_model.detection_model_wrapper import DetectionModelWrapper


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



def view_surfaces(surface_dict):
    import cv2
    for k, v in surface_dict.items():
        cv2.imshow(k, v)
    cv2.waitKey(1)



if __name__ == '__main__':
    from core.utils.simulator_utils.carla_agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
    from core.utils.simulator_utils.carla_agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
    from core.utils.simulator_utils.carla_agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
    import random
    from simple_rl_train_with_detection import draw_detection_result

    parser = argparse.ArgumentParser(description='simple-rl train')
    parser.add_argument('-n', '--name', type=str, default='simple-rl', help='experiment name')
    parser.add_argument('-p', '--policy', default='dqn', choices=['dqn', 'ppo', 'td3', 'sac', 'ddpg'], help='RL policy')
    parser.add_argument('-d', '--ding-cfg', default=None, help='DI-engine config path')
    args = parser.parse_args()
    cfg = get_cfg(args)

    env_cfg = cfg.env
    host = "localhost"
    port = 9000
    detector = DetectionModelWrapper()
    env = SimpleCarlaEnv(cfg=env_cfg, host=host, port=port)
    env.reset()
    tot_target_reached = 0
    world = env._simulator._world
    spawn_points = world.get_map().get_spawn_points()
    print("The planner id is:", env.hero_player.id)
    planner_id = env.hero_player.id
    agent = BehaviorAgent(env.hero_player, behavior="normal")
    num_min_waypoints = 21
    loop = True

    random.shuffle(spawn_points)

    if spawn_points[0].location != agent.vehicle.get_location():
        destination = spawn_points[0].location
    else:
        destination = spawn_points[1].location

    agent.set_destination(agent.vehicle.get_location(), destination, clean=True)

    try:
        while True:
            agent.update_information(world)
            obs = env.get_observations()

            view_surfaces(obs['birdview_initial_dict'])

            point_cloud = np.c_[obs['toplidar']['points'], np.ones(obs['toplidar']['points'].shape[0])]
            d_st = time.time()
            detection_res = detector.forward([{'points': point_cloud}])
            d_ed = time.time()
            print("detection time:", d_ed - d_st)

            map_width = cfg.env.simulator.obs[0].size[0]
            map_height = cfg.env.simulator.obs[0].size[1]
            pixel_ahead = cfg.env.simulator.obs[0].pixels_ahead_vehicle
            pixel_per_meter = cfg.env.simulator.obs[0].pixels_per_meter

            detection_surface = draw_detection_result(detection_res[0],
                                                      map_width, map_height, pixel_ahead, pixel_per_meter)
            view_surfaces(detection_surface)
            # Set new destination when target has been reached
            if len(agent.get_local_planner().waypoints_queue) < num_min_waypoints and loop:
                agent.reroute(spawn_points)

            elif len(agent.get_local_planner().waypoints_queue) == 0 and not loop:
                print("Target reached, mission accomplished...")
                break

            speed_limit = env.hero_player.get_speed_limit()
            agent.get_local_planner().set_speed(speed_limit)

            control = agent.run_step()
            print("detected:{}, id={}, throttle={},steer={}".format(detection_res[0]['pred_boxes'].shape[0],
                                                                    planner_id,
                                                                    control.throttle,
                                                                    control.steer))
            env.hero_player.apply_control(control)
            world.tick()
            env._simulator._bev_wrapper.tick()
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)