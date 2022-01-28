import sys
sys.path.append("/home/xlju/carla-0.9.11-py3.7-linux-x86_64.egg")
# system
import argparse
import numpy as np
from functools import partial
import copy
import pygame
import torch
import carla
import logging

# ding
from ding.envs import SyncSubprocessEnvManager, BaseEnvManager
from ding.policy import DQNPolicy, PPOPolicy, TD3Policy, SACPolicy, DDPGPolicy
from ding.worker import BaseLearner, SampleSerialCollector, AdvancedReplayBuffer, NaiveReplayBuffer
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn

# rl model
from demo.simple_rl.model import DQNRLModel, PPORLModel, TD3RLModel, SACRLModel, DDPGRLModel
from demo.simple_rl.env_wrapper import DiscreteEnvWrapper, ContinuousEnvWrapper

# utils
from core.utils.data_utils.bev_utils import unpack_birdview
from core.utils.others.ding_utils import compile_config
from core.utils.others.ding_utils import read_ding_config
from core.envs import SimpleCarlaEnv, BenchmarkEnvWrapper
from core.utils.others.tcp_helper import parse_carla_tcp
from core.eval import SerialEvaluator
from noisy_planning.debug_utils import TestTimer

# other module
from noisy_planning.detection_model.detection_model_wrapper import DetectionModelWrapper
from tensorboardX import SummaryWriter


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
        env_manager=SyncSubprocessEnvManager,
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


def validate_point_size(point_frm):
    point = point_frm['points']
    original_points_num = point_frm['lidar_pt_num']
    if point.shape[0] < original_points_num:
        print("[Warning] use less lidar point caused by fixed_pt_num: "
              "{}/{}.".format(point.shape[0], original_points_num))
    if point.shape[-1] == 3:
        # add a dim:
        print("[Warning] use point cloud without intensity... add intensity value: 1.0")
        point = torch.concat([point, torch.ones([point.shape[0], 1], dtype=point.dtype, device=point.device)], dim=1)
        return point

    elif point.shape[-1] == 4:
        return point
    else:
        raise NotImplementedError


def make_image(x):
    return np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)


def det_number2label(num):
    label_dict = {
        1: "vehicle",
        2: "walker"
    }
    return label_dict[num]


def draw_detection_result(pred, map_width, map_height, pixel_ahead, map_pixels_per_meter):
    # element need to fixed from map_utils.py
    # ensure the operation is same with the code inside env
    # note : the lidar coordinate must be the same with carla, which means rotation=0
    from core.utils.simulator_utils.map_utils import COLOR_BLACK, COLOR_WHITE
    walker_color = COLOR_WHITE
    vehicle_color = COLOR_WHITE
    assert pixel_ahead < map_height
    map_world_offset_x = map_width / 2.0
    map_world_offset_y = pixel_ahead

    def world2pixel(w_x, w_y):
        x = map_pixels_per_meter * w_x + map_world_offset_x
        y = map_pixels_per_meter * w_y + map_world_offset_y
        return x, y

    def trans_corners(corner_list, rot, trans_x, trans_y):
        for i in corner_list:
            i_0 = i.x * np.cos(rot) - i.y * np.sin(rot)
            i_1 = i.x * np.sin(rot) + i.y * np.cos(rot)
            # for carla coordinate
            i.y = -(i_0 + trans_x)
            i.x = i_1 + trans_y

    vehicle_surface = pygame.Surface((map_width, map_height))
    walker_surface = pygame.Surface((map_width, map_height))
    vehicle_surface.fill(COLOR_BLACK)
    walker_surface.fill(COLOR_BLACK)
    boxes = pred['pred_boxes'].cpu().numpy()
    labels = pred['pred_labels'].cpu().numpy()

    for bb, label in zip(boxes, labels):
        # bb to corner
        w_, h_, l_ = bb[3], bb[4], bb[5]
        bb_extension = carla.Vector3D(w_ / 2.0, h_ / 2.0, l_ / 2.0)
        corners = [
            carla.Location(x=-bb_extension.x, y=-bb_extension.y),
            carla.Location(x=-bb_extension.x, y=bb_extension.y),
            carla.Location(x=bb_extension.x, y=bb_extension.y),
            carla.Location(x=bb_extension.x, y=-bb_extension.y)
        ]
        trans_corners(corners, bb[6], float(bb[0]), float(bb[1]))

        corners = [world2pixel(p.x, p.y) for p in corners]
        if det_number2label(label) == "walker":
            pygame.draw.polygon(walker_surface, walker_color, corners)
        elif det_number2label(label) == 'vehicle':
            pygame.draw.polygon(vehicle_surface, vehicle_color, corners)

    return {
        "det_vehicle_surface": make_image(vehicle_surface),
        "det_walker_surface": make_image(walker_surface)
    }


def visualize_points(points):
    import open3d as od
    point_cloud = od.geometry.PointCloud()
    point_cloud.points = od.utility.Vector3dVector(points[:, 0:3].reshape(-1, 3))
    od.visualization.draw_geometries([point_cloud], width=800, height=600)


def detection_process(data_list, detector, env_cfg):
    # 1. extract batch
    batch_points = []
    for i in data_list:
        p_frm_cur = i['obs']['lidar_points']
        p_frm_nxt = i['next_obs']['lidar_points']
        batch_points.append({'points': validate_point_size(p_frm_cur)})
        batch_points.append({'points': validate_point_size(p_frm_nxt)})
        # visualize_points(p_frm_cur)

    # 2. inference
    detection_res = detector.forward(batch_points)

    # 3. distribute and draw obs on bev
    for inx, (i, j1, j2) in enumerate(zip(data_list, detection_res[::2], detection_res[1::2])):
        # i['lidar_points'] = "processed"
        i['obs'].pop('lidar_points')
        i['next_obs'].pop('lidar_points')
        # i['detection'] = {
        #     'current': j1,
        #     'next': j2
        # }
        # draw obs
        map_width = env_cfg.simulator.obs[0].size[0]
        map_height = env_cfg.simulator.obs[0].size[1]
        pixel_ahead = env_cfg.simulator.obs[0].pixels_ahead_vehicle
        pixel_per_meter = env_cfg.simulator.obs[0].pixels_per_meter

        detection_surface = draw_detection_result(j1,
                                                  map_width, map_height, pixel_ahead, pixel_per_meter)
        # substitute the 2,3 dim of bev using detection results
        vehicle_dim = detection_surface['det_vehicle_surface']
        walker_dim = detection_surface['det_walker_surface']
        vehicle_dim[vehicle_dim > 0] = 1
        walker_dim[walker_dim > 0] = 1
        device_here = i['obs']['birdview'].device
        dtype_here = i['obs']['birdview'].dtype
        vehicle_dim = torch.Tensor(vehicle_dim, device=device_here).to(dtype_here)
        walker_dim = torch.Tensor(walker_dim, device=device_here).to(dtype_here)
        i['obs']['birdview'][:, :, 2] = vehicle_dim
        i['obs']['birdview'][:, :, 3] = walker_dim


        detection_surface = draw_detection_result(j2,
                                                  map_width, map_height, pixel_ahead, pixel_per_meter)
        # substitute the 2,3 dim of bev using detection results
        vehicle_dim = detection_surface['det_vehicle_surface']
        walker_dim = detection_surface['det_walker_surface']
        vehicle_dim[vehicle_dim > 0] = 1
        walker_dim[walker_dim > 0] = 1
        vehicle_dim = torch.Tensor(vehicle_dim, device=device_here).to(dtype_here)
        walker_dim = torch.Tensor(walker_dim, device=device_here).to(dtype_here)
        i['next_obs']['birdview'][:, :, 2] = vehicle_dim
        i['next_obs']['birdview'][:, :, 3] = walker_dim



def post_processing_data_collection(data_list, detector, env_cfg):
    assert isinstance(data_list, list)

    # unpack_birdview
    unpack_birdview(data_list)

    if not env_cfg.enable_detector:
        return
    # detection
    assert isinstance(detector, DetectionModelWrapper)
    max_batch_size = env_cfg.detector.max_batch_size
    # get mini-batches
    data_list_size = len(data_list)
    pivots = [i for i in range(0, data_list_size, max_batch_size)] + [data_list_size]
    seg_num = len(pivots) - 1
    for i in range(seg_num):
        print('[DET]processing minibatch-{}...'.format(i))
        detection_process(data_list[pivots[i]: pivots[i + 1]], detector, env_cfg)


timer = TestTimer()


def main(args, seed=0):
    cfg = get_cfg(args)
    tcp_list = parse_carla_tcp(cfg.server)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    assert len(tcp_list) >= collector_env_num + evaluator_env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(
            collector_env_num + evaluator_env_num, len(tcp_list)
    )

    if args.policy == 'dqn':
        wrapped_env = wrapped_discrete_env
    else:
        wrapped_env = wrapped_continuous_env

    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper.collect, *tcp_list[i]) for i in range(collector_env_num)],
        cfg=cfg.env.manager.collect,
    )
    evaluate_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper.eval, *tcp_list[collector_env_num + i]) for i in range(evaluator_env_num)],
        cfg=cfg.env.manager.eval,
        )

    # detector
    timer.st_point("Init_detector")
    detection_model = None
    if cfg.env.enable_detector:
        print("[MAIN]Detector enabled.")
        detection_model = DetectionModelWrapper(cfg=cfg.env.detector)
    else:
        print("[MAIN]Detector not enabled.")
    timer.ed_point("Init_detector")

    # Uncomment this to add save replay when evaluation
    # evaluate_env.enable_save_replay(cfg.env.replay_path)

    collector_env.seed(seed)
    evaluate_env.seed(seed)
    set_pkg_seed(seed)

    policy_cls, model_cls = get_cls(args.policy)
    model = model_cls(**cfg.policy.model)
    policy = policy_cls(cfg.policy, model=model)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    timer.st_point("Init_collector")
    collector = SampleSerialCollector(cfg.policy.collect.collector,
                                      collector_env,
                                      policy.collect_mode,
                                      tb_logger,
                                      exp_name=cfg.exp_name)
    timer.ed_point("Init_collector")

    timer.st_point("Init_evaluator")
    evaluator = SerialEvaluator(cfg.policy.eval.evaluator,
                                evaluate_env,
                                policy.eval_mode,
                                tb_logger,
                                exp_name=cfg.exp_name)
    timer.ed_point("Init_evaluator")
    if cfg.policy.get('priority', False):
        replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    else:
        replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    if args.policy == 'dqn':
        eps_cfg = cfg.policy.other.eps
        epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    learner.call_hook('before_run')
    timer.st_point("Pre_collect")
    if args.policy != 'ppo':
        if args.policy == 'dqn':
            eps = epsilon_greedy(collector.envstep)
            new_data = collector.collect(n_sample=cfg.policy.collect.pre_sample_num,
                                         train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        else:
            new_data = collector.collect(n_sample=cfg.policy.collect.pre_sample_num,
                                         train_iter=learner.train_iter)
        timer.st_point("post_processing")
        post_processing_data_collection(new_data, detection_model, cfg.env)
        timer.ed_point("post_processing")

        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
    timer.ed_point("Pre_collect")

    while True:
        timer.st_point("whole_cycle")
        print('[MAIN]learner.train_iter={}'.format(learner.train_iter))
        if evaluator.should_eval(learner.train_iter):
            print('[EVAL]Enter evaluation.')
            timer.st_point("eval")
            stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            timer.ed_point("eval")
            if stop:
                break
        print('[MAIN]Enter collection. _default_n_sample={}'.format(collector._default_n_sample))
        timer.st_point("collect")
        if args.policy == 'dqn':
            eps = epsilon_greedy(collector.envstep)
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        else:
            new_data = collector.collect(train_iter=learner.train_iter)
        timer.ed_point("collect")

        # unpack_birdview(new_data)
        timer.st_point("post_processing")
        post_processing_data_collection(new_data, detection_model, cfg.env)
        timer.ed_point("post_processing")

        if args.policy == 'ppo':
            learner.train(new_data, collector.envstep)
        else:
            update_per_collect = len(new_data) // cfg.policy.learn.batch_size * 4
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            for i in range(update_per_collect):
                train_data = replay_buffer.sample(cfg.policy.learn.batch_size, learner.train_iter)
                if train_data is not None:
                    train_data = copy.deepcopy(train_data)
                    unpack_birdview(train_data)
                    timer.st_point("learner.train")
                    learner.train(train_data, collector.envstep)
                    timer.ed_point("learner.train")
                if args.policy == 'dqn':
                    replay_buffer.update(learner.priority_info)
        timer.ed_point("whole_cycle")
    learner.call_hook('after_run')

    collector.close()
    evaluator.close()
    learner.close()
    if args.policy != 'ppo':
        replay_buffer.close()
  
    print('finish')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple-rl train')
    parser.add_argument('-n', '--name', type=str, default='simple-rl', help='experiment name')
    parser.add_argument('-p', '--policy', default='dqn', choices=['dqn', 'ppo', 'td3', 'sac', 'ddpg'], help='RL policy')
    parser.add_argument('-d', '--ding-cfg', default=None, help='DI-engine config path')
    
    args = parser.parse_args()
    main(args)
