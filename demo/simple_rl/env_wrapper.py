import torch
import numpy as np
from typing import Dict, Any, List
import math
import gym

from core.envs import BaseDriveEnv
from ding.torch_utils.data_helper import to_ndarray

import cv2
def plot_pcl(points):
    points = points[:, :3]
    pixel_size = 0.08
    cx = 800
    cy = 600
    img_width = 1600
    img_height = 1200
    img = np.zeros((img_height, img_width, 3), np.uint8)

    pt_num = points.shape[0]
    valid_pt_num = 0
    for i in range(pt_num):
        x = points[i, 0]
        y = points[i, 1]
        # to adapt to carla ordinate: exchange x and y
        py = int(x / pixel_size + cx)
        px = int(y / pixel_size + cy)
        range_ = (x ** 2 + y ** 2) ** 0.5
        theta = np.arctan2(y, x)
        if abs(theta) > np.pi / 2.0 or abs(y) < 0.0001:
            continue

        if py < 0 or py >= img_height or px < 0 or px >= img_width:
            continue
        valid_pt_num += 1
        cv2.circle(img, (px, py), 1, (255, 0, 255), 1)

    py = int(cx)
    px = int(cy)
    cv2.circle(img, (px, py), 2, (0, 0, 255), 2)
    print("valid=", valid_pt_num)
    return cv2.flip(img, 0)


def process_line_lidar(points):
    points = points[:, :3]
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    inx,  = ((abs(azimuth) < np.pi / 2.0) & (abs(points[:, 1]) > 0.0001)).nonzero()
    points = points[inx, :]
    azimuth = np.round(azimuth[inx] / np.pi * 180.0).astype(int)

    ranges = (points[:, 0] ** 2 + points[:, 1] ** 2) ** 0.5
    max_dis = 20.0
    slots = {i: max_dis for i in range(-90, 91)}
    for i, j in zip(azimuth, ranges):
        if j < slots[i]:
            slots[i] = j
    laser_obs = []
    for i in range(-90, 91):
        laser_obs.append(slots[i])
    return np.array(laser_obs).reshape(-1, 1)


DEFAULT_ACC_LIST = [
    (0, 1),
    (0.25, 0),
    (0.75, 0),
]
DEFAULT_STEER_LIST = [
    -0.8,
    -0.5,
    -0.2,
    0,
    0.2,
    0.5,
    0.8,
]
# birdview dimension, dim
# bev_data['road'],1
# bev_data['lane'],1
# bev_data['traffic'],3
# bev_data['vehicle'],1
# bev_data['pedestrian'],1
# bev_data['hero'],1
# bev_data['route'],1


def get_obs_out(obs):
    # print('speed', obs['speed'])
    # print('velocity_local', obs['velocity_local'])
    # print('acceleration_local', obs['acceleration_local'])
    # print('heading_diff', obs['heading_diff'])
    # print('last_steer', obs['last_steer'])
    # print('collide_wall', obs['collide_wall'])
    # print('collide_obj', obs['collide_obj'])
    # print('waypoint_curvature', obs['waypoint_curvature'])

    laser_obs = process_line_lidar(obs['linelidar'])
    obs_out = {
        # 'birdview': obs['birdview'][..., [0, 1, 5, 6, 8, 7]],
        # 'speed': (obs['speed'] / 25).astype(np.float32),

        'velocity_local': np.array(obs['velocity_local'] / 20.0).reshape(-1, 1),
        'acceleration_local': np.array(obs['acceleration_local'] / 20.0).reshape(-1, 1),
        'heading_diff': np.array(obs['heading_diff'] / 60.0).reshape(-1, 1),
        'last_steer': np.array(np.clip(obs['last_steer'], -1.0, 1.0)).reshape(-1, 1),
        'collide_wall': np.array(obs['collide_wall']).reshape(-1, 1),
        'collide_obj': np.array(obs['collide_obj']).reshape(-1, 1),
        'way_curvature': np.array(obs['waypoint_curvature'] / 10.0).reshape(-1, 1),
        'laser_obs': laser_obs / 20.0,
        # 'bev_obj': obs['birdview'][..., 5:6] + obs['birdview'][..., 6:7],
        # 'bev_road': obs['birdview'][..., 0:1] + obs['birdview'][..., 1:2],


#        'bev_elements': obs['birdview_initial_dict']
    }
    if 'camera_vis' in obs.keys():
        obs_out['camera_vis'] = obs['camera_vis']

    if 'toplidar' in obs.keys():
        obs_out['lidar_points'] = obs['toplidar']
    return obs_out


class DiscreteEnvWrapper(gym.Wrapper):

    def __init__(self, env: BaseDriveEnv, acc_list: List = None, steer_list: List = None) -> None:
        super().__init__(env)
        self._acc_list = acc_list
        if acc_list is None:
            self._acc_list = DEFAULT_ACC_LIST
        self._steer_list = steer_list
        if steer_list is None:
            self._steer_list = DEFAULT_STEER_LIST

    def reset(self, *args, **kwargs) -> Any:
        obs = super().reset(*args, **kwargs)
        obs_out = get_obs_out(obs)
        return obs_out

    def step(self, id):
        if isinstance(id, torch.Tensor):
            id = id.item()
        id = np.squeeze(id)
        assert id < len(self._acc_list) * len(self._steer_list), (id, len(self._acc_list) * len(self._steer_list))
        mod_value = len(self._acc_list)
        acc = self._acc_list[id % mod_value]
        steer = self._steer_list[id // mod_value]
        action = {
            'steer': steer,
            'throttle': acc[0],
            'brake': acc[1],
        }
        obs, reward, done, info = super().step(action)
        obs_out = get_obs_out(obs)
        return obs_out, reward, done, info

    def __repr__(self) -> str:
        return repr(self.env)


class MultiDiscreteEnvWrapper(gym.Wrapper):

    def __init__(self, env: BaseDriveEnv, acc_list: List = None, steer_list: List = None) -> None:
        super().__init__(env)
        self._acc_list = acc_list
        if acc_list is None:
            self._acc_list = DEFAULT_ACC_LIST
        self._steer_list = steer_list
        if steer_list is None:
            self._steer_list = DEFAULT_STEER_LIST

    def reset(self, *args, **kwargs) -> Any:
        obs = super().reset(*args, **kwargs)
        obs_out = get_obs_out(obs)
        return obs_out

    def step(self, action_ids):
        action_ids = to_ndarray(action_ids, dtype=int)
        action_ids = np.squeeze(action_ids)
        acc_id = action_ids[0]
        steer_id = action_ids[1]
        assert acc_id < len(self._acc_list), (acc_id, len(self._acc_list))
        assert steer_id < len(self._steer_list), (steer_id, len(self._steer_list))
        acc = self._acc_list[acc_id]
        steer = self._steer_list[steer_id]
        action = {
            'steer': steer,
            'throttle': acc[0],
            'brake': acc[1],
        }
        obs, reward, done, info = super().step(action)
        obs_out = get_obs_out(obs)
        return obs_out, reward, done, info

    def __repr__(self) -> str:
        return repr(self.env)


class ContinuousEnvWrapper(gym.Wrapper):

    def reset(self, *args, **kwargs) -> Any:
        obs = super().reset(*args, **kwargs)
        obs_out = get_obs_out(obs)
        return obs_out

    def step(self, action):
        action = to_ndarray(action)
        action = np.squeeze(action)
        steer = action[0]
        acc = action[1]
        if acc > 0:
            throttle, brake = acc, 0
        else:
            throttle, brake = 0, -acc

        action = {
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
        }
        obs, reward, done, info = super().step(action)
        obs_out = get_obs_out(obs)
        return obs_out, reward, done, info

    def __repr__(self) -> str:
        return repr(self.env)
