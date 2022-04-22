'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:carla utils for DI-drive
'''
import logging

import numpy as np
import math

import carla

BACKGROUND = [0, 0, 0]


def control_to_signal(control):
    for k, v in control.items():
        if k in ['steer', 'throttle', 'brake', 'manual_gear_shift', 'gear']:
            control[k] = float(v)
    control_signal = carla.VehicleControl()
    control_signal.steer = control['steer'] if 'steer' in control else 0.0
    control_signal.throttle = control['throttle'] if 'throttle' in control else 0.0
    control_signal.brake = control['brake'] if 'brake' in control else 0.0
    if 'manual_gear_shift' in control:
        control_signal.manual_gear_shift = control['manual_gear_shift']
    if 'gear' in control:
        control_signal.gear = control['gear']
    return control_signal


def signal_to_control(signal):
    control = {
        'steer': signal.steer,
        'throttle': signal.throttle,
        'brake': signal.brake,
        'manual_gear_shift': signal.manual_gear_shift,
        'gear': signal.gear,
    }
    return control


def compute_angle(vec1, vec2):
    arr1 = np.array([vec1.x, vec1.y, vec1.z])
    arr2 = np.array([vec2.x, vec2.y, vec2.z])
    cosangle = arr1.dot(arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
    angle = min(np.pi / 2, np.abs(np.arccos(cosangle)))
    return angle


def get_birdview(bev_data):
    birdview = [
        bev_data['road'],
        bev_data['lane'],
        bev_data['traffic'],
        bev_data['vehicle'],
        bev_data['pedestrian'],
        bev_data['hero'],
        bev_data['route'],
    ]
    birdview = [x if x.ndim == 3 else x[..., None] for x in birdview]
    birdview = np.concatenate(birdview, 2)
    birdview[birdview > 0] = 1

    return birdview


def visualize_birdview(birdview):
    """
    0 road
    1 lane
    2 red light
    3 yellow light
    4 green light
    5 vehicle
    6 pedestrian
    7 hero
    8 route
    """
    bev_render_colors = [
        (85, 87, 83),
        (211, 215, 207),
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 0),
        (252, 175, 62),
        (173, 74, 168),
        (32, 74, 207),
        (41, 239, 41),
    ]
    h, w, c = birdview.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[...] = BACKGROUND
    index_list = []
    for i in [0, 1, 2, 3, 4, 8, 5, 6, 7]:
        if i < c:
            index_list.append(i)

    for i in index_list:
        canvas[birdview[:, :, i] > 0.5] = bev_render_colors[i]

    return canvas


def calculate_speed(actor):
    """
    Method to calculate the velocity of a actor
    """
    speed_squared = actor.get_velocity().x ** 2
    speed_squared += actor.get_velocity().y ** 2
    speed_squared += actor.get_velocity().z ** 2
    return math.sqrt(speed_squared)


def convert_waypoint_to_transform(waypoint_vec):
    transform_vec = []
    for waypoint_tuple in waypoint_vec:
        transform_vec.append((waypoint_tuple[0].transform, waypoint_tuple[1]))

    return transform_vec


def lane_mid_distance(waypoint_location_list, location):
    num = min(len(waypoint_location_list), 5)  # use next 4 lines for lane mid esitimation
    if num <= 1:
        return 0
    #waypoint_location_list = 0.99 * waypoint_location_list[:-1] + 0.01 * waypoint_location_list[1:]
    start = waypoint_location_list[:num - 1, :2]  # start points of the 4 lines
    end = waypoint_location_list[1:num, :2]  # end   points of the 4 lines
    rotate = np.array([[0.0, -1.0], [1.0, 0.0]])
    normal_vec = (end - start).dot(rotate)
    loc = location[None, :2]
    dis = np.min(np.abs(np.sum(normal_vec * (loc - start), axis=1)) / np.sqrt(np.sum(normal_vec * normal_vec, axis=1)))
    return dis


def get_lane_dis(waypoints, x, y):
  """
  Calculate distance from (x, y) to waypoints.
  :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
  :param x: x position of vehicle
  :param y: y position of vehicle
  :return: a tuple of the distance and the closest waypoint orientation
  """
  dis_min = 1000
  waypt = waypoints[0]
  for pt in waypoints:
    d = np.sqrt((x-pt[0])**2 + (y-pt[1])**2)
    if d < dis_min:
      dis_min = d
      waypt=pt
  vec = np.array([x - waypt[0], y - waypt[1]])
  lv = np.linalg.norm(np.array(vec))
  w = np.array([np.cos(waypt[2]/180*np.pi), np.sin(waypt[2]/180*np.pi)])
  cross = np.cross(w, vec/lv)
  dis = - lv * cross
  return dis, w


def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


def get_lane_marker_dis(way_points, x, y):
    """
    Args:
        way_points: the front waypoints on the route
        x:  current location.x
        y: current location.y

    Returns:
        most_left_lanemarker_dis, most_right_lanemarker_dis
    """
    most_left_lanemarker_loc = []
    most_right_lanemarker_loc = []
    for cur_way_point in way_points:
        most_left_lane = cur_way_point
        most_right_lane = cur_way_point

        # get left
        while most_left_lane.lane_change == carla.LaneChange.Both or most_left_lane.lane_change == carla.LaneChange.Left:
            most_left_lane = most_left_lane.get_left_lane()

        # get right
        while most_right_lane.lane_change == carla.LaneChange.Both or most_right_lane.lane_change == carla.LaneChange.Right:
            most_right_lane = most_right_lane.get_right_lane()

        road_left_side = lateral_shift(most_left_lane.transform, -most_left_lane.lane_width * 0.5)
        road_right_side = lateral_shift(most_right_lane.transform, most_right_lane.lane_width * 0.5)
        most_left_lanemarker_loc.append(road_left_side)
        most_right_lanemarker_loc.append(road_right_side)
    most_left_lanemarker_dis = [((i.x - x) ** 2 + (i.y - y) ** 2) ** 0.5 for i in most_left_lanemarker_loc]
    most_right_lanemarker_dis = [((i.x - x) ** 2 + (i.y - y) ** 2) ** 0.5 for i in most_right_lanemarker_loc]
    return most_left_lanemarker_dis, most_right_lanemarker_dis


def get_neibor_obj_bev_box(actors_with_transforms, hero_x, hero_y, hero_yaw, hero_id, range_scope=30.0):
    # output 20 obj
    vehicles = []
    traffic_lights = []
    walkers = []
    hero_yaw_rad = np.deg2rad(hero_yaw)
    for actor_with_transform in actors_with_transforms:
        actor = actor_with_transform[0]
        if actor.id == hero_id:
            continue
        if 'vehicle' in actor.type_id:
            vehicles.append(actor_with_transform)
        elif 'traffic_light' in actor.type_id:
            traffic_lights.append(actor_with_transform)
        elif 'walker' in actor.type_id:
            walkers.append(actor_with_transform)
    valid_relative_corners = []
    valid_num = 0
    for v in vehicles:
        # Compute bounding box points under global coordinate
        bb = v[0].bounding_box.extent
        corners = [
            carla.Location(x=-bb.x, y=-bb.y),
            carla.Location(x=-bb.x, y=bb.y),
            carla.Location(x=bb.x, y=bb.y),
            carla.Location(x=bb.x, y=-bb.y)
        ]
        v[1].transform(corners)
        # get any kind of relative position is ok
        in_range = False
        relative_corners = []
        for i in corners:
            dx, dy = i.x - hero_x, i.y - hero_y
            # trans by hero_yaw
            dx_n = dx * np.cos(hero_yaw_rad) - dy * np.sin(hero_yaw_rad)
            dy_n = dx * np.sin(hero_yaw_rad) + dy * np.cos(hero_yaw_rad)
            if np.sqrt(dy ** 2 + dx ** 2) < range_scope:
                in_range = True
            relative_corners += [dx_n, dy_n]
        if in_range:
            assert len(relative_corners) == 8
            valid_relative_corners += relative_corners
            valid_num += 1
    if valid_num < 20:
        valid_relative_corners += [0] * (20 - valid_num) * 8
    else:
        valid_relative_corners = valid_relative_corners[:8 * 20]

    return valid_relative_corners, valid_num





