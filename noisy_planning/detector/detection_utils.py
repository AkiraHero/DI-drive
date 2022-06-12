import torch
import carla
import pygame
import numpy as np
import copy
from core.utils.simulator_utils.fake_laser_sensor import get_bourndary_points

def validate_point_size(point_frm):
    point = point_frm
    # point = point_frm['points']
    # original_points_num = point_frm['lidar_pt_num']
    # if point.shape[0] < original_points_num:
    #     print("[Warning] use less lidar point caused by fixed_pt_num: "
    #           "{}/{}.".format(point.shape[0], original_points_num))
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


def get_corner_nearest_dis(corners, hero_x, hero_y, hero_yaw_rad):
    min_dist = np.inf
    min_dist_inx = 0
    front_ = False
    for inx, i_ in enumerate(corners):
        dx_, dy_ = i_.x - hero_x, i_.y - hero_y
        dist = np.sqrt(dy_ ** 2 + dx_ ** 2)
        if dist < min_dist:
            min_dist = dist
            min_dist_inx = inx
    i_ = corners[min_dist_inx]
    dx_, dy_ = i_.x - hero_x, i_.y - hero_y
    dx_n_ = dx_ * np.cos(hero_yaw_rad) - dy_ * np.sin(hero_yaw_rad)
    if dx_n_ > 0:
        front_ = True
    return min_dist, front_


def draw_detection_result(pred, map_width, map_height, pixel_ahead, map_pixels_per_meter, model_name):
    # element need to fixed from map_utils.py
    # ensure the operation is same with the code inside env
    # note : the lidar coordinate must be the same with carla, which means rotation=0


    # openpcd detection coordinate: front x right y
    # adas_pro baseline : front x left y

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
            i.y = -(i_0 + trans_x)  # rear y
            i.x = i_1 + trans_y    # right x

    vehicle_surface = pygame.Surface((map_width, map_height))
    walker_surface = pygame.Surface((map_width, map_height))
    vehicle_surface.fill(COLOR_BLACK)
    walker_surface.fill(COLOR_BLACK)

    boxes = None
    pred_corners_list = None
    labels = None
    if 'b-' in model_name:
        pred_corners_list = pred['pred_corners']
        labels = np.array(pred['pred_labels'])
    else:
        boxes = pred['pred_boxes'].cpu().numpy()
        labels = pred['pred_labels'].cpu().numpy()

    '''
    for fake laser reading
    '''
    box_pt_xs = []
    box_pt_ys = []
    range_scope = 30.0
    azimuth_resolution = 1.0 #deg
    azimuth_range_min = -90.0
    azimuth_range_max = 90.0
    azimuth_slot_num = int((azimuth_range_max - azimuth_range_min) // azimuth_resolution) + 1
    ranges_slot = [range_scope for i in range(azimuth_slot_num)]
    point_slot = [(0., 0.) for i in range(azimuth_slot_num)]
    '''
    =================
    '''

    num_obj = len(labels)
    for inx in range(num_obj):
        if 'b-' in model_name:
            label = labels[inx]
            pred_corners = pred_corners_list[inx]
            corners = [
                carla.Location(x=float(-pred_corners[2][1]), y=float(-pred_corners[2][0])),  # left-rear
                carla.Location(x=float(-pred_corners[3][1]), y=float(-pred_corners[3][0])),  # right-rear
                carla.Location(x=float(-pred_corners[0][1]), y=float(-pred_corners[0][0])),  # right-front
                carla.Location(x=float(-pred_corners[1][1]), y=float(-pred_corners[1][0]))   # left-front
            ]
        else:
            bb = boxes[inx]
            label = labels[inx]
            # bb to corner
            w_, h_, l_ = bb[3], bb[4], bb[5]
            bb_extension = carla.Vector3D(w_ / 2.0, h_ / 2.0, l_ / 2.0)
            corners = [
                carla.Location(x=-bb_extension.x, y=-bb_extension.y),
                carla.Location(x=-bb_extension.x, y=bb_extension.y),
                carla.Location(x=bb_extension.x, y=bb_extension.y),
                carla.Location(x=bb_extension.x, y=-bb_extension.y)
            ]
            trans_corners(corners, bb[6], float(bb[0]), float(bb[1]))  # front x, right y -> rear y, right x

        '''
        for fake laser reading
        '''
        # get any kind of relative position is ok
        in_range = False
        distance, front = get_corner_nearest_dis(corners, 0, 0, np.pi / 2)
        if distance < range_scope and front:
            in_range = True
        yaw_delta = np.pi / 2.0
        if in_range:
            # feature1: relative corners: 8 dim normalized
            relative_corners = []
            for i in corners:
                # illness coordinate.... need to tease apart: bev/laser and all....
                dx = -i.y
                dy = i.x
                relative_corners.append([dx, dy])
            box_pt_x, box_pt_y = get_bourndary_points(relative_corners)
            box_pt_xs.append(box_pt_x)
            box_pt_ys.append(box_pt_y)
        '''
        =================
        '''


        corners = [world2pixel(p.x, p.y) for p in corners]
        if det_number2label(label) == "walker":
            pygame.draw.polygon(walker_surface, walker_color, corners)
        elif det_number2label(label) == 'vehicle':
            pygame.draw.polygon(vehicle_surface, vehicle_color, corners)

    '''
    for fake laser reading
    '''
    if len(box_pt_xs):
        box_pt_xs = np.concatenate(box_pt_xs)
        box_pt_ys = np.concatenate(box_pt_ys)
        ranges = np.sqrt(np.square(box_pt_xs) + np.square(box_pt_ys))
        thetas = np.rad2deg(np.arctan2(box_pt_ys, box_pt_xs))
        for inx, (r_, th_) in enumerate(zip(ranges, thetas)):
            if th_ < azimuth_range_min:
                continue
            if th_ > azimuth_range_max:
                continue
            slot_inx = round((th_ - azimuth_range_min) // azimuth_resolution)
            if slot_inx < len(ranges_slot) and ranges_slot[slot_inx] > r_:
                ranges_slot[slot_inx] = r_
                point_slot[slot_inx] = (box_pt_xs[inx], box_pt_ys[inx])
    point_slot = np.array([i for i in point_slot if i is not None]).reshape(-1, 2)
    ranges_slot = np.array(ranges_slot)
    '''
    =================
    '''

    return {
        "det_vehicle_surface": make_image(vehicle_surface),
        "det_walker_surface": make_image(walker_surface),
        "det_fake_laser_points": point_slot,
        "det_fake_laser_ranges": ranges_slot
    }


def visualize_points(points):
    import open3d as od
    point_cloud = od.geometry.PointCloud()
    point_cloud.points = od.utility.Vector3dVector(points[:, 0:3].reshape(-1, 3))
    od.visualization.draw_geometries([point_cloud], width=800, height=600)


def check_obs_id(data_list):
    cur_ids = [id(i['obs']) for i in data_list]
    next_ids = [id(i['next_obs']) for i in data_list]
    print("cur_ids:", cur_ids)
    print("next_ids:", next_ids)


def detection_process(data_list, detector, env_bev_obs_cfg, model_name, keep_ini=False):
    # 1. extract batch
    batch_data = []
    for i in data_list:
        if 'baseline_data' in i.keys():
            batch_data.append(i['baseline_data'])
        else:
            p_frm = i['lidar_points']
            batch_data.append({'points': validate_point_size(p_frm)})
            # visualize_points(p_frm_cur)


    # 2. inference
    detection_res = detector.forward(batch_data)

    # 3. distribute and draw obs on bev
    for inx, (i, j) in enumerate(zip(data_list, detection_res)):
        # i['lidar_points'] = "processed"
        # print("i['obs']:", i['obs'].keys())
        i.pop('lidar_points')
        # i['detection'] = {
        #     'current': j1,
        #     'next': j2
        # }
        # draw obs
        map_width = env_bev_obs_cfg.size[0]
        map_height = env_bev_obs_cfg.size[1]
        pixel_ahead = env_bev_obs_cfg.pixels_ahead_vehicle
        pixel_per_meter = env_bev_obs_cfg.pixels_per_meter

        detection_surface = draw_detection_result(j,
                                                  map_width, map_height, pixel_ahead, pixel_per_meter, model_name)
        # substitute the 2,3 dim of bev using detection results
        vehicle_dim = detection_surface['det_vehicle_surface']
        walker_dim = detection_surface['det_walker_surface']
        vehicle_dim[vehicle_dim > 0] = 1
        walker_dim[walker_dim > 0] = 1
        # device_here = i['birdview'].device
        # dtype_here = i['birdview'].dtype
        # vehicle_dim = torch.Tensor(vehicle_dim, device=device_here).to(dtype_here)
        # walker_dim = torch.Tensor(walker_dim, device=device_here).to(dtype_here)
        if keep_ini:
            i['ini_vehicle_dim'] = copy.deepcopy(i['birdview'][:, :, 2])
            i['ini_walker_dim'] = copy.deepcopy(i['birdview'][:, :, 3])
        i['gt_vehicle'] = copy.deepcopy(i['birdview'][:, :, 2])
        i['gt_pedestrian'] = copy.deepcopy(i['birdview'][:, :, 3])

        i['birdview'][:, :, 2] = vehicle_dim
        i['birdview'][:, :, 3] = walker_dim
        i['detected'] = 1.0 # avoid batch collate error, use float

        '''
        for fake laser reading
        '''
        i['gt_fake_laser_pts'] = copy.deepcopy(i['fake_laser_pts'])
        i['gt_fake_line_laser_ranges'] = copy.deepcopy(i['fake_line_laser_ranges'])
        i['fake_laser_pts'] = detection_surface['det_fake_laser_points']
        i['fake_line_laser_ranges'] = detection_surface['det_fake_laser_ranges'].reshape(-1, 1) / 30.0
        '''
        =================
        '''

