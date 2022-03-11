import torch
import carla
import pygame
import numpy as np
from collections import OrderedDict

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


def check_obs_id(data_list):
    cur_ids = [id(i['obs']) for i in data_list]
    next_ids = [id(i['next_obs']) for i in data_list]
    print("cur_ids:", cur_ids)
    print("next_ids:", next_ids)


def detection_process(data_list, detector, env_bev_obs_cfg, keep_ini=False):
    # 1. extract batch
    batch_points = []
    unique_ids = OrderedDict()
    for i in data_list:
        if 'lidar_points' in i.keys():
            unique_ids[id(i)] = i
        else:
            assert i['detected'] == 1.0 

    for i in unique_ids.values():
        p_frm = i['lidar_points']
        batch_points.append({'points': validate_point_size(p_frm)})
        # visualize_points(p_frm_cur)

    # 2. inference
    detection_res = detector.forward(batch_points)

    # 3. distribute and draw obs on bev
    for inx, (i, j) in enumerate(zip(unique_ids.values(), detection_res)):
        i.pop('lidar_points')
        map_width = env_bev_obs_cfg.size[0]
        map_height = env_bev_obs_cfg.size[1]
        pixel_ahead = env_bev_obs_cfg.pixels_ahead_vehicle
        pixel_per_meter = env_bev_obs_cfg.pixels_per_meter

        detection_surface = draw_detection_result(j,
                                                  map_width, map_height, pixel_ahead, pixel_per_meter)
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
            i['ini_vehicle_dim'] = i['birdview'][:, :, 2]
            i['ini_walker_dim'] = i['birdview'][:, :, 3]
        if i['birdview'].shape[-1] == 3:
            i['birdview'][:, :, 1] = vehicle_dim + walker_dim
        else:
            i['birdview'][:, :, 2] = vehicle_dim
            i['birdview'][:, :, 3] = walker_dim
        i['detected'] = 1.0 # avoid batch collate error, use float

    # 4. checkall
    for i in data_list:
        assert i['detected'] == 1.0
        assert 'lidar_points' not in i.keys()
