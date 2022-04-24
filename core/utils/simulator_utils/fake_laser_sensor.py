import numpy as np
import carla



def get_bourndary_points(corners, edge_pts=50):
    assert len(corners) == 4
    bound_points_x = []
    bound_points_y = []
    for i in range(4):
        st = corners[i]
        ed = corners[(i + 1) % 4]
        x = [st[0], ed[0]]
        y = [st[1], ed[1]]
        if st[0] > ed[0]:
            x = x[::-1]
            y = y[::-1]
        x_new = np.linspace(x[0], x[1], edge_pts, endpoint=False)
        y_new = np.interp(x_new, x, y)
        bound_points_x.append(x_new)
        bound_points_y.append(y_new)
    bound_points_x = np.concatenate(bound_points_x)
    bound_points_y = np.concatenate(bound_points_y)
    return bound_points_x, bound_points_y



def get_neibor_obj_laser_reading(actors_with_transforms,  hero_id, range_scope=30.0):
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

    azimuth_resolution = 1.0 #deg
    azimuth_range_min = -90.0
    azimuth_range_max = 90.0
    azimuth_slot_num = int((azimuth_range_max - azimuth_range_min) // azimuth_resolution) + 1
    ranges_slot = [range_scope for i in range(azimuth_slot_num)]
    point_slot = [(0.0, 0.0) for i in range(azimuth_slot_num)]


    vehicles = []
    traffic_lights = []
    walkers = []

    hero_transform = None
    hero_bb = None
    for actor_with_transform in actors_with_transforms:
        actor = actor_with_transform[0]
        if actor.id == hero_id:
            hero_transform = actor_with_transform[1]
            hero_bb = actor.bounding_box.extent
            continue
        if 'vehicle' in actor.type_id:
            vehicles.append(actor_with_transform)
        elif 'traffic_light' in actor.type_id:
            traffic_lights.append(actor_with_transform)
        elif 'walker' in actor.type_id:
            walkers.append(actor_with_transform)

    hero_x = hero_transform.location.x
    hero_y = hero_transform.location.y
    # !!!!!!!!!!!!!!!!! pay attention to the hero yaw!!!!!!!
    hero_yaw = -hero_transform.rotation.yaw
    hero_yaw_rad = np.deg2rad(hero_yaw)

    box_pt_xs = []
    box_pt_ys = []
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
        distance, front = get_corner_nearest_dis(corners, hero_x, hero_y, hero_yaw_rad)
        if distance < range_scope and front:
            in_range = True

        if in_range:
            # feature1: relative corners: 8 dim normalized
            relative_corners = []
            for i in corners:
                dx, dy = i.x - hero_x, i.y - hero_y
                # trans by hero_yaw
                dx_n = dx * np.cos(hero_yaw_rad) - dy * np.sin(hero_yaw_rad)
                dy_n = dx * np.sin(hero_yaw_rad) + dy * np.cos(hero_yaw_rad)
                relative_corners.append([dx_n, dy_n])
            box_pt_x, box_pt_y = get_bourndary_points(relative_corners)
            box_pt_xs.append(box_pt_x)
            box_pt_ys.append(box_pt_y)
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
    return point_slot, ranges_slot



if __name__ ==  '__main__':
    import matplotlib.pyplot as plt
    import time
    points = [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1]
    ]
    st = time.time()
    new_x, new_y = get_bourndary_points(points)
    ed = time.time()
    print("cost:", ed - st)
    plt.plot(new_x, new_y, '.')
    plt.show()



    pass






