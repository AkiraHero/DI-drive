import copy
import sys

import cv2

sys.path.append("/home/xlju/carla-0.9.11-py3.7-linux-x86_64.egg")

import carla
import pygame
from core.utils.simulator_utils.map_utils import MapImage

def save_spw(filename):
    import pickle
    pts = []
    for i in spawn_points:
        pt = [i.location.x, i.location.y, i.rotation.yaw]
        pts.append(pt)
    with open(filename, 'wb') as f:
        pickle.dump(pts, f)

def mouse_click(ent, x, y, flags=None, param=None):
    if ent == cv2.EVENT_LBUTTONDBLCLK:
        dis_list = []
        for i in spawn_points:
            map_loc = map_image.world_to_pixel(i.location)
            dis_ = ((map_loc[0] - x/mapscale) ** 2 + (map_loc[1] - y/mapscale) ** 2) ** 0.5
            dis_list.append(dis_)
        inx = dis_list.index(min(dis_list))
        print("{},{:.2f},{:.2f},{:.2f}，dis={:.2f}".format(inx, spawn_points[inx].location.x, spawn_points[inx].location.y, spawn_points[inx].rotation.yaw, min(dis_list)))

if __name__ == '__main__':
    pygame.init()
    width = 400
    height = 300
    display = pygame.display.set_mode((width, height), 0, 32)
    pygame.display.flip()

    # set up carla client
    client = carla.Client("localhost", 9000)
    tm_port = 12345
    town = 'TOWN01'
    sync = True
    delta_seconds = 0.05
    tm = client.get_trafficmanager(tm_port)
    tm.set_global_distance_to_leading_vehicle(2.0)
    tm.set_hybrid_physics_mode(True)
    world = client.load_world(town)
    map = world.get_map()

    settings = world.get_settings()
    if settings.synchronous_mode is not sync:
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = delta_seconds
        world.apply_settings(settings)

    spawn_points = map.get_spawn_points()

    map_image = MapImage(world, map, 10)
    patch_map = map_image.big_map_surface.convert()
    # for i in spawn_points:
    #     map_loc = map_image.world_to_pixel(i.location)
    #     pygame.draw.circle(patch_map, (255, 0, 0), map_loc, 30)
    mapscale = 1.000001
    map_image.scale_map(mapscale)
    width = int(map_image.big_map_surface.get_width() * map_image.scale)
    lane_surface = pygame.transform.smoothscale(map_image.big_lane_surface, (width, width))
    lane_img = pygame.surfarray.array3d(lane_surface)
    road_img = pygame.surfarray.array3d(map_image.surface)
    patch_map = pygame.transform.smoothscale(patch_map, (width, width))
    for i in spawn_points:
        map_loc = map_image.world_to_pixel(i.location)
        pygame.draw.circle(patch_map, (255, 0, 0), map_loc, 30)
    spawn_points_img = pygame.surfarray.array3d(patch_map)
    #
    # cv2.imshow("road_map", road_img)
    # cv2.imshow("lane_map", lane_img  )
    lane_inx = lane_img>0
    spawn_points_img[lane_inx] = 0
    cv2.imshow("spawn_points_img", spawn_points_img)
    cv2.imwrite("town1_spw.png", spawn_points_img)
    save_spw("spw_pts.pickle")
    cv2.setMouseCallback("spawn_points_img", mouse_click)
    cv2.waitKey(-1)

    sync = False
    settings = world.get_settings()
    if settings.synchronous_mode is not sync:
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = delta_seconds
        world.apply_settings(settings)
    pass