import logging

import numpy as np
import cv2
import struct
import time

# import open3d as od
# vis = od.visualization.Visualizer()
# vis.create_window()
#
# def visualize_points(points):
#     point_cloud = od.geometry.PointCloud()
#     point_cloud.points = od.utility.Vector3dVector(points[:, 0:3].reshape(-1, 3))
#     od.visualization.draw_geometries([point_cloud], width=800, height=600)


def plot_pcl(points):
    points = points[:, :3]
    pixel_size = 0.08
    cx = 800
    cy = 600
    img_width = 1600
    img_height = 1200
    img = np.zeros((img_height, img_width, 3), np.uint8)

    pt_num = points.shape[0]
    for i in range(pt_num):
        x = points[i, 0]
        y = points[i, 1]
        # to adapt to carla ordinate: exchange x and y
        py = int(x / pixel_size + cx)
        px = int(y / pixel_size + cy)

        if py < 0 or py >= img_height or px < 0 or px >= img_width:
            continue
        cv2.circle(img, (px, py), 1, (255, 0, 255), 1)
    return cv2.flip(img, 0)


def read_kitti_bin(file_name):
    with open(file_name, 'rb') as f:
        buf = f.read()
        num = len(buf) // 4
        numbers = struct.unpack("f"*num, buf)
        return numbers


class TestTimer:
    def __init__(self):
        self.logger = logging.getLogger("TestTimer")
        if len(self.logger.handlers) == 0:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        self.time_dict = {}

    def st_point(self, tag):
        self.time_dict[tag] = time.time()

    def ed_point(self, tag):
        if tag not in self.time_dict.keys():
            return
        cur = time.time()
        time_diff = cur - self.time_dict[tag]
        self.logger.info("[{}] {}s".format(tag, time_diff))
        self.time_dict.pop(tag)

def generate_general_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(thread)0x- %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    return logger


