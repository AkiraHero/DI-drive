import logging

import torch.nn as nn
from easydict import EasyDict
import yaml
import numpy as np
import torch
import time



# point_cloud_range = np.array([0, -40, -3, 70.4, 40, 1]).astype(np.float32)
# voxel_size = np.array([0.05, 0.05, 0.1])
# grid_size = np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size).astype(np.int64)
DATA_INFO_KITTI_PVRCNN = EasyDict(dict(
    class_names=['Car', 'Pedestrian', 'Cyclist'],
    point_feature_encoder=dict(
        num_point_features=4, #carla produce only xyz without intensity
    ),
    grid_size=None,
    point_cloud_range=None,
    voxel_size=None,
    depth_downsample_factor=None
))

DATA_INFO_KITTI_POINTPILLAR = EasyDict(dict(
    class_names=['Car', 'Pedestrian', 'Cyclist'],
    point_feature_encoder=dict(
        num_point_features=4, #carla produce only xyz without intensity
    ),
    grid_size=None,
    point_cloud_range=None,
    voxel_size=None,
    depth_downsample_factor=None
))


# DEFAULT_DETECTION_CFG = dict(
#     model_repo="openpcdet",
#     model_name="pvrcnn",
#     repo_config_file="/home/xlju/Project/Model_behavior/DI-drive/noisy_planning/config/openpcdet_config/pv_rcnn.yaml",
#     # ckpt="/home/xlju/Downloads/pointpillar_7728.pth",
#     ckpt="/home/xlju/Downloads/pv_rcnn_8369.pth",
#     data_config=DATA_INFO_KITTI_PVRCNN
# )

DEFAULT_DETECTION_CFG = dict(
    model_repo="openpcdet",
    model_name="pointpillar",
    repo_config_file="/home/xlju/Project/Model_behavior/DI-drive/noisy_planning/config/openpcdet_config/pointpillar.yaml",
    ckpt="/home/xlju/Downloads/pointpillar_7728.pth",
    data_config=DATA_INFO_KITTI_POINTPILLAR
)



class EmbeddedDetectionModelBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg.model_name

    def preprocess_data(self, data):
        raise NotImplementedError

    def load_model_config(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError


class OpenpcdetModel(EmbeddedDetectionModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.load_model_config()
        from pcdet.models import build_network
        from pcdet.utils import common_utils
        from pcdet.datasets.processor.data_processor import DataProcessor
        from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
        point_cloud_range = np.array(self.cfg.repo_config.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
        num_point_features = 4

        self.data_processor = DataProcessor(self.cfg.repo_config.DATA_CONFIG.DATA_PROCESSOR,
                                            point_cloud_range=point_cloud_range,
                                            training=False,
                                            num_point_features=num_point_features
                                            )
        self.point_feature_encoder = PointFeatureEncoder(self.cfg.repo_config.DATA_CONFIG.POINT_FEATURE_ENCODING,
                                                         point_cloud_range=point_cloud_range)
        DATA_INFO = cfg.data_config
        DATA_INFO.voxel_size = self.cfg.repo_config.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE
        DATA_INFO.point_cloud_range = point_cloud_range
        DATA_INFO.grid_size = np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) / DATA_INFO.voxel_size).astype(np.int64)


        self.model = build_network(model_cfg=cfg.repo_config.MODEL,
                                   num_class=len(cfg.repo_config.CLASS_NAMES),
                                   dataset=EasyDict(DATA_INFO))
        self.logger = common_utils.create_logger()
        # load checkpoint
        self.model.load_params_from_file(filename=cfg.ckpt, logger=self.logger, to_cpu=False)
        self.model.cuda()
        self.model.eval()

    def load_model_config(self):
        from pcdet.config import cfg_from_yaml_file
        # load config of models
        self.cfg.repo_config = cfg_from_yaml_file(self.cfg.repo_config_file, EasyDict())


    def preprocess_data(self, data):
        data = self.point_feature_encoder.forward(data)
        data = self.data_processor.forward(data_dict=data)
        return data


    def forward(self, data_dict_list):
        with torch.no_grad():

            for inx, data_dict in enumerate(data_dict_list):
                data_dict = self.preprocess_data(data_dict)

            from pcdet.datasets.dataset import DatasetTemplate
            from pcdet.models import load_data_to_gpu
            batch_dict = DatasetTemplate.collate_batch(data_dict_list)
            load_data_to_gpu(batch_dict)
            for i in range(10):
                st3 = time.time()
                det_res = self.model(batch_dict)
                st4 = time.time()
                print("total inference time", st4 - st3)
            return det_res

    def merge2batch(self):
        pass




class DetectionModelWrapper:
    def __init__(self, cfg=None):
        if cfg is None:
            self.detection_cfg = EasyDict(DEFAULT_DETECTION_CFG)
        else:
            self.detection_cfg = cfg
        self.model = None
        self.init_detection_model(self.detection_cfg)

    def init_detection_model(self, cfg):
        if cfg.model_repo == "openpcdet":
            self.model = OpenpcdetModel(cfg)

    def forward(self, data):
        return self.model(data)
