import logging

import torch.nn as nn
from easydict import EasyDict
import yaml
import numpy as np

DEFAULT_DETECTION_CFG = dict(
    model_repo="openpcdet",
    model_name="pvrcnn",
    repo_config_file="/home/xlju/Project/Model_behavior/DI-drive/noisy_planning/config/openpcdet_config/pv_rcnn.yaml",
    data_process_config_file="/home/xlju/Project/Model_behavior/DI-drive/noisy_planning/config/openpcdet_config/kitti_dataset.yaml",
    ckpt="/home/xlju/Downloads/pv_rcnn_8369.pth"
)


point_cloud_range = np.array([0, -40, -3, 70.4, 40, 1]).astype(np.float32)
voxel_size = np.array([0.05, 0.05, 0.1])
grid_size = np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size).astype(np.int64)
DATA_INFO_KITTI_PVRCNN = dict(
    class_names=['Car', 'Pedestrian', 'Cyclist'],
    point_feature_encoder=dict(
        num_point_features=3, #carla produce only xyz without intensity
    ),
    grid_size=grid_size,
    point_cloud_range=point_cloud_range,
    voxel_size=voxel_size,
    depth_downsample_factor=None
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

        self.data_processor = DataProcessor(self.cfg.repo_config.dp_config.DATA_PROCESSOR,
                                            point_cloud_range=point_cloud_range,
                                            training=False,
                                            num_point_features=EasyDict(DATA_INFO_KITTI_PVRCNN).point_feature_encoder.num_point_features
                                            )
        self.point_feature_encoder = PointFeatureEncoder(self.cfg.repo_config.dp_config.POINT_FEATURE_ENCODING,
                                                         point_cloud_range=point_cloud_range)
        self.model = build_network(model_cfg=cfg.repo_config.MODEL,
                                   num_class=len(cfg.repo_config.CLASS_NAMES),
                                   dataset=EasyDict(DATA_INFO_KITTI_PVRCNN))
        self.logger = common_utils.create_logger()
        # load checkpoint
        self.model.load_params_from_file(filename=cfg.ckpt, logger=self.logger, to_cpu=False)
        self.model.cuda()
        self.model.eval()

    def load_model_config(self):
        # load config of models
        new_config = None
        with open(self.cfg.repo_config_file, 'r') as f:
            try:
                new_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                new_config = yaml.load(f)

        with open(self.cfg.data_process_config_file, 'r') as f:
            dp_config = None
            try:
                dp_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                dp_config = yaml.load(f)
            new_config['dp_config'] = dp_config
        self.cfg.repo_config = new_config


    def preprocess_data(self, data):
        data = self.point_feature_encoder.forward(data)
        data = self.data_processor.forward(data_dict=data)
        return data


    def forward(self, data_dict):
        data_dict = self.preprocess_data(data_dict)
        # todo
        # here need to get a batch then put onto gpu...
        # data should be summarized to a batch.. not like this..
        # detection model should be independent with env... because it need batch as input, so that it need to be
        # a plugin, env interface must be changed, at least in its child class..
        return self.model(data_dict)






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
        data_dict = {
            "points": data['toplidar']
        }
        return self.model(data_dict)
