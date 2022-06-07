import os
import numpy as np
import torch
import torch.nn as nn

from easydict import EasyDict

from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.models import load_data_to_gpu

# DEFAULT_DETECTION_CFG = dict(
#     model_repo="openpcdet",
#     model_name="pointpillar",
#     repo_config_file="config/openpcdet_config/pointpillar_carla.yaml",
#     ckpt="",
#     data_config=dict(
#         class_names=['Car', 'Pedestrian'],
#         point_feature_encoder=dict(
#             num_point_features=4,
#         ),
#         depth_downsample_factor=None
#     ),
#     score_thres={
#         "vehicle": 0.6,
#         "walker": 0.4
#     }
# )

DEFAULT_POINTPILLAR_CFG = dict(
    model_repo="openpcdet",
    model_name="pointpillar",
    repo_config_file="config/openpcdet_config/pointpillar/pointpillar_carla.yaml",
    ckpt="detector_ckpt/pointpillar_checkpoint_epoch_120.pth",
    max_batch_size=32,
    data_config=dict(
        class_names=['Car', 'Pedestrian'],
        point_feature_encoder=dict(
            num_point_features=4,
        ),
        depth_downsample_factor=None
    ),
    score_thres={
        "vehicle": 0.6,
        "walker": 0.5
    },
)

DEFAULT_CENTERPOINT_CFG = dict(
    model_repo="openpcdet",
    model_name="centerpoint",
    repo_config_file="config/openpcdet_config/centerpoint/centerpoint.yaml",
    ckpt="detector_ckpt/centerpoint_checkpoint_epoch_120.pth",
    max_batch_size=32,
    data_config=dict(
        class_names=['Car', 'Pedestrian'],
        # point_feature_encoder=dict(
        #     num_point_features=4,
        # ),
        # depth_downsample_factor=None
    ),
    score_thres={
        "vehicle": 0.6,
        "walker": 0.5
    },
)

DEFAULT_PVRCNN_CFG = dict(
    model_repo="openpcdet",
    model_name="pvrcnn",
    repo_config_file="config/openpcdet_config/pvrcnn/pv_rcnn_carla.yaml",
    ckpt="detector_ckpt/pvrcnn_checkpoint_epoch_120.pth",
    max_batch_size=32,
    data_config=dict(
        class_names=['Car', 'Pedestrian'],
        # point_feature_encoder=dict(
        #     num_point_features=4,
        # ),
        # depth_downsample_factor=None
    ),
    score_thres={
        "vehicle": 0.6,
        "walker": 0.5
    },
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
        point_cloud_range = np.array(self.cfg.repo_config.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
        num_point_features = 4

        self.data_processor = DataProcessor(self.cfg.repo_config.DATA_CONFIG.DATA_PROCESSOR,
                                            point_cloud_range=point_cloud_range,
                                            training=False,
                                            num_point_features=num_point_features
                                            )
        self.point_feature_encoder = PointFeatureEncoder(self.cfg.repo_config.DATA_CONFIG.POINT_FEATURE_ENCODING,
                                                         point_cloud_range=point_cloud_range)
        data_info = cfg.data_config
        voxel_size = self.cfg.repo_config.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE
        data_info.update(
            dict(
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                grid_size=np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size).astype(np.int64)
            )
        )

        self.score_thres = cfg.score_thres
        self.model = build_network(model_cfg=cfg.repo_config.MODEL,
                                   num_class=len(cfg.repo_config.CLASS_NAMES),
                                   dataset=EasyDict(data_info))
        self.logger = common_utils.create_logger()
        # load checkpoint
        self.model.load_params_from_file(filename=cfg.ckpt, logger=self.logger, to_cpu=False)
        self.model.cuda()
        self.model.eval()

    def load_model_config(self):
        # load config of models
        cfg_file = self.cfg.repo_config_file
        parent_folder = os.path.dirname(os.path.dirname(__file__))
        if not os.path.isfile(self.cfg.repo_config_file):
            cfg_file = os.path.join(parent_folder, self.cfg.repo_config_file)
            assert os.path.isfile(cfg_file)
        self.cfg.repo_config = cfg_from_yaml_file(cfg_file, EasyDict())
        if not os.path.isfile(self.cfg.repo_config.DATA_CONFIG._BASE_CONFIG_):
            self.cfg.repo_config.DATA_CONFIG._BASE_CONFIG_ = \
                os.path.join(parent_folder, self.cfg.repo_config.DATA_CONFIG._BASE_CONFIG_)
            assert os.path.isfile(self.cfg.repo_config.DATA_CONFIG._BASE_CONFIG_)
        # make up the ckpt path
        if not os.path.isfile(self.cfg.ckpt):
            self.cfg.ckpt = os.path.join(parent_folder, self.cfg.ckpt)

    def preprocess_data(self, data):
        data = self.point_feature_encoder.forward(data)
        data = self.data_processor.forward(data_dict=data)
        return data

    def post_process(self, batch_dt):
        for det_res in batch_dt:
            scores = det_res['pred_scores']
            labels = det_res['pred_labels']
            valid_inx_vehicle = ((scores > self.score_thres['vehicle']) & (labels == 1)).nonzero()
            valid_inx_walker = ((scores > self.score_thres['walker']) & (labels == 2)).nonzero()
            valid_inx = torch.cat([valid_inx_vehicle, valid_inx_walker], dim=0).squeeze(1)
            det_res['pred_scores'] = det_res['pred_scores'][valid_inx, ...]
            det_res['pred_labels'] = det_res['pred_labels'][valid_inx, ...]
            det_res['pred_boxes'] = det_res['pred_boxes'][valid_inx, ...]

    def forward(self, data_dict_list):
        with torch.no_grad():
            for inx, data_dict in enumerate(data_dict_list):
                data_dict = self.preprocess_data(data_dict)
            batch_dict = DatasetTemplate.collate_batch(data_dict_list)
            load_data_to_gpu(batch_dict)
            # img = plot_pcl(data_dict_list[0]['points'][:, :3])
            # cv2.imshow("bev-pcl", img)
            # cv2.waitKey(1)
            det_res, _ = self.model(batch_dict)
            self.post_process(det_res)
            return det_res


class DetectionModelWrapper:
    support_model = ['pointpillar', 'pvrcnn', 'centerpoint']
    def __init__(self, model_name):
        if model_name not in self.support_model:
            raise NotImplementedError
        self.detection_cfg = None
        if model_name == "pointpillar":
            self.detection_cfg = EasyDict(DEFAULT_POINTPILLAR_CFG)
        elif model_name == "pvrcnn":
            self.detection_cfg = EasyDict(DEFAULT_PVRCNN_CFG)
        elif model_name == "centerpoint":
            self.detection_cfg = EasyDict(DEFAULT_CENTERPOINT_CFG)
        else:
            raise NotImplementedError
        self.model = None
        self.init_detection_model(self.detection_cfg)

    def init_detection_model(self, cfg):
        if cfg.model_repo == "openpcdet":
            self.model = OpenpcdetModel(cfg)

    def forward(self, data):
        return self.model(data)
