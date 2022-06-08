from easydict import EasyDict
from noisy_planning.detector.openpcdet_model import OpenpcdetModel
from noisy_planning.detector.openpcdet_centerpoint_model import OpenpcdetCenterpointModel


default_max_batchsize = 16
DEFAULT_POINTPILLAR_CFG = dict(
    model_repo="openpcdet",
    model_name="pointpillar",
    repo_config_file="config/openpcdet_config/pointpillar/pointpillar_carla.yaml",
    ckpt="detector_ckpt/pointpillar_checkpoint_epoch_120.pth",
    max_batch_size=default_max_batchsize,
    data_config=dict(
        class_names=['Car', 'Pedestrian'],
        point_feature_encoder=dict(
            num_point_features=4,
        ),
        depth_downsample_factor=None
    ),
    score_thres={
        "vehicle": 0.7,
        "walker": 0.5
    },
)

DEFAULT_CENTERPOINT_CFG = dict(
    model_repo="openpcdet_centerpoint",
    model_name="centerpoint",
    repo_config_file="config/openpcdet_config/centerpoint/centerpoint.yaml",
    ckpt="detector_ckpt/centerpoint_checkpoint_epoch_120.pth",
    max_batch_size=default_max_batchsize,
    data_config=dict(
        class_names=['Car', 'Pedestrian'],
        point_feature_encoder=dict(
            num_point_features=4,
        ),
        depth_downsample_factor=None
    ),
    score_thres={
        "vehicle": 0.7,
        "walker": 0.5
    },
)

DEFAULT_PVRCNN_CFG = dict(
    model_repo="openpcdet",
    model_name="pvrcnn",
    repo_config_file="config/openpcdet_config/pvrcnn/pv_rcnn_carla.yaml",
    ckpt="detector_ckpt/pvrcnn_checkpoint_epoch_120.pth",
    max_batch_size=default_max_batchsize,
    data_config=dict(
        class_names=['Car', 'Pedestrian'],
        point_feature_encoder=dict(
            num_point_features=4,
        ),
        depth_downsample_factor=None
    ),
    score_thres={
        "vehicle": 0.7,
        "walker": 0.5
    },
)



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
        elif cfg.model_repo == 'openpcdet_centerpoint':
            self.model = OpenpcdetCenterpointModel(cfg)
        else:
            raise NotImplementedError

    def forward(self, data):
        return self.model(data)
