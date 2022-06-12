from easydict import EasyDict
from noisy_planning.detector.openpcdet_model import OpenpcdetModel
from noisy_planning.detector.openpcdet_centerpoint_model import OpenpcdetCenterpointModel
from noisy_planning.detector.perception_imitator_wrapper import PerceptionImitatorWrapper

default_max_batchsize = 32
DEFAULT_POINTPILLAR_CFG = dict(
    model_repo="openpcdet",
    model_name="pointpillar",
    repo_config_file="config/openpcdet_config/pointpillar/pointpillar_carla.yaml",
    ckpt="detector_ckpt/pointpillar_checkpoint_epoch_146.pth",
    max_batch_size=default_max_batchsize,
    data_config=dict(
        class_names=['Car', 'Pedestrian'],
        point_feature_encoder=dict(
            num_point_features=4,
        ),
        depth_downsample_factor=None
    ),
    score_thres={
        "vehicle": 0.73,
        "walker": 0.33
    },
)

DEFAULT_CENTERPOINT_CFG = dict(
    model_repo="openpcdet_centerpoint",
    model_name="centerpoint",
    repo_config_file="config/openpcdet_config/centerpoint/centerpoint.yaml",
    ckpt="detector_ckpt/centerpoint_checkpoint_epoch_156.pth",
    max_batch_size=default_max_batchsize,
    data_config=dict(
        class_names=['Car', 'Pedestrian'],
        point_feature_encoder=dict(
            num_point_features=4,
        ),
        depth_downsample_factor=None
    ),
    score_thres={
        "vehicle": 0.75,
        "walker": 0.45
    },
)

DEFAULT_PVRCNN_CFG = dict(
    model_repo="openpcdet",
    model_name="pvrcnn",
    repo_config_file="config/openpcdet_config/pvrcnn/pv_rcnn_carla.yaml",
    ckpt="detector_ckpt/pvrcnn_checkpoint_epoch_171.pth",
    max_batch_size=default_max_batchsize,
    data_config=dict(
        class_names=['Car', 'Pedestrian'],
        point_feature_encoder=dict(
            num_point_features=4,
        ),
        depth_downsample_factor=None
    ),
    score_thres={
        "vehicle": 0.78,
        "walker": 0.50
    },
)

BASELINE_POINTPILLAR_CFG = dict(
    model_repo="baseline",
    model_name="b-pointpillar",
    repo_config_file="config/admodelpro_config/sample_baseline",
    ckpt="detector_ckpt/baseline_pp.pt",
    max_batch_size=default_max_batchsize,
)


BASELINE_PVRCNN_CFG = dict(
    model_repo="baseline",
    model_name="b-pvrcnn",
    repo_config_file="config/admodelpro_config/sample_baseline",
    ckpt="detector_ckpt/baseline_pvrcnn.pt",
    max_batch_size=default_max_batchsize,
)


BASELINE_CENTERPOINT_CFG = dict(
    model_repo="baseline",
    model_name="b-centerpoint",
    repo_config_file="config/admodelpro_config/sample_baseline",
    ckpt="detector_ckpt/baseline_cp.pt",
    max_batch_size=default_max_batchsize,
)




class DetectionModelWrapper:
    support_model = ['pointpillar', 'pvrcnn', 'centerpoint', 'b-pointpillar', 'b-pvrcnn', 'b-centerpoint']
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name not in self.support_model:
            raise NotImplementedError
        self.detection_cfg = None
        if model_name == "pointpillar":
            self.detection_cfg = EasyDict(DEFAULT_POINTPILLAR_CFG)
        elif model_name == "pvrcnn":
            self.detection_cfg = EasyDict(DEFAULT_PVRCNN_CFG)
        elif model_name == "centerpoint":
            self.detection_cfg = EasyDict(DEFAULT_CENTERPOINT_CFG)
        elif model_name == 'b-pointpillar':
            self.detection_cfg = EasyDict(BASELINE_POINTPILLAR_CFG)
        elif model_name == 'b-pvrcnn':
            self.detection_cfg = EasyDict(BASELINE_PVRCNN_CFG)
        elif model_name == 'b-centerpoint':
            self.detection_cfg = EasyDict(BASELINE_CENTERPOINT_CFG)
        else:
            raise NotImplementedError
        self.model = None
        self.init_detection_model(self.detection_cfg)

    def get_model_name(self):
        return self.model_name

    def init_detection_model(self, cfg):
        if cfg.model_repo == "openpcdet":
            self.model = OpenpcdetModel(cfg)
        elif cfg.model_repo == 'openpcdet_centerpoint':
            self.model = OpenpcdetCenterpointModel(cfg)
        elif cfg.model_repo == 'baseline':
            self.model = PerceptionImitatorWrapper(cfg)
        else:
            raise NotImplementedError

    def forward(self, data):
        return self.model(data)
