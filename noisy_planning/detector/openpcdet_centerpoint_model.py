import os
import numpy as np
import torch

from easydict import EasyDict
from noisy_planning.detector.embedded_detection_model_base import EmbeddedDetectionModelBase

# for pcdet
from pcdet_centerpoint.models import build_network
from pcdet_centerpoint.utils import common_utils
from pcdet_centerpoint.datasets.processor.data_processor import DataProcessor
from pcdet_centerpoint.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pcdet_centerpoint.config import cfg_from_yaml_file
from pcdet_centerpoint.datasets.dataset import DatasetTemplate
from pcdet_centerpoint.models import load_data_to_gpu


class OpenpcdetCenterpointModel(EmbeddedDetectionModelBase):
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


