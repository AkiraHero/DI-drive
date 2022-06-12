import sys

import numpy as np

sys.path.append("/home/akira/Project/Model_behaviour")

from noisy_planning.detector.embedded_detection_model_base import EmbeddedDetectionModelBase

from ADModel_Pro.factory.model_factory import ModelFactory
from ADModel_Pro.utils.config.Configuration import Configuration
from ADModel_Pro.utils.postprocess import filter_pred
import torch
import logging


class PerceptionImitatorWrapper(EmbeddedDetectionModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.config_dir = cfg.repo_config_file
        self.paras_file = cfg.ckpt
        self.config = Configuration()
        self.config.load_config(self.config_dir)

        # instantiating all modules by non-singleton factory
        self.model = ModelFactory.get_model(self.config.model_config)
        paras = torch.load(self.paras_file)
        logging.error("[PerceptionImitatorWrapper] loading paras from file: " + self.paras_file)

        self.model.load_model_paras(paras)
        self.model.set_decode(True)
        self.model.set_eval()
        self.model.set_device("cuda:0")

    def load_data_to_gpu(self, data_):
        # collate
        data_ = np.stack(data_)
        return torch.from_numpy(data_).float().cuda()

    def forward(self, data_):
        with torch.no_grad():
            data_gpu = self.load_data_to_gpu(data_)
            batch_pred, batch_features = self.model(data_gpu)
            batchsize = batch_pred.shape[0]
            result_list = []
            for i in range(batchsize):
                pred, features = batch_pred[i], batch_features[i]
                pred.squeeze_(0)
                features.squeeze_(0)
                cls_pred = pred[0, ...]
                corners, scores = filter_pred(self.config, pred)
                corners = np.array(corners)
                scores = np.array(scores)
                obj_num = corners.shape[0]
                # only vehicle boxes corners, without class output
                result_list.append({'pred_labels': np.array([1] * obj_num), 'pred_corners': corners})
        return result_list
