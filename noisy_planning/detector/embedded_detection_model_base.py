import torch.nn as nn

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