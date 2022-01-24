from core.envs.simple_carla_env import SimpleCarlaEnv
from noisy_planning.detection_model.detection_model_wrapper import DetectionModelWrapper
import numpy as np

class CarlaEnvWithDetection(SimpleCarlaEnv):
    def __init__(self, **kwargs):
        super(CarlaEnvWithDetection, self).__init__(**kwargs)
        self.detection_model = DetectionModelWrapper()

    def get_observations(self):
        obs = super(CarlaEnvWithDetection, self).get_observations()
        data_dict = {
            "points": np.c_[obs['toplidar'], np.ones(obs['toplidar'].shape[0])]
        }
        fake_dict_list = []
        for i in range(16):
            fake_dict = {}
            fake_dict['points'] = np.random.random([60000, 4])


        det_results = self.detection_model.forward([data_dict]*16)
        obs.update({"detection_results": det_results})
        return obs