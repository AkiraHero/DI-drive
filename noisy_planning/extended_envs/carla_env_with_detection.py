from core.envs.simple_carla_env import SimpleCarlaEnv
from noisy_planning.detection_model.detection_model_wrapper import DetectionModelWrapper


class CarlaEnvWithDetection(SimpleCarlaEnv):
    def __init__(self, **kwargs):
        super(CarlaEnvWithDetection, self).__init__(**kwargs)
        self.detection_model = DetectionModelWrapper()

    def get_observations(self):
        obs = super(CarlaEnvWithDetection, self).get_observations()
        det_results = self.detection_model.forward(obs)
        obs.update({"detection_results": det_results})
        return obs