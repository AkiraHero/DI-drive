import copy
import os
import torch
import numpy as np
from typing import Any, Dict, List, Optional

from core.eval.base_evaluator import BaseEvaluator
from ding.torch_utils.data_helper import to_tensor
from ding.utils import build_logger
from tensorboardX import SummaryWriter

from noisy_planning.detector.detection_model_wrapper import DetectionModelWrapper
from noisy_planning.detector.detection_utils import detection_process


class SingleCarlaEvaluatorWithDet(BaseEvaluator):
    """
    Carla evaluator used to evaluate a single environment. It is mainly used to visualize the
    evaluation results. It uses a environment in DI-engine form and can be rendered in the runtime.

    :Arguments:
        - cfg (Dict): Config dict
        - env (Any): Carla env, should be in DI-engine form
        - policy (Any): the policy to pe evaluated
        - exp_name (str, optional): Name of the experiments. Used to build logger. Defaults to 'default_experiment'.
        - instance_name (str, optional): Name of the evaluator. Used to build logger. Defaults to 'single_evaluator'.

    :Interfaces: reset, eval, close

    :Properties:
        - env (BaseDriveEnv): Environment used to evaluate.
        - policy (Any): Policy instance to interact with envs.
    """

    config = dict(
        # whether calling 'render' each step
        render=False,
        # whether transform obs into tensor manually
        transform_obs=False,
    )

    def __init__(
            self,
            cfg: Dict,
            env: Any,
            policy: Any,
            tb_logger: Optional['SummaryWriter'] = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'single_evaluator',
            detector=None,
            bev_obs_config=None
    ) -> None:
        super().__init__(cfg, env, policy, tb_logger=tb_logger, exp_name=exp_name, instance_name=instance_name)
        self._render = self._cfg.render
        self._transform_obs = self._cfg.transform_obs

        self._detection_model = detector
        self._bev_obs_config = bev_obs_config

    def close(self) -> None:
        """
        Close evaluator. It will close the EnvManager
        """
        self._env.close()

    def reset(self) -> None:
        pass

    def eval(self, reset_param: Dict = None) -> float:
        """
        Running one episode evaluation with provided reset params.

        :Arguments:
            - reset_param (Dict, optional): Reset parameter for environment. Defaults to None.

        :Returns:
            bool: Whether evaluation succeed.
        """
        self._policy.reset([0])
        eval_reward = 0
        success = False
        use_det_policy = None
        if "use_det_policy" in reset_param.keys():
            use_det_policy = reset_param["use_det_policy"]
        if reset_param is not None:
            obs = self._env.reset(**reset_param)
        else:
            obs = self._env.reset()

        with self._timer:
            while True:
                # insert detection
                obs_using_detection = None
                if self._detection_model is not None:
                    obs_using_detection = copy.deepcopy(obs)
                    data_list = [obs_using_detection]
                    if len(data_list):
                        self.insert_detection_result(data_list)

                if self._render:
                    self._env.render(obs_with_det=obs_using_detection)

                if self._transform_obs:
                    obs = to_tensor(obs, dtype=torch.float32)
                    if obs_using_detection:
                        obs_using_detection = to_tensor(obs_using_detection, dtype=torch.float32)
                if use_det_policy:
                    actions = self._policy.forward({0: obs})
                else:
                    actions = self._policy.forward({0: obs_using_detection})
                action = actions[0]['action']
                timestep = self._env.step(action)
                obs = timestep.obs
                if timestep.info.get('abnormal', False):
                    # If there is an abnormal timestep, reset all the related variables(including this env).
                    self._policy.reset(**reset_param)
                    action = np.array([0.0, 0.0, 0.0])
                    timestep = self._env.step(action)

                if timestep.done:
                    eval_reward = timestep.info['final_eval_reward']
                    success = timestep.info['success']
                    break

        duration = self._timer.value
        info = {
            'evaluate_time': duration,
            'eval_reward': eval_reward,
            'success': success,
        }
        print(
            "[EVALUATOR] Evaluation ends:\n{}".format(
                '\n'.join(['\t{}: {:.3f}'.format(k, v) for k, v in info.items()])
            )
        )
        print("[EVALUATOR] Evaluate done!")
        return info


    def insert_detection_result(self, data_list):
        if self._detection_model is None:
            return
        # detection
        assert isinstance(self._detection_model, DetectionModelWrapper)
        detection_process(data_list,
                          self._detection_model,
                          self._bev_obs_config,
                          keep_ini=True)