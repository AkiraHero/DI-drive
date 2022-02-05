# system
import copy
import time

from collections import namedtuple
from easydict import EasyDict
from typing import Optional, Tuple

# ding
from ding.worker import BaseLearner
from ding.torch_utils import auto_checkpoint

# utils
from core.utils.data_utils.bev_utils import unpack_birdview

# other module
from noisy_planning.detector.detection_model_wrapper import DetectionModelWrapper
from tensorboardX import SummaryWriter
from noisy_planning.detector.detection_utils import detection_process


class CarlaLearner(BaseLearner):
    def __init__(self,
                 cfg: EasyDict,
                 policy: namedtuple = None,
                 tb_logger: Optional['SummaryWriter'] = None,  # noqa
                 dist_info: Tuple[int, int] = None,
                 exp_name: Optional[str] = 'default_experiment',
                 instance_name: Optional[str] = 'learner',
                 ):
        exp_name_with_time = exp_name + '_' + time.strftime("%Y-%m-%d-%H-%M-%S")
        super(CarlaLearner, self).__init__(cfg.learner, policy, tb_logger, dist_info, exp_name_with_time, instance_name)
        self._batch_size = cfg.batch_size
        self._collector = None
        self._collector_config = None
        self._replay_buffer = None
        self._evaluator = None
        self._detection_model = None
        self._detection_batch_size = 16
        self._bev_obs_config = None
        self._epsilon_greedy = None
        self._policy_name = None

    def set_policy_name(self, n):
        self._policy_name = n

    def set_epsilon_greedy(self, eps_func):
        self._epsilon_greedy = eps_func

    def set_detection_model(self, mdl, batch_size, obs_config):
        self._detection_model = mdl
        self._detection_batch_size = batch_size
        self._bev_obs_config = obs_config

    def set_collector(self, collector, collector_config):
        self._collector_config = collector_config
        self._collector = collector

    def set_evaluator(self, evaluator):
        self._evaluator = evaluator

    def set_replay_buffer(self, buf):
        self._replay_buffer = buf

    def check_element(self):
        assert self._collector is not None
        assert self._collector_config is not None
        assert self._replay_buffer is not None
        assert self._policy_name is not None

    def post_processing_data_collection(self, data_list):
        assert isinstance(data_list, list)

        # unpack_birdview
        unpack_birdview(data_list)

        if self._detection_model is None:
            return
        # detection
        assert isinstance(self._detection_model, DetectionModelWrapper)
        max_batch_size = self._detection_batch_size

        # get unique datalist
        data_list_dict = {id(i['obs']): i['obs'] for i in data_list}
        data_list_dict.update({id(i['next_obs']): i['next_obs'] for i in data_list})
        obs_list = [i for i in data_list_dict.values()]

        # get mini-batches
        obs_list_size = len(obs_list)
        pivots = [i for i in range(0, obs_list_size, max_batch_size)] + [obs_list_size]
        seg_num = len(pivots) - 1
        for i in range(seg_num):
            self.logger.info('[DET]processing minibatch-{}...'.format(i))
            detection_process(obs_list[pivots[i]: pivots[i + 1]], self._detection_model, self._bev_obs_config)

        # debug: check detection process
        for i in data_list:
            assert i['obs']['birdview_using_detection'] is True
            assert i['next_obs']['birdview_using_detection'] is True

    @auto_checkpoint
    def start(self) -> None:
        self.check_element()
        self._end_flag = False
        self._learner_done = False
        # before run hook
        self.call_hook('before_run')

        # learning process
        if self._policy_name != 'ppo':
            if self._policy_name == 'dqn':
                eps = self._epsilon_greedy(self._collector.envstep)
                new_data = self._collector.collect(n_sample=self._collector_config.pre_sample_num,
                                                   train_iter=self.train_iter, policy_kwargs={'eps': eps})
            else:
                new_data = self._collector.collect(n_sample=self._collector_config.pre_sample_num,
                                                   train_iter=self.train_iter)
            self.post_processing_data_collection(new_data)
            self._replay_buffer.push(new_data, cur_collector_envstep=self._collector.envstep)

        while True:
            self.logger.info('learner.train_iter={}'.format(self.train_iter))
            try:
                if self._evaluator and self._evaluator.should_eval(self.train_iter):
                    self.logger.info('[EVAL]Enter evaluation.')
                    stop, rate = self._evaluator.eval(self.save_checkpoint, self.train_iter, self._collector.envstep)
                    if stop:
                        break
            except Exception as e:
                self.logger.error("Fail to do evaluation...")
                self.logger.error(str(e))
            self.logger.info('Enter collection. _default_n_sample={}'.format(self._collector._default_n_sample))

            if self._policy_name == 'dqn':
                eps = self._epsilon_greedy(self._collector.envstep)
                new_data = self._collector.collect(train_iter=self.train_iter, policy_kwargs={'eps': eps})
            else:
                new_data = self._collector.collect(train_iter=self.train_iter)

            # unpack_birdview(new_data)
            self.post_processing_data_collection(new_data)

            if self._policy_name == 'ppo':
                self.train(new_data, self._collector.envstep)
            else:
                update_per_collect = len(new_data) // self._batch_size * 4
                self._replay_buffer.push(new_data, cur_collector_envstep=self._collector.envstep)
                for i in range(update_per_collect):
                    train_data = self._replay_buffer.sample(self._batch_size, self.train_iter)
                    if train_data is not None:
                        train_data = copy.deepcopy(train_data)
                        unpack_birdview(train_data)
                        self.train(train_data, self._collector.envstep)
                    if self._policy_name == 'dqn':
                        self._replay_buffer.update(self.priority_info)
            self.logger.info(
                "........................................cycle end.........................................")

        self._learner_done = True
        # after run hook
        self.call_hook('after_run')