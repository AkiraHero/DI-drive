# system
import copy
import pickle
import time

from collections import namedtuple
from easydict import EasyDict
from typing import Optional, Tuple

# ding
from ding.worker import BaseLearner
from ding.torch_utils import auto_checkpoint
from ding.worker.learner.learner_hook import LearnerHook, register_learner_hook
from ding.utils.file_helper import read_file

# utils
from core.utils.data_utils.bev_utils import unpack_birdview

# other module
from tensorboardX import SummaryWriter

from noisy_planning.utils.debug_utils import TestTimer
timer = TestTimer()


class LoadCkptHookWithoutIter(LearnerHook):
    """
    Overview:
        Hook to load checkpoint
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: EasyDict = EasyDict(), **kwargs) -> None:
        """
        Overview:
            Init LoadCkptHook.
        Arguments:
            - ext_args (:obj:`EasyDict`): Extended arguments. Use ``ext_args.freq`` to set ``load_ckpt_freq``.
        """
        super().__init__(*args, **kwargs)
        self._load_path = ext_args['load_path']

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            Load checkpoint to learner. Checkpoint info includes policy state_dict and iter num.
        Arguments:
            - engine (:obj:`BaseLearner`): The BaseLearner to load checkpoint to.
        """
        path = self._load_path
        if path == '':  # not load
            return
        state_dict = read_file(path)
        if 'last_iter' in state_dict:
            last_iter = state_dict.pop('last_iter')
            engine.info("last iter={}, but we would not load it for replay buffer logic.".format(str(last_iter)))
            # engine.last_iter.update(last_iter)
        engine.policy.load_state_dict(state_dict)
        engine.info('{} load ckpt in {}'.format(engine.instance_name, path))


class CarlaLearner(BaseLearner):
    def __init__(self,
                 cfg: EasyDict,
                 policy: namedtuple = None,
                 tb_logger: Optional['SummaryWriter'] = None,  # noqa
                 dist_info: Tuple[int, int] = None,
                 exp_name: Optional[str] = 'default_experiment',
                 instance_name: Optional[str] = 'learner',
                 ):
        register_learner_hook("load_ckpt_without_iter", LoadCkptHookWithoutIter)
        super(CarlaLearner, self).__init__(cfg.learner, policy, tb_logger, dist_info, exp_name, instance_name)
        self._batch_size = cfg.batch_size
        self._collector = None
        self._collector_config = None
        self._replay_buffer = None
        self._evaluator = None
        self._epsilon_greedy = None
        self._policy_name = None

    def set_policy_name(self, n):
        self._policy_name = n

    def set_epsilon_greedy(self, eps_func):
        self._epsilon_greedy = eps_func

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

    @staticmethod
    def check_batch_data(data_list):
        pass

    def post_processing_data_collection(self, data_list):
        assert isinstance(data_list, list)
        # unpack_birdview
        unpack_birdview(data_list)

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
            timer.st_point("Current Loop")
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

            timer.st_point("New Data Collection")
            if self._policy_name == 'dqn':
                eps = self._epsilon_greedy(self._collector.envstep)
                new_data = self._collector.collect(train_iter=self.train_iter, policy_kwargs={'eps': eps})
            else:
                new_data = self._collector.collect(train_iter=self.train_iter)
            timer.ed_point("New Data Collection")

            timer.st_point("New Data PostProcessing")
            # unpack_birdview(new_data)
            self.post_processing_data_collection(new_data)
            timer.ed_point("New Data PostProcessing")

            timer.st_point("Sample and Training")
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
                        try:
                            self.train(train_data, self._collector.envstep)
                        except Exception as e:
                            with open("debug_data_pickle.pp", 'wb') as f:
                                pickle.dump(train_data, f)
                            raise e
                    if self._policy_name == 'dqn':
                        self._replay_buffer.update(self.priority_info)
            timer.ed_point("Sample and Training")
            timer.ed_point("Current Loop")
            self.logger.info(
                "........................................cycle end.........................................")

        self._learner_done = True
        # after run hook
        self.call_hook('after_run')

