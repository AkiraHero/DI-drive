from typing import Any, List, Dict, Callable
from easydict import EasyDict
import os
import traceback
import logging
import time
import numpy as np
from collections import namedtuple

from ding.envs import SyncSubprocessEnvManager
from ding.envs.env_manager.subprocess_env_manager import is_abnormal_timestep, ShmBufferContainer
from ding.envs.env_manager.base_env_manager import retry_wrapper, EnvState
from ding.utils import ENV_MANAGER_REGISTRY
from ding.utils import PropagatingThread
from ding.envs.env.base_env import BaseEnvTimestep

from noisy_planning.utils.debug_utils import generate_general_logger, TestTimer
from noisy_planning.detector.detection_model_wrapper import DetectionModelWrapper
from noisy_planning.detector.detection_utils import detection_process

'''
compatable with Ding v0.2.1
'''
timer = TestTimer()



class ObsStack:
    def __init__(self, maxlen: int, fill=True):
        assert maxlen > 0
        self._max_len = maxlen
        self._fill = fill
        self._stack = []
        self._push_cnt = 0

    def push(self, data):
        self._push_cnt += 1
        # ## debug
        # data['birdview'][0][0][0] = self._push_cnt
        # ##
        if self._fill and len(self._stack) == 0:
            self._stack = [data] * self._max_len
        else:
            self._stack.append(data)
        cur_len = len(self._stack)
        if cur_len > self._max_len:
            self._stack = self._stack[-self._max_len:]
        # ids = [id(i) for i in self._stack]
        # print("[debug-push]" + str(ids))
        # pass

    def clear(self):
        self._stack.clear()
        self._push_cnt = 0

    def get_data(self):
        data_res = self._stack[::-1] # return reversed data
        # ids = [id(i) for i in data_res]
        # print("[debug-get]" + str(ids))
        return data_res

@ENV_MANAGER_REGISTRY.register('carla_subprocess')
class CarlaSyncSubprocessEnvManager(SyncSubprocessEnvManager):
    def __init__(
            self,
            env_fn: List[Callable],
            cfg: EasyDict = EasyDict({}),
            env_ports=None,
            detector=None,
            detection_max_batch_size=None,
            bev_obs_config=None,
            obs_stack_len=1,
    ) -> None:
        super(CarlaSyncSubprocessEnvManager, self).__init__(env_fn, cfg)
        self._env_reset_try_num = {}
        self.logger = generate_general_logger("[Manager]")
        self.logger.setLevel(logging.INFO)

        self._detection_model = detector
        self._detection_batch_size = detection_max_batch_size
        self._bev_obs_config = bev_obs_config
        assert len(env_ports) == len(env_fn)
        self._env_ports = env_ports
        # overload
        self._obs_stack_len = 5

    def _create_state(self) -> None:
        r"""
        Overview:
            Fork/spawn sub-processes(Call ``_create_env_subprocess``) and create pipes to transfer the data.
        """
        self._env_episode_count = {env_id: 0 for env_id in range(self.env_num)}
        self._ready_obs = {env_id: ObsStack(maxlen=self._obs_stack_len) for env_id in range(self.env_num)}
        self._env_ref = self._env_fn[0]()
        self._reset_param = {i: {} for i in range(self.env_num)}
        if self._shared_memory:
            obs_space = self._env_ref.info().obs_space
            shape = obs_space.shape
            dtype = np.dtype(obs_space.value['dtype']) if obs_space.value is not None else np.dtype(np.float32)
            self._obs_buffers = {env_id: ShmBufferContainer(dtype, shape) for env_id in range(self.env_num)}
        else:
            self._obs_buffers = {env_id: None for env_id in range(self.env_num)}
        self._pipe_parents, self._pipe_children = {}, {}
        self._subprocesses = {}
        for env_id in range(self.env_num):
            self._create_env_subprocess(env_id)
        self._waiting_env = {'step': set()}
        self._closed = False

    def _check_data(self, data: Dict, close: bool = False) -> bool:
        exceptions = []
        abnormal_env_ids = []
        for i, d in data.items():
            if isinstance(d, Exception):
                self._env_states[i] = EnvState.ERROR
                exceptions.append(d)
                abnormal_env_ids.append(i)
        # when receiving env Exception, env manager will safely close and raise this Exception to caller
        if len(exceptions) > 0:
            self.logger.error("There are {} env return exception in data..".format(len(exceptions)))
            self.logger.error("The exceptions are:")
            for env_id, ex_ in zip(abnormal_env_ids, exceptions):
                self.logger.error("env_id={} give exceptions as :{}".format(env_id, str(ex_)))

            if not close:
                self.logger.error("We will not close the abnormal envs...")
                for env_id in abnormal_env_ids:
                    empty_data_info = dict(
                        comm_err=True,
                        abnormal=True,
                    )
                    empty_data = BaseEnvTimestep(None, None, None, empty_data_info)
                    data[env_id] = empty_data
                return False
            if close:
                self.logger.error("We will close the abnormal envs...")
                self.close()
                raise exceptions[0]
        return True

    def step(self, actions: Dict[int, Any]) -> Dict[int, namedtuple]:
        """
        Overview:
            Step all environments. Reset an env if done.
        Arguments:
            - actions (:obj:`Dict[int, Any]`): {env_id: action}
        Returns:
            - timesteps (:obj:`Dict[int, namedtuple]`): {env_id: timestep}. Timestep is a \
                ``BaseEnvTimestep`` tuple with observation, reward, done, env_info.
        Example:
            >>>     actions_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
            >>>     timesteps = env_manager.step(actions_dict):
            >>>     for env_id, timestep in timesteps.items():
            >>>         pass

        .. note::

            - The env_id that appears in ``actions`` will also be returned in ``timesteps``.
            - Each environment is run by a subprocess separately. Once an environment is done, it is reset immediately.
        """
        self._check_closed()
        env_ids = list(actions.keys())
        assert all([self._env_states[env_id] == EnvState.RUN for env_id in env_ids]
                   ), 'current env state are: {}, please check whether the requested env is in reset or done'.format(
            {env_id: self._env_states[env_id]
             for env_id in env_ids}
        )

        ready_conn = []
        ready_env_ids = []
        broken_env_err_dict = {}
        for env_id, act in actions.items():
            try:
                self._pipe_parents[env_id].send(['step', [act], {}])
                ready_conn.append(self._pipe_parents[env_id])
                ready_env_ids.append(env_id)
            except Exception as e:
                broken_env_err_dict[env_id] = '\nEnv {} step [Exception] in sending signal:\n'.format(env_id) \
                                              + ''.join(traceback.format_tb(e.__traceback__)) + repr(e)

        # ===     This part is different from async one.     ===
        # === Because operate in this way is more efficient. ===
        timesteps = {}
        for env_id, p in zip(ready_env_ids, ready_conn):
            try:
                self.logger.debug("try to receive step info from env_id={}".format(env_id))
                if self._pipe_parents[env_id].poll(self._connect_timeout):
                    data_recv_ = p.recv()
                    timesteps[env_id] = data_recv_
                    self.logger.debug("[Done] try to receive step info from env_id={}".format(env_id))
                else:
                    broken_env_err_dict[env_id] = "\nEnv {} step [Timeout] in receiving signal:\n".format(env_id) \
                                                  + "timeout with max connection timeout={}".format(
                        self._connect_timeout)
            except Exception as e:
                broken_env_err_dict[env_id] = '\nEnv {} step [Exception] in receiving signal:\n'.format(env_id) \
                                              + ''.join(traceback.format_tb(e.__traceback__)) + repr(e)

        # fill data for abnormal env
        if len(broken_env_err_dict.keys()):
            self.logger.error("========== Print errors in send-recv process ==========")
            long_err_str = ''
            for k, v in broken_env_err_dict.items():
                long_err_str += v
            self.logger.error(long_err_str)
            self.logger.error("====================   Print End   ====================")
            self.logger.error("============= Process above will be reset =============")

        for ab_env_id, env_err in broken_env_err_dict.items():
            # these envs have to be reset by collector, for it should clean all trajectories...
            empty_data_info = dict(
                comm_err=True,
                abnormal=True,
            )
            self._env_states[env_id] = EnvState.ERROR
            empty_data = BaseEnvTimestep(None, None, None, empty_data_info)
            timesteps[ab_env_id] = empty_data

        self._check_data(timesteps)
        # ======================================================

        if self._shared_memory:
            for i, (env_id, timestep) in enumerate(timesteps.items()):
                timesteps[env_id] = timestep._replace(obs=self._obs_buffers[env_id].get())

        ############################## perform detection #????????????############
        if self._detection_model is not None:
            data_list = []
            for env_id, ts in timesteps.items():
                if not is_abnormal_timestep(ts):
                    v = ts.obs
                    if v is not None:
                        if 'detected' not in v.keys() or v['detected'] != 1.0:
                            data_list.append(v)
            if len(data_list):
                # timer.st_point("det_step")
                # self.logger.warning("perform det in step with list len={}".format(len(data_list)))
                self.insert_detection_result(data_list)
                # timer.ed_point("det_step")
        ############################## perform detection ##############################

        done_ids = []
        for env_id, timestep in timesteps.items():
            if is_abnormal_timestep(timestep):
                self._env_states[env_id] = EnvState.ERROR
            if timestep.done:
                done_ids.append(env_id)
                self._env_episode_count[env_id] += 1
            self._ready_obs[env_id].push(timestep.obs)
        for env_id, timestep in timesteps.items():
            stacked_history_obs = self._ready_obs[env_id].get_data()
            # ids = [id(i) for i in stacked_history_obs]
            # print("[debug-stacked_history_obs]" + str(ids))
            timesteps[env_id] = BaseEnvTimestep(stacked_history_obs, timestep.reward, timestep.done, timestep.info)
            # ids = [id(i) for i in timesteps[env_id].obs]
            # print("[debug-timesteps[env_id].obs]" + str(ids))
            pass

        # reset done env
        for env_id in done_ids:
            if self._env_episode_count[env_id] < self._episode_num and self._auto_reset:
                self._env_states[env_id] = EnvState.RESET
                reset_thread = PropagatingThread(target=self._reset, args=(env_id,), name='regular_reset')
                reset_thread.daemon = True
                reset_thread.start()
                if self._force_reproducibility:
                    reset_thread.join()
            else:
                self._env_states[env_id] = EnvState.DONE
        return timesteps

    def _restart_thread(self, env_id):
        self._pipe_parents[env_id].close()
        if self._subprocesses[env_id].is_alive():
            self._subprocesses[env_id].terminate()
        # reset the subprocess
        self._create_env_subprocess(env_id)

    def close(self) -> None:
        """
        Overview:
            CLose the env manager and release all related resources.
        """
        if self._closed:
            return
        self._closed = True
        self._env_ref.close()
        for _, p in self._pipe_parents.items():
            try:
                p.send(['close', None, None])
            except:
                pass
        for _, p in self._pipe_parents.items():
            try:
                p.recv()
            except:
                pass
        for i in range(self._env_num):
            self._env_states[i] = EnvState.VOID
        # disable process join for avoiding hang
        # for p in self._subprocesses:
        #     p.join()
        for _, p in self._subprocesses.items():
            p.terminate()
        for _, p in self._pipe_parents.items():
            p.close()

    def _reset(self, env_id: int) -> None:
        verbose = False

        # @retry_wrapper(max_retry=self._max_retry, waiting_time=self._retry_waiting_time)
        def reset_fn() -> bool:
            self._env_reset_try_num[env_id] += 1
            if self._env_reset_try_num[env_id] > 1:
                self.logger.error(
                    "[RESET]: Resetting env={} for {} time(s).".format(env_id, self._env_reset_try_num[env_id]))
            have_to_be_restart = False
            if self._pipe_parents[env_id].poll():
                try:
                    recv_data = self._pipe_parents[env_id].recv()
                except Exception as e:
                    if verbose:
                        self.logger.error("[RESET]time={} Env {} poll success, but recv error! have to be restart!!"
                                          .format(self._env_reset_try_num[env_id], env_id)
                                          + '\nEnv {} reset [Exception] in receiving signal:\n'.format(env_id) \
                                          + ''.join(traceback.format_tb(e.__traceback__)) + repr(e))
                    have_to_be_restart = True
            # if self._reset_param[env_id] is None, just reset specific env, not pass reset param
            if not have_to_be_restart:
                reset_paras = {}
                if self._reset_param[env_id] is not None:
                    assert isinstance(self._reset_param[env_id], dict), type(self._reset_param[env_id])
                    reset_paras = self._reset_param[env_id]
                try:
                    self._pipe_parents[env_id].send(['reset', [], reset_paras])
                    if verbose:
                        self.logger.error("[RESET]time={} Env {} send reset suc!!"
                                          .format(self._env_reset_try_num[env_id], env_id))
                    if self._pipe_parents[env_id].poll(self._connect_timeout):
                        obs = self._pipe_parents[env_id].recv()
                        if verbose:
                            self.logger.error("[RESET]time={} Env {} send/recv reset suc!!"
                                              .format(self._env_reset_try_num[env_id], env_id))
                        if self._check_data({env_id: obs}, close=False):
                            if self._shared_memory:
                                obs = self._obs_buffers[env_id].get()
                            self._ready_obs[env_id].clear()
                            self._ready_obs[env_id].push(obs)
                            self._env_reset_try_num[env_id] = 0
                            # self.logger.info("[RESET] Env={}, reset_params={}".format(env_id, str(reset_paras)))
                            return True
                        else:
                            if verbose:
                                self.logger.error("[RESET]time={} Env {} send/recv reset suc, but checkdata fail!!"
                                                  .format(self._env_reset_try_num[env_id], env_id))
                            have_to_be_restart = True
                    else:
                        if verbose:
                            self.logger.error("[RESET]time={} Env {} send reset suc, but recv timeout!!"
                                              .format(self._env_reset_try_num[env_id], env_id))
                        have_to_be_restart = True
                except Exception as e:
                    if verbose:
                        self.logger.error("[RESET]time={} Env {} send reset error! have to be restart!!"
                                          .format(self._env_reset_try_num[env_id], env_id)
                                          + '\nEnv {} reset [Exception] in receiving signal:\n'.format(env_id) \
                                          + ''.join(traceback.format_tb(e.__traceback__)) + repr(e))
                    have_to_be_restart = True

            if not self._closed and have_to_be_restart:
                self.logger.warning("Fail to soft reset, try to restart env={}".format(env_id))
                # terminate the old subprocess
                self._pipe_parents[env_id].close()
                if self._subprocesses[env_id].is_alive():
                    self._subprocesses[env_id].terminate()
                if self._env_ports:
                    cmd_ = "lsof -i:{}|grep python|awk \'{{print \"kill -9 \" $2}}\'|sh".format(
                        self._env_ports[env_id][1])
                    self.logger.error("System cmd:{}".format(cmd_))
                    kill_suc = os.system(cmd_)
                    if 0 != kill_suc:
                        self.logger.error("Fail to kill the port process by system, err_code={}...".format(kill_suc))
                # reset the subprocess
                self._create_env_subprocess(env_id)
            return False

        try:
            if env_id not in self._env_reset_try_num.keys():
                self._env_reset_try_num[env_id] = 0
            success = False
            while not success:
                success = reset_fn()
                if self._closed:
                    self.logger.error("Reseting interrupted for recv closing signal...")
                    break
            # Because each thread updates the corresponding env_id value, they won't lead to a thread-safe problem.
            self._env_states[env_id] = EnvState.RUN
            self.logger.info("Env {} reset success!".format(env_id))
        except Exception as e:
            self.logger.error("Ready to close for unonymous exception...")
            self.close()
            raise e

    def insert_detection_result(self, data_list):
        if self._detection_model is None:
            return
        # detection
        assert isinstance(self._detection_model, DetectionModelWrapper)
        max_batch_size = self._detection_batch_size

        # get unique datalist
        obs_list = data_list

        # get mini-batches
        obs_list_size = len(obs_list)
        pivots = [i for i in range(0, obs_list_size, max_batch_size)] + [obs_list_size]
        seg_num = len(pivots) - 1
        for i in range(seg_num):
            self.logger.debug('[DET]processing minibatch-{}...'.format(i))
            detection_process(obs_list[pivots[i]: pivots[i + 1]], self._detection_model, self._bev_obs_config)

    '''
    Note: error in property is dangerous, which may lead to another undesired property seeking. Be careful.
    '''

    @property
    def ready_obs(self) -> Dict[int, Any]:
        """
        Overview:
            Get the next observations.
        Return:
            A dictionary with observations and their environment IDs.
        Note:
            The observations are returned in np.ndarray.
        Example:
            >>>     obs_dict = env_manager.ready_obs
            >>>     actions_dict = {env_id: model.forward(obs) for env_id, obs in obs_dict.items())}
        """
        try:
            no_done_env_idx = [i for i, s in self._env_states.items() if s != EnvState.DONE]
            sleep_count = 0
            while not any([self._env_states[i] == EnvState.RUN for i in no_done_env_idx]):
                if sleep_count % 1000 == 0:
                    self.logger.warning(
                        'VEC_ENV_MANAGER: all the not done envs are resetting, sleep {} times'.format(sleep_count)
                    )
                time.sleep(0.001)
                sleep_count += 1
            res_dict = {i: self._ready_obs[i].get_data() for i in self.ready_env}

            ############################## perform detection ##############################
            if self._detection_model is not None:
                data_list = []
                for v in res_dict.values():
                    if isinstance(v, dict):
                        if 'detected' not in v.keys() or v['detected'] != 1.0:
                            data_list.append(v)
                    if isinstance(v, list):
                        for v_ in v:
                            if 'detected' not in v_.keys() or v_['detected'] != 1.0:
                                data_list.append(v_)

                if len(data_list):
                    timer.st_point("det_ready_obs")
                    self.logger.warning("perform det in ready_obs with list len={}".format(len(data_list)))
                    self.insert_detection_result(data_list)
                    timer.ed_point("det_ready_obs")

                # check detection process stamp
                for k, v in res_dict.items():
                    if isinstance(v, dict):
                        if not v['detected']:
                            raise TypeError("Detection needed...")
                    if isinstance(v, list):
                        for v_ in v:
                            if not v_['detected']:
                                raise TypeError("Detection needed...")
            ############################## perform detection ##############################
            return res_dict
        except Exception as e:
            self.logger.error('\nRead Property error in ENV Manager:\n'
                              + ''.join(traceback.format_tb(e.__traceback__)) + repr(e))
            return {}
