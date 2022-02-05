from typing import Any, List, Dict, Callable
from easydict import EasyDict

import traceback
from collections import namedtuple

from ding.envs import SyncSubprocessEnvManager
from ding.envs.env_manager.subprocess_env_manager import is_abnormal_timestep
from ding.envs.env_manager.base_env_manager import retry_wrapper, EnvState
from ding.utils import ENV_MANAGER_REGISTRY
from ding.utils import PropagatingThread
from ding.envs.env.base_env import BaseEnvTimestep

from noisy_planning.utils.debug_utils import generate_general_logger

'''
compatable with Ding v0.2.1
'''


@ENV_MANAGER_REGISTRY.register('carla_subprocess')
class CarlaSyncSubprocessEnvManager(SyncSubprocessEnvManager):
    def __init__(
            self,
            env_fn: List[Callable],
            cfg: EasyDict = EasyDict({}),
    ) -> None:
        super(CarlaSyncSubprocessEnvManager, self).__init__(env_fn, cfg)
        self._env_reset_try_num = {}
        self.logger = generate_general_logger("[Manager]")

    def _check_data(self, data: Dict, close: bool = False) -> None:
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
                self.logger.error("env_id={} give exceptions as :{}".format(env_id, str(ex)))

            if not close:
                self.logger.error("We will not close the abnormal envs...")
                for env_id in abnormal_env_ids:
                    empty_data_info = dict(
                        comm_err=True
                    )
                    empty_data = BaseEnvTimestep(None, None, None, empty_data_info)
                    data[env_id] = empty_data
            if close:
                self.logger.error("We will close the abnormal envs...")
                self.close()
                raise exceptions[0]

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
        broken_envs = []
        for env_id, act in actions.items():
            try:
                self._pipe_parents[env_id].send(['step', [act], {}])
                ready_conn.append(self._pipe_parents[env_id])
                ready_env_ids.append(env_id)
            except BrokenPipeError as e:
                broken_envs.append(env_id)
                self.logger.debug("env_id={} got error when performing sending in step().".format(env_id))
            except Exception as e:
                self.logger.error('VEC_ENV_MANAGER: env {} step error in sending'.format(env_id))
                self.logger.error(
                    '\nEnv Process step Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e))

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
                    broken_envs.append(env_id)
                    self.logger.debug("[Timeout] try to receive step info from env_id={}, timeout={}".format(env_id,
                                                                                                             self._connect_timeout))
            except EOFError as e:
                broken_envs.append(env_id)
                self.logger.debug("env_id={} got error when performing recv in step().".format(env_id))
            except Exception as e:
                self.logger.error('VEC_ENV_MANAGER: env {} step error'.format(env_id))
                self.logger.error(
                    '\nEnv Process step Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e))

        # fill data for abnormal env
        if len(broken_envs):
            self.logger.error("{} envs to be reset due to comm error!".format(len(broken_envs)))
        for ab_env_id in broken_envs:
            empty_data_info = dict(
                comm_err=True
            )
            empty_data = BaseEnvTimestep(None, None, None, empty_data_info)
            timesteps[ab_env_id] = empty_data

        self._check_data(timesteps)
        # ======================================================

        if self._shared_memory:
            for i, (env_id, timestep) in enumerate(timesteps.items()):
                timesteps[env_id] = timestep._replace(obs=self._obs_buffers[env_id].get())
        for env_id, timestep in timesteps.items():
            if is_abnormal_timestep(timestep):
                self._env_states[env_id] = EnvState.ERROR
                continue
            if 'comm_err' in timestep.info.keys() and timestep.info['comm_err']:
                self._env_states[env_id] = EnvState.RESET
                reset_thread = PropagatingThread(target=self._reset, args=(env_id,), name='regular_reset')
                reset_thread.daemon = True
                reset_thread.start()
                if self._force_reproducibility:
                    reset_thread.join()
            if timestep.done:
                self._env_episode_count[env_id] += 1
                if self._env_episode_count[env_id] < self._episode_num and self._auto_reset:
                    self._env_states[env_id] = EnvState.RESET
                    reset_thread = PropagatingThread(target=self._reset, args=(env_id,), name='regular_reset')
                    reset_thread.daemon = True
                    reset_thread.start()
                    if self._force_reproducibility:
                        reset_thread.join()
                else:
                    self._env_states[env_id] = EnvState.DONE
            else:
                self._ready_obs[env_id] = timestep.obs
        return timesteps

    def _reset(self, env_id: int) -> None:

        @retry_wrapper(max_retry=self._max_retry, waiting_time=self._retry_waiting_time)
        def reset_fn():
            self._env_reset_try_num[env_id] += 1
            if self._env_reset_try_num[env_id] > 1:
                self.logger.error(
                    "VEC_ENV_MANAGER: Resetting env={} for {} time(s).".format(env_id, self._env_reset_try_num[env_id]))
            if self._pipe_parents[env_id].poll():
                recv_data = self._pipe_parents[env_id].recv()
                raise Exception("unread data left before sending to the pipe: {}".format(repr(recv_data)))
            # if self._reset_param[env_id] is None, just reset specific env, not pass reset param
            if self._reset_param[env_id] is not None:
                assert isinstance(self._reset_param[env_id], dict), type(self._reset_param[env_id])
                self._pipe_parents[env_id].send(['reset', [], self._reset_param[env_id]])
            else:
                self._pipe_parents[env_id].send(['reset', [], {}])
            self.logger.debug("VEC_ENV_MANAGER:sent resend order fot env_id={}.".format(env_id))

            if not self._pipe_parents[env_id].poll(self._connect_timeout):
                # terminate the old subprocess
                self._pipe_parents[env_id].close()
                if self._subprocesses[env_id].is_alive():
                    self._subprocesses[env_id].terminate()
                # reset the subprocess
                self._create_env_subprocess(env_id)
                raise TimeoutError("env reset timeout")  # Leave it to retry_wrapper to try again

            self.logger.debug("[reset_fn]try to receive step info from env_id={}".format(env_id))
            obs = self._pipe_parents[env_id].recv()
            self.logger.debug("[Done][reset_fn]try to receive step info from env_id={}".format(env_id))

            self._check_data({env_id: obs}, close=False)
            if self._shared_memory:
                obs = self._obs_buffers[env_id].get()
            # Because each thread updates the corresponding env_id value, they won't lead to a thread-safe problem.
            self._env_states[env_id] = EnvState.RUN
            self._ready_obs[env_id] = obs

        if env_id not in self._env_reset_try_num.keys():
            self._env_reset_try_num[env_id] = 0

        try:
            reset_fn()
            self._env_reset_try_num[env_id] = 0
        except EOFError as e:
            # the thread is closing... restart it
            # logging.error('VEC_ENV_MANAGER: env {} reset error'.format(env_id))
            # logging.error('\nEnv Process Reset Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e))
            self.logger.error("VEC_ENV_MANAGER: prepare to restart the dead subprocess due to EOF!")
            self._pipe_parents[env_id].close()
            if self._subprocesses[env_id].is_alive():
                self._subprocesses[env_id].terminate()
            self._create_env_subprocess(env_id)
            return self._reset(env_id)
        except TimeoutError as e:
            # the thread is closing... restart it
            # logging.error('VEC_ENV_MANAGER: env {} reset error'.format(env_id))
            # logging.error('\nEnv Process Reset Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e))
            self.logger.debug("VEC_ENV_MANAGER: prepare to restart the dead subprocess due to TimeoutError!")
            return self._reset(env_id)
        except Exception as e:
            self.logger.error('VEC_ENV_MANAGER: env {} reset error'.format(env_id))
            self.logger.error(
                '\nEnv Process Reset Exception:\n' + ''.join(traceback.format_tb(e.__traceback__)) + repr(e))
            if self._closed:  # exception cased by main thread closing parent_remote
                self.logger.error("Ready to return, for self._closed is true!")
                return
            else:
                self.logger.error("Ready to close for unonymous exception...")
                self.close()
                raise e
