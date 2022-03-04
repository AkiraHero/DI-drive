from ding.worker import SampleSerialCollector
from typing import Optional, Any, List
import torch

from ding.utils import one_time_warning
from ding.torch_utils import to_tensor, to_ndarray
from ding.worker.collector.base_serial_collector import to_tensor_transitions
from noisy_planning.utils.debug_utils import generate_general_logger
import logging


class SampleTailCollector(SampleSerialCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = generate_general_logger("[Manager]")
        self.logger.setLevel(logging.INFO)
        self._tail_len = 300
        if "tail_len" in args[0].keys():
            self._tail_len = args[0].tail_len
        self.logger.warning("SampleTailCollector only truncate the tail with max_len={}!".format(self._tail_len))



    def collect(self,
                n_sample: Optional[int] = None,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None) -> List[Any]:
        """
                Overview:
                    Collect `n_sample` data with policy_kwargs, which is already trained `train_iter` iterations
                Arguments:
                    - n_sample (:obj:`int`): the number of collecting data sample
                    - train_iter (:obj:`int`): the number of training iteration
                    - policy_kwargs (:obj:`dict`): the keyword args for policy forward
                Returns:
                    - return_data (:obj:`List`): A list containing training samples.
                """
        if n_sample is None:
            if self._default_n_sample is None:
                raise RuntimeError("Please specify collect n_sample")
            else:
                n_sample = self._default_n_sample
        if n_sample % self._env_num != 0:
            one_time_warning(
                "Please make sure env_num is divisible by n_sample: {}/{}, which may cause convergence \
                problems in a few algorithms".format(n_sample, self._env_num)
            )
        if policy_kwargs is None:
            policy_kwargs = {}
        collected_sample = 0
        return_data = []
        cnt = 0
        while collected_sample < n_sample:
            cnt += 1
            if cnt % 100 == 0:
                self.logger.info("collector cnt={}, sample={}/{}".format(cnt, collected_sample, n_sample))
            with self._timer:
                # Get current env obs.
                obs = self._env.ready_obs
                # Policy forward.
                self._obs_pool.update(obs)
                if self._transform_obs:
                    obs = to_tensor(obs, dtype=torch.float32)
                policy_output = self._policy.forward(obs, **policy_kwargs)
                self._policy_output_pool.update(policy_output)
                # Interact with env.
                actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)

            # TODO(nyz) this duration may be inaccurate in async env
            interaction_duration = self._timer.value / len(timesteps)

            # TODO(nyz) vectorize this for loop
            for env_id, timestep in timesteps.items():
                with self._timer:
                    if timestep.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        # suppose there is no reset param, just reset this env
                        self._env.reset({env_id: None})
                        self._policy.reset([env_id])
                        self._reset_stat(env_id)
                        self._logger.info('env_id {}, abnormal step {}', env_id, timestep.info)
                        continue
                    transition = self._policy.process_transition(
                        self._obs_pool[env_id], self._policy_output_pool[env_id], timestep
                    )
                    # ``train_iter`` passed in from ``serial_entry``, indicates current collecting model's iteration.
                    transition['collect_iter'] = train_iter
                    self._traj_buffer[env_id].append(transition)  # NOTE
                    self._env_info[env_id]['step'] += 1
                    self._total_envstep_count += 1
                    # prepare data
                    if timestep.done:  #or len(self._traj_buffer[env_id]) == self._traj_len:
                        # Episode is done or traj_buffer(maxlen=traj_len) is full.
                        transitions = to_tensor_transitions(self._traj_buffer[env_id])
                        train_sample = self._policy.get_train_sample(transitions)

                        # only want tail
                        total_samples = len(train_sample)
                        tail_len_desired = min(self._tail_len, total_samples)
                        train_sample = train_sample[total_samples - tail_len_desired: total_samples]

                        return_data.extend(train_sample)
                        self._total_train_sample_count += len(train_sample)
                        self._env_info[env_id]['train_sample'] += len(train_sample)
                        collected_sample += len(train_sample)
                        self._traj_buffer[env_id].clear()

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration

                # If env is done, record episode info and reset
                if timestep.done:
                    self._total_episode_count += 1
                    reward = timestep.info['final_eval_reward']
                    info = {
                        'reward': reward,
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                        'train_sample': self._env_info[env_id]['train_sample'],
                    }
                    self._episode_info.append(info)
                    # Env reset is done by env_manager automatically
                    self._policy.reset([env_id])
                    self._reset_stat(env_id)
        # log
        self._output_log(train_iter)
        # on-policy reset
        if self._on_policy:
            for env_id in range(self._env_num):
                self._reset_stat(env_id)

        return return_data[:n_sample]