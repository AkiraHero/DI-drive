import os
import numpy as np
from collections import defaultdict
import torch
from typing import Dict, Any, List, Optional, Callable, Tuple
import pickle
import datetime

from .base_evaluator import BaseEvaluator
from core.data.benchmark import ALL_SUITES
from ding.envs import BaseEnvManager
from ding.torch_utils.data_helper import to_tensor
from ding.utils import build_logger, EasyTimer
from noisy_planning.utils.debug_utils import auto_signal_handler


class SerialEvaluator(BaseEvaluator):
    """
    Evaluator used to serially evaluate a policy for defined times. It is mainly used when training a policy to get the
    evaluator performance frequently and store the best iterations. Different from serial evaluator in `DI-engine`, this
    evaluator compares the performance of iterations by the success rate rather than rewards. You can provide a
    tensorboard logger to save scalars when training.

    Note:
        Env manager must run WITH auto reset.

    :Arguments:
        - cfg (Dict): Config dict.
        - env (BaseEnvManager): Env manager used to evaluate.
        - policy (Any): Policy to evaluate. Must have ``forward`` method.
        - tb_logger (SummaryWriter, optional): Tensorboard writter to store values in tensorboard. Defaults to None.
        - exp_name (str, optional): Name of the experiments. Used to build logger. Defaults to 'default_experiment'.
        - instance_name (str, optional): [description]. Defaults to 'serial_evaluator'.

    :Interfaces: reset, eval, close, should_eval

    :Properties:
        - env (BaseEnvManager): Env manager with several environments used to evaluate.
        - policy (Any): Policy instance to interact with envs.
    """

    config = dict(
        # whether transform obs into tensro manually
        transform_obs=False,
        # evaluate every "eval_freq" training iterations.
        eval_freq=100,
        # evaluate times in each evaluation
        n_episode=10,
        # stop value of success rate
        stop_rate=1,
        # max steps to evaluate to avoid too long sequences
        env_max_steps=2000,
        eval_once = False
    )

    def __init__(
            self,
            cfg: Dict,
            env: BaseEnvManager,
            policy: Any,
            tb_logger: Optional['SummaryWriter'] = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'serial_evaluator',
    ) -> None:
        super().__init__(cfg, env, policy, tb_logger=tb_logger, exp_name=exp_name, instance_name=instance_name)
        self._transform_obs = self._cfg.transform_obs
        self._default_n_episode = self._cfg.n_episode
        self._stop_rate = self._cfg.stop_rate
        self._env_max_steps = self._cfg.env_max_steps

        self._last_eval_iter = 0
        self._max_success_rate = 0
        self._eval_once = self._cfg.eval_once
        self._eval_all_result = []
        self._exp_name = exp_name
        self.episode_count = 0
        if self._eval_once:
            self._logger.error("[EVAL]==================================Using eval once===========================================")

    @property
    def env(self) -> BaseEnvManager:
        return self._env_manager

    @env.setter
    def env(self, _env_manager: BaseEnvManager) -> None:
        assert _env_manager._auto_reset, "auto reset for env manager should be opened!"
        self._end_flag = False
        self._env_manager = _env_manager
        self._env_manager.launch()
        self._env_num = self._env_manager.env_num

    def close(self) -> None:
        """
        Close the collector and the env manager if not closed.
        """
        if self._close_flag:
            return
        self._close_flag = True
        self._env_manager.close()
        if self._tb_logger is not None:
            self._tb_logger.flush()
            self._tb_logger.close()

    def reset(self) -> None:
        """
        Reset evaluator and policies.
        """
        self._policy.reset([i for i in range(self._env_num)])
        self._last_eval_iter = 0
        self._max_success_rate = 0

    def should_eval(self, train_iter: int) -> bool:
        """
        Judge if the training iteration is at frequency value to run evaluation.

        :Arguments:
            - train_iter (int): Current training iteration

        :Returns:
            bool: Whether should run iteration
        """
        if (train_iter - self._last_eval_iter) < self._cfg.eval_freq and train_iter != 0:
            return False
        self._last_eval_iter = train_iter
        return True

    def save_eval_result(self):
        if self._eval_once:
            res_file_name =  os.path.join(self._exp_name, 'eval_result_episode{}_{}.pickle'.format(self.episode_count, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
            with open(res_file_name, 'wb') as f:
                pickle.dump(self._eval_all_result, f)
                self._logger.error("[EVAL]Dumped eval result:" + res_file_name)

    @auto_signal_handler("save_eval_result")
    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            policy_kwargs: Optional[Dict] = None,
            n_episode: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Run evaluation with provided policy arguments. It will evaluate all available episodes of the benchmark suite
        unless `episode_per_suite` is set in config.

        :Arguments:
            - save_ckpt_fn (Callable, optional): Function to save ckpt. Will be called if at best performance.
                Defaults to None.
            - train_iter (int, optional): Current training iterations. Defaults to -1.
            - envstep (int, optional): Current env steps. Defaults to -1.
            - policy_kwargs (Dict, optional): Additional arguments in policy forward. Defaults to None.
            - n_episode: (int, optional): Episodes to eval. By default it is set in config.

        :Returns:
            Tuple[bool, float]: Whether reach stop value and success rate.
        """
        if policy_kwargs is None:
            policy_kwargs = dict()
        if n_episode is None:
            n_episode = self._default_n_episode
        self._logger.error("[EVAL]we gonna to eval on {} episodes".format(n_episode))
        assert n_episode is not None, "please indicate eval n_episode"
        self._env_manager.reset()
        self._policy.reset([i for i in range(self._env_num)])

        self.episode_count = 0
        self._eval_all_result.clear()

        env_steps = {}
        with self._timer:
            total_step = 0
            while self.episode_count < n_episode:
                obs = self._env_manager.ready_obs
                if self._transform_obs:
                    obs = to_tensor(obs, dtype=torch.float32)
                policy_output = self._policy.forward(obs, **policy_kwargs)
                actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                timesteps = self._env_manager.step(actions)
                total_step += 1
                if total_step % 200 == 0:
                    self._logger.error("[EVAL]now is step {} of the env manager, cnt:{}/{}!".format(total_step, self.episode_count, n_episode))
                for env_id, t in timesteps.items():
                    if env_id not in env_steps.keys():
                        env_steps[env_id] = 0
                    if t.info.get('abnormal', False):
                        env_steps[env_id] = 0
                        self._logger.warning("step {}, process timestep of env={}, it is abnormal".format(total_step, env_id))
                        self._env_manager.reset({env_id: None})
                        self._policy.reset([env_id])
                        continue
                    if t.info['stuck']:
                        self._policy.reset([env_id])
                        env_steps[env_id] = 0
                        self.episode_count += 1
                        self._logger.info(
                            "[EVALUATOR] env {} stop episode for it is stucked".format(env_id))
                        continue
                    if env_steps[env_id] > self._env_max_steps:
                        self.episode_count += 1
                        result = {
                            'stuck': t.info['stuck'],
                            'step': int(t.info['tick']),
                        }
                        self._logger.info(
                            "[EVALUATOR] env {} stop episode for it is too long,"
                            " stuck: {}, current episode: {}, step: {}".format(
                                env_id, result['stuck'], self.episode_count, env_steps[env_id]
                            ))
                        self._policy.reset([env_id])
                        env_steps[env_id] = 0
                        continue

                    if t.done:
                        self._policy.reset([env_id])
                        env_steps[env_id] = 0
                        result = {
                            'reward': t.info['final_eval_reward'],
                            'success': t.info['success'],
                            'step': int(t.info['tick']),
                            'failure_reason': t.info['failure_reason'],
                        }
                        if 'suite_name' in t.info.keys():
                            result['suite_name'] = t.info['suite_name']
                        self.episode_count += 1
                        self._eval_all_result.append(result)
                        self._logger.info(
                            "[EVALUATOR] env {} finish episode, final reward: {}, current episode: {}".format(
                                env_id, result['reward'], self.episode_count
                            )
                        )
                    env_steps[env_id] += 1
                if self._env_manager.done:
                    break
        self.save_eval_result()

        duration = self._timer.value
        episode_reward = [i['reward'] for i in self._eval_all_result]
        envstep_count = np.sum([i['step'] for i in self._eval_all_result])
        success_count = np.sum([i['success'] for i in self._eval_all_result])

        suite_cnt = {}
        for d in self._eval_all_result:
            if 'suite_name' in d.keys():
                suite_name = d['suite_name']
                if suite_name not in suite_cnt.keys():
                    suite_cnt[suite_name] = {'suc':0, 'total':0, 'failure_reason':[], 'rewards':[]}
                suite_cnt[suite_name]['rewards'].append(d['reward'])
                if d['success']:
                    suite_cnt[suite_name]['suc'] += 1
                else:
                    suite_cnt[suite_name]['failure_reason'].append(d['failure_reason'])
                suite_cnt[suite_name]['total'] += 1
        # self._total_duration += duration

        success_rate = 0 if self.episode_count == 0 else success_count / self.episode_count
        info = {
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'avg_envstep_per_episode': envstep_count / n_episode,
            'evaluate_time': duration,
            'avg_time_per_episode': duration / n_episode,
            'success_rate': success_rate,
        }
        if len(episode_reward):
            info.update(
                {
                    'reward_mean': np.mean(episode_reward),
                    'reward_std': np.std(episode_reward),
                }
            )
        # for k, v in suite_cnt.items():
        #     info.update({'suite_{}_count'.format(k): v['total'], 'suite_{}_suc_rate'.format(k): v['suc'] / v['total']})


        # update success info
        # 1. suite
        for k, v in suite_cnt.items():
            self._tb_logger.add_scalar('{}_iter_suc_info/suite_{}/episode_num'.format(self._instance_name, k), v['total'], train_iter)
            self._tb_logger.add_scalar('{}_iter_suc_info/suite_{}/suc_rate'.format(self._instance_name, k), v['suc'] / v['total'], train_iter)
            self._tb_logger.add_scalar('{}_iter_suc_info/suite_{}/mean_reward'.format(self._instance_name, k), np.mean(v['rewards']), train_iter)
            failure_reason_dict = {}
            failure_num = v['total'] - v['suc']
            for rea_ in v['failure_reason']:
                if rea_ is not None:
                    if rea_ not in failure_reason_dict.keys():
                        failure_reason_dict[rea_] = 0
                    failure_reason_dict[rea_] += 1
            failure_rate_dict = {i: j / failure_num for i, j in failure_reason_dict.items()}
            self._tb_logger.add_scalars('{}_iter_suc_info/suite_{}/fail_reason_rate'.format(self._instance_name, k), failure_rate_dict, train_iter)
            self._tb_logger.add_scalars('{}_iter_suc_info/suite_{}/fail_reason_num'.format(self._instance_name, k), failure_reason_dict, train_iter)
        # 2. total
        total_num = 0
        total_suc = 0
        total_fail = 0
        total_failure_reason_dict = {}
        all_rewards = []
        for k, v in suite_cnt.items():
            total_num += v['total']
            total_suc += v['suc']
            total_fail += v['total'] - v['suc']
            all_rewards += v['rewards']
            for rea_ in v['failure_reason']:
                if rea_ is not None:
                    if rea_ not in total_failure_reason_dict.keys():
                        total_failure_reason_dict[rea_] = 0
                    total_failure_reason_dict[rea_] += 1
        total_failure_rate_dict = {i: j / total_fail for i, j in total_failure_reason_dict.items()}
        self._tb_logger.add_scalars('{}_iter_suc_info/total/fail_reason_rate'.format(self._instance_name), total_failure_rate_dict, train_iter)
        self._tb_logger.add_scalars('{}_iter_suc_info/total/fail_reason_num'.format(self._instance_name), total_failure_reason_dict, train_iter)
        self._tb_logger.add_scalar('{}_iter_suc_info/total/mean_reward'.format(self._instance_name), np.mean(all_rewards), train_iter)





        self._logger.info(self._logger.get_tabulate_vars_hor(info))
        if self._tb_logger is not None:
            for k, v in info.items():
                if k in ['train_iter', 'ckpt_name', 'each_reward']:
                    continue
                if not np.isscalar(v):
                    continue
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)

        if not self._eval_once and success_rate > self._max_success_rate:
            if save_ckpt_fn:
                save_ckpt_fn('ckpt_best.pth.tar')
            self._max_success_rate = success_rate
        stop_flag = success_rate > self._stop_rate and train_iter > 0
        if stop_flag:
            self._logger.info(
                "[EVALUATOR] " +
                "Current success rate: {} is greater than stop rate: {}".format(success_rate, self._stop_rate) +
                ", so the training is converged."
            )
        print("===================================")
        print(str(info))
        print("===================================")
        return stop_flag, success_rate
