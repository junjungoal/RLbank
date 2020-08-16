import os
from time import time
from collections import defaultdict, OrderedDict
import gzip
import pickle

import h5py
import torch
import wandb
import numpy as np
import moviepy.editor as mpy
from tqdm import tqdm, trange

from rl.policies import get_actor_critic_by_name
from rl.trainers.base_trainer import BaseTrainer
from rl.rollout import Rollout
from util.logger import logger
from util.pytorch import get_ckpt_path, count_parameters
from util.info import Info


class OnPolicyTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        config = self._config
        num_batches = config.num_batches

        # load checkpoint
        step, update_iter = self._load_ckpt()

        logger.info("Start training at step=%d", step)
        pbar = tqdm(initial=step, total=config.max_global_step, desc=config.run_name)

        env = self._env
        rollout = Rollout()
        episode = 0
        st_time = time()
        st_step = step
        num_updates = int(config.max_global_step // config.rollout_length // config.num_processes)
        while step < config.max_global_step:
            ob = env.reset()
            done = False
            ep_len = 0
            ep_rew = 0
            reward_infos = [Info() for _ in range(config.num_processes)]
            ep_info = Info()
            train_info = {}
            update_linear_schedule(self._agent._actor_optim,
                                   update_iter, num_updates,
                                   self._agent._actor_optim.lr)
            update_linear_schedule(self._agent._critic_optim,
                                   update_iter, num_updates,
                                   self._agent._critic_optim.lr)
            for _ in range(config.rollout_length):
                transition = {}
                ac, ac_before_activation, log_prob, vpred = self._agent.act(ob, pred_value=True, return_log_prob=True)
                rollout.add({'ob': ob, 'ac': ac, 'ac_before_activation': ac_before_activation, 'vpred': vpred, 'log_prob': log_prob})
                ob, reward, dones , infos = env.step(ac)
                ep_rew += reward
                pbar.update(1)
                for i, (info, done) in enumerate(zip(infos, dones)):
                    reward_infos[i].add(info)
                    if done:
                        reward_info_dict = reward_infos[i].get_dict(reduction='sum', only_scalar=True)
                        ep_info.add(reward_info_dict)
                rollout.add({'done': dones, 'rew': reward, 'ob_next': ob})

                ep_len += 1
                step += 1

            vpred = self._agent.value(ob)
            rollout.add({'vpred': vpred})
            self._agent.store_episode(rollout.get())
            train_info = self._agent.train()

            ep_info = ep_info.get_dict(only_scalar=True)

            logger.info("Rollout %d: %s", update_iter,
            {k: v for k, v in ep_info.items()
             if np.isscalar(v)})
            update_iter += 1

            if update_iter % config.log_interval == 0:
                logger.info("Update networks %d", update_iter)
                train_info.update({
                    'sec': (time() - st_time) / config.log_interval,
                    'steps_per_sec': (step - st_step) / (time() - st_time),
                    'update_iter': update_iter
                })
                st_time = time()
                st_step = step
                self._log_train(step, train_info, ep_info)

            if update_iter % config.evaluate_interval == 0:
                logger.info("Evaluate at %d", update_iter)
                _, info, vids = self._evaluate(step=step, record=config.record)
                self._log_test(step, info, vids)

            if update_iter % config.ckpt_interval == 0:
                self._save_ckpt(step, update_iter)
