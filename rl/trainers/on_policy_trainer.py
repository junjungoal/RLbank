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
        init_step = step
        init_episode = 0
        while step < config.max_global_step:
            ob = env.reset()
            done = False
            ep_len = 0
            ep_rew = 0
            reward_infos = [Info() for _ in range(config.num_processes)]
            ep_info = [Info() for _ in range(config.num_processes)]
            train_info = {}
            for _ in range(config.rollout_length):
                transition = {}
                ac, ac_before_activation, vpred = self._agent.act(ob, pred_value=True)
                rollout.add({'ob': ob, 'ac': ac, 'ac_before_activation': ac_before_activation, 'vpred': vpred})
                ob, reward, dones, infos = env.step(ac)
                ep_rew += reward
                for i, (info, done) in enumerate(zip(infos, dones)):
                    reward_infos[i].add(info)
                    if done:
                        reward_info_dict = reward_info[i].get_dict(reduction='sum', only_scalar=True)
                        ep_info[i].add(reward_info_dict)
                rollout.add({'done': done, 'rew': reward, 'ob_next': ob})

                ep_len += 1
                step += 1

            vpred = self._agent.value(ob)
            rollout.add({'vpred': vpred})
            self._agent.store_episode(rollout.get())
            pbar.update(1)
            if step % config.log_interval == 0:
                logger.info("Update networks %d", update_iter)
            train_info = self._agent.train()
            update_iter += 1
            # ep_info = ep_info.get_dict(only_scalar=True)

            # if episode % config.log_interval == 0:
            #     train_info.update({
            #         'sec': (time() - st_time) / config.log_interval,
            #         'steps_per_sec': (step - st_step) / (time() - st_time),
            #         'update_iter': update_iter
            #     })
            #     st_time = time()
            #     st_step = step
            #     self._log_train(step, train_info, ep_info)
            #
            # if episode % config.evaluate_interval == 0:
            #     logger.info("Evaluate at %d", update_iter)
            #     rollout, info, vids = self._evaluate(step=step, record=config.record)
            #     self._log_test(step, info, vids)
            #
            # if update_iter % config.ckpt_interval == 0 and init_step > config.init_steps:
            #     self._save_ckpt(step, update_iter)
            #
            #
