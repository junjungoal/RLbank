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

class OffPolicyTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        config = self._config
        num_batches = config.num_batches

        # load checkpoint
        step, update_iter = self._load_ckpt()

        logger.info("Start training at step=%d", step)
        pbar = tqdm(initial=step, total=config.max_global_step, desc=config.run_name)
        ep_info = defaultdict(list)

        env = self._env
        rollout = Rollout()
        episode = 0
        while step < config.max_global_step:
            ob = env.reset()
            done = False
            ep_len = 0
            reward_info = Info()
            ep_info = Info()
            while not done and ep_len < env.max_episode_steps:
                transition = {}
                if step < config.start_steps:
                    ac, ac_before_activation = self._agent.act(ob)
                else:
                    ac = env.action_space.sample()
                    ac_before_activation = None
                rollout.add({'ob': ob, 'ac': ac, 'ac_before_activation': ac_before_activation})
                ob, reward, done, info = env.step(ac)
                rollout.add({'done': done, 'rew': reward, 'ob': ob})
                ep_len += 1
                step += 1
                ep_rew = reward
                reward_info.add(info)
                pbar.update(1)

                self._agent.store_episode(rollout.get())
                if step % config.log_interval == 0:
                    logger.info("Update networks %d", update_iter)
                train_info = self._agent.train()
                update_iter += 1

            ep_info.add({'len': ep_len, 'ep_rew': ep_rew})
            reward_info_dict = reward_info.get_dict(reduction='sum', only_scalar=True)
            ep_info.add(reward_info_dict)
            logger.info("Ep %d rollout: %s", episode,
                        {k: v for k, v in reward_info_dict.items()
                         if np.isscalar(v)})
            episode += 1

            if episode % config.log_interval == 0:
                if isinstance(v, list):
                    ep_info[k].extend(v)
                else:
                    ep_info[k].append(v)
                train_info.update({
                    'sec': (time() - st_time) / config.log_interval,
                    'steps_per_sec': (step - st_step) / (time() - st_time),
                    'update_iter': update_iter
                })
                st_time = time()
                st_step = step
                self._log_train(step, train_info, ep_info)

            if update_iter % config.ckpt_interval == 0:
                self._save_ckpt(step, update_iter)



    def evaluate(self):
        raise NotImplementedError
