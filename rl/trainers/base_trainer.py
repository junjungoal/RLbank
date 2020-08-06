import os
from time import time
from collections import defaultdict, OrderedDict
import gzip
import pickle
import copy

import cv2
import h5py
import torch
import wandb
import numpy as np
import moviepy.editor as mpy
from tqdm import tqdm, trange
import gym
from gym import envs
from gym import spaces, error
# from gym.wrappers import TimeLimit

from rl.policies import get_actor_critic_by_name
from util.pytorch import get_ckpt_path, count_parameters, to_tensor
from util.logger import logger
from env.base import EnvWrapper
from rl.rollout import Rollout
from util.info import Info

def get_agent_by_name(algo):
    if algo == "sac":
        from rl.agents.sac_agent import SACAgent
        return SACAgent
    # elif algo == "ppo":
    #     from rl.ppo_agent import PPOAgent
    #     return PPOAgent
    else:
        raise NotImplementedError

class BaseTrainer(object):
    def __init__(self, config):
        self._config = config

        all_envs = envs.registry.all()
        env_ids = [env_spec.id for env_spec in all_envs]
        if config.env in env_ids:
            self._env = EnvWrapper(gym.make(config.env), config)
            self._env_eval = EnvWrapper(gym.make(config.env), copy.copy(config))
        else:
            self._env = gym.make(config.env, **config.__dict__)
            self._env_eval = gym.make(config.env, **copy.copy(config).__dict__)

        # get actor and critic networks
        actor, critic = get_actor_critic_by_name(config.policy)

        ac_space = self._env.action_space
        ob_space = self._env.observation_space

        # build up networks
        self._agent = get_agent_by_name(config.algo)(
            config, ob_space, ac_space, actor, critic
        )

        if self._config.is_train:
            exclude = ['device']
            if not self._config.wandb:
                os.environ['WANDB_MODE'] = 'dryrun'

            # WANDB user or team name
            entity = config.entity
            # WANDB project name
            project = config.project

            wandb.init(
                resume=config.run_name,
                project=project,
                config={k: v for k, v in config.__dict__.items() if k not in exclude},
                dir=config.log_dir,
                entity=entity,
                notes=config.notes
            )

    def _save_ckpt(self, ckpt_num, update_iter):
        """
        Save checkpoint to log directory.
        Args:
            ckpt_num: number appended to checkpoint name. The number of
                environment step is used in this code.
            update_iter: number of policy update. It will be used for resuming training.
        """
        ckpt_path = os.path.join(self._config.log_dir, 'ckpt_%08d.pt' % ckpt_num)
        state_dict = {'step': ckpt_num, 'update_iter': update_iter}
        state_dict['agent'] = self._agent.state_dict()
        torch.save(state_dict, ckpt_path)
        logger.warn("Save checkpoint: %s", ckpt_path)

        replay_path = os.path.join(self._config.log_dir, 'replay_%08d.pkl' % ckpt_num)
        with gzip.open(replay_path, 'wb') as f:
            replay_buffers = {'replay': self._agent.replay_buffer()}
            pickle.dump(replay_buffers, f)

    def _load_ckpt(self, ckpt_num=None):
        """
        Loads checkpoint with index number @ckpt_num. If @ckpt_num is None,
        it loads and returns the checkpoint with the largest index number.
        """
        ckpt_path, ckpt_num = get_ckpt_path(self._config.log_dir, ckpt_num)

        if ckpt_path is not None:
            logger.warn('Load checkpoint %s', ckpt_path)
            ckpt = torch.load(ckpt_path)
            self._agent.load_state_dict(ckpt['agent'])

            if self._config.is_train:
                replay_path = os.path.join(self._config.log_dir, 'replay_%08d.pkl' % ckpt_num)
                logger.warn('Load replay_buffer %s', replay_path)
                with gzip.open(replay_path, 'rb') as f:
                    replay_buffers = pickle.load(f)
                    self._agent.load_replay_buffer(replay_buffers['replay'])

            return ckpt['step'], ckpt['update_iter']
        else:
            logger.warn('Randomly initialize models')
            return 0, 0

    def _log_train(self, step, train_info, ep_info):
        """
        Logs training and episode information to wandb.
        Args:
            step: the number of environment steps.
            train_info: training information to log, such as loss, gradient.
            ep_info: episode information to log, such as reward, episode time.
        """
        for k, v in train_info.items():
            if np.isscalar(v) or (hasattr(v, 'shape') and np.prod(v.shape) == 1):
                wandb.log({'train_rl/%s' % k: v}, step=step)
            else:
                wandb.log({'train_rl/%s' % k: [wandb.Image(v)]}, step=step)

        for k, v in ep_info.items():
            wandb.log({'train_ep/%s' % k: np.mean(v)}, step=step)
            wandb.log({'train_ep_max/%s' % k: np.max(v)}, step=step)

    def _log_test(self, step, ep_info, vids=None):
        """
        Logs episode information during testing to wandb.
        Args:
            step: the number of environment steps.
            ep_info: episode information to log, such as reward, episode time.
        """
        if self._config.is_train:
            for k, v in ep_info.items():
                wandb.log({'test_ep/%s' % k: np.mean(v)}, step=step)
            if vids is not None:
                self.log_videos(vids.transpose((0, 1, 4, 2, 3)), 'test_ep/video', step=step)


    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def _evaluate(self, step=None, record=False, idx=None):
        env = self._env_eval
        vids = []
        for i in range(self._config.num_record_samples):
            ob = env.reset()
            done = False
            ep_len = 0
            reward_info = Info()
            ep_info = Info()
            rollout = Rollout()
            frames = []
            if record: frames.append(self._create_frame(env))
            while not done and ep_len < env.max_episode_steps:
                transition = {}
                ac, ac_before_activation = self._agent.act(ob)
                rollout.add({'ob': ob, 'ac': ac, 'ac_before_activation': ac_before_activation})
                ob, reward, done, info = env.step(ac)
                rollout.add({'done': done, 'rew': reward, 'ob_next': ob})
                ep_len += 1
                step += 1
                ep_rew = reward
                reward_info.add(info)
                if record:
                    frames.append(self._create_frame(env, info))

            ep_info.add({'len': ep_len, 'ep_rew': ep_rew})
            reward_info_dict = reward_info.get_dict(reduction='sum', only_scalar=True)
            ep_info.add(reward_info_dict)
            if record:
                ep_info = ep_info.get_dict(only_scalar=True)
                ep_rew = ep_info['ep_rew']
                ep_success = 's' if ep_info['episode_success'] else 'f'
                fname = "{}_step_{:011d}_{}_r_{}_{}.mp4".format(
                    self._config.env, step, idx if idx is not None else i,
                    ep_rew, ep_success)
                self._save_video(fname, frames)
                vids.append(frames)
            if idx is not None:
                break
        logger.info("rollout: %s",
                    {k: v for k, v in reward_info_dict.items()
                     if np.isscalar(v)})
        return rollout, ep_info, np.array(vids)

    def _create_frame(self, env, info={}):
        color = (200, 200, 200)

        text = "{:4} {}".format(env._episode_length,
                                env._episode_reward)

        frame = env.render('rgb_array') * 255.0
        fheight, fwidth = frame.shape[:2]
        frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

        if self._config.record_caption:
            font_size = 0.4
            thickness = 1
            offset = 12
            x, y = 5, fheight + 10
            cv2.putText(frame, text,
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (255, 255, 0), thickness, cv2.LINE_AA)
            for i, k in enumerate(info.keys()):
                v = info[k]
                key_text = '{}: '.format(k)
                (key_width, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                    font_size, thickness)

                cv2.putText(frame, key_text,
                            (x, y + offset * (i + 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size, (66, 133, 244), thickness, cv2.LINE_AA)

                cv2.putText(frame, str(v),
                            (x + key_width, y + offset * (i + 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size, (255, 255, 255), thickness, cv2.LINE_AA)
        return frame

    def _save_video(self, fname, frames, fps=15.):
        """ Saves @frames into a video with file name @fname. """
        path = os.path.join(self._config.record_dir, fname)

        def f(t):
            frame_length = len(frames)
            new_fps = 1./(1./fps + 1./frame_length)
            idx = min(int(t*new_fps), frame_length-1)
            return frames[idx]

        video = mpy.VideoClip(f, duration=len(frames)/fps+2)

        video.write_videofile(path, fps, verbose=False)
        logger.warn("[*] Video saved: {}".format(path))

    def log_videos(self, vids, name, fps=15, step=None):
        """Logs videos to WandB in mp4 format.
        Assumes list of numpy arrays as input with [time, channels, height, width]."""
        assert len(vids[0].shape) == 4 and vids[0].shape[1] == 3
        assert isinstance(vids[0], np.ndarray)
        log_dict = {name: [wandb.Video(vid, fps=fps, format="mp4") for vid in vids]}
        wandb.log(log_dict) if step is None else wandb.log(log_dict, step=step)
