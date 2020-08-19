from collections import defaultdict, OrderedDict
from time import time

import numpy as np
from util.gym import observation_size, action_size
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from util.image import random_crop, center_crop_image
import copy

class ReplayBuffer:
    """ Replay Buffer. """

    def __init__(self, config, ob_space, ac_space):
        self._config = config
        self._size = config.buffer_size

        # memory management
        self._idx = 0
        self._current_size = 0

        # create the buffer to store info
        self._buffers= defaultdict(list)
        self._obs = {k: np.empty((self._size, *ob_space[k].shape)) for k in ob_space.spaces.keys()}
        self._obs_next = {k: np.empty((self._size, *ob_space[k].shape)) for k in ob_space.spaces.keys()}
        self._actions = {k: np.empty((self._size, action_size(ac_space[k]))) for k in ac_space.spaces.keys()}
        self._rewards = np.empty((self._size, 1))
        self._terminals = np.empty((self._size, 1))

    def clear(self):
        self._idx = 0
        self._current_size = 0
        self._buffers = defaultdict(list)

    def store_sample(self, rollout):
        for k in self._obs.keys():
            self._obs[k][self._idx] = rollout['ob'][0][k]
            self._obs_next[k][self._idx] = rollout['ob_next'][0][k]
        for k in self._actions.keys():
            self._actions[k][self._idx] = rollout['ac'][0][k]
        self._rewards[self._idx] = rollout['rew'][0]
        self._terminals[self._idx] = rollout['done'][0]

        idx = self._idx = (self._idx + 1) % self._size
        if self._current_size < self._size:
            self._current_size += 1

    def store_episode(self, rollout):
        """ Stores the episode. """
        idx = self._idx = (self._idx + 1) % self._size
        for k in self._obs.keys():
            for i, data in enumerate(rollout['ob']):
                if self._current_size + i > self._size:
                    self._obs[k][idx] = data[k]
                    self._obs_next[k][idx] = data[k]
                else:
                    self._obs[k][self._current_size+i] = data[k]
                    self._obs_next[k][self._current_size+i] = data[k]
        for k in self._actions.keys():
            for i, data in enumerate(rollout['ac']):
                if self._current_size  + i > self._size:
                    self._actions[k][idx] = data[k]
                else:
                    self._actions[k][self._current_size+i] = data[k]

        for i in range(len(rollout['rew'])):
            if self._current_size > self._size:
                self._rewards[idx] = rollout['rew'][i]
                self._terminals[idx] = rollout['done'][i]
                idx = self._idx = (self._idx + 1) % self._size
            else:
                self._rewards[self._current_size+i] = rollout['rew'][i]
                self._terminals[self._current_size+i] = rollout['done'][i]
                self._current_size += 1

    def sample_cpc(self, batch_size):
        idxs = np.random.randint(0, self._current_size, batch_size)
        ob = OrderedDict([(k, v[idxs]) for k, v in self._obs.items()])
        ob_next = OrderedDict([(k, v[idxs]) for k, v in self._obs_next.items()])
        pos = copy.deepcopy(ob)
        ob['default'] = random_crop(ob['default'], self._config.img_height)
        ob_next['default'] = random_crop(ob_next['default'], self._config.img_height)
        pos['default'] = random_crop(pos['default'], self._config.img_height)

        cpc_kwargs = dict(ob_anchor=ob, ob_pos=pos, time_anchor=None, time_pos=None)

        done = self._terminals[idxs]
        rew = self._rewards[idxs]
        ac = OrderedDict([(k, v[idxs]) for k, v in self._actions.items()])
        return {'ob': ob,
               'ob_next': ob_next,
               'done': done,
               'rew': rew,
               'ac': ac,
               'cpc_kwargs': cpc_kwargs}


    def sample(self, batch_size):
        """ Samples the data from the replay buffer. """
        # sample transitions
        idxs = np.random.randint(0, self._current_size, batch_size)
        ob = OrderedDict([(k, v[idxs]) for k, v in self._obs.items()])
        ob_next = OrderedDict([(k, v[idxs]) for k, v in self._obs_next.items()])
        done = self._terminals[idxs]
        rew = self._rewards[idxs]
        ac = OrderedDict([(k, v[idxs]) for k, v in self._actions.items()])
        return {'ob': ob,
               'ob_next': ob_next,
               'done': done,
               'rew': rew,
               'ac': ac}

    def state_dict(self):
        return self._buffers

    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        self._current_size = len(self._buffers['ac'])



class MultiProcessReplayBuffer:
    def __init__(self, config, num_processes, ob_space, ac_space):
        self._idx = 0
        self._current_size = 0
        self._config = config
        self._size = config.rollout_length
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._num_processes = num_processes

        self._obs = {k: np.empty((buffer_size, num_processes, observation_size(ob_space[k]))) for k in ob_space.spaces.keys()}
        self._obs_next = {k: np.empty((buffer_size, num_processes, observation_size(ob_space[k]))) for k in ob_space.spaces.keys()}
        self._actions = {k: np.empty((buffer_size, num_processes, action_size(ac_space[k]))) for k in ac_space.spaces.keys()}
        self._ac_before_activation = {k: np.empty((buffer_size, num_processes, action_size(ac_space[k]))) for k in ac_space.spaces.keys()}
        self._rewards = np.empty((buffer_size, num_processes, 1))
        self._terminals = np.empty((buffer_size, num_processes, 1))
        self._vpreds = np.empty((buffer_size, num_processes, 1))
        self._adv = np.empty((buffer_size, num_processes, 1))
        self._ret = np.empty((buffer_size, num_processes, 1))
        self._log_prob = np.empty((buffer_size, num_processes, 1))

    def store_episode(self, rollout):
        """ Stores the episode. """
        idx = self._idx = (self._idx + 1) % self._size
        for k in self._obs.keys():
            for i, data in enumerate(rollout['ob']):
                if self._current_size + i > self._size:
                    self._obs[k][idx] = data[k]
                    self._obs_next[k][idx] = data[k]
                else:
                    self._obs[k][self._current_size+i] = data[k]
                    self._obs_next[k][self._current_size+i] = data[k]
        for k in self._actions.keys():
            for i, data in enumerate(rollout['ac']):
                if self._current_size  + i > self._size:
                    self._actions[k][idx] = data[k]
                    self._ac_before_activation[k][idx] = data[k]
                else:
                    self._actions[k][self._current_size+i] = data[k]
                    self._ac_before_activation[k][self._current_size+i] = data[k]

        for i in range(len(rollout['rew'])):
            if self._current_size > self._size:
                self._rewards[idx] = rollout['rew'][i].reshape(-1, 1)
                self._terminals[idx] = rollout['done'][i].reshape(-1, 1)
                self._vpreds[idx] = rollout['vpred'][i].reshape(-1, 1)
                self._adv[idx] = rollout['adv'][i].reshape(-1, 1)
                self._ret[idx] = rollout['ret'][i].reshape(-1, 1)
                self._log_prob[idx] = rollout['log_prob'][i].reshape(-1, 1)
                idx = self._idx = (self._idx + 1) % self._size
            else:
                self._rewards[self._current_size] = rollout['rew'][i].reshape(-1, 1)
                self._terminals[self._current_size] = rollout['done'][i].reshape(-1, 1)
                self._vpreds[self._current_size] = rollout['vpred'][i].reshape(-1, 1)
                self._adv[self._current_size] = rollout['adv'][i].reshape(-1, 1)
                self._ret[self._current_size] = rollout['ret'][i].reshape(-1, 1)
                self._log_prob[self._current_size] = rollout['log_prob'][i].reshape(-1, 1)
                self._current_size += 1

    def sample(self, batch_size):
        """ Samples the data from the replay buffer. """
        # sample transitions
        num_processes = self._rewards.shape[1]
        idxs = np.random.randint(0, self._size*num_processes, batch_size)
        transitions = {}
        ob = OrderedDict([(k, v.reshape(self._size*num_processes, -1)[idxs]) for k, v in self._obs.items()])
        # ob_next = OrderedDict([(k, v[idxs]) for k, v in self._obs_next.items()])
        # reward = self._rewards.reshape((self._size*num_processes, -1))[idxs]
        done = self._terminals.reshape((self._size*num_processes, -1))[idxs]
        action = OrderedDict([(k, v.reshape(self._size*num_processes, -1)[idxs]) for k, v in self._actions.items()])
        ac_before_activation = OrderedDict([(k, v.reshape(self._size*num_processes, -1)[idxs]) for k, v in self._ac_before_activation.items()])
        # vpred = self._vpreds.reshape((self._size*num_processes, -1))[idxs]
        adv = self._adv.reshape((self._size*num_processes, -1))[idxs]
        ret = self._ret.reshape((self._size*num_processes, -1))[idxs]
        log_prob = self._log_prob.reshape((self._size*num_processes, -1))[idxs]

        return {
            'ob': ob,
            'ac': action,
            'ac_before_activation': ac_before_activation,
            'done': done,
            'ret': ret,
            'adv': adv,
            'log_prob': log_prob
        }


    def generator(self, batch_size):
        num_processes = self._rewards.shape[1]
        sampler = BatchSampler(SubsetRandomSampler(range(self._size*num_processes)),
                               batch_size,
                               drop_last=True)
        for idxs in sampler:
            transitions = {}
            ob = OrderedDict([(k, v.reshape(self._size*num_processes, -1)[idxs]) for k, v in self._obs.items()])
            # ob_next = OrderedDict([(k, v[idxs]) for k, v in self._obs_next.items()])
            # reward = self._rewards.reshape((self._size*num_processes, -1))[idxs]
            done = self._terminals.reshape((self._size*num_processes, -1))[idxs]
            action = OrderedDict([(k, v.reshape(self._size*num_processes, -1)[idxs]) for k, v in self._actions.items()])
            ac_before_activation = OrderedDict([(k, v.reshape(self._size*num_processes, -1)[idxs]) for k, v in self._ac_before_activation.items()])
            vpred = self._vpreds.reshape((self._size*num_processes, -1))[idxs]
            adv = self._adv.reshape((self._size*num_processes, -1))[idxs]
            ret = self._ret.reshape((self._size*num_processes, -1))[idxs]
            log_prob = self._log_prob.reshape((self._size*num_processes, -1))[idxs]
            yield {
                'ob': ob,
                'ac': action,
                'ac_before_activation': ac_before_activation,
                'done': done,
                'ret': ret,
                'adv': adv,
                'vpred': vpred,
                'log_prob': log_prob
            }

    def clear(self):
        self._idx = 0
        self._current_size = 0
        buffer_size = self._size
        num_processes = self._num_processes
        self._obs = {k: np.empty((buffer_size, num_processes, observation_size(self._ob_space[k]))) for k in self._ob_space.spaces.keys()}
        self._obs_next = {k: np.empty((buffer_size, num_processes, observation_size(self._ob_space[k]))) for k in self._ob_space.spaces.keys()}
        self._actions = {k: np.empty((buffer_size, num_processes, action_size(self._ac_space[k]))) for k in self._ac_space.spaces.keys()}
        self._ac_before_activation = {k: np.empty((buffer_size, num_processes, action_size(self._ac_space[k]))) for k in self._ac_space.spaces.keys()}
        self._rewards = np.empty((buffer_size, num_processes, 1))
        self._terminals = np.empty((buffer_size, num_processes, 1))
        self._vpreds = np.empty((buffer_size, num_processes, 1))
        self._adv = np.empty((buffer_size, num_processes, 1))
        self._ret = np.empty((buffer_size, num_processes, 1))
        self._log_prob = np.empty((buffer_size, num_processes, 1))
