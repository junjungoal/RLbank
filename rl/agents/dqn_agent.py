# SAC training code reference
# https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym import spaces

from rl.dataset import ReplayBuffer, RandomSampler
from rl.agents.base_agent import BaseAgent
from util.logger import logger
from util.gym import action_size, observation_size
from util.pytorch import optimizer_cuda, count_parameters, to_tensor

class DQNAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space,
                 dqn):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space
        # build up networks
        self._dqn = dqn(config, ob_space, ac_space)
        self._network_cuda(config.device)

        self._dqn_optim = optim.Adam(self._dqn.parameters(), lr=config.lr_actor)
        sampler = RandomSampler()
        self._buffer = ReplayBuffer(config.buffer_size,
                                    sampler.sample_func,
                                    ob_space,
                                    ac_space)


    def _log_creation(self):
        logger.info("Creating a DQN agent")
        logger.info("The DQN has %d parameters".format(count_parameters(self._dqn)))

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)

    def store_sample(self, rollouts):
        self._buffer.store_sample(rollouts)

    def _network_cuda(self, device):
        self._dqn.to(device)

    def state_dict(self):
        return {
            'dqn_state_dict': self._dqn.state_dict(),
            'dqn_optim_state_dict': self._dqn_optim.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._dqn.load_state_dict(ckpt['dqn_state_dict'])

        self._network_cuda(self._config.device)
        self._dqn_optim.load_state_dict(ckpt['dqn_optim_state_dict'])
        optimizer_cuda(self._dqn_optim, self._config.device)

    def train(self):
        for _ in range(self._config.num_batches):
            transitions = self._buffer.sample(self._config.batch_size)
            train_info = self._update_network(transitions)

        return train_info

    def act_log(self, o):
        raise NotImplementedError

    def act(self, o):
        o = to_tensor(o, self._config.device)
        q_value = self._dqn(o)
        action = OrderedDict([('default', q_value.max(1)[1].item())])
        return action, None

    def _update_network(self, transitions):
        info = {}

        # pre-process observations
        o, o_next = transitions['ob'], transitions['ob_next']

        bs = len(transitions['done'])
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions['ac'])
        ac = ac['default'].to(torch.long)

        done = _to_tensor(transitions['done']).reshape(bs, 1)
        rew = _to_tensor(transitions['rew']).reshape(bs, 1)

        with torch.no_grad():
            q_next_values = self._dqn(o)
            q_next_value = q_next_values.max(1)[0]
            target_q_value = rew + \
                (1-done)  * self._config.discount_factor * q_next_value
            target_q_value = target_q_value.detach()

        q_values = self._dqn(o)
        q_value = q_values.gather(1, ac[:, 0].unsqueeze(1)).squeeze(1)
        info['target_q'] = target_q_value.mean().cpu().item()
        info['real_q'] = q_value.mean().cpu().item()
        loss = (q_value - target_q_value).pow(2).mean()
        self._dqn_optim.zero_grad()
        loss.backward()
        self._dqn_optim.step()
        return info


