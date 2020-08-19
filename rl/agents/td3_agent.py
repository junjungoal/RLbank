from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl.dataset import ReplayBuffer
from rl.agents.base_agent import BaseAgent
from util.logger import logger
from util.pytorch import optimizer_cuda, count_parameters, to_tensor
from util.gym import action_size, observation_size
from gym import spaces

class TD3Agent(BaseAgent):
    def __init__(self, config, ob_space, ac_space,
                 actor, critic):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        self._log_alpha = [torch.zeros(1, requires_grad=True, device=config.device)]
        self._alpha_optim = [optim.Adam([self._log_alpha[0]], lr=config.lr_actor)]

        self._actor = actor(self._config, self._ob_space,
                              self._ac_space, self._config.tanh_policy, deterministic=True)
        self._actor_target = actor(self._config, self._ob_space,
                              self._ac_space, self._config.tanh_policy, deterministic=True)
        self._actor_target.load_state_dict(self._actor.state_dict())
        self._critic1 = critic(config, ob_space, ac_space)
        self._critic2 = critic(config, ob_space, ac_space)
        self._critic1_target = critic(config, ob_space, ac_space)
        self._critic2_target = critic(config, ob_space, ac_space)
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())

        self._network_cuda(config.device)

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor)
        self._critic1_optim = optim.Adam(self._critic1.parameters(), lr=config.lr_critic)
        self._critic2_optim = optim.Adam(self._critic2.parameters(), lr=config.lr_critic)

        self._buffer = ReplayBuffer(config,
                                    ob_space,
                                    ac_space)

        self._ounoise = OUNoise(action_size(ac_space))

        self._log_creation()

    def _log_creation(self):
        logger.info('creating a TD3 agent')
        logger.info('the actor has %d parameters', count_parameters(self._actor))
        logger.info('the critic1 has %d parameters', count_parameters(self._critic1))
        logger.info('the critic2 has %d parameters', count_parameters(self._critic2))

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)

    def store_sample(self, rollouts):
        self._buffer.store_sample(rollouts)

    def state_dict(self):
        return {
            'actor_state_dict': self._actor.state_dict(),
            'critic1_state_dict': self._critic1.state_dict(),
            'critic2_state_dict': self._critic2.state_dict(),
            'actor_optim_state_dict': self._actor_optim.state_dict(),
            'critic1_optim_state_dict': self._critic1_optim.state_dict(),
            'critic2_optim_state_dict': self._critic2_optim.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._actor.load_state_dict(ckpt['actor_state_dict'])
        self._actor_target.load_state_dict(self._actor.state_dict())
        self._critic1.load_state_dict(ckpt['critic1_state_dict'])
        self._critic2.load_state_dict(ckpt['critic2_state_dict'])
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())
        self._network_cuda(self._config.device)

        self._actor_optim.load_state_dict(ckpt['actor_optim_state_dict'])
        self._critic1_optim.load_state_dict(ckpt['critic1_optim_state_dict'])
        self._critic2_optim.load_state_dict(ckpt['critic2_optim_state_dict'])
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic1_optim, self._config.device)
        optimizer_cuda(self._critic2_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._actor_target.to(device)
        self._critic1.to(device)
        self._critic2.to(device)
        self._critic1_target.to(device)
        self._critic2_target.to(device)

    def train(self):
        config = self._config
        for i in range(config.num_batches):
            transitions = self._buffer.sample(config.batch_size)
            train_info = self._update_network(transitions, step=i)
            self._soft_update_target_network(self._actor_target, self._actor, self._config.polyak)
            self._soft_update_target_network(self._critic1_target, self._critic1, self._config.polyak)
            self._soft_update_target_network(self._critic2_target, self._critic2, self._config.polyak)
        return train_info

    def act_log(self, ob):
        return self._actor.act_log(ob)

    def act(self, ob, is_train=True):
        ob = to_tensor(ob, self._config.device)
        ac, activation = self._actor.act(ob, is_train=is_train)
        if is_train:
            for k, space in self._ac_space.spaces.items():
                if isinstance(space, spaces.Box):
                    ac[k] += self._config.noise_scale*np.random.randn(len(ac[k]))
                    ac[k] = np.clip(ac[k], self._ac_space[k].low, self._ac_space[k].high)
        return ac, activation

    def target_act(self, ob, is_train=True):
        ac, activation = self._actor_target.act(ob, is_train=is_train)
        return ac, activation

    def target_act_log(self, ob):
        return self._actor_target.act_log(ob)

    def _update_network(self, transitions, step=0):
        config = self._config
        info = {}

        o, o_next = transitions['ob'], transitions['ob_next']
        bs = len(transitions['done'])
        _to_tensor = lambda x: to_tensor(x, config.device)
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions['ac'])

        done = _to_tensor(transitions['done']).reshape(bs, 1)
        rew = _to_tensor(transitions['rew']).reshape(bs, 1)

        ## Actor loss
        actions_real, _ = self.act_log(o)
        actor_loss = -self._critic1(o, actions_real).mean()
        info['actor_loss'] = actor_loss.cpu().item()

        ## Critic loss
        with torch.no_grad():
            actions_next, _ = self.target_act_log(o_next)
            for k, space in self._ac_space.spaces.items():
                if isinstance(space, spaces.Box):
                    epsilon = torch.randn_like(actions_next[k]) * self._config.noise_scale
                    epsilon = torch.clamp(epsilon, -config.noise_clip, config.noise_clip)
                    actions_next[k] += epsilon
            q_next_value1 = self._critic1_target(o_next, actions_next)
            q_next_value2 = self._critic2_target(o_next, actions_next)
            q_next_value = torch.min(q_next_value1, q_next_value2)
            target_q_value = rew + (1.-done) * config.discount_factor * q_next_value
            target_q_value = target_q_value.detach()

        real_q_value1 = self._critic1(o, ac)
        real_q_value2 = self._critic2(o, ac)

        critic1_loss = 0.5 * (target_q_value - real_q_value1).pow(2).mean()
        critic2_loss = 0.5 * (target_q_value - real_q_value2).pow(2).mean()

        info['min_target_q'] = target_q_value.min().cpu().item()
        info['target_q'] = target_q_value.mean().cpu().item()
        info['min_real1_q'] = real_q_value1.min().cpu().item()
        info['min_real2_q'] = real_q_value2.min().cpu().item()
        info['real1_q'] = real_q_value1.mean().cpu().item()
        info['rea2_q'] = real_q_value2.mean().cpu().item()
        info['critic1_loss'] = critic1_loss.cpu().item()
        info['critic2_loss'] = critic2_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        # update the critics
        self._critic1_optim.zero_grad()
        critic1_loss.backward()
        self._critic1_optim.step()

        self._critic2_optim.zero_grad()
        critic2_loss.backward()
        self._critic2_optim.step()


        return info

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
