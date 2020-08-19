# SAC training code reference
# https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym import spaces

from rl.dataset import ReplayBuffer
from rl.agents.base_agent import BaseAgent
from rl.policies.curl import CURL
from util.logger import logger
from util.gym import action_size, observation_size
from util.pytorch import optimizer_cuda, count_parameters, to_tensor

class SACAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space,
                 actor, critic):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        self._log_alpha = torch.tensor(np.log(config.alpha), requires_grad=True, device=config.device)
        self._alpha_optim = optim.Adam([self._log_alpha], lr=config.lr_actor)

        # build up networks
        self._actor = actor(config, ob_space, ac_space, config.tanh_policy)
        self._critic1 = critic(config, ob_space, ac_space)
        self._critic2 = critic(config, ob_space, ac_space)

        self._target_entropy = -action_size(self._actor._ac_space)

        # build up target networks
        self._critic1_target = critic(config, ob_space, ac_space)
        self._critic2_target = critic(config, ob_space, ac_space)
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())

        if config.policy == 'cnn':
            self._critic2.base.copy_conv_weights_from(self._critic1.base)
            self._actor.base.copy_conv_weights_from(self._critic1.base)

            if config.unsup_algo == 'curl':
                self._curl = CURL(config, ob_space, ac_space, self._critic1, self._critic1_target)
                self._encoder_optim = optim.Adam(self._critic1.base.parameters(), lr=config.lr_encoder)
                self._cpc_optim = optim.Adam(self._curl.parameters(), lr=config.lr_encoder)


        self._network_cuda(config.device)

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor)
        self._critic1_optim = optim.Adam(self._critic1.parameters(), lr=config.lr_critic)
        self._critic2_optim = optim.Adam(self._critic2.parameters(), lr=config.lr_critic)

        self._buffer = ReplayBuffer(config,
                                    ob_space,
                                    ac_space)


    def _log_creation(self):
        logger.info("Creating a SAC agent")
        logger.info("The actor has %d parameters".format(count_parameters(self._actor)))
        logger.info('The critic1 has %d parameters', count_parameters(self._critic1))
        logger.info('The critic2 has %d parameters', count_parameters(self._critic2))

    def store_sample(self, rollouts):
        self._buffer.store_sample(rollouts)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._critic1.to(device)
        self._critic2.to(device)
        self._critic1_target.to(device)
        self._critic2_target.to(device)
        if self._config.policy == 'cnn' and self._config.unsup_algo == 'curl':
            self._curl.to(device)

    def state_dict(self):
        ret = {
            'log_alpha': self._log_alpha.cpu().detach().numpy(),
            'actor_state_dict': self._actor.state_dict(),
            'critic1_state_dict': self._critic1.state_dict(),
            'critic2_state_dict': self._critic2.state_dict(),
            'alpha_optim_state_dict': self._alpha_optim.state_dict(),
            'actor_optim_state_dict': self._actor_optim.state_dict(),
            'critic1_optim_state_dict': self._critic1_optim.state_dict(),
            'critic2_optim_state_dict': self._critic2_optim.state_dict(),
        }
        if self._config.policy == 'cnn' and self._config.unsup_algo == 'curl':
            ret['curl_state_dict'] = self._curl.state_dict()
            ret['encoder_optim_state_dict'] = self._encoder_optim.state_dict()
            ret['cpc_optim_state_dict'] = self._cpc_optim.state_dict()

    def load_state_dict(self, ckpt):
        self._log_alpha.data = torch.tensor(ckpt['log_alpha'], requires_grad=True,
                                            device=self._config.device)
        self._actor.load_state_dict(ckpt['actor_state_dict'])
        self._critic1.load_state_dict(ckpt['critic1_state_dict'])
        self._critic2.load_state_dict(ckpt['critic2_state_dict'])
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())


        self._alpha_optim.load_state_dict(ckpt['alpha_optim_state_dict'])
        self._actor_optim.load_state_dict(ckpt['actor_optim_state_dict'])
        self._critic1_optim.load_state_dict(ckpt['critic1_optim_state_dict'])
        self._critic2_optim.load_state_dict(ckpt['critic2_optim_state_dict'])
        optimizer_cuda(self._alpha_optim, self._config.device)
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic1_optim, self._config.device)
        optimizer_cuda(self._critic2_optim, self._config.device)

        if self._config.policy == 'cnn' and self._config.unsup_algo == 'curl':
            self._curl.load_state_dict(ckpt['curl_state_dict'])
            self._encoder_optim.load_state_dict(ckpt['encoder_optim_state_dict'])
            self._cpc_optim.load_state_dict(ckpt['cpc_optim_state_dict'])
            optimizer_cuda(self._encoder_optim, self._config.device)
            optimizer_cuda(self._cpc_optim, self._config.device)

        self._network_cuda(self._config.device)

    def train(self):
        for _ in range(self._config.num_batches):
            if self._config.policy == 'cnn' and self._config.unsup_algo == 'curl':
                transitions = self._buffer.sample_cpc(self._config.batch_size)
            else:
                transitions = self._buffer.sample(self._config.batch_size)
            train_info = self._update_network(transitions)
            self._soft_update_target_network(self._critic1_target, self._critic1, self._config.polyak)
            self._soft_update_target_network(self._critic2_target, self._critic2, self._config.polyak)

        return train_info

    def act_log(self, o):
        return self._actor.act_log(o)

    def _update_critic(self, o, ac, rew, o_next, done):
        info = {}
        alpha = self._log_alpha.exp()
        with torch.no_grad():
            actions_next, log_pi_next = self.act_log(o_next)
            q_next_value1 = self._critic1_target(o_next, actions_next)
            q_next_value2 = self._critic2_target(o_next, actions_next)
            q_next_value = torch.min(q_next_value1, q_next_value2) - alpha * log_pi_next
            target_q_value = rew * self._config.reward_scale + \
                (1-done)  * self._config.discount_factor * q_next_value
            target_q_value = target_q_value.detach()

        # q loss
        real_q_value1 = self._critic1(o, ac)
        real_q_value2 = self._critic2(o, ac)
        critic1_loss = 0.5 * (target_q_value - real_q_value1).pow(2).mean()
        critic2_loss = 0.5 * (target_q_value - real_q_value2).pow(2).mean()

        info['min_target_q'] = target_q_value.min().cpu().item()
        info['target_q'] = target_q_value.mean().cpu().item()
        info['min_real1_q'] = real_q_value1.min().cpu().item()
        info['min_real2_q'] = real_q_value2.min().cpu().item()
        info['real1_q'] = real_q_value1.mean().cpu().item()
        info['real2_q'] = real_q_value2.mean().cpu().item()
        info['critic1_loss'] = critic1_loss.cpu().item()
        info['critic2_loss'] = critic2_loss.cpu().item()
        return info

    def _update_actor_and_alpha(self, o):
        info = {}
        actions_real, log_pi = self.act_log(o)
        alpha_loss = -(self._log_alpha * (log_pi + self._target_entropy).detach()).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()
        alpha = self._log_alpha.exp()

        # actor loss
        entropy_loss = (alpha * log_pi).mean()
        actor_loss = -torch.min(self._critic1(o, actions_real),
                                self._critic2(o, actions_real)).mean()
        info['entropy_alpha'] = alpha.cpu().item()
        info['entropy_loss'] = entropy_loss.cpu().item()
        info['actor_loss'] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()
        return info

    def _update_cpc(self, o_anchor, o_pos, cpc_kwargs):
        info = {}
        z_a = self._curl.encode(o_anchor)
        z_pos = self._curl.encode(o_pos, ema=True)
        logits = self._curl.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self._config.device)
        cpc_loss = F.cross_entropy(logits, labels)
        info['cpc_loss'] = cpc_loss.cpu().item()

        self._encoder_optim.zero_grad()
        self._cpc_optim.zero_grad()
        cpc_loss.backward()
        self._encoder_optim.step()
        self._cpc_optim.step()
        return info

    def _update_network(self, transitions):
        info = {}

        # pre-process observations
        o, o_next = transitions['ob'], transitions['ob_next']

        bs = len(transitions['done'])
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions['ac'])

        done = _to_tensor(transitions['done']).reshape(bs, 1)
        rew = _to_tensor(transitions['rew']).reshape(bs, 1)
        # update alpha
        critic_info = self._update_critic(o, ac, rew, o_next, done)
        info.update(critic_info)
        actor_alpha_info = self._update_actor_and_alpha(o)
        info.update(actor_alpha_info)

        if self._config.policy == 'cnn' and self._config.unsup_algo == 'curl':

            cpc_kwargs = transitions['cpc_kwargs']
            o_anchor = _to_tensor(cpc_kwargs['ob_anchor'])
            o_pos = _to_tensor(cpc_kwargs['ob_pos'])
            cpc_info = self._update_cpc(o_anchor, o_pos, cpc_kwargs)
            info.update(cpc_info)

        return info


