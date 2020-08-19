from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.dataset import ReplayBuffer, MultiProcessReplayBuffer
from rl.agents.base_agent import BaseAgent
from util.logger import logger
from util.pytorch import optimizer_cuda, count_parameters, \
    obs2tensor, to_tensor, list2dict


class PPOAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space,
                 actor, critic):
        super().__init__(config, ob_space)

        self._ac_space = ac_space
        # build up networks
        self._actor = actor(config, ob_space, ac_space, config.tanh_policy, activation='tanh')
        self._old_actor = actor(config, ob_space, ac_space, config.tanh_policy, activation='tanh')
        self._critic = critic(config, ob_space, activation='tanh')
        self._network_cuda(config.device)

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=config.lr_critic)

        # self._buffer = ReplayBuffer(['ob', 'ac', 'done', 'rew', 'ret', 'adv', 'ac_before_activation'],
        self._buffer = MultiProcessReplayBuffer(config,
                                    config.num_processes,
                                    ob_space, ac_space)


        logger.info('Creating a PPO agent')
        logger.info('The actor has %d parameters', count_parameters(self._actor))
        logger.info('The critic has %d parameters', count_parameters(self._critic))

    def store_episode(self, rollouts):
        self._compute_gae(rollouts)
        self._buffer.store_episode(rollouts)

    def store_sample(self, rollout):
        self._buffer.store_sample(rollout)

    def _compute_gae(self, rollouts):
        """ Computes GAE from @rollouts. """
        config = self._config
        T = len(rollouts['done'])
        vpred = np.array(rollouts['vpred']).squeeze(-1)
        assert len(vpred) == T + 1

        done = rollouts['done']
        rew = rollouts['rew']
        adv = np.empty((T, config.num_processes) , 'float32')
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - done[t]
            delta = rew[t] + self._config.discount_factor * vpred[t + 1] * nonterminal - vpred[t]
            adv[t] = lastgaelam = delta + self._config.discount_factor * self._config.gae_lambda * nonterminal * lastgaelam

        ret = adv + vpred[:-1]

        assert np.isfinite(adv).all()
        assert np.isfinite(ret).all()

        # update rollouts
        if adv.std() == 0:
            rollouts['adv'] = (adv * 0)
        else:
            rollouts['adv'] = ((adv - adv.mean()) / adv.std())
        rollouts['ret'] = ret

    def state_dict(self):
        return {
            'actor_state_dict': self._actor.state_dict(),
            'critic_state_dict': self._critic.state_dict(),
            'actor_optim_state_dict': self._actor_optim.state_dict(),
            'critic_optim_state_dict': self._critic_optim.state_dict(),
            'ob_norm_state_dict': self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._actor.load_state_dict(ckpt['actor_state_dict'])
        self._critic.load_state_dict(ckpt['critic_state_dict'])
        self._ob_norm.load_state_dict(ckpt['ob_norm_state_dict'])
        self._network_cuda(self._config.device)

        self._actor_optim.load_state_dict(ckpt['actor_optim_state_dict'])
        self._critic_optim.load_state_dict(ckpt['critic_optim_state_dict'])
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._old_actor.to(device)
        self._critic.to(device)

    def train(self):
        self._soft_update_target_network(self._old_actor, self._actor, 0.0)
        for _ in range(self._config.ppo_epoch):
            generator = self._buffer.generator(self._config.batch_size)
            for transitions in generator:
                train_info = self._update_network(transitions)

        self._buffer.clear()

        return train_info

    def _update_network(self, transitions):
        info = {}

        # pre-process observations
        o = transitions['ob']
        # o = self.normalize(o)
        bs = len(transitions['done'])
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        ac = _to_tensor(transitions['ac'])
        a_z = _to_tensor(transitions['ac_before_activation'])
        ret = _to_tensor(transitions['ret']).reshape(bs, 1)
        adv = _to_tensor(transitions['adv']).reshape(bs, 1)
        vpred = _to_tensor(transitions['vpred']).reshape(bs, 1)
        old_log_pi = _to_tensor(transitions['log_prob']).reshape(bs, 1)

        log_pi, ent = self._actor.act_log(o, a_z)
        # old_log_pi, _ = self._old_actor.act_log(o, a_z)
        # if old_log_pi.min() < -100:
        #     import ipdb; ipdb.set_trace()

        # the actor loss
        entropy_loss = self._config.entropy_loss_coeff * ent.mean()
        ratio = torch.exp(log_pi - old_log_pi.detach())
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self._config.clip_param,
                            1.0 + self._config.clip_param) * adv
        actor_loss = -torch.min(surr1, surr2).mean()

        if not np.isfinite(ratio.cpu().detach()).all() or not np.isfinite(adv.cpu().detach()).all():
            import ipdb; ipdb.set_trace()
        info['entropy_loss'] = entropy_loss.cpu().item()
        info['actor_loss'] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # the q loss
        value_pred = self._critic(o)
        value_pred_clipped = vpred + (value_pred - vpred).clamp(-self._config.clip_param, self._config.clip_param)
        value_loss = (value_pred-ret).pow(2)
        value_loss_clipped = (value_pred_clipped-ret).pow(2)
        value_loss = self._config.value_loss_coeff * torch.max(value_loss, value_loss_clipped).mean()

        # value_loss = self._config.value_loss_coeff * (ret - value_pred).pow(2).mean()

        info['value_target'] = ret.mean().cpu().item()
        info['value_predicted'] = value_pred.mean().cpu().item()
        info['value_loss'] = value_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        if self._config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._config.max_grad_norm)
        self._actor_optim.step()

        # update the critic
        self._critic_optim.zero_grad()
        value_loss.backward()
        if self._config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._critic.parameters(), self._config.max_grad_norm)
        self._critic_optim.step()

        # include info from policy
        info.update(self._actor.info)

        return info
