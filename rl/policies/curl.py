# reference: https://github.com/MishaLaskin/curl/blob/master/curl_sac.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CURL(nn.Module):
    def __init__(self, config, ob_space, ac_space, critic, critic_target):
        super().__init__()
        self._config = config
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._encoder = critic.base
        self._encoder_target = critic_target.base

        self._w = nn.Parameter(torch.rand(config.rl_hid_size, config.rl_hid_size))

    def encode(self, ob, detach=False, ema=False):
        if ema:
            with torch.no_grad():
                z_out = self._encoder_target(ob)
        else:
            z_out = self._encoder(ob)

        if detach:
            z_out = z_out.detach()

        return z_out

    def compute_logits(self, z_a, z_pos):
        Wz = torch.matmul(self.W, z_pos.T)
        logits = torch.matmul(z_a, Wz)
        logits = logits - torch.max(logits, 1)[0][:, None]


