from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from gym import spaces
from util.gym import observation_size, action_size
from rl.policies.utils import CNN, MLP

class DQN(nn.Module):
    def __init__(self, config, ob_space, ac_space, rl_hid_size=None):
        super(DQN, self).__init__()
        if rl_hid_size is None:
            rl_hid_size = config.rl_hid_size

        input_dim = observation_size(ob_space)
        self._ac_space = ac_space
        self._ob_space = ob_space

        self.fc = MLP(config, input_dim, rl_hid_size, [rl_hid_size]*config.actor_num_hid_layers, activation='relu')
        self._activation_fn = getattr(F, 'relu')
        self.out = MLP(config, rl_hid_size, self._ac_space['default'].n, activation='relu')

    def forward(self, ob):
        inp = list(ob.values())
        if len(inp[0].shape) == 1:
            inp = [x.unsqueeze(0) for x in inp]
        inp = torch.cat(inp, dim=-1)
        out = self._activation_fn(self.fc(inp))
        return self.out(out)

# class CnnDQN(nn.Module):
#     def __init__(self, inputs_shape, num_actions):
#         super(CnnDQN, self).__init__()
#
#         self.inut_shape = inputs_shape
#         self.num_actions = num_actions
#
#         self.features = nn.Sequential(
#             nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )
#
#         self.fc = nn.Sequential(
#             nn.Linear(self.features_size(), 512),
#             nn.ReLU(),
#             nn.Linear(512, self.num_actions)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
#
#     def features_size(self):
#         return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)
