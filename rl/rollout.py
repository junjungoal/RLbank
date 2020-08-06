from collections import defaultdict

import numpy as np
import torch
import cv2

from util.logger import logger


class Rollout(object):
    """ Rollout storage. """
    def __init__(self):
        self._history = defaultdict(list)

    def add(self, data):
        for key, value in data.items():
            self._history[key].append(value)

    def get(self):
        batch = {}
        batch['ob'] = self._history['ob']
        batch['ac'] = self._history['ac']
        batch['ac_before_activation'] = self._history['ac_before_activation']
        batch['done'] = self._history['done']
        batch['rew'] = self._history['rew']
        batch['ob_next'] = self._history['ob_next']
        self._history = defaultdict(list)
        return batch
