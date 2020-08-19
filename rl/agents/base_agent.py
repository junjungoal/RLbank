from collections import OrderedDict
from util.pytorch import to_tensor
from util.image import center_crop_image
import copy

class BaseAgent(object):
    def __init__(self, config, ob_space):
        self._config = config

    def act(self, ob, is_train=True, pred_value=False, return_log_prob=False):
        if self._config.policy == 'cnn' and ob['default'].shape[-1] != self._config.img_height:
            ob = copy.deepcopy(ob)
            ob['default'] = center_crop_image(ob['default'], self._config.img_height)
        ret = self._actor.act(ob, is_train=is_train, return_log_prob=return_log_prob)
        ret = list(ret)
        if pred_value:
            ret.append(self.value(ob))
        return tuple(ret)

    def value(self, ob):
        return self._critic(to_tensor(ob, self._config.device)).detach().cpu().numpy()


    def store_episode(self, rollouts):
        raise NotImplementedError()

    def replay_buffer(self):
        return self._buffer.state_dict()

    def load_replay_buffer(self, state_dict):
        self._buffer.load_state_dict(state_dict)

    def train(self):
        raise NotImplementedError

    def _soft_update_target_network(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * param.data +
                                    tau * target_param.data)

    def _copy_target_network(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)



