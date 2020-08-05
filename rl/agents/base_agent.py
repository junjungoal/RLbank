from collections import OrderedDict

class BaseAgent(object):
    def __init__(self, config, ob_space):
        self._config = config

    def act(self, ob, is_train=True):
        ac, activation = self._actor.act(ob, is_train=is_train)
        return ac, activation

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



