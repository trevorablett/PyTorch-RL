from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))

TransitionWithAux = namedtuple('TransitionWithAux', ('state', 'action', 'mask', 'next_state',
                                       'reward', 'aux_state', 'aux_next_state'))


class Memory(object):
    def __init__(self, include_aux_state=False):
        self.memory = []
        if include_aux_state:
            self.trans_tuple = TransitionWithAux
        else:
            self.trans_tuple = Transition

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(self.trans_tuple(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.trans_tuple(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return self.trans_tuple(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)
