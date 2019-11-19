from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))


class Memory(object):
    def __init__(self, extra_fields=()):
        self.memory = []
        fields = ['state', 'action', 'mask', 'next_state', 'reward']
        fields.extend(extra_fields)
        self.trans_tuple = namedtuple('Transition', tuple(fields))

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
