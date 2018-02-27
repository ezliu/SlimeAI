# Adapted from OpenAI Gym Baselines
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import collections
import numpy as np
import random


class Experience(collections.namedtuple(
        'Experience', ['state', 'action', 'reward', 'next_state', 'done'])):
    pass


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, experience):
        if self._next_idx >= len(self._storage):
            self._storage.append(experience)
        else:
            self._storage[self._next_idx] = experience
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Args:
            batch_size (int): How many transitions to sample.

        Returns:
            list[Experience]: sampled experiences, not necessarily unique
        """
        indices = np.random.choice(range(len(self._storage)), size=batch_size)
        return [self._storage[i] for i in indices]
