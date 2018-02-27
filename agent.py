import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from action import Action
from utils import GPUVariable


class Agent(object):
    def __init__(self, num_actions, epsilon=1.):
        self._Q = DQN(num_actions, StateEmbedder())
        self._target_Q = DQN(num_actions, StateEmbedder())
        self._epsilon = epsilon

    def act(self, state):
        state = GPUVariable(torch.FloatTensor(np.expand_dims(state, 0)))
        q_values = self._Q(state)
        return Action(epsilon_greedy(q_values, self._epsilon)[0])

    def update_from_experiences(self, experiences):
        pass


class DQN(nn.Module):
    def __init__(self, num_actions, state_embedder):
        super(DQN, self).__init__()
        self._state_embedder = state_embedder
        self._q_values = nn.Linear(self._state_embedder.embed_dim, num_actions)

    def forward(self, states):
        """Returns Q-values for each of the states.

        Args:
            states (FloatTensor): shape (batch_size, ...)

        Returns:
            FloatTensor: (batch_size, num_actions)
        """
        return self._q_values(self._state_embedder(states))


class StateEmbedder(nn.Module):
    def __init__(self):
        super(StateEmbedder, self).__init__()
        self._layer1 = nn.Linear(12, 128)
        self._layer2 = nn.Linear(128, 128)
        self._layer3 = nn.Linear(128, 128)
        self._layer4 = nn.Linear(128, 128)

    def forward(self, states):
        """Embeds the states

        Args:
            states (FloatTensor): shape (batch_size, 12)

        Returns:
            FloatTensor: (batch_size, 128)
        """
        hidden = F.relu(self._layer1(states))
        hidden = F.relu(self._layer2(hidden))
        hidden = F.relu(self._layer3(hidden))
        return F.relu(self._layer4(hidden))

    @property
    def embed_dim(self):
        return 128


def epsilon_greedy(q_values, epsilon):
    """Returns the index of the highest q value with prob 1 - epsilon,
    otherwise uniformly at random with prob epsilon.

    Args:
        q_values (Variable[FloatTensor]): (batch_size, num_actions)
        epsilon (float)

    Returns:
        list[int]: actions
    """
    batch_size, num_actions = q_values.size()
    _, max_indices = torch.max(q_values, 1)
    max_indices =  max_indices.cpu().data.numpy()
    actions = []
    for i in xrange(batch_size):
        if random.random() > epsilon:
            actions.append(max_indices[i])
        else:
            actions.append(random.randint(0, num_actions - 1))
    return actions
