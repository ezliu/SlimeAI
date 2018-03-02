import abc
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from action import Action
from utils import GPUVariable


class Agent(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def act(self, state):
        """Takes a single state and returns an action to play in that state.

        Args:
            state (np.array)

        Returns:
            int
        """
        raise NotImplementedError()


class RandomAgent(Agent):
    def __init__(self, num_actions):
        super(RandomAgent, self).__init__()
        self._num_actions = num_actions

    def act(self, state):
        return Action(random.randint(0, self._num_actions - 1))


class DQNAgent(nn.Module):
    def __init__(self, num_actions, epsilon=1.):
        super(DQNAgent, self).__init__()
        self._Q = DQN(num_actions, StateEmbedder())
        self._target_Q = DQN(num_actions, StateEmbedder())
        self._epsilon = epsilon

    def act(self, state):
        state = GPUVariable(torch.FloatTensor(np.expand_dims(state, 0)))
        q_values = self._Q(state)
        self._epsilon -= 0.0000009  # TODO: Put a real schedule here
        self._epsilon = max(self._epsilon, 0.1)
        return Action(epsilon_greedy(q_values, self._epsilon)[0])

    def update_from_experiences(self, experiences, take_grad_step):
        gamma = 0.99  # TODO: Fix this

        batch_size = len(experiences)
        states = GPUVariable(torch.FloatTensor(
            np.array([np.array(e.state) for e in experiences])))
        actions = GPUVariable(torch.LongTensor(
            np.array([np.array(e.action) for e in experiences])))
        next_states = GPUVariable(torch.FloatTensor(
            np.array([np.array(e.next_state) for e in experiences])))
        rewards = GPUVariable(
            torch.FloatTensor(np.array([e.reward for e in experiences])))

        # (batch_size,) 1 if was not done, otherwise 0
        not_done_mask = GPUVariable(
                torch.FloatTensor(np.array([1 - e.done for e in experiences])))

        current_state_q_values = self._Q(states).gather(
                1, actions.unsqueeze(1))

        # DDQN
        best_actions = torch.max(self._Q(next_states), 1)[1].unsqueeze(1)
        next_state_q_values = self._target_Q(
                next_states).gather(1, best_actions).squeeze(1)
        targets = rewards + gamma * (next_state_q_values * not_done_mask)
        targets.detach_()  # Don't backprop through targets

        loss_fn = nn.MSELoss()
        take_grad_step(self._Q, loss_fn(current_state_q_values, targets))

    def sync_target(self):
        """Syncs the target Q values with the current Q values"""
        self._target_Q.load_state_dict(self._Q.state_dict())


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
