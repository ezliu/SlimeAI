import abc
import collections
import numpy as np
import random
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from action import Action
from instance import ObservationMode
from torch.nn.utils import clip_grad_norm
from utils import GPUVariable


class Agent(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name=None):
        super(Agent, self).__init__()
        if name is None:
            self._name = ''.join(
                random.choice(string.ascii_uppercase + string.digits)
                for _ in range(10))
        else:
            self._name = name

    @abc.abstractmethod
    def act(self, state, test=False):
        """Takes a single state and returns an action to play in that state.

        Args:
            state (np.array)
            test (bool): True in test mode

        Returns:
            int
        """
        raise NotImplementedError()

    @property
    def name(self):
        return self._name

    @property
    def stats(self):
        return {}

    def reset(self):
        pass


class RandomAgent(Agent):
    def __init__(self, num_actions, name=None):
        super(RandomAgent, self).__init__(name)
        self._num_actions = num_actions

    def act(self, state, test=False):
        return Action(random.randint(0, self._num_actions - 1))


class DQNAgent(Agent):
    def __init__(self, num_actions, epsilon_schedule, observation_mode,
                 lr=0.00025, max_grad_norm=10., name=None):
        super(DQNAgent, self).__init__(name)
        if observation_mode == ObservationMode.PIXEL:
            embedder = PixelStateEmbedder
        elif observation_mode == ObservationMode.RAM:
            embedder = StructuredStateEmbedder
        else:
            raise ValueError(
                    "{} not a valid observation mode".format(observation_mode))

        self._Q = DQN(num_actions, embedder())
        self._target_Q = DQN(num_actions, embedder())
        self._epsilon_schedule = epsilon_schedule
        self._epsilon = 1.
        self._max_q = collections.deque(maxlen=1000)
        self._min_q = collections.deque(maxlen=1000)
        self._avg_loss = collections.deque(maxlen=1000)
        self._avg_loss.append(0.)
        self._optimizer = optim.Adam(self._Q.parameters(), lr=lr)
        self._grad_clip_norm = max_grad_norm

    def act(self, state, test=False):
        state = GPUVariable(torch.FloatTensor(np.expand_dims(state, 0)))
        q_values = self._Q(state)
        if test:
            epsilon = 0.05
        else:
            epsilon = self._epsilon_schedule.get_epsilon()
        self._epsilon = epsilon
        self._max_q.append(torch.max(q_values).cpu().data.numpy())
        self._min_q.append(torch.min(q_values).cpu().data.numpy())
        return Action(epsilon_greedy(q_values, epsilon)[0])

    def update_from_experiences(self, experiences):
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
        loss = loss_fn(current_state_q_values, targets)
        self._avg_loss.append(loss.cpu().data.numpy())
        self._take_grad_step(loss)

    def _take_grad_step(self, loss):
        """Try to take a gradient step w.r.t. loss.

        If the gradient is finite, takes a step. Otherwise, does nothing.

        Args:
            loss (Variable): a differentiable scalar variable
            max_grad_norm (float): gradient norm is clipped to this value.

        Returns:
            finite_grads (bool): True if the gradient was finite.
            grad_norm (float): norm of the gradient (BEFORE clipping)
        """
        self._optimizer.zero_grad()
        loss.backward()

        # clip according to the max allowed grad norm
        grad_norm = clip_grad_norm(
                self.parameters(), self._grad_clip_norm, norm_type=2)
        # (this returns the gradient norm BEFORE clipping)

        self._optimizer.step()

        return grad_norm


    def sync_target(self):
        """Syncs the target Q values with the current Q values"""
        self._target_Q.load_state_dict(self._Q.state_dict())

    @property
    def stats(self):
        return {"Epsilon": self._epsilon,
                "Max Q": sum(self._max_q) / len(self._max_q),
                "Min Q": sum(self._min_q) / len(self._min_q),
                "Avg Loss": sum(self._avg_loss) / len(self._avg_loss)}


class EnsembleDQNAgent(Agent):
    """Set of DQNAgents. Each episode is rolled out with a random
    agent. Updating updates each of the ensembles.
    """
    def __init__(self, dqn_agents):
        super(EnsembleDQNAgent, self).__init__()
        self._agents = dqn_agents
        self._chosen_index = 0

    def reset(self):
        self._chosen_index = random.randint(0, len(self._agents) - 1)

    def act(self, state, test=False):
        return self._agents[self._chosen_index].act(state, test)

    def update_from_experiences(self, experiences):
        for agent in self._agents:
            agent.update_from_experiences(experiences)

    def sync_target(self):
        for agent in self._agents:
            agent.sync_target()

    @property
    def stats(self):
        stats = {}
        for k in self._agents[0].stats.iterkeys():
            stats[k] = sum(agent.stats[k] for agent in self._agents) / len(self._agents)
        return stats


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


class StructuredStateEmbedder(nn.Module):
    def __init__(self):
        super(StructuredStateEmbedder, self).__init__()
        self._layer1 = nn.Linear(14, 128)
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


class PixelStateEmbedder(nn.Module):
    def __init__(self):
        super(PixelStateEmbedder, self).__init__()

        self._layer1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # Pad so that layer2 outputs 10 x 10 x 64
        self._layer2_pad = nn.ZeroPad2d((1, 2, 1, 2))
        self._layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._layer3 = nn.Conv2d(64, 64, padding=(1, 1),
                                 kernel_size=3, stride=1)
        self._layer4 = nn.Linear(10 * 10 * 64, 512)

    def forward(self, states):
        """Embeds the states

        Args:
            states (FloatTensor): shape (batch_size, 84, 84, 3)

        Returns:
            FloatTensor: (batch_size, 512)
        """
        # Re-arrange to (batch, channels, width, height)
        states = states.transpose(1, 3).transpose(2, 3)

        hidden = F.relu(self._layer1(states))
        hidden = F.relu(self._layer2(self._layer2_pad(hidden)))
        hidden = F.relu(self._layer3(hidden))
        hidden = hidden.view((-1, 10 * 10 * 64))  # flatten
        return F.relu(self._layer4(hidden))

    @property
    def embed_dim(self):
        return 512


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
    max_indices = max_indices.cpu().data.numpy()
    actions = []
    for i in xrange(batch_size):
        if random.random() > epsilon:
            actions.append(int(max_indices[i]))
        else:
            actions.append(random.randint(0, num_actions - 1))
    return actions
