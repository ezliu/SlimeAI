from agent import DQNAgent, RandomAgent
from instance import Instance, ObservationMode
from time import sleep
from replay import ReplayBuffer, Experience
from schedule import LinearSchedule
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm
import collections
import os
import torch.optim as optim
import torch

######################################################
###### Configs
######################################################
# TODO: Move

NUM_LEADERS = 5
LEADER_DIR = "leaders"
GRAVEYARD_DIR = "graveyard"
SAVE_FREQ = 100000
TRAIN_FRAMES = 500000
OBSERVATION_MODE = ObservationMode.RAM

env = Instance(OBSERVATION_MODE)

def take_grad_step(model, loss, max_grad_norm=float('inf')):
    """Try to take a gradient step w.r.t. loss.

    If the gradient is finite, takes a step. Otherwise, does nothing.

    Args:
        loss (Variable): a differentiable scalar variable
        max_grad_norm (float): gradient norm is clipped to this value.

    Returns:
        finite_grads (bool): True if the gradient was finite.
        grad_norm (float): norm of the gradient (BEFORE clipping)
    """
    optimizer.zero_grad()
    loss.backward()

    # clip according to the max allowed grad norm
    grad_norm = clip_grad_norm(model.parameters(), max_grad_norm, norm_type=2)
    # (this returns the gradient norm BEFORE clipping)

    optimizer.step()

    return grad_norm


def evaluate(challenger, leader, num_episodes=10):
    """Rolls out num_episodes episodes and returns the average score of the
    challenger against the leaders.

    Arg:
        challenger (EnsembleDQNAgent)
        leader (EnsembleDQNAgent)
        num_episodes (int)
    """
    episode_rewards = []
    for _ in xrange(num_episodes):
        episode_reward = 0.
        states = env.reset()
        challenger.reset()
        leader.reset()
        while True:
            action1 = challenger.act(states[0], True)
            action2 = leader.act(states[1], True)
            next_states, reward, done = env.step(action1, action2)
            episode_reward += reward
            states = next_states

            if done:
                break
        episode_rewards.append(episode_reward)
    return sum(episode_rewards) / len(episode_rewards)


def purge_round():
    candidate_leaders_map = {}  # {filename --> agent}

    # Load in all of the leaders
    for leader_checkpoint in os.listdir(LEADER_DIR):
        path = os.path.join(LEADER_DIR, leader_checkpoint)
        candidate_leader = DQNAgent(6, LinearSchedule(0.05, 0.05, 1))
        candidate_leader.load_state_dict(torch.load(path))
        candidate_leaders_map[leader_checkpoint] = candidate_leader

    candidate_scores = []  # list[(filename, score)]
    filenames, candidate_leaders = zip(*candidate_leaders_map.items())
    for i, (filename, candidate_leader) in enumerate(candidate_leaders):
        leaders = EnsembleDQNAgent(
                candidate_leaders[:i] + candidate_leaders[i + 1:])
        candidate_scores.append((filename, evaluate(candidate_leader, leaders)))
    sorted_scores = sorted(candidate_scores, key=lambda x:x[1])

    for filename, score in sorted_scores[NUM_LEADERS:]:
        print "PURGING ({}, {})".format(filename, score)
        leader_path = os.path.join(LEADER_DIR, filename)
        graveyard_path = os.path.join(GRAVEYARD_DIR, filename)
        os.rename(path, graveyard_path)


def challenger_round():
    challenger = DQNAgent(6, LinearSchedule(1., 0.1, 500000))
    leader = DQNAgent(6, LinearSchedule(0.1, 0.1, 500000))
    leader_checkpoints = os.listdir(LEADER_DIR)
    if len(leader_checkpoints) > 0:
        filename = os.path.join(LEADER_DIR, leader_checkpoints[0])
        print "LOADING CHECKPOINT: {}".format(filename)
        challenger.load_state_dict(torch.load(filename))
        leader.load_state_dict(torch.load(filename))
    optimizer = optim.Adam(challenger.parameters(), lr=0.00025)
    challenger = EnsembleDQNAgent([challenger])
    leader = EnsembleDQNAgent([leader])
    replay_buffer = ReplayBuffer(1000000)

    rewards = collections.deque(maxlen=1000)
    frames = 0  # number of training frames seen
    episodes = 0  # number of training episodes that have been played
    with tqdm(total=TRAIN_FRAMES) as progress:
        # Each loop completes a single episode
        while frames < 100000000:
            states = env.reset()
            challenger.reset()
            leader.reset()
            episode_reward = 0.
            episode_frames = 0
            # Each loop completes a single step, duplicates _evaluate() to
            # update at the appropriate frame #s
            for _ in xrange(10000):
                frames += 1
                episode_frames += 1
                action1 = challenger.act(states[0])
                action2 = leader.act(states[1])
                next_states, reward, done = env.step(action1, action2)
                episode_reward += reward

                # NOTE: state and next_state are LazyFrames and must be
                # converted to np.arrays
                replay_buffer.add(
                    Experience(states[0], action1._action_index, reward, next_states[0], done))
                states = next_states

                if len(replay_buffer) > 50000 and \
                        frames % 4 == 0:
                    experiences = replay_buffer.sample(32)
                    challenger.update_from_experiences(experiences, take_grad_step)

                if frames % 10000 == 0:
                    challenger.sync_target()

                if frames % SAVE_FREQ == 0:
                    print "SAVING CHECKPOINT"
                    torch.save(challenger.state_dict(), LEADER_DIR + "/test")

                if done:
                    break

            if episodes % 100 == 0:
                print "Evaluation: {}".format(evaluate(challenger, leader, 5))
            print "Episode reward: {}".format(episode_reward)
            episodes += 1
            rewards.append(episode_reward)
            stats = {}
            stats["Avg Episode Reward"] = float(sum(rewards)) / len(rewards)
            stats["Num Episodes"] = episodes
            stats["Replay Buffer Size"] = len(replay_buffer)
            progress.set_postfix(stats, refresh=False)
            progress.update(episode_frames)
            episode_frames = 0

challenger_round()
