from agent import DQNAgent, RandomAgent, EnsembleDQNAgent, JavaScriptAgent
from instance import Instance, ObservationMode
from time import sleep
from replay import ReplayBuffer, Experience
from schedule import LinearSchedule
from tqdm import tqdm
import collections
import numpy as np
import os
import random
import torch.optim as optim
import torch
import sys

######################################################
###### Configs
######################################################

NUM_LEADERS = 1
GRAD_CLIP_NORM = 10.
LEADER_DIR = "leaders"
GRAVEYARD_DIR = "graveyard"
SAVE_FREQ = 100000
TRAIN_FRAMES = 500000
EPISODES_EVALUATE_TRAIN = 1
EPISODES_EVALUATE_PURGE = 100
MAX_EPISODE_LENGTH = 1000
EPS_START = 0.5
EPS_END = 0.1
LR = 0.00025
OBSERVATION_MODE = ObservationMode.RAM
SEED = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def challenger_round():
    right = JavaScriptAgent(6)

    left = DQNAgent(6, LinearSchedule(0.0001, 0.0001, 500000))
    leader_checkpoints = os.listdir(LEADER_DIR)
    leader_path = os.path.join(LEADER_DIR, leader_checkpoints[0])
    print "LOADING CHECKPOINT: {}".format(leader_path)

    left.load_state_dict(torch.load(leader_path))

    replay_buffer = ReplayBuffer(1000000)
    rewards = collections.deque(maxlen=1000)
    frames = 0  # number of training frames seen
    episodes = 0  # number of training episodes that have been played

    with tqdm(total=TRAIN_FRAMES) as progress:
        # Each loop completes a single episode
        while frames < TRAIN_FRAMES:
            states = env.reset()
            # sleep(0.5)
            right.reset()
            left.reset()
            episode_reward = 0.
            episode_frames = 0
            # Each loop completes a single step, duplicates _evaluate() to
            # update at the appropriate frame #s
            for _ in xrange(MAX_EPISODE_LENGTH):
                frames += 1
                episode_frames += 1
                action1 = left.act(states[0])
                action2 = right.act(states[1])
                next_states, reward, done = env.step(action1, action2)
                sleep(0.1)
                episode_reward += reward

                # NOTE: state and next_state are LazyFrames and must be
                # converted to np.arrays
                replay_buffer.add(
                    Experience(states[0], action1._action_index, reward,
                               next_states[0], done))
                states = next_states

                if done:
                    break


            print "Episode reward: {}".format(episode_reward)
            episodes += 1
            rewards.append(episode_reward)
            stats = left.stats
            stats["Avg Episode Reward"] = float(sum(rewards)) / len(rewards)
            stats["Num Episodes"] = episodes
            stats["Replay Buffer Size"] = len(replay_buffer)
            progress.set_postfix(stats, refresh=False)
            progress.update(episode_frames)
            episode_frames = 0

if len(sys.argv) != 2:
    print "Usage: python jsAgent.py 0/1/2/3"
    print "0 for human"
    print "1 for Easy Hard Coded AI"
    print "2 for Medium Hard Coded AI"
    print "3 for DIfficult Hard Coded AI"
OPPONENT = int(sys.argv[1]) - 1
env = Instance(OBSERVATION_MODE, opponent=OPPONENT)

while True:
    challenger_round()
