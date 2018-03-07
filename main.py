from agent import DQNAgent, RandomAgent, EnsembleDQNAgent, HumanAgent
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

######################################################
###### Configs
######################################################
# TODO: Move

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

env = Instance(OBSERVATION_MODE)

def evaluate(challenger, leader, num_episodes=10):
    """Rolls out num_episodes episodes and returns the average score of the
    challenger against the leaders.

    Arg:
        challenger (EnsembleDQNAgent)
        leader (EnsembleDQNAgent)
        num_episodes (int)
    """
    episode_rewards = []
    for _ in tqdm(xrange(num_episodes), desc="Evaluating"):
        episode_reward = 0.
        states = env.reset()
        challenger.reset()
        leader.reset()
        for _ in xrange(MAX_EPISODE_LENGTH):
            action1 = challenger.act(states[0], True)
            action2 = leader.act(states[1], True)
            next_states, reward, done = env.step(action1, action2, True)
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
        candidate_leader = DQNAgent(
                6, LinearSchedule(0.05, 0.05, 1), lr=LR, max_grad_norm=GRAD_CLIP_NORM, name=leader_checkpoint)
        candidate_leader.load_state_dict(torch.load(path))
        candidate_leaders_map[leader_checkpoint] = candidate_leader

    candidate_scores = []  # list[(filename, score)]
    filenames, candidate_leaders = zip(*candidate_leaders_map.items())
    for i, (filename, candidate_leader) in enumerate(
            zip(filenames, candidate_leaders)):
        print "EVALUATING {}".format(candidate_leader.name)
        leaders = EnsembleDQNAgent(
                candidate_leaders[:i] + candidate_leaders[i + 1:])
        candidate_scores.append(
            (filename,
             evaluate(candidate_leader, leaders, EPISODES_EVALUATE_PURGE)))
    sorted_scores = sorted(candidate_scores, key=lambda x:x[1])

    for filename, score in sorted_scores[NUM_LEADERS:]:
        print "PURGING ({}, {})".format(filename, score)
        leader_path = os.path.join(LEADER_DIR, filename)
        graveyard_path = os.path.join(GRAVEYARD_DIR, filename)
        os.rename(leader_path, graveyard_path)


def challenger_round():
    #challengers = []
    leaders = []
    leader_checkpoints = os.listdir(LEADER_DIR)
    #challenger_parameters = []
    for i in xrange(NUM_LEADERS):
        #challenger = DQNAgent(6, LinearSchedule(EPS_START, EPS_END, TRAIN_FRAMES))
        if i == 0:
            leader = DQNAgent(6, LinearSchedule(0.01, 0.01, 500000))
            leader_path = os.path.join(LEADER_DIR, leader_checkpoints[i])
            print "LOADING CHECKPOINT: {}".format(leader_path)
            #challenger.load_state_dict(torch.load(leader_path))
            leader.load_state_dict(torch.load(leader_path))
        #else:
        #    leader = RandomAgent(6)
        #    print "INITIALIZING NEW CHALLENGER AND LEADER"
        #challenger_parameters += challenger.parameters()
        #challengers.append(challenger)
        leaders.append(leader)

    challenger = HumanAgent(6)
    #challenger = DQNAgent(
    #        6, LinearSchedule(EPS_START, EPS_END, TRAIN_FRAMES),
    #        lr=LR, max_grad_norm=GRAD_CLIP_NORM)
    #leader = DQNAgent(6, LinearSchedule(0.01, 0.01, 500000))
    #leader_checkpoints = os.listdir(LEADER_DIR)
    #leader_path = os.path.join(LEADER_DIR, leader_checkpoints[1])
    #print "LOADING CHECKPOINT: {}".format(leader_path)
    ##challenger.load_state_dict(torch.load(leader_path))
    #leader.load_state_dict(torch.load(leader_path))
    #challenger = EnsembleDQNAgent(challengers)
    leader = EnsembleDQNAgent(leaders)
    replay_buffer = ReplayBuffer(1000000)
    rewards = collections.deque(maxlen=1000)
    frames = 0  # number of training frames seen
    episodes = 0  # number of training episodes that have been played
    with tqdm(total=TRAIN_FRAMES) as progress:
        # Each loop completes a single episode
        while frames < TRAIN_FRAMES:
            states = env.reset()
            #sleep(0.5)
            challenger.reset()
            leader.reset()
            episode_reward = 0.
            episode_frames = 0
            # Each loop completes a single step, duplicates _evaluate() to
            # update at the appropriate frame #s
            for _ in xrange(MAX_EPISODE_LENGTH):
                frames += 1
                episode_frames += 1
                action1 = challenger.act(states[0])
                action2 = leader.act(states[1])
                next_states, reward, done = env.step(action1, action2, True)
                sleep(0.02)
                episode_reward += reward

                # NOTE: state and next_state are LazyFrames and must be
                # converted to np.arrays
                replay_buffer.add(
                    Experience(states[0], action1._action_index, reward,
                               next_states[0], done))
                states = next_states

                if len(replay_buffer) > 50000 and \
                        frames % 4 == 0:
                    experiences = replay_buffer.sample(32)
                    challenger.update_from_experiences(experiences)

                if frames % 10000 == 0:
                    challenger.sync_target()

                if frames % SAVE_FREQ == 0:
                    # TODO: Don't access internals
                    #for agent in challenger._agents:
                    #    path = os.path.join(
                    #            LEADER_DIR, agent.name + "-{}".format(frames))
                    #    print "SAVING CHECKPOINT TO: {}".format(path)
                    #    torch.save(agent.state_dict(), path)
                    path = os.path.join(
                            LEADER_DIR, challenger.name + "-{}".format(frames))
                    torch.save(challenger.state_dict(), path)

                if frames >= TRAIN_FRAMES:
                    break

                if done:
                    break

            if episodes % 300 == 0:
                print "Evaluation: {}".format(
                        evaluate(challenger, leader, EPISODES_EVALUATE_TRAIN))
            print "Episode reward: {}".format(episode_reward)
            episodes += 1
            rewards.append(episode_reward)
            stats = challenger.stats
            stats["Avg Episode Reward"] = float(sum(rewards)) / len(rewards)
            stats["Num Episodes"] = episodes
            stats["Replay Buffer Size"] = len(replay_buffer)
            progress.set_postfix(stats, refresh=False)
            progress.update(episode_frames)
            episode_frames = 0

while True:
    challenger_round()
    purge_round()
