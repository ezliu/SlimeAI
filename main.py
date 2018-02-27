from agent import Agent
from instance import Instance, ObservationMode
from time import sleep
from replay import ReplayBuffer, Experience
from tqdm import tqdm
import collections

env = Instance(ObservationMode.RAM)
p1 = Agent(6)
p2 = Agent(6)
replay_buffer = ReplayBuffer(1000000)

rewards = collections.deque(maxlen=1000)
frames = 0  # number of training frames seen
episodes = 0  # number of training episodes that have been played
with tqdm(total=100000000) as progress:
    # Each loop completes a single episode
    while frames < 100000000:
        states = env.reset()
        episode_reward = 0.
        episode_frames = 0
        # Each loop completes a single step, duplicates _evaluate() to
        # update at the appropriate frame #s
        for _ in xrange(10000):
            frames += 1
            episode_frames += 1
            action1 = p1.act(states[0])
            action2 = p2.act(states[1])
            next_states, reward, done = env.step(action1, action2)
            episode_reward += reward

            # NOTE: state and next_state are LazyFrames and must be
            # converted to np.arrays
            replay_buffer.add(
                Experience(states[0], action1, reward, next_states[0], done))
            states = next_states

            if len(replay_buffer) > 50000 and \
                    frames % 4 == 0:
                experiences = replay_buffer.sample(32)
                p1.update_from_experiences(experiences, take_grad_step)

            if frames % 100 == 0:
                p1.sync_target()

            if done:
                break

        episodes += 1
        rewards.append(episode_reward)
        stats = {}
        stats["Avg Episode Reward"] = float(sum(rewards)) / len(rewards)
        stats["Num Episodes"] = episodes
        progress.set_postfix(stats, refresh=False)
        progress.update(episode_frames)
        episode_frames = 0
