from agent import Agent
from instance import Instance, ObservationMode
from time import sleep

env = Instance(ObservationMode.RAM)
p1 = Agent(6)
p2 = Agent(6)

for _ in xrange(100):
    states = env.reset()
    for _ in xrange(10000):
        action1 = p1.act(states[0])
        action2 = p2.act(states[1])
        next_states, reward, done = env.step(action1, action2)
        states = next_states
        if done:
            break
