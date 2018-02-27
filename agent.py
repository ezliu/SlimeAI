import random
from action import Action


class Agent(object):
    def __init__(self):
        pass

    def act(self, state):
        return Action(random.randint(0, 5))

    def update_from_experiences(self, experiences):
        pass
