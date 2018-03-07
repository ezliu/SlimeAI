import numpy as np
import os
import random
from state import State
from selenium import webdriver


class ObservationMode:
    PIXEL = 0
    RAM = 1


class Instance(object):
    def __init__(self, observation_mode, headless=False, opponent=-1, render=0):
        """
        Args:
            observation_mode (int): See ObservationMode
            headless (bool): True runs Chrome in headless mode
        """
        if "SLIME_URL" not in os.environ:
            assert False, "SLIME_URL" + " environmental variable must be set."
        self._url = os.environ["SLIME_URL"]

        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('headless')
            options.add_argument('disable-gpu')
            options.add_argument('no-sandbox')
        else:
            options.add_argument('app={}'.format(self._url))

        self._driver = webdriver.Chrome(chrome_options=options)
        self._driver.implicitly_wait(5)
        if headless:
            self._driver.get(self._url)

        self._driver.execute_script('return config({}, {});'.format(
            opponent, render))

    # TODO: skip to every 4 frames
    def step(self, action1, action2):
        """Takes an action, returns the next state.

        Args:
            action1 (Action): action from player 1
            action2 (Action): action from player 2

        Returns:
            next_states ((np.array, np.array)): next_states[0] is next state in
                reference frame of player 1, next_state[1] is next state in
                reference frame of player 2
            reward (float): in reference frame of player 1
                (reward for player 2 = -reward for player 1)
            done (bool): True if the episode is over
        """
        # TODO: Max state over last observations
        response = self._driver.execute_script(
                'return step({}, {}, 4);'.format(
                    action1.to_list(True), action2.to_list(False)))
        next_states = State(response)
        #next_states1 = response["player1"] + response["ball"] + response["player2"]
        #next_states2 = response["player2"] + response["ball"] + response["player1"]
        #print next_states.p1_state
        #print next_states.p2_state
        return (next_states.p1_state, next_states.p2_state), response["reward"], \
                response["done"]

    def reset(self):
        """Starts a new episode and returns the first state.

        Returns:
            states ((np.array, np.array)): (player 1 state, player 2 state)
        """
        response = self._driver.execute_script('return reset({});'.format(
            random.random()))
        next_states = State(response)
        #next_states1 = response["player1"] + response["ball"] + response["player2"]
        #next_states2 = response["player2"] + response["ball"] + response["player1"]
        return (next_states.p1_state, next_states.p2_state)
