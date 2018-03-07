import base64
import numpy as np
import os
import random
from cStringIO import StringIO
from PIL import Image
from state import State
from selenium import webdriver


class ObservationMode:
    PIXEL = 0
    RAM = 1


class Instance(object):
    def __init__(self, observation_mode, headless=False, render=False):
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
        self._observation_mode = observation_mode
        if observation_mode == ObservationMode.PIXEL:
            assert render, "Render must be true in pixel mode"

        if render:
            self._driver.execute_script("RENDER_ENV = true;")

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
        next_states = (next_states.p1_state, next_states.p2_state)
        #next_states1 = response["player1"] + response["ball"] + response["player2"]
        #next_states2 = response["player2"] + response["ball"] + response["player1"]
        #print next_states.p1_state
        #print next_states.p2_state
        #pixel_state = self._get_screenshot()
        #self._downsample_and_grayscale(pixel_state)
        if self._observation_mode == ObservationMode.PIXEL:
            pixel_state = self._get_screenshot()
            next_states = (pixel_state, np.flip(pixel_state, 1))
        return next_states, response["reward"], \
                response["done"]

    def _get_screenshot(self):
        png_data = base64.b64decode(self._driver.execute_script("return getPixels();"))
        pil_image = Image.open(StringIO(png_data)).convert("RGB")
        np_image = np.array(pil_image).astype(np.float32)
        return np_image

    def _downsample_and_grayscale(self, screenshot):
        # screenshot is (375, 750, 3)
        screenshot = np.dot(
                screenshot.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        screenshot = np.array(Image.fromarray(screenshot).resize((84, 84),
            resample=Image.BILINEAR), dtype=np.float32)
        screenshot = screenshot.reshape((84, 84))
        import scipy.misc
        scipy.misc.imsave("tmp1.png", screenshot)
        assert False
        return screenshot

    def reset(self):
        """Starts a new episode and returns the first state.

        Returns:
            states ((np.array, np.array)): (player 1 state, player 2 state)
        """
        response = self._driver.execute_script(
                'return reset({});'.format(random.random()))
        next_states = State(response)
        #next_states1 = response["player1"] + response["ball"] + response["player2"]
        #next_states2 = response["player2"] + response["ball"] + response["player1"]
        return (next_states.p1_state, next_states.p2_state)
