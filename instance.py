import base64
import numpy as np
import os
import random
import scipy.misc
from collections import deque
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
            render = True
            self._k = 4  # Number of frames to stack
            self._frames1 = deque([], maxlen=self._k)
            self._frames2 = deque([], maxlen=self._k)

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
        if self._observation_mode == ObservationMode.PIXEL:
            pixel_state = self._downsample_and_grayscale(self._get_screenshot())
            self._frames1.append(pixel_state)
            self._frames2.append(np.flip(pixel_state, 1))
            #scipy.misc.imsave("tmp-{}.png".format(self._step), pixel_state.squeeze(-1))
            next_states = (LazyFrames(list(self._frames1)),
                           LazyFrames(list(self._frames2)))
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
        screenshot = screenshot.reshape((84, 84, 1))
        #scipy.misc.imsave("tmp1.png", screenshot)
        return screenshot

    def reset(self):
        """Starts a new episode and returns the first state.

        Returns:
            states ((np.array, np.array)): (player 1 state, player 2 state)
        """
        response = self._driver.execute_script(
                'return reset({});'.format(random.random()))
        next_states = State(response)
        next_states = (next_states.p1_state, next_states.p2_state)
        if self._observation_mode == ObservationMode.PIXEL:
            pixel_state = self._downsample_and_grayscale(self._get_screenshot())
            for _ in xrange(self._k):
                self._frames1.append(pixel_state)
                self._frames2.append(np.flip(pixel_state, 1))
            next_states = (LazyFrames(list(self._frames1)),
                           LazyFrames(list(self._frames2)))
        return next_states


# Adapted from OpenAI Gym baselines
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are
        only stored once.  It exists purely to optimize memory usage which can
        be huge for DQN's 1M frames replay buffers.  This object should only be
        converted to numpy array before being passed to the model.  You'd not
        believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=2)
        if dtype is not None:
            out = out.astype(dtype)
        return out
