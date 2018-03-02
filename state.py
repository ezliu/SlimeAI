import copy
import numpy as np


class State(object):
    def __init__(self, response):
        self._player1 = response["player1"]
        self._player2 = response["player2"]
        self._ball = response["ball"]

    def _reflect(self, position_velocity):
        position_velocity[0] *= -1
        position_velocity[2] *= -1
        return position_velocity

    @property
    def p2_state(self):
        player1_reflected = self._reflect(copy.copy(self._player1))
        player2_reflected = self._reflect(copy.copy(self._player2))
        ball_reflected = self._reflect(copy.copy(self._ball))
        return np.concatenate(
                [player2_reflected, player1_reflected, ball_reflected])

    @property
    def p1_state(self):
        return np.concatenate([self._player1, self._player2, self._ball])
