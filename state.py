import copy


class State(object):
    def __init__(self, player1, player2, ball):
        self._player1 = copy.copy(player1)
        self._player2 = copy.copy(player2)
        self._ball = copy.copy(ball)
        self._player1_reflected = self._reflect(copy.copy(player1))
        self._player2_reflected = self._reflect(copy.copy(player2))
        self._ball_reflected = self._reflect(copy.copy(ball))

    def _reflect(self, position_velocity):
        position_velocity["x"] *= -1
        position_velocity["velocityX"] *= -1

    @property
    def p2_state(self):
        return (self._player1_reflected,
                self._player2_reflected,
                self._ball_reflected)

    @property
    def p1_state(self):
        return (self._player1, self._player2, self._ball)
