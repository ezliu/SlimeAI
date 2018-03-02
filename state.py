import copy


class State(object):
    def __init__(self, player1, player2, ball):
        self._player1 = player1
        self._player2 = player2
        self._ball = ball

    def _reflect(self, position_velocity):
        position_velocity["x"] *= -1
        position_velocity["velocityX"] *= -1

    @property
    def p2_state(self):
        player1_reflected = self._reflect(copy.copy(self._player1))
        player2_reflected = self._reflect(copy.copy(self._player2))
        ball_reflected = self._reflect(copy.copy(self._ball))
        return (player1_reflected,
                player2_reflected,
                ball_reflected)

    @property
    def p1_state(self):
        return (self._player1, self._player2, self._ball)
