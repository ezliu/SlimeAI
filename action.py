class Action(object):
    def __init__(self, action_index):
        """Converts an int action_index to the appropriate type of Action.

        Args:
            action_index (int): an int 0 <= action_index <= 5
                0 ==> go away from net
                1 ==> jump away from net
                2 ==> jump
                3 ==> jump toward net
                4 ==> go toward net
                5 ==> no-op
        """
        self._action_index = action_index
        self._P1_KEYLIST = [
                ([1, 0, 0], "left"),
                ([1, 1, 0], "left jump"),
                ([0, 1, 0], "jump"),
                ([0, 1, 1], "jump right"),
                ([0, 0, 1], "right"),
                ([0, 0, 0], "no-op")
        ]


    def to_list(self, first_player):
        """Converts the Action to an list to feed to JS. If first_player=True,
        returns the the array in player one's reference, otherwise returns in
        player two's reference.

        Args:
            first_player (bool): True ==> player 1 reference

        Returns:
            keylist (list[int]): keylist[0] = left, keylist[1] = up,
                keylist[2] = right, 1 if pressed, 0 if not pressed
        """
        keylist, _ = self._P1_KEYLIST[self._action_index]

        # Convert to p2 reference
        if not first_player:
            left = keylist[2]
            keylist[2] = keylist[0]
            keylist[0] = left
        return keylist

    def __str__(self):
        _, human_readable = self._P1_KEYLIST[self._action_index]
        return "Action({})".format(human_readable)
    __repr__ = __str__
