class Action(object):
    @classmethod
    def from_int(action_index):
        """Converts an int action_index to the appropriate type of Action.

        Args:
            action_index (int): an int 0 <= action_index <= 5
                0 ==> go away from net
                1 ==> jump away from net
                2 ==> jump
                3 ==> jump toward net
                4 ==> go toward net
                5 ==> no-op

        Returns:
            Action
        """
        pass

    def to_list(self, first_player):
        """Converts the Action to an list to feed to JS. If first_player=True,
        returns the the array in player one's reference, otherwise returns in
        player two's reference.

        Args:
            first_player (bool): True ==> player 1 reference

        Returns:
            keylist (list[bool]): keylist[0] = left, keylist[1] = up,
                keylist[2] = right
        """
        pass
