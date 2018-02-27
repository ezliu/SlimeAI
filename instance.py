class Instance(object):
    def __init__(self, mode, headless=False):
        """
        Args:
            observation_mode (string): either pixel or ram
            headless (bool): True runs Chrome in headless mode
        """
        pass

    def step(self, action1, action2):
        """Takes an action, returns the next state.
        
        Args:
            action (Action)
            agent (int): 0 or 1

        Returns:
            next_states ((np.array, np.array)): next_states[0] is next state in
                reference frame of player 1, next_state[1] is next state in
                reference frame of player 2
            reward (float): in reference frame of player 1
                (reward for player 2 = -reward for player 1)
            done (bool): True if the episode is over
        """
        pass

    def reset(self):
        """Starts a new episode and returns the first state.
        
        Returns:
            states ((np.array, np.array)): (player 1 state, player 2 state)
        """
        pass
