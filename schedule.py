class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, total_steps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self._epsilon        = eps_begin
        self._eps_begin      = eps_begin
        self._eps_end        = eps_end
        self._total_steps    = total_steps

    def get_epsilon(self):
        """Updates epsilon by one step and returns the new value

        Returns:
            float
        """
        step_size = float(self._eps_begin - self._eps_end) / self._total_steps
        self._epsilon = max(self._eps_end, self._epsilon - step_size)
        return self._epsilon
