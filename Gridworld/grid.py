import numpy as np

class GridWorld(object):
    """ Gridworld defined by m x n matrix with
    terminal states at top left corner and bottom right corner.
    State transitions are deterministic; attempting to move
    off the grid leaves the state unchanged, and rewards are -1 on
    each step.

    In this implementation we model the environment as a system of 
    equations to be solved, rather than as a game to be played.
    """
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.grid = np.zeros((m,n))
        self.stateSpace = [i+1 for i in range(self.m*self.n-2)]
        self.stateSpacePlus = [i for i in range(self.m*self.n)]        
        self.actionSpace = {'up': -self.m, 'down': self.m, 
                            'left': -1, 'right': 1}
        self.p = self.initP() # probability functions  

    def initP(self):
        """ construct state transition probabilities for
        use in value function. P(s', r|s, a) is a dictionary
        with keys corresponding to the functional arguments.
        values are either 1 or 0.
        Translations that take agent off grid leave the state unchanged.
        (s', r|s, a)
        (1, -1|1, 'up') = 1
        (1, -1|2, 'left') = 1
        (1, -1|3, 'left') = 0        
        """
        P = {}
        def offGridMove(newState, oldState):
            # if we move into a row not in the grid
            if newState not in self.stateSpacePlus:
                return True
            # if we're trying to wrap around to next row
            elif oldState % self.m == 0 and newState  % self.m == self.m - 1:
                return True
            elif oldState % self.m == self.m - 1 and newState % self.m == 0:
                return True
            else:
                return False

        for state in self.stateSpace:
            for action in self.actionSpace:
                resultingState = state + self.actionSpace[action]
                key = (state, -1, state, action) if offGridMove(resultingState, state) \
                                                 else (resultingState, -1, state, action)
                P[key] = 1
        return P