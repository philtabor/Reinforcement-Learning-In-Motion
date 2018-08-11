import numpy as np

class WindyGrid(object):
    def __init__(self, m, n, wind):
        self.grid = np.zeros((m,n))
        self.m = m
        self.n = n
        self.stateSpace = [i for i in range(self.m*self.n)]        
        self.stateSpace.remove(28)        
        self.stateSpacePlus = [i for i in range(self.m*self.n)]
        self.actionSpace = {'U': -self.m, 'D': self.m, 
                            'L': -1, 'R': 1}
        self.possibleActions = ['U', 'D', 'L', 'R']
        self.agentPosition = 0
        self.wind = wind

    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace 

    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.m
        y = self.agentPosition % self.n
        return x, y
    
    def setState(self, state):
        x, y = self.getAgentRowAndColumn() 
        self.grid[x][y] = 0            
        self.agentPosition = state        
        x, y = self.getAgentRowAndColumn() 
        self.grid[x][y] = 1 

    def offGridMove(self, newState, oldState):
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

    def step(self, action):
        agentX, agentY = self.getAgentRowAndColumn()
        if agentX > 0:
            resultingState = self.agentPosition + self.actionSpace[action] + \
                            self.wind[agentY] * self.actionSpace['U']
        else:
            resultingState = self.agentPosition + self.actionSpace[action]
        reward = -1 if not self.isTerminalState(resultingState) else 0
        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
            return resultingState, reward, self.isTerminalState(resultingState), action
        else:
            return self.agentPosition, reward, self.isTerminalState(self.agentPosition), None

    def reset(self):
        self.agentPosition = 0
        return self.agentPosition, False

    def render(self):
        print('------------------------------------------')
        for row in self.grid:
            for col in row:
                if col == 0:
                    print('-', end='\t')
                elif col == 1:
                    print('X', end='\t')
            print('\n')
        print('------------------------------------------')