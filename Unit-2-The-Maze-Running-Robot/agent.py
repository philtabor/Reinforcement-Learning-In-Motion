import numpy as np 

actionSpace = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1)}

class Agent(object):
    def __init__(self, maze, alpha=0.15, randomFactor=0.2):        
        self.stateHistory = [((0,0), 0)]   
        self.G = {}  # present value of expected future rewards
        self.randomFactor = randomFactor
        self.alpha = alpha   
        self.initReward(maze.allowedStates)  

    def chooseAction(self, state, allowedMoves):           
        maxG = -10e15    
        nextMove = None 
        randomN = np.random.random()
        if randomN < self.randomFactor:
            nextMove = np.random.choice(allowedMoves)          
        else:            
            for action in allowedMoves:
                newState = tuple([sum(x) for x in zip(state, actionSpace[action])])                                                          
                if self.G[newState] >= maxG:
                    maxG = self.G[newState]
                    nextMove = action            
        return nextMove

    def printG(self):
        for i in range(6):            
            for j in range(6):
                if (i,j) in self.G.keys():
                    print('%.6f' % self.G[(i,j)], end='\t')
                else:
                    print('X', end='\t\t')
            print('\n')

    def updateStateHistory(self, state, reward):
        self.stateHistory.append((state, reward))

    def initReward(self, allowedStates):
        for state in allowedStates:     
            self.G[state] = np.random.uniform(low=-1.0, high=-0.1)

    def learn(self):
        target = 0 # we only learn when we beat the maze

        for prev, reward in reversed(self.stateHistory):                    
            self.G[prev] = self.G[prev] + self.alpha * (target - self.G[prev])            
            target += reward

        self.stateHistory = []
        self.randomFactor -= 10e-5