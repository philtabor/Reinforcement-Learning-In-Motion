import numpy as np
def printV(V, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):            
            state = grid.m * idx + idy 
            print('%.2f' % V[state], end='\t')
        print('\n')
    print('--------------------')

def printPolicy(policy, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):            
            state = grid.m * idx + idy 
            if state != grid.m * grid.n - 1:
                print(policy[state], end='\t')
            else:
                print('\t')
        print('\n')
    print('--------------------')    

def printQ(Q, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):            
            state = grid.m * idx + idy            
            if state != grid.m * grid.n - 1:
                vals = [np.round(Q[state,action], 3) for action in grid.possibleActions]
                print(vals, end='\t')
        print('\n')
    print('--------------------')

def sampleReducedActionSpace(grid, action):
    actions = grid.possibleActions[:]
    actions.remove(action)
    sample = np.random.choice(actions)
    return sample