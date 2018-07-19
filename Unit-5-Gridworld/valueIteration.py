import numpy as np
from utils import printV

def iterateValues(grid, V, policy, GAMMA, THETA):
    converged = False
    while not converged:
        DELTA = 0
        for state in grid.stateSpace:
            oldV = V[state]
            newV = []            
            for action in grid.actionSpace:           
                for key in grid.p:
                    (newState, reward, oldState, act) = key
                    if state == oldState and action == act:
                        newV.append(grid.p[key]*(reward+GAMMA*V[newState]))                          
            newV = np.array(newV)
            bestV = np.where(newV == newV.max())[0]
            bestState = np.random.choice(bestV)
            V[state] = newV[bestState]
            DELTA = max(DELTA, np.abs(oldV-V[state]))
            converged = True if DELTA < THETA else False

    for state in grid.stateSpace:
        newValues = []
        actions = []
        for action in grid.actionSpace:
            for key in grid.p:
                (newState, reward, oldState, act) = key
                if state == oldState and action == act:
                    newValues.append(grid.p[key]*(reward+GAMMA*V[newState]))
            actions.append(action)
        newValues = np.array(newValues)
        bestActionIDX = np.where(newValues == newValues.max())[0]        
        #bestActions = actions[np.random.choice(bestActionIDX)]
        bestActions = actions[bestActionIDX[0]]
        policy[state] = bestActions

    return V, policy