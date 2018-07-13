import numpy as np
from utils import printV

def iterateValues(grid, V, policy, GAMMA, THETA):
    converged = False
    while not converged:
        for state in grid.stateSpace:
            oldV = V[state]
            newV = []            
            for action in grid.actionSpace:
                total = 0 
                grid.setState(state)
                newState, reward, _, _ = grid.step(action)
                key = (newState, reward, state, action) 
                newV.append(grid.p[key]*(reward+GAMMA*V[newState]))                
            newV = np.array(newV)
            bestV = np.where(newV == newV.max())[0]
            bestState = np.random.choice(bestV)
            V[state] = newV[bestState]
            converged = True if np.abs(oldV-V[state]) < THETA else False

    for state in grid.stateSpace:
        actionValues = []
        actions = []
        for action in grid.actionSpace:
            total = 0
            grid.setState(state)
            newState, reward, _, _ = grid.step(action)
            key = (newState, reward, state, action)
            total = grid.p[key]*(reward+GAMMA*V[newState])           
            actionValues.append(total)
            actions.append(action)
        actionValues = np.array(actionValues)
        bestActionIDX = np.where(actionValues == actionValues.max())[0]
        #bestActions = actions[np.random.choice(bestActionIDX)]
        bestActions = actions[bestActionIDX[0]]
        policy[state] = bestActions

    return V, policy