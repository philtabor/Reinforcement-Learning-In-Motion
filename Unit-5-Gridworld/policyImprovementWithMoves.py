import numpy as np

def improvePolicy(grid, V, policy, GAMMA):
    stable = True
    newPolicy = {}
    for state in grid.stateSpace:       
        oldActions = policy[state]                
        value = []
        newAction = []
        for action in policy[state]:
            grid.setState(state)
            weight = 1 / len(policy[state])
            newState, reward, _, _ = grid.step(action)
            key = (newState, reward, state, action)
            value.append(np.round(weight*grid.p[key]*(reward+GAMMA*V[newState]), 2))
            newAction.append(action)
        value = np.array(value)        
        best = np.where(value == value.max())[0]        
        bestActions = [newAction[item] for item in best] 
        newPolicy[state] = bestActions

        if oldActions != bestActions:
            stable = False
        
    return stable, newPolicy