import numpy as np

def improvePolicy(grid, V, policy, GAMMA):
    stable = True
    newPolicy = {}
    for state in grid.stateSpace:       
        oldActions = policy[state]                
        value = []
        newAction = []
        for action in policy[state]:
            weight = 1 / len(policy[state])       
            for key in grid.p:
                (newState, reward, oldState, act) = key
                # We're given state and action, want new state and reward
                if oldState == state and act == action:
                    value.append(np.round(weight*grid.p[key]*(reward+GAMMA*V[newState]), 2))
                    newAction.append(action)
        value = np.array(value)        
        best = np.where(value == value.max())[0]        
        bestActions = [newAction[item] for item in best] 
        newPolicy[state] = bestActions

        if oldActions != bestActions:
            stable = False
        
    return stable, newPolicy