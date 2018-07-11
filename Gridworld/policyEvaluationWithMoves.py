import numpy as np

def evaluatePolicy(grid, V, policy, GAMMA, THETA):
    # policy evaluation for the random choice in gridworld
    converged = False
    while not converged:
        for state in grid.stateSpace:
            oldV = V[state]
            total = 0
            weight = 1 / len(policy[state])           
            for action in policy[state]:
                grid.setState(state)
                newState, reward, _, _ = grid.step(action)
                key = (newState, reward, state, action)
                total += weight*grid.p[key]*(reward+GAMMA*V[newState])
            V[state] = total
            if np.abs(oldV-V[state]) < THETA:
                converged = True
            else:
                converged = False
    return V