from grid import GridWorld
from policyEvaluation import evaluatePolicy
from policyImprovement import improvePolicy
from valueIteration import iterateValues
from utils import printV

if __name__ == '__main__':
    grid = GridWorld(4,4)
    THETA = 10e-6
    GAMMA = 1.0
    
    # initialize V(s)
    V = {}
    for state in grid.stateSpacePlus:        
        V[state] = 0
    
    # Initialize policy
    policy = {}
    for state in grid.stateSpace:
        policy[state] = [key for key in grid.actionSpace.keys()]    
    
    # Initial policy evaluation
    V = evaluatePolicy(grid, V, policy, GAMMA, THETA)
    printV(V, grid)

    # Iterative policy evaluation
    stable = False
    while not stable:
        V = evaluatePolicy(grid, V, policy, GAMMA, THETA)
        printV(V, grid)
        stable, policy = improvePolicy(grid, V, policy, GAMMA)
    
    printV(V, grid)
    for state in policy:
        print(state, policy[state])       
    print('\n---------------\n')
    
    # initialize V(s)
    V = {}
    for state in grid.stateSpacePlus:        
        V[state] = 0
    # Reinitialize policy
    policy = {}
    for state in grid.stateSpace:
        policy[state] = [key for key in grid.actionSpace.keys()] 

    # 2 round of value iteration ftw
    for i in range(2):
        V, policy = iterateValues(grid, V, policy, GAMMA, THETA)
    
    printV(V, grid)
    
    for state in policy:
        print(state, policy[state])
    print()   

    