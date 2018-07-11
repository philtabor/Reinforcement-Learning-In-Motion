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
    
    policy = {}
    for state in grid.stateSpace:
        policy[state] = [key for key in grid.actionSpace.keys()]    
        
    # main loop for policy improvement
    """
    stable = False
    while not stable:
        V = evaluatePolicy(grid, V, policy, GAMMA, THETA)
        stable, policy = improvePolicy(grid, V, policy, GAMMA)       
    V = evaluatePolicy(grid, V, policy, GAMMA, THETA)    
    """
    for i in range(2):
        V, policy = iterateValues(grid, V, policy, GAMMA, THETA)
    
    for state in policy:
        print(state, policy[state])
    print()
    
    printV(V, grid)   