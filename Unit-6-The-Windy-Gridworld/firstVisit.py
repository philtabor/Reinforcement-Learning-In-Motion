from windygrid import WindyGrid
import numpy as np
from utils import printV

if __name__ == '__main__':
    grid = WindyGrid(6,6, wind=0.1)
    GAMMA = 0.9

    policy = {}
    for state in grid.stateSpace:
        policy[state] = [key for key in grid.actionSpace.keys()]

    V = {}
    for state in grid.stateSpacePlus:
        V[state] = 0

    returns = {}
    for state in grid.stateSpace:
        returns[state] = []

    for i in range(500):        
        observation, done = grid.reset()
        memory = []
        while not done:
            # attempt to follow the policy
            action = np.random.choice(policy[observation])            
            observation_, reward, done, info = grid.step(action)
            memory.append((observation, action, reward))
            observation = observation_
        
        G = 0
        statesVisited = []
        last = True
        for state, action, reward in reversed(memory):
            G = GAMMA*G + reward            
            if last:
                last = False
            else:

                if state not in statesVisited:
                    returns[state].append(G)                
                    V[state] = np.mean(returns[state])                
                statesVisited.append(state)
    
    printV(V, grid)