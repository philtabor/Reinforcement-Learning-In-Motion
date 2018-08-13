from windygrid import WindyGrid
import numpy as np
from utils import printV

if __name__ == '__main__':
    grid = WindyGrid(6,6, wind=[0, 0, 1, 2, 1, 0])
    GAMMA = 0.9

    policy = {}
    for state in grid.stateSpace:
        policy[state] = grid.possibleActions

    V = {}
    for state in grid.stateSpacePlus:
        V[state] = 0

    
    returns = {}
    for state in grid.stateSpace:
        returns[state] = []
    
    for i in range(500):        
        observation, done = grid.reset()
        memory = []
        statesReturns = []
        if i % 50 == 0:
            print('starting episode', i)
        while not done:
            # attempt to follow the policy
            action = np.random.choice(policy[observation])            
            observation_, reward, done, info = grid.step(action)
            memory.append((observation, action, reward))
            observation = observation_
        
        G = 0
        
        last = True
        for state, action, reward in reversed(memory): 
            if last:
                last = False
            else:
                statesReturns.append((state,G))
            G = GAMMA*G + reward

        statesReturns.reverse()
        statesVisited = []
        for state, G in statesReturns:
            if state not in statesVisited:
                returns[state].append(G)
                V[state] = np.mean(returns[state])
                statesVisited.append(state)
    printV(V, grid)