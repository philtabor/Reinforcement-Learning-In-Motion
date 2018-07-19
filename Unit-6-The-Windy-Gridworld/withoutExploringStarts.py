from windygrid import WindyGrid
from utils import printPolicy, printQ
import numpy as np

if __name__ == '__main__':
    grid = WindyGrid(6,6, wind=0.1)
    GAMMA = 0.9
    eps = 0.2

    Q = {}
    returns = {}
    for state in grid.stateSpacePlus:
        for action in grid.actionSpace.keys():
            Q[(state, action)] = 0
            returns[(state,action)] = []

    policy = {}
    for state in grid.stateSpace:
        policy[state] = np.random.choice(grid.possibleActions)

    for i in range(10000):
        if i % 2500 == 0:
            print('starting episode', i)   
        observation, done = grid.reset()       
        memory = []
        steps = 0
        while not done:
            rand = np.random.random()            
            action = policy[observation] if rand < (1 - eps) else np.random.choice(grid.possibleActions)
            observation_, reward, done, info = grid.step(action)
            memory.append((observation, action, reward))
            observation = observation_

        G = 0
        statesAndActions = []
        last = True # start at t = T - 1
        for state, action, reward in reversed(memory):                        
            G = GAMMA*G + reward
            if last:
                last = False
            else:                
                if (state, action) not in statesAndActions:
                    returns[(state,action)].append(G)
                    Q[(state,action)] = np.round(np.mean(returns[state,action]),2)
                    statesAndActions.append((state,action))
                    values = np.array([Q[(state,a)] for a in grid.possibleActions])
                    best = np.where(values == values.max())[0]
                    best = np.random.choice(best)
                    policy[state] = grid.possibleActions[best]          
    printPolicy(policy,grid)