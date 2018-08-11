from windygrid import WindyGrid
from utils import printPolicy, printQ
import numpy as np

if __name__ == '__main__':
    grid = WindyGrid(6,6, wind=[0, 0, 1, 2, 1, 0])
    GAMMA = 0.9
    EPS = 0.1

    Q = {}
    returns = {}
    pairsVisited = {}
    for state in grid.stateSpacePlus:
        for action in grid.actionSpace.keys():
            Q[(state, action)] = 0
            returns[(state,action)] = 0
            pairsVisited[(state,action)] = 0

    policy = {}
    for state in grid.stateSpace:
        policy[state] = np.random.choice(grid.possibleActions)

    for i in range(100000):
        if i % 5000 == 0:
            print('starting episode', i)   
        observation, done = grid.reset()       
        memory = []
        steps = 0
        while not done:
            rand = np.random.random()            
            action = policy[observation]
            observation_, reward, done, info = grid.step(action)
            steps += 1
            if steps > 50 and not done:
                done = True
                reward = -steps
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
                    pairsVisited[(state,action)] += 1
                    returns[(state,action)] += (1 / pairsVisited[(state,action)])*(G-returns[(state,action)])                   
                    Q[(state,action)] = returns[(state,action)]
                    statesAndActions.append((state,action))
                    rand = np.random.random()
                    if rand < 1 - EPS:
                        values = np.array([Q[(state,a)] for a in grid.possibleActions])
                        best = np.random.choice(np.where(values==values.max())[0])
                        policy[state] = grid.possibleActions[best]
                    else:
                        policy[state] = np.random.choice(grid.possibleActions)
    printQ(Q, grid)
    printPolicy(policy,grid)