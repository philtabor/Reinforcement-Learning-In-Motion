from windygrid import WindyGrid
from utils import printPolicy, printQ
import numpy as np
 
if __name__ == '__main__':
    grid = WindyGrid(6,6, wind=[0, 0, 1, 2, 1, 0])
    GAMMA = 0.9

    Q = {}
    returns = {}
    pairsVisited = {}
    for state in grid.stateSpacePlus:
        for action in grid.possibleActions:
            Q[(state, action)] = 0
            returns[(state,action)] = 0
            pairsVisited[(state,action)] = 0

    policy = {}
    for state in grid.stateSpace:
        policy[state] = np.random.choice(grid.possibleActions)
    
    for i in range(1000000):
        if i % 50000 == 0:
            print('starting episode', i)   
        observation = np.random.choice(grid.stateSpace)
        action = np.random.choice(grid.possibleActions)
        grid.setState(observation)
        observation_, reward, done, info = grid.step(action)
        memory = [(observation, action, reward)]
        steps = 0
        while not done:
            action = policy[observation_]
            steps += 1            
            observation, reward, done, info = grid.step(action)
            if steps > 50 and not done:
                done = True
                reward = -steps
            memory.append((observation_, action, reward))
            observation_ = observation

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
                    values = np.array([Q[(state,a)] for a in grid.possibleActions])
                    best = np.argmax(values)
                    policy[state] = grid.possibleActions[best]
    printQ(Q, grid)
    printPolicy(policy,grid)    