from windygrid import WindyGrid
from utils import printPolicy, printQ, sampleReducedActionSpace
import numpy as np

if __name__ == '__main__':
    grid = WindyGrid(6,6, wind=[0,0,1,2,1,0])
    GAMMA = 0.9

    Q = {}
    C = {}
    for state in grid.stateSpacePlus:
        for action in grid.possibleActions:
            Q[(state,action)] = 0
            C[(state,action)] = 0
    
    targetPolicy = {}
    for state in grid.stateSpace:
        targetPolicy[state] = np.random.choice(grid.possibleActions)

    for i in range(1000):
        if i % 100 == 0:
            print(i)            
        behaviorPolicy = {}
        for state in grid.stateSpace:
            behaviorPolicy[state] = grid.possibleActions
        memory = []
        observation, done = grid.reset()
        steps = 0
        while not done:
            action = np.random.choice(behaviorPolicy[observation])
            observation_, reward, done, info = grid.step(action)
            steps += 1
            if steps > 25:
                done = True
                reward = -steps
            memory.append((observation, action, reward))
            observation = observation_
        memory.append((observation, action, reward))
        
        G = 0
        W = 1
        last = True
        for (state, action, reward) in reversed(memory):            
            if last:
                last = False
            else:
                C[state,action] += W
                Q[state,action] += (W / C[state,action])*(G-Q[state,action])
                prob = 1 if action in targetPolicy[state] else 0
                W *= prob/(1/len(behaviorPolicy[state]))
                if W == 0:
                    break
            G = GAMMA*G + reward
    printQ(Q, grid)