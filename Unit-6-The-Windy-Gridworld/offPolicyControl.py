from windygrid import WindyGrid
from utils import printPolicy, printQ, sampleReducedActionSpace
import numpy as np

if __name__ == '__main__':
    grid = WindyGrid(6,6, wind=[0, 0, 1, 2, 1, 0])
    GAMMA = 0.9
    EPS = 0.1

    Q = {}
    C = {}
    for state in grid.stateSpacePlus:
        for action in grid.possibleActions:
            Q[(state,action)] = 0
            C[(state,action)] = 0
    
    targetPolicy = {}
    for state in grid.stateSpace:
        vals = np.array([Q[state, a] for a in grid.possibleActions])
        argmax = np.argmax(vals)
        targetPolicy[state] = grid.possibleActions[argmax]

    for i in range(3000000):
        if i % 100000 == 0:
            print('starting episode', i)                      
        behaviorPolicy = {}
        for state in grid.stateSpace:
            rand = np.random.random()
            if rand < 1 - EPS:
                behaviorPolicy[state] = [targetPolicy[state]]
            else:
                behaviorPolicy[state] = grid.possibleActions
        memory = []
        observation, done = grid.reset()
        steps = 0
        while not done:
            action = np.random.choice(behaviorPolicy[state])
            observation_, reward, done, info = grid.step(action)
            steps += 1             
            if steps > 25 and not done:
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
                C[(state,action)] += W
                Q[(state,action)] += (W / C[(state,action)])*(G-Q[(state,action)])
                vals = np.array([Q[(state, a)] for a in grid.possibleActions])
                argmax = np.argmax(vals)
                targetPolicy[state] = grid.possibleActions[argmax]                
                if action != targetPolicy[state]:
                    break
                if len(behaviorPolicy[state]) == 1:
                    prob = 1 - EPS
                else:
                    prob = EPS / len(behaviorPolicy[state])             
                W *= 1/prob
            G = GAMMA*G + reward

    printPolicy(targetPolicy, grid)