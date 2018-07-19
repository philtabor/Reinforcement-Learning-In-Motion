from windygrid import WindyGrid
from utils import printPolicy, printQ
import numpy as np
 
if __name__ == '__main__':
    grid = WindyGrid(6,6, wind=0.0)
    GAMMA = 0.9

    Q = {}
    returns = {}
    for state in grid.stateSpacePlus:
        for action in grid.actionSpace.keys():
            Q[(state, action)] = 0
            returns[(state,action)] = []

    policy = {}
    for state in grid.stateSpace:
        policy[state] = grid.possibleActions

    for i in range(10000):
        if i % 1000 == 0:
            print('starting episode', i)   
        observation = np.random.choice(grid.stateSpace)
        action = np.random.choice(grid.possibleActions)   
        grid.setState(observation)
        observation_, reward, done, info = grid.step(action)
        memory = [(observation, action, reward)]
        steps = 0
        while not done:
            action = np.random.choice(policy[observation_])
            steps += 1                
            observation_, reward, done, info = grid.step(action)
            if steps > 100 and not done:
                done = True
                reward = -100                       
            memory.append((observation_, info, reward))

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
        for state in grid.stateSpace:
            values = np.array([Q[(state,a)] for a in grid.possibleActions])                
            best = np.where(values == values.max())[0]
            policy[state] = [grid.possibleActions[k] for k in best]

    printQ(Q, grid)
    printPolicy(policy,grid)