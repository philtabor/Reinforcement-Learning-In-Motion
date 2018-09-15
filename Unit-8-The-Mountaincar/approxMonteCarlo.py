import numpy as np
import gym
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self, alpha, stateSpace):
        self.ALPHA = alpha
        self.weights = {}
        self.stateSpace = stateSpace
        for state in stateSpace:
            self.weights[state] = 0
    
    def calculateV(self, state):  
        v = self.weights[state]
        return v

    def updateWeights(self, G, state, t):
        value = self.calculateV(state)
        self.weights[state] += self.ALPHA/t*(G - value)

def aggregateState(posBins, velBins, obs):
    pos = int(np.digitize(obs[0], posBins))
    vel = int(np.digitize(obs[1], velBins))
    state = (pos, vel)
    return state

def policy(vel):
    #_, velocity = state
    # 0 - backward, 1 - none, 2 - forward
    if vel < 4: 
        return 0
    elif vel >= 4: 
        return 2
    
if __name__ == '__main__':
    GAMMA = 1.0
    env = gym.make('MountainCar-v0') 
    
    posBins = np.linspace(-1.2, 0.5, 8)
    velBins = np.linspace(-0.07, 0.07, 8)

    stateSpace = []
    for i in range(1,9):
        for j in range(1,9):
            stateSpace.append((i,j))

    numEpisodes = 20000
    nearExit = np.zeros((3, int(numEpisodes/1000)))
    leftSide = np.zeros((3, int(numEpisodes/1000)))
    x = [i for i in range(nearExit.shape[1])]

    for k, LR in enumerate([0.1, 0.01, 0.001]):
        dt = 1.0
        model = Model(LR, stateSpace)        
        for i in range(numEpisodes):
            if i % 1000 == 0:
                print('start episode', i)
                idx = i // 1000
                state = aggregateState(posBins, velBins, (0.43, 0.054))
                nearExit[k][idx] = model.calculateV(state)        
                state = aggregateState(posBins, velBins, (-1.1, 0.001))
                leftSide[k][idx] = model.calculateV(state)            
                dt += 0.1
            observation = env.reset()
            done = False
            memory = [] 

            while not done:
                state = aggregateState(posBins, velBins, observation)
                action = policy(state[1])
                observation_, reward, done, _ = env.step(action)           
                memory.append((state, action, reward))
                observation = observation_ 
            state = aggregateState(posBins, velBins, observation)
            memory.append((state, action, reward))

            G = 0
            last = True
            statesReturns = []        
            for state, action, reward in reversed(memory):
                if last:
                    last = False
                else:
                    statesReturns.append((state, G))
                G = GAMMA*G + reward

            statesReturns.reverse()
            statesVisited = []
            for state, G in statesReturns:                                
                if state not in statesVisited:
                    model.updateWeights(G, state, dt)
                    statesVisited.append(state)

    plt.subplot(221)
    plt.plot(x, nearExit[0], 'r--')
    plt.plot(x, nearExit[1], 'g--')
    plt.plot(x, nearExit[2], 'b--')
    plt.title('near exit, moving right')
    plt.subplot(222)    
    plt.plot(x, leftSide[0], 'r--')
    plt.plot(x, leftSide[1], 'g--')
    plt.plot(x, leftSide[2], 'b--')
    plt.title('left side, moving right')
    plt.legend(('alpha = 0.1', 'alpha = 0.01', 'alpha = 0.001'))
    plt.show()