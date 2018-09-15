import numpy as np
import matplotlib.pyplot as plt
import gym

def getBins(nBins=8, nLayers=8):
    # construct the asymmetric bins
    posTileWidth = (0.5 + 1.2)/nBins*0.5
    velTileWidth = (0.07 + 0.07)/nBins*0.5
    posBins = np.zeros((nLayers,nBins))
    velBins = np.zeros((nLayers,nBins))
    for i in range(nLayers):
        posBins[i] = np.linspace(-1.2+i*posTileWidth, 0.5+i*posTileWidth/2, nBins)
        velBins[i] = np.linspace(-0.07+3*i*velTileWidth, 0.07+3*i*posTileWidth/2, nBins)    
    return posBins, velBins    

def tileState(posBins, velBins,  obs, nTiles=8, nLayers=8):
    position, velocity = obs
    # 8 tilings of 8x8 grid   
    tiledState = np.zeros(nTiles*nTiles*nTiles)
    for row in range(nLayers):
        if position > posBins[row][0] and position < posBins[row][nTiles-1]:
            if velocity > velBins[row][0] and velocity < velBins[row][nTiles-1]:
                x = np.digitize(position, posBins[row])
                y = np.digitize(velocity, velBins[row])                
                idx = (x+1)*(y+1)+row*nTiles**2-1
                tiledState[idx] = 1.0
            else:
                break
        else:
            break            
    return tiledState

class Model(object):
    def __init__(self, alpha, gamma, nStates):
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.weights = np.zeros(nStates)
    
    def calculateV(self, state):     
        v = self.weights.dot(state)
        return v

    def updateWeights(self, R, state, state_, t):        
        value = self.calculateV(state) 
        value_ = self.calculateV(state_)
        self.weights += self.ALPHA/t*(R + self.GAMMA*value_ - value)*state

def policy(velocity):
    # 0 - backward, 1 - none, 2 - forward
    if velocity < 0:
        return 0
    elif velocity >= 0:
        return 2

if __name__ == '__main__':
    GAMMA = 1.0
    env = gym.make('MountainCar-v0')

    posBins, velBins = getBins()
    numEpisodes = 20000
    nearExit = np.zeros((3, int(numEpisodes/1000)))
    leftSide = np.zeros((3, int(numEpisodes/1000)))
    x = [i for i in range(nearExit.shape[1])]

    for k, ALPHA in enumerate([1e-1, 1e-2, 1e-3]):
        model = Model(ALPHA, GAMMA, 8*8*8)
        dt = 1.0
        for i in range(numEpisodes):
            if i % 1000 == 0:
                print('alpha', LR, 'start episode', i)
                idx = i // 1000
                tiledState = tileState(posBins, velBins, (0.43, 0.054))
                nearExit[k][idx] = model.calculateV(tiledState)        
                tiledState = tileState(posBins, velBins, (-1.1, 0.001))
                leftSide[k][idx] = model.calculateV(tiledState)
            if i % 100 == 0:
                dt += 10
            observation = env.reset()
            done = False 
            while not done:
                state = tileState(posBins, velBins, observation)
                action = policy(observation[1])
                observation_, reward, done, _ = env.step(action)
                state_ = tileState(posBins, velBins, observation_)
                model.updateWeights(reward, state, state_, dt)
                observation = observation_ 

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
    plt.legend(('alpha = 1e-1', 'alpha = 1e-2', 'alpha = 1e-3'))
    plt.show()    