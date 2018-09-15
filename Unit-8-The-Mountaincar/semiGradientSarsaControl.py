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

def tileStateAndAction(posBins, velBins, action, obs, nTiles=8, nLayers=8, nActions=3):
    position, velocity = obs
    # 8 tilings of 8x8 grid, with 3 actions each    
    tiledState = np.zeros(nTiles*nTiles*nTiles*nActions)
    for row in range(nLayers):
        if position > posBins[row][0] and position < posBins[row][nTiles-1]:
            if velocity > velBins[row][0] and velocity < velBins[row][nTiles-1]:
                x = np.digitize(position, posBins[row])
                y = np.digitize(velocity, velBins[row])                
                idx = (x+1)*(y+1)+row*nTiles**2-1+action*nLayers*nTiles**2
                tiledState[idx] = 1.0
            else:
                break
        else:
            break            
    return tiledState

class Model(object):
    def __init__(self, alpha, gamma, nStates, nActions):
        self.ALPHA = alpha
        self.GAMMA = gamma        
        self.weights = np.zeros(nStates*nActions)
        self.actions = [i for i in range(nActions)]

    def calculateQ(self, stateAction):
        return self.weights.dot(stateAction)
    
    def updateWeights(self, R, stateAction, stateAction_, t):
        value = self.calculateQ(stateAction) 
        value_ = self.calculateQ(stateAction_)
        self.weights += self.ALPHA/t*(R + self.GAMMA*value_ - value)*stateAction

if __name__ == '__main__':     
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    GAMMA = 1

    posBins, velBins = getBins()
    numEpisodes = 500
    numRuns = 10
    epLengths = np.zeros((3, numEpisodes, numRuns))
    x = [i for i in range(epLengths.shape[1])]

    for k, ALPHA in enumerate([0.01, 0.1, 0.2]):
        for j in range(numRuns):
            model = Model(ALPHA, GAMMA, 8*8*8, 3)
            EPSILON = 0.1#1.0
            dt = 1.0
            print('alpha', ALPHA, 'run ', j)
            for i in range(numEpisodes):
                if i % 100 == 0:
                    dt += 1
                steps = 0
                done = False
                observation = env.reset()

                rand = np.random.random()
                if rand < 1 - EPSILON:
                    values = []
                    for a in model.actions:
                        sa = tileStateAndAction(posBins, velBins, a, observation)
                        values.append(model.calculateQ(sa))
                    values = np.array(values)
                    best = np.argmax(values)
                    action = model.actions[best]
                else:
                    action = int(np.random.choice(model.actions))

                while not done:
                    stateAction = tileStateAndAction(posBins, velBins, action, observation)
                    observation_, reward, done, _ = env.step(action)
                    steps += 1
                    
                    if done and steps < env._max_episode_steps:
                        q = model.calculateQ(stateAction)
                        model.weights += model.ALPHA/dt*(reward-q)*stateAction
                        break

                    rand = np.random.random()
                    if rand < 1 - EPSILON:
                        values = []
                        for a_ in model.actions:                        
                            sa = tileStateAndAction(posBins, velBins, a_, observation)
                            values.append(model.calculateQ(sa))
                        values = np.array(values)                    
                        best = np.argmax(values)
                        action_ = model.actions[best]
                    else:
                        action_ = int(np.random.choice(model.actions))
                    
                    stateAction_ = tileStateAndAction(posBins, velBins, action_, observation_)
                    model.updateWeights(reward, stateAction, stateAction_, dt)
                    action = action_
                    observation = observation_
                    
                #EPSILON -= 2 / numEpisodes if EPSILON > 0 else 0
                epLengths[k][i][j] = steps

    averaged1 = np.mean(epLengths[0], axis=1)    
    averaged2 = np.mean(epLengths[1], axis=1)
    averaged3 = np.mean(epLengths[2], axis=1)

    plt.plot(averaged1, 'r--')
    plt.plot(averaged2, 'b--')
    plt.plot(averaged3, 'g--')

    plt.legend(('alpha = 0.01', 'alpha = 0.1', 'alpha = 0.2'))
    plt.show()