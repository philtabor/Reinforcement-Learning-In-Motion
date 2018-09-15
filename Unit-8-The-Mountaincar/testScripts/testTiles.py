import numpy as np

def getBins():
    # construct the asymmetric bins
    posTileWidth = (0.5 + 1.2)/2*0.5
    velTileWidth = (0.07 + 0.07)/2*0.5
    posBins = np.zeros((2, 2))
    velBins = np.zeros((2, 2))
    for i in range(2):
        posBins[i] = np.linspace(-1.2+i*posTileWidth, 0.5+i*posTileWidth/2, 2)
        velBins[i] = np.linspace(-0.07+3*i*velTileWidth, 0.07+3*i*posTileWidth/2, 2)    
    return posBins, velBins    

def tileState(posBins, velBins, obs):
    position, velocity = obs    
    tiledState = np.zeros(2*2*2)
    for row in range(2):
        if position > posBins[row][0] and position < posBins[row][1]:
            if velocity > velBins[row][0] and velocity < velBins[row][1]:
                x = np.digitize(position, posBins[row])
                y = np.digitize(velocity, velBins[row])
                idx = (x+1)*(y+1)+row*4-1
                tiledState[idx] = 1.0
            else:
                break
        else:
            break
    return tiledState

posBins, velBins = getBins()

observation = [-0.553432, 0.03835]
print(observation)
tile = tileState(posBins, velBins, observation)
print(tile)