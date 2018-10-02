import numpy as np
from util import plotRunningAverage
import gym
from gym import wrappers

def maxAction(Q, state):    
    values = np.array([Q[state,a] for a in range(2)])
    action = np.argmax(values)
    return action

#discretize the spaces
poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 10)
poleThetaVelSpace = np.linspace(-4, 4, 10)
cartPosSpace = np.linspace(-2.4, 2.4, 10)
cartVelSpace = np.linspace(-4, 4, 10)

def getState(observation):
    cartX, cartXdot, cartTheta, cartThetadot = observation
    cartX = int(np.digitize(cartX, cartPosSpace))
    cartXdot = int(np.digitize(cartXdot, cartVelSpace))
    cartTheta = int(np.digitize(cartTheta, poleThetaSpace))
    cartThetadot = int(np.digitize(cartThetadot, poleThetaVelSpace))

    return (cartX, cartXdot, cartTheta, cartThetadot)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # model hyperparameters
    ALPHA = 0.1
    GAMMA = 1.0    
    EPS = 1.0

    #construct state space
    states = []
    for i in range(len(cartPosSpace)+1):
        for j in range(len(cartVelSpace)+1):
            for k in range(len(poleThetaSpace)+1):
                for l in range(len(poleThetaVelSpace)+1):
                    states.append((i,j,k,l))

    Q = {}
    for state in states:
        for action in range(2):
            Q[state, action] = 0

    numGames = 50000
    totalRewards = np.zeros(numGames)
    for i in range(numGames):
        if i % 5000 == 0:
            print('starting game', i)
        # cart x position, cart velocity, pole theta, pole velocity
        observation = env.reset()        
        state = getState(observation)
        rand = np.random.random()
        action = maxAction(Q, state) if rand < (1-EPS) else env.action_space.sample()
        done = False
        epRewards = 0
        while not done:
            observation_, reward, done, info = env.step(action)
            epRewards += reward
            state_ = getState(observation_)
            rand = np.random.random()
            action_ = maxAction(Q, state_) if rand < (1-EPS) else env.action_space.sample()            
            Q[state,action] = Q[state,action] + ALPHA*(reward + GAMMA*Q[state_,action_] - Q[state,action])
            state, action = state_, action_            
        if EPS - 2 / numGames > 0:
            EPS -= 2 / numGames
        else:
            EPS = 0
        totalRewards[i] = epRewards
    plotRunningAverage(totalRewards)