import numpy as np
import gym
import matplotlib.pyplot as plt
from util import plotRunningAverageComparison

def maxActionQ1Q2(Q1, Q2, state):    
    values = np.array([Q1[state,a] + Q2[state,a] for a in range(2)])
    action = np.argmax(values)
    return action

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
    EPS = 0.1

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
    totalRewardsQLearning = np.zeros(numGames)
    for i in range(numGames):
        if i % 5000 == 0:
            print('starting Qlearning game ', i)
        done = False
        epRewards = 0
        observation = env.reset()
        while not done:
            state = getState(observation)
            rand = np.random.random()
            action = maxAction(Q,state) if rand < (1-EPS) else env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            epRewards += reward
            state_ = getState(observation_)
            action_ = maxAction(Q,state_)
            Q[state,action] = Q[state,action] + ALPHA*(reward + GAMMA*Q[state_,action_] - Q[state,action])
            observation = observation_  
        totalRewardsQLearning[i] = epRewards
   
    
    Q1, Q2 = {}, {}
    for state in states:
        for action in range(2):
            Q1[state,action] = 0
            Q2[state,action] = 0
    totalRewardsDQ = np.zeros(numGames)
    for i in range(numGames):
        if i % 5000 == 0:
            print('starting double q learning game ', i)
        done = False
        epRewards = 0
        observation = env.reset()
        while not done:
            state = getState(observation)
            rand = np.random.random()
            action = maxActionQ1Q2(Q1,Q2,state) if rand < (1-EPS) else env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            epRewards += reward
            state_ = getState(observation_)
            rand = np.random.random()
            if rand <= 0.5:
                action_ = maxAction(Q1,state_)
                Q1[state,action] = Q1[state,action] + ALPHA*(reward + GAMMA*Q2[state_,action_] - Q1[state,action])
            elif rand > 0.5:
                action_ = maxAction(Q2,state_)
                Q2[state,action] = Q2[state,action] + ALPHA*(reward + GAMMA*Q1[state_,action_] - Q2[state,action])
            observation = observation_
        totalRewardsDQ[i] = epRewards    
    labels=['Q Learning', 'Double Q Learning']
    plotRunningAverageComparison(totalRewardsQLearning, totalRewardsDQ, labels)