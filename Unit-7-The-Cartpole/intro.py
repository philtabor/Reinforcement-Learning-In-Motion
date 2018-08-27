import numpy as np
import matplotlib.pyplot as plt
import gym

def SimplePolicy(state):
    action = 0 if state < 0 else 1
    return action

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    totalRewards = []
    for i in range(1000):
        # cart x position, cart velocity, pole theta, pole velocity
        observation = env.reset()
        done = False
        epRewards = 0    
        while not done:      
            a = SimplePolicy(observation[2])            
            observation_, reward, done, info = env.step(a)
            epRewards += reward
            observation = observation_
            env.render()
        totalRewards.append(epRewards)
    
    plt.plot(totalRewards)
    plt.show()    