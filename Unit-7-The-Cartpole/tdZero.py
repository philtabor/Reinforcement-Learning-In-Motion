import numpy as np
import matplotlib.pyplot as plt
import gym

def SimplePolicy(state):
    action = 0 if state < 5 else 1
    return action

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ALPHA = 0.1
    GAMMA = 1.0

    #discretize the space
    states = np.linspace(-0.20943951, 0.20943951, 10)

    V = {}
    for state in range(len(states)+1):
        V[state] = 0

    totalRewards = []
    for i in range(1000):
        # cart x position, cart velocity, pole theta, pole velocity
        observation = env.reset()
        done = False
        epRewards = 0    
        while not done:      
            s = int(np.digitize(observation[2], states))
            a = SimplePolicy(s)            
            observation_, reward, done, info = env.step(a)
            epRewards += reward
            s_ = int(np.digitize(observation_[2], states))            
            V[s] = V[s] + ALPHA*(reward + GAMMA*V[s_] - V[s])
            observation = observation_
        totalRewards.append(epRewards)
    
    for state in V:
        print(state, '%.3f' % V[state])
    
    plt.plot(totalRewards)
    plt.show()