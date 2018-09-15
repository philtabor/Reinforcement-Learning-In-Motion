import numpy as np
import gym
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    numGames = 1000
    rewards = np.zeros(numGames)
    for i in range(numGames):
        observation = env.reset()
        done = False
        epRewards = 0
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            epRewards += reward
            #env.render()
        rewards[i] = epRewards

    plt.plot(rewards)
    plt.show()