import numpy as np
from matplotlib import pyplot as plt

class Bandit(object):
    def __init__(self, numArms, trueRewards, epsilon, initialQ, mode):
        self.Q = [initialQ for i in range(numArms)]
        self.N = [0 for i in range(numArms)]
        self.numArms = numArms
        self.epsilon = epsilon
        self.trueRewards = trueRewards
        self.lastAction = None
        self.mode = mode

    def pull(self):
        rand = np.random.random()
        if rand <= self.epsilon:
            whichArm = np.random.choice(self.numArms)
        elif rand > self.epsilon:
            a = np.array([approx for approx in self.Q])
            whichArm = np.random.choice(np.where(a == a.max())[0])
        self.lastAction = whichArm 

        #self.trueRewards = [reward + 0.1*np.random.randn() for reward in self.trueRewards]

        return np.random.randn() + self.trueRewards[whichArm]
    
    def updateMean(self, sample):
        whichArm = self.lastAction
        self.N[whichArm] += 1
        if self.mode == 'sample-average':
            self.Q[whichArm] = self.Q[whichArm] + 1.0/self.N[whichArm]*(sample - self.Q[whichArm]) 
        elif self.mode == 'constant':
            self.Q[whichArm] = self.Q[whichArm] + 0.1*(sample - self.Q[whichArm])         

def simulate(numArms, epsilon, numPulls, initialQ, mode):    
    rewardHistory = np.zeros(numPulls)
    for j in range(2000):
        rewards = [np.random.randn() for _ in range(numActions)]
        bandit = Bandit(numArms, rewards, epsilon, initialQ, mode)
        if j % 200 == 0:
            print(j)
        for i in range(numPulls):        
            reward = bandit.pull()
            bandit.updateMean(reward)
            rewardHistory[i] += reward

    average = rewardHistory / 2000
    return average

if __name__ == '__main__':
    numActions = 5    
    run1 = simulate(numActions, epsilon=0.1, numPulls=1000, initialQ=0, mode='constant')
    run2 = simulate(numActions, epsilon=0.0, numPulls=1000, initialQ=10, mode='constant')
    plt.plot(run1, 'b--', run2, 'r--')
    plt.legend(['Realistic epsilon greedy', 'Optimistic pure greedy'])
    plt.show()