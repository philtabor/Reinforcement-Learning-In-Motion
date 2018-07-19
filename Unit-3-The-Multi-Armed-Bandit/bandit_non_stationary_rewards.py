import numpy as np
from matplotlib import pyplot as plt

class Bandit(object):
    def __init__(self, numArms, trueRewards, epsilon, mode):
        self.Q = [0 for i in range(numArms)]
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
       
        self.trueRewards = [reward + 0.1*np.random.randn() for reward in self.trueRewards]       

        return np.random.randn() + self.trueRewards[whichArm]
    
    def updateMean(self, sample):
        whichArm = self.lastAction
        self.N[whichArm] += 1
        if self.mode == 'sample-average':
            self.Q[whichArm] = self.Q[whichArm] + 1.0/self.N[whichArm]*(sample - self.Q[whichArm]) 
        elif self.mode == 'constant':
            self.Q[whichArm] = self.Q[whichArm] + 0.1*(sample - self.Q[whichArm])         

def simulate(numArms, epsilon, numPulls, mode):
    rewardHistory = np.zeros(numPulls)
    for j in range(2000):
        if j % 100 == 0:
            print(j)
        rewards = [np.random.randn() for _ in range(numActions)]
        bandit = Bandit(numArms, rewards, epsilon, mode)        
        for i in range(numPulls):        
            reward = bandit.pull()
            bandit.updateMean(reward)
            rewardHistory[i] += reward
    average = rewardHistory / 2000
    return average

if __name__ == '__main__':
    numActions = 5   
    run1 = simulate(numActions, epsilon=0.1, numPulls=10000, mode='sample-average')
    run2 = simulate(numActions, epsilon=0.1, numPulls=10000, mode='constant')
    plt.plot(run1, 'b--', run2, 'r--')
    plt.legend(['sample-average', 'constant alpha'])
    plt.show()
