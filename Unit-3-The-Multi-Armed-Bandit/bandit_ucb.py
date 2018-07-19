import numpy as np
from matplotlib import pyplot as plt

class Bandit(object):
    def __init__(self, numArms, trueRewards, epsilon, C, initialQ, mode):
        self.Q = [initialQ for _ in range(numArms)]
        self.N = [0 for _ in range(numArms)]
        self.numArms = numArms
        self.epsilon = epsilon
        self.trueRewards = trueRewards
        self.lastAction = None
        self.mode = mode
        self.steps = 0
        self.C = C

    def pull(self):
        rand = np.random.random()
        if self.C == 0: # vanilla epsilon-greedy        
            if rand <= self.epsilon:           
                whichArm = np.random.choice(self.numArms)
            elif rand > self.epsilon:
                a = np.array([approx for approx in self.Q])
                whichArm = np.random.choice(np.where(a == a.max())[0]) 
                        
        elif self.C > 0: # UCB
            a = np.zeros(self.numArms)
            for idx, approx in enumerate(self.Q):      
                if self.N[idx] != 0:          
                    a[idx] = approx + self.C * np.sqrt(np.log(self.steps)/self.N[idx])
                elif self.N[idx] == 0:
                    whichArm = idx
                    break
            else:
                whichArm = np.random.choice(np.where(a == a.max())[0])            
                    
        self.lastAction = whichArm       
        self.steps += 1       
        return np.random.randn() + self.trueRewards[whichArm]
    
    def updateMean(self, sample):
        whichArm = self.lastAction
        self.N[whichArm] += 1
        if self.mode == 'sample-average':
            self.Q[whichArm] = self.Q[whichArm] + 1.0/self.N[whichArm]*(sample - self.Q[whichArm]) 
        elif self.mode == 'constant':
            self.Q[whichArm] = self.Q[whichArm] + 0.1*(sample - self.Q[whichArm])         

def simulate(numArms, epsilon, numPulls, C, initialQ, mode):
    rewardHistory = np.zeros(numPulls)
    for j in range(2000):
        rewards = [np.random.randn() for _ in range(numActions)]
        bandit = Bandit(numArms, rewards, epsilon, C, initialQ, mode)         
        if j % 200 == 0:
            print(j)
        for i in range(numPulls):        
            reward = bandit.pull()
            bandit.updateMean(reward)
            rewardHistory[i] += reward
   
    average = rewardHistory / 2000
    return average

if __name__ == '__main__':
    numActions = 10    
    run1 = simulate(numActions, epsilon=0.1, numPulls=1000, C=0, initialQ=0, mode='constant')
    run2 = simulate(numActions, epsilon=0.0, numPulls=1000, C=2, initialQ=0, mode='constant')    
    plt.plot(run1, 'b--', run2, 'r--')
    plt.legend(['Epsilon greedy', 'UCB C = 2'])
    plt.show()
