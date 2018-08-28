import numpy as np
import matplotlib.pyplot as plt

def plotRunningAverage(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

def plotRunningAverageComparison(Algo1, Algo2, labels=None):
    N_1 = len(Algo1)
    N_2 = len(Algo2)
    runningAvgAlgo1 = np.empty(N_1)
    runningAvgAlgo2 = np.empty(N_2)
    for t in range(N_1):
        runningAvgAlgo1[t] = np.mean(Algo1[max(0, t-100):(t+1)])
        runningAvgAlgo2[t] = np.mean(Algo2[max(0, t-100):(t+1)])

    plt.plot(runningAvgAlgo1, 'r--')
    plt.plot(runningAvgAlgo2, 'b--')
    plt.title("Running Average")
    if labels:
        plt.legend((labels[0], labels[1]))
    plt.show()    